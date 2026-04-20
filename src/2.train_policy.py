import argparse
import datetime
import random
import numpy

from Datasets.get_dataset import *
from getters import find_latest, load_function_encoder, get_coefficients
from neuromancer.modules.activations import activations
from neuromancer.system import Node, System
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

from Callbacks import TensorboardCallback, ProgressBarCallback, ListCallback, EvalCallback
from Policies.Policy import Policy
from getters import get_policy

if __name__ == "__main__":

    # training arguments
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of dynamical systems to train on") 
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train over") 
    parser.add_argument("--horizon", type=int, default=50, help="Prediction horizon")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use.")
    parser.add_argument("--log_dir", type=str, default="logs/policy", )
    parser.add_argument("--n_layers", type=int, default=4,)
    parser.add_argument("--n_hidden", type=int, default=256, )
    parser.add_argument("--fe_load_path", type=str, default="latest", help="Path to the function encoder model. If 'latest', it will find the latest model in the log directory.")
    parser.add_argument("--dataset", type=str, default="VanDerPol", ) 
    parser.add_argument("--policy_type", type=str, default="adaptive", )
    args = parser.parse_args()

    # create a logdir
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = os.path.join(args.log_dir, args.dataset, args.policy_type, f"seed_{args.seed}", datetime)
    
    print(f"Training with arguments: {args}")

    # Set the random seed for reproducibility
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    # Fetch a dataset
    dataset, _ = get_function_encoder_dataset(args)
    
    args.fe_load_path = "logs/function_encoder/Vanderpol/seed_0/2025-10-21_21-08-35"
    print(f"Using latest function encoder model from {args.fe_load_path}")

    # load a model, to be used as dynamics with no gradients.
    model = load_function_encoder(
        load_path=args.fe_load_path,
        device=args.device,
    )

    # get a large set of coefficients from the training dataset
    coefficients, hp = get_coefficients(dataset, args, model)

    # Training dataset generation
    train_data, dev_data = get_trajectory_dataset(args, coefficients, hp)

    # prepare to train
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size,  collate_fn=dev_data.collate_fn, shuffle=False)


    # create the learned dynamics model
    dt = torch.tensor([[dataset.dt]], device=args.device)
    def model_wrapper(x, u, c):
        ret = (
                # Function encoder expects a leading batch dimension, and a separate dt for every point.
                model((x.unsqueeze(1), u.unsqueeze(1), dt.expand(x.shape[0], 1)), coefficients=c)
                + x.unsqueeze(1)
        ).squeeze(1)
        return ret
    model_node = Node(model_wrapper, ["x", "u", 'c'], ["x"], name="model")

    # create the neural net control policy
    policy = get_policy(args, dataset, coefficients, len(model.basis_functions.basis_functions),)


    # create the closed loop system
    cl_system = System([policy, model_node], nsteps=args.horizon)
    objectives, constraints = train_data.get_constraints_objectives(args.device)
    components = [cl_system]
    loss = PenaltyLoss(objectives, constraints)
    problem = Problem(components, loss)

    # create callbacks
    cb1 = TensorboardCallback(args.log_dir)
    cb2 = ProgressBarCallback(len(train_loader) * args.num_epochs)
    cb3 = EvalCallback(dataset, dev_data, model, cb1.summary_writer)
    callback = ListCallback([cb1, cb2, cb3])

    # train the policy
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-3)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        optimizer=optimizer,
        epochs=args.num_epochs,
        train_metric='train_loss',
        warmup=50,
        device=args.device,
        callback=callback,
        # log_dir=args.log_dir,
    )
    
    best_model = trainer.train()

    # save the policy
    os.makedirs(args.log_dir, exist_ok=True)
    trainer.model.load_state_dict(best_model)
    torch.save(trainer.model.state_dict(), os.path.join(args.log_dir, "policy.pth"))

    # simulate a trajectory in the actual system
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=1)
        hidden_parameter, y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = next(iter(dataloader))
        y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = (
            y0.to(args.device),
            u0.to(args.device),
            dt.to(args.device),
            y1.to(args.device),
            y0_example.to(args.device),
            u0_example.to(args.device),
            dt_example.to(args.device),
            y1_example.to(args.device),
        )
        coefficients, _ = model.compute_coefficients((y0_example, u0_example, dt_example), y1_example)
        dev_data.rollout_real_trajectory(hidden_parameter, coefficients, cl_system.nodes[0], args.log_dir)
