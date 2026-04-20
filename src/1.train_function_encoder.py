import argparse
import random
import datetime
import numpy
import torch
import tqdm
import os

from function_encoder.utils.training import train_step
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Datasets.get_dataset import get_function_encoder_dataset
from getters import create_function_encoder

torch.cuda.set_device(1)
torch.set_printoptions(precision=16)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":

    # training arguments
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument("--grad_steps", type=int, default=10_000, help="Number of training epochs") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training") 
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use.")
    parser.add_argument("--n_basis", type=int, default=11,) 
    parser.add_argument("--n_layers", type=int, default=4,)
    parser.add_argument("--n_hidden", type=int, default=77,)
    parser.add_argument("--log_dir", type=str, default="logs/function_encoder", )
    parser.add_argument("--dataset", type=str, default="Vanderpol", ) 
    parser.add_argument("--use_residual", type=bool, default=False)
    args = parser.parse_args()

    # create a logdir
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = os.path.join(args.log_dir, args.dataset, f"seed_{args.seed}", datetime)

    # create a summary writer
    logger = SummaryWriter(args.log_dir)

    # Set the random seed for reproducibility
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Fetch a dataset
    train_dataset, eval_dataset = get_function_encoder_dataset(args)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    dataloader_iter = iter(dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    eval_dataloader_iter = iter(dataloader)
    # state_weights = train_dataset.weights.to(args.device)
    # initialize the model
    model = create_function_encoder(
        state_size=train_dataset.state_size,
        action_size=train_dataset.action_size,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_basis=args.n_basis,
        use_residual=args.use_residual,
        device=args.device,
    )

    # MSE loss function
    def train_loss_function(model, batch):

        _, y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = batch

        # change device
        y0 = y0.to(args.device)
        u0 = u0.to(args.device)
        dt = dt.to(args.device)
        y1 = y1.to(args.device)
        y0_example = y0_example.to(args.device)
        u0_example = u0_example.to(args.device)
        dt_example = dt_example.to(args.device)
        y1_example = y1_example.to(args.device)

        # compute coefficients
        coefficients, _ = model.compute_coefficients((y0_example, u0_example, dt_example), y1_example)
        
        pred = model((y0, u0, dt), coefficients=coefficients)
        pred_loss = torch.nn.functional.mse_loss(pred, y1)

        # residual loss
        if args.use_residual:
            residual = model.residual_function((y0, u0, dt))
            residual_loss = torch.nn.functional.mse_loss(residual, y1)
        else:
            residual_loss = torch.tensor(0.0, device=args.device)

        return pred_loss + residual_loss

    def eval_loss_function(model, batch):
        _, y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = batch

        # change device
        y0 = y0.to(args.device)
        u0 = u0.to(args.device)
        dt = dt.to(args.device)
        y1 = y1.to(args.device)
        y0_example = y0_example.to(args.device)
        u0_example = u0_example.to(args.device)
        dt_example = dt_example.to(args.device)
        y1_example = y1_example.to(args.device)

        # compute coefficients
        coefficients, _ = model.compute_coefficients((y0_example, u0_example, dt_example), y1_example)

        # # basis function loss
        pred = model((y0, u0, dt), coefficients=coefficients)
        pred_loss = torch.nn.functional.mse_loss(pred, y1)
        return pred_loss


    # train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tqdm.trange(args.grad_steps) as tqdm_bar:
        for epoch in tqdm_bar:
            # get data
            batch = next(dataloader_iter)

            # train
            loss = train_step(model, optimizer, batch, train_loss_function)

            # eval (MSE only)
            with torch.no_grad():
                batch = next(eval_dataloader_iter)
                eval_loss = eval_loss_function(model, batch)

            # log
            tqdm_bar.set_postfix_str(f"Loss: {eval_loss:.2e}")
            logger.add_scalar("loss/eval", eval_loss, epoch)
            logger.add_scalar("loss/train", loss, epoch)

            if epoch % 1000 == 0:
                # save a checkpoint
                checkpoint_path = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

    # save the model
    arch_params = {"n_basis": args.n_basis,
                "n_layers": args.n_layers,
                "n_hidden": args.n_hidden,
                "state_size": train_dataset.state_size,
                "action_size": train_dataset.action_size,
                "use_residual": args.use_residual,
                }
    torch.save(model.state_dict(), os.path.join(args.log_dir, "model.pth"))
    torch.save(arch_params, os.path.join(args.log_dir, "arch_params.pth"))

    # plot the result.
    train_dataset.plot(model, args)