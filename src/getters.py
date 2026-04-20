import os

import torch
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE
from neuromancer import Node
from neuromancer.modules.activations import activations
from torch.utils.data import DataLoader

from Integrator import rk4_step, ODEFunc
from Policies.LinearOperatorPolicy import LinearOperatorPolicy
from Policies.NonlinearOperatorPolicy import NonlinearOperatorPolicy
from Policies.Policy import Policy


def find_latest(args):
    """Find the latest function encoder model based on the log directory."""
    log_dir = "logs/function_encoder"
    dataset_dir = os.path.join(log_dir, args.dataset, f"seed_{args.seed}",)

    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist.")

    # Get all subdirectories in the dataset directory
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if len(subdirs) == 0:
        raise ValueError(f"No subdirectories found in {dataset_dir}. Train a FE on this dataset. ")

    # sort subdirectories by modified time
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(dataset_dir, d)), reverse=True)
    latest_subdir = subdirs[0]
    latest_path = os.path.join(dataset_dir, latest_subdir)
    assert os.path.exists(os.path.join(latest_path, "model.pth")), \
        f"Model file not found in {latest_path}. Make sure the model is trained and saved correctly."
    return latest_path

def create_function_encoder(state_size, action_size, n_hidden, n_layers, n_basis, use_residual, device):
    layer_sizes = [state_size + action_size + 1] + [n_hidden] * n_layers + [state_size]
    basis_functions = BasisFunctions(
        *[
            NeuralODE(
                ode_func=ODEFunc(model=MLP(layer_sizes=layer_sizes)),
                integrator=rk4_step,
            )
            for _ in range(n_basis)
        ]
    )
    if use_residual:
        residual = NeuralODE(
                ode_func=ODEFunc(model=MLP(layer_sizes=layer_sizes)),
                integrator=rk4_step,
            )
    else:
        residual = None
    model = FunctionEncoder(basis_functions, residual).to(device)
    return model


def load_function_encoder(load_path, device, requires_grad=True):
    params = torch.load(os.path.join(load_path, "arch_params.pth"))
    model = create_function_encoder(
        state_size=params['state_size'],
        action_size=params['action_size'],
        n_hidden=params['n_hidden'],
        n_layers=params['n_layers'],
        n_basis=params['n_basis'],
        use_residual=params.get('use_residual', False),
        device=device,
    )
    model.load_state_dict(torch.load(os.path.join(load_path, "model.pth")))

    # disable gradients if we arent training
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
 
    return model


def get_coefficients(dataset, args, model):
    # compute a large set of coefficients corresponding to different dynamical systems
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=args.num_envs)

        # get batches of data
        hp , y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = next(iter(dataloader))

        # change device
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

        # get coefficients
        coefficients, _ = model.compute_coefficients((y0_example, u0_example, dt_example), y1_example)
        coefficients = coefficients.detach()
        return coefficients, hp



def get_policy(args, dataset, coefficients, n_basis):

    # create the neural net control policy
    if args.policy_type == "adaptive":
        input_size = dataset.state_size + dataset.reference_size + n_basis
        input_keys = ['x', 'r', 'c']
        
        coefficient_mean = torch.mean(coefficients, dim=0)
        coefficient_std = torch.std(coefficients, dim=0)

        
        net = Policy(coefficient_mean, coefficient_std,
                     insize=input_size,
                     outsize=dataset.action_size,
                     hsizes=[args.n_hidden] * args.n_layers,
                     nonlin=activations['gelu'],
                     min=dataset.action_bounds[0].to(args.device),
                     max=dataset.action_bounds[1].to(args.device),
                     ).to(args.device)
        
        
    elif args.policy_type == "robust":
        input_size = dataset.state_size + dataset.reference_size
        input_keys = ['x', 'r']
        net = Policy(None, None,
                     insize=input_size,
                     outsize=dataset.action_size,
                     hsizes=[args.n_hidden] * args.n_layers,
                     nonlin=activations['gelu'],
                     min=dataset.action_bounds[0].to(args.device),
                     max=dataset.action_bounds[1].to(args.device),
                     ).to(args.device)
    elif args.policy_type == "linear":
        input_keys = ['x', 'r', 'c']
        net = LinearOperatorPolicy(
            state_size=dataset.state_size,
            action_size=dataset.action_size,
            reference_size=dataset.reference_size,
            dynamics_basis_size=n_basis,
            hsizes=[args.n_hidden] * args.n_layers,
            activation=activations['gelu'],
            action_min=dataset.action_bounds[0].to(args.device),
            action_max=dataset.action_bounds[1].to(args.device),
        ).to(args.device)
    elif args.policy_type == "nonlinear":
        coefficient_mean = torch.mean(coefficients, dim=0)
        coefficient_std = torch.std(coefficients, dim=0)
        input_keys = ['x', 'r', 'c']
        net = NonlinearOperatorPolicy(
            coefficient_mean=coefficient_mean,
            coefficient_std=coefficient_std,
            state_size=dataset.state_size,
            action_size=dataset.action_size,
            reference_size=dataset.reference_size,
            dynamics_basis_size=n_basis,
            hsizes=[args.n_hidden] * args.n_layers,
            activation=activations['gelu'],
            action_min=dataset.action_bounds[0].to(args.device),
            action_max=dataset.action_bounds[1].to(args.device),
        ).to(args.device)
    else:
        raise ValueError(f"Unknown policy type {args.policy_type}")




    policy = Node(net, input_keys, ['u'], name='policy')
    return policy