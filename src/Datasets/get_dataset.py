from Datasets.VanDerPolDataset import *
from Datasets.GlycolyticOscillatorDataset import GlycolyticOscillatorDataset, GlycolyticOscillatorTrajectoryDataset
from Datasets.TwoTankDataset import TwoTankDataset, TwoTankTrajectoryDataset

def get_function_encoder_dataset(args):
    if args.dataset.lower() == "vanderpol":
        train_dataset = VanDerPolDataset(n_points=900, n_example_points=100, dt_range=(0.1, 0.1))
        eval_dataset = VanDerPolDataset(n_points=900, n_example_points=100, dt_range=(0.1, 0.1))
    elif args.dataset.lower() == "glycolyticoscillator":
        train_dataset = GlycolyticOscillatorDataset(n_points=900, n_example_points=100, dt_range=(0.01, 0.01))
        eval_dataset = GlycolyticOscillatorDataset(n_points=900, n_example_points=100, dt_range=(0.01, 0.01))
    elif args.dataset.lower() == "twotank":
        train_dataset = TwoTankDataset(n_points=900, n_example_points=100, dt_range=(1, 1))
        eval_dataset = TwoTankDataset(n_points=900, n_example_points=100, dt_range=(1, 1))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_dataset, eval_dataset


def get_trajectory_dataset(args, coefficients, hp=None):
    batches_per_epoch = args.batch_size * args.num_envs
    if args.dataset == "VanDerPol":
        train_dataset = VanDerPolTrajectoryDataset(
            dt=0.1,
            coefficients=coefficients,
            horizon=args.horizon,
            name="train",
            batches_per_epoch=batches_per_epoch,
        )
        dev_dataset = VanDerPolTrajectoryDataset(
            dt=0.1,
            coefficients=coefficients,
            horizon=args.horizon,
            name="dev",
            batches_per_epoch=batches_per_epoch,
        )
    elif args.dataset == "TwoTank":
        train_dataset = TwoTankTrajectoryDataset(
            coefficients=coefficients,
            horizon=args.horizon,
            name="train",
            batches_per_epoch=batches_per_epoch,
            dt=1.0,
        )
        dev_dataset = TwoTankTrajectoryDataset(
            coefficients=coefficients,
            horizon=args.horizon,
            name="dev",
            batches_per_epoch=batches_per_epoch,
            dt=1.0,
        )
    elif args.dataset == "GlycolyticOscillator":
        train_dataset = GlycolyticOscillatorTrajectoryDataset(
            coefficients=coefficients,
            horizon=args.horizon,
            name="train",
            batches_per_epoch=batches_per_epoch,
            dt=0.01,
        )
        dev_dataset = GlycolyticOscillatorTrajectoryDataset(
            coefficients=coefficients,
            horizon=args.horizon,
            name="dev",
            batches_per_epoch=batches_per_epoch,
            dt=0.01,
        )
    else: 
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_dataset, dev_dataset
