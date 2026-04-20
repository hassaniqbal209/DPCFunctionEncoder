"""
Callback classes for versatile behavior in the Trainer object at specified checkpoints.
"""
import torch
from neuromancer import Callback
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm


class TensorboardCallback(Callback):
    """
    Callback base class which allows for bare functionality of Trainer
    """
    def __init__(self, logdir):
        self.summary_writer = SummaryWriter(logdir)
        self.step = 0

    def end_batch(self, trainer, output):
        loss = output[trainer.train_metric].item()
        self.summary_writer.add_scalar("loss/train", loss, self.step)
        self.step += 1
        if loss > 10000:
            print("LOSS divergence")


class ProgressBarCallback(Callback):
    """
    Callback for displaying a progress bar during training.
    """
    def __init__(self, total_steps):
        self.progress_bar = tqdm.tqdm(total=total_steps, desc="Training...")

    def end_batch(self, trainer, output):
        self.progress_bar.update(1)

    def end_train(self, trainer, output):
        self.progress_bar.close()

class ListCallback(Callback):
    def __init__(self, callbacks):
        self.cbs = callbacks

    def begin_train(self, trainer):
        for cb in self.cbs:
            cb.begin_train(trainer)

    def begin_epoch(self, trainer, output):
        for cb in self.cbs:
            cb.begin_epoch(trainer, output)

    def begin_eval(self, trainer, output):
        for cb in self.cbs:
            cb.begin_eval(trainer, output)

    def end_batch(self, trainer, output):
        for cb in self.cbs:
            cb.end_batch(trainer, output)

    def end_eval(self, trainer, output):
        for cb in self.cbs:
            cb.end_eval(trainer, output)

    def end_epoch(self, trainer, output):
        for cb in self.cbs:
            cb.end_epoch(trainer, output)

    def end_train(self, trainer, output):
        for cb in self.cbs:
            cb.end_train(trainer, output)

    def begin_test(self, trainer):
        for cb in self.cbs:
            cb.begin_test(trainer)

    def end_test(self, trainer, output):
        for cb in self.cbs:
            cb.end_test(trainer, output)

class EvalCallback(Callback):
    def __init__(self,
                 function_encoder_dataset,
                 trajectory_dataset,
                 function_encoder,
                 summary_writer,
                 eval_frequency=1):
        self.function_encoder_dataset = function_encoder_dataset
        self.trajectory_dataset = trajectory_dataset
        self.function_encoder = function_encoder
        self.step = 0
        self.eval_frequency = eval_frequency
        self.summary_writer = summary_writer

    def end_batch(self, trainer, output):
        self.step += 1
        if self.step % self.eval_frequency == 0:
            with torch.no_grad():
                dataloader = DataLoader(self.function_encoder_dataset, batch_size=100)
                hidden_parameter, y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = next(iter(dataloader))
                device = next(self.function_encoder.parameters()).device
                y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = (
                    y0.to(device),
                    u0.to(device),
                    dt.to(device),
                    y1.to(device),
                    y0_example.to(device),
                    u0_example.to(device),
                    dt_example.to(device),
                    y1_example.to(device),
                )
                coefficients, _ = self.function_encoder.compute_coefficients((y0_example, u0_example, dt_example), y1_example)
                policy = trainer.model.nodes[0].nodes[0]
                # performance = self.trajectory_dataset.rollout_real_trajectory(hidden_parameter, coefficients, policy)
                # Log the performance metric to TensorBoard
                # self.summary_writer.add_scalar("eval/performance", performance, self.step)
