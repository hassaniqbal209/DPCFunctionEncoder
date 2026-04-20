from neuromancer.modules import blocks
import torch

class Policy(blocks.MLP_bounds):

    def __init__(self, coefficient_mean, coefficient_std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coefficient_mean = coefficient_mean
        if coefficient_std is not None:
            self.coefficient_std = torch.clamp(coefficient_std, 1e-6)
        else:
            self.coefficient_std = coefficient_std

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        # NOTE inputs = [state, reference, coefficients] for adaptive policy
        # we want to do some preprocessing to the coefficients to help the model learn.
        if self.coefficient_mean is not None:
            x[..., -self.coefficient_mean.shape[0]:] = (x[..., -self.coefficient_mean.shape[0]:] - self.coefficient_mean) / self.coefficient_std

        # normal forward pass
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return self.method(x, self.min, self.max)