from neuromancer.modules import blocks
import torch
from neuromancer.modules.functions import bounds_scaling, bounds_clamp


class NonlinearOperatorPolicy(torch.nn.Module):
    """
    A linear operator policy that maps a dynamics function to a policy function.
    """

    bound_methods = {"sigmoid_scale": bounds_scaling, "relu_clamp": bounds_clamp}

    def __init__(self,
                 coefficient_mean,
                 coefficient_std,
                 state_size,
                 action_size,
                 reference_size,
                 dynamics_basis_size,
                 hsizes,
                 activation,
                 action_min,
                 action_max,
                 method="sigmoid_scale",):
        super().__init__()

        # store state and action sizes
        self.state_size = state_size
        self.action_size = action_size
        self.reference_size = reference_size
        self.dynamics_basis_size = dynamics_basis_size
        self.policy_basis_size = self.dynamics_basis_size + 1 # note this +1 helps debugging, since they are different sizes. If you make it equal, debugging is harder.
        self.method = self._set_method(method)
        self.min = action_min
        self.max = action_max

        # coefficients
        self.coefficient_mean = coefficient_mean
        if coefficient_std is not None:
            self.coefficient_std = torch.clamp(coefficient_std, 1e-6)
        else:
            self.coefficient_std = coefficient_std



        # create the dynamics basis \pi: S -> A^k
        layers = []
        layers.append(torch.nn.Linear(self.state_size + self.reference_size, hsizes[0]))
        for i in range(len(hsizes) - 1):
            layers.append(activation())
            layers.append(torch.nn.Linear(hsizes[i], hsizes[i + 1]))
        layers.append(activation())
        layers.append(torch.nn.Linear(hsizes[-1], self.policy_basis_size * self.action_size))
        self.policy_space = torch.nn.Sequential(*layers)

        # create the linear operator O:dynamics function -> policy function
        layers2 = []
        layers2.append(torch.nn.Linear(self.dynamics_basis_size, hsizes[0]))
        for i in range(len(hsizes) - 1):
            layers2.append(activation())
            layers2.append(torch.nn.Linear(hsizes[i], hsizes[i + 1]))
        layers2.append(activation())
        layers2.append(torch.nn.Linear(hsizes[-1], self.policy_basis_size))
        self.nonlinear_operator = torch.nn.Sequential(*layers2)


    def _set_method(self, method):
        if method in self.bound_methods.keys():
            return self.bound_methods[method]
        else:
            assert callable(method), (
                f"Method, {method} must be a key in {self.bound_methods} "
                f"or a differentiable callable."
            )
            return method


    def forward(self, x, r, c):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        # first, take the coefficients and map to the policy basis
        policy_coefficients = self.nonlinear_operator(c)

        # next, output a space of policies from the policy network
        ins = torch.cat((x, r), dim=-1)
        policy_space = self.policy_space(ins)
        policy_space = policy_space.view(-1, self.action_size, self.policy_basis_size)

        # now, compute the policy by multiplying the coefficients with the policy space
        actions = torch.einsum("...k, ...ak -> ...a", policy_coefficients, policy_space)

        # return, making sure to clamp
        return self.method(actions, self.min, self.max)