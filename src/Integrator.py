from typing import Callable
import torch

def create_cross_terms(x: torch.Tensor) -> torch.Tensor:
    """Create cross product terms for angular rates."""
    wx, wy, wz = x[..., 9], x[..., 10], x[..., 11]
    cross_terms = torch.stack([wy * wz,   wx * wz,   wx * wy], dim=-1)
    return torch.cat([x, cross_terms], dim=-1)

def rk4_step(func:Callable,
             x:torch.tensor,
             u:torch.tensor,
             dt:torch.tensor,
             **ode_kwargs) -> torch.tensor:
    """Runge-Kutta 4th order ODE integrator for a single step."""
    t = torch.zeros_like(dt, device=dt.device)
    k1 = func(t, x, u, **ode_kwargs)
    k2 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k1, u, **ode_kwargs)
    k3 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k2, u, **ode_kwargs)
    k4 = func(t + dt, x + dt.unsqueeze(-1) * k3, u, **ode_kwargs)
    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)


class ODEFunc(torch.nn.Module):
  def __init__(self, model: torch.nn.Module):
    super(ODEFunc, self).__init__()
    self.model = model

  def forward(self, t, x, u):
    state = torch.cat([t.unsqueeze(-1), x, u], dim=-1)
    return self.model(state)



