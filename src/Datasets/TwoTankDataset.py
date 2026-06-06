import os.path

import torch
import matplotlib.pyplot as plt

from Datasets.BaseDatasets import *
from Integrator import rk4_step
from neuromancer import variable, pltCL, pltPhase
from torch.utils.data import DataLoader, default_collate

import copy
from tqdm import trange
from TT_casadi_solver import run_mpc_simulation
import numpy as np
import time

RK4_NSTEPS = 1

def two_tank_dynamics(t, x, u, c1, c2):


    x1 = x[..., 0:1]  # Tank 1 level
    x2 = x[..., 1:2]  # Tank 2 level
    p = u[..., 0:1]  # pump modulation input
    v = u[..., 1:2]  # valve opening input

    dx1 = c1 * (1.0 - v) * p - c2 * torch.sqrt(x1)  # Tank 1 dynamics
    dx2 = c1 * v * p + c2 * torch.sqrt(x1) - c2 * torch.sqrt(x2)  # Tank 2 dynamics

    mask = (x1 + dx1 > 1.0)
    dx1[mask] = 0
    mask = (x2 + dx2 > 1.0)
    dx2[mask] = 0
    mask = (x1 + dx1 < 0.0)
    dx1[mask] = 0
    mask = (x2 + dx2 < 0.0)
    dx2[mask] = 0

    return torch.cat([dx1, dx2], dim=-1,)

class TwoTankDataset(BaseFunctionEncoderDataset):
    def __init__(
        self,
        c1_range: float = [0.06, 0.1],#[0.01, 0.1], #[0.75, 1.0],
        c2_range: float = [0.01, 0.06], #[0.1, 0.4],
        dt_range: tuple = (1.0, 1.0),
        rk4_nsteps: int = RK4_NSTEPS,
        **base_dataset_kwargs,
    ):
        
        state_bounds = torch.tensor([[0.01, 0.01], [1.0, 1.0]])
        action_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        initial_state_bounds = torch.tensor([[0.01, 0.01], [1.0, 1.0]])

        super().__init__(state_size=2,
                         action_size=2,
                         reference_size=2,
                         state_bounds=state_bounds,
                         action_bounds=action_bounds,
                         dt=dt_range[0],
                         **base_dataset_kwargs)
        
        self.c1_range = c1_range
        self.c2_range = c2_range
        self.dt_range = dt_range
        self.initial_state_bounds = initial_state_bounds
        self.rk4_nsteps = rk4_nsteps  # Number of RK4 steps to take for each integration

    def __iter__(self):
        while True:
            total_points = self.n_example_points + self.n_points
            # select c1 and c2
            c1 = torch.empty(1).uniform_(*self.c1_range)
            c2 = torch.empty(1).uniform_(*self.c2_range)

            # Generate random initial conditions
            _y0 = torch.rand(total_points, 2) * (self.state_bounds[1] - self.state_bounds[0]) + self.state_bounds[0]

            # Generate random control inputs
            _u0 = torch.rand(total_points, 2) * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]

            # Generate random time steps
            _dt = torch.empty(total_points).uniform_(*self.dt_range)

            # Integrate
            _ym = _y0.clone()
            for _ in range(self.rk4_nsteps):
                _ym = rk4_step(two_tank_dynamics, _y0, _u0, _dt, c1=c1, c2=c2) + _ym
            _y_change = _ym - _y0

            # Split the data
            y0_example = _y0[: self.n_example_points]
            u0_example = _u0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points] * self.rk4_nsteps
            ychange_example = _y_change[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            u0 = _u0[self.n_example_points :]
            dt = _dt[self.n_example_points :] * self.rk4_nsteps
            ychange = _y_change[self.n_example_points :]

            yield {"c1":c1, "c2":c2}, y0, u0, dt, ychange, y0_example, u0_example, dt_example, ychange_example

    def plot(self, model, args):

        model.eval()
        with torch.no_grad():
            # Generate a single batch of functions for plotting
            nrows = 3 
            ncols = 3
            num_plots = nrows * ncols  
            dataloader = DataLoader(self, batch_size=num_plots)
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

            hp, y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = batch
            c1 = hp["c1"].to(args.device)
            c2 = hp["c2"].to(args.device)
            y0 = y0.to(args.device)
            u0 = u0.to(args.device)
            dt = dt.to(args.device)
            y1 = y1.to(args.device)
            y0_example = y0_example.to(args.device)
            u0_example = u0_example.to(args.device)
            dt_example = dt_example.to(args.device)
            y1_example = y1_example.to(args.device)

            # Precompute the coefficients for the batch
            coefficients, G = model.compute_coefficients((y0_example, u0_example, dt_example), y1_example)

            s = self.dt_range[0] #self.dt  # Time step for simulation
            n = int(100 / (s*self.rk4_nsteps))  # Number of steps to simulate, e.g., 10 seconds n = 100
            y_tensor = torch.zeros((num_plots, n+1, self.state_size), device=args.device)
            
            fig, ax = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))

            for i in range(nrows):
                for j in range(ncols):

                    # Plot a single trajectory
                    _c1 = c1[i * ncols + j]
                    _c2 = c2[i * ncols + j]
                    _y0 = torch.rand(1, 2, device=args.device) * (self.initial_state_bounds[1].to(args.device) - self.initial_state_bounds[0].to(args.device)) + self.initial_state_bounds[0].to(args.device)

                    # We use the coefficients that we computed before
                    _c = coefficients[i * ncols + j].unsqueeze(0)
                    _dt = torch.tensor([s], device=args.device)

                    # Integrate the true trajectory
                    x = _y0.clone()
                    u = torch.rand(n, 2, device=args.device) * (self.action_bounds[1].to(args.device) - self.action_bounds[0].to(args.device)) + self.action_bounds[0].to(args.device)
                    
                    y = [x]
                    for k in trange(n, desc="Integrating true trajectory {} out of {}".format(i * ncols + j + 1, num_plots), leave=True):
                        
                        for _ in range(self.rk4_nsteps):
                            x = rk4_step(two_tank_dynamics, x, u[k].unsqueeze(0), _dt, c1=_c1, c2=_c2) + x
                        y.append(x)

                    if torch.isnan(x).any():
                        print(f"Warning: NaN values found in trajectory at index {i * ncols + j}!")
                        
                    y = torch.cat(y, dim=0)
                    y = y.detach().cpu().numpy()

                    # Integrate the predicted trajectory
                    x = _y0.clone()
                    x = x.unsqueeze(1)
                    _dt = _dt.unsqueeze(0)

                    pred = [x]
                    for k in trange(n, desc="Integrating predicted trajectory {} out of {}".format(i * ncols + j + 1, num_plots), leave=True):
                        x = model((x, u[k].unsqueeze(0).unsqueeze(0), _dt * self.rk4_nsteps), coefficients=_c) + x
                        pred.append(x)
                    pred = torch.cat(pred, dim=1)
                    pred = pred.detach().cpu().numpy()

                    _t = []
                    _p = []
                    times = torch.arange(0, n + 1) * s  * self.rk4_nsteps
                    colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 colors
                    for k in range(2):
                        color = colors[k % len(colors)]
                        (t_line,) = ax[i, j].plot(times, y[:, k], color=color, linestyle='-', label=f"True x{k+1}")
                        (p_line,) = ax[i, j].plot(times, pred[0, :, k], color=color, linestyle='--', label=f"Pred x{k+1}")
                        ax[i, j].set_title(f"c1={_c1.item():.2f}, c2={_c2.item():.2f}, u=({u[0,0].item():.2f}, {u[0,1].item():.2f})", fontsize=6)
                        ax[i, j].set_xlabel("Time", fontsize=6)
                        _t.append(t_line)
                        _p.append(p_line)

                    ax[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)
                    # xlabel every 2 seconds i.e. every 2/s steps
                    ax[i, j].set_xticks(times[::int(2/(s * self.rk4_nsteps))])
                    # ax[i, j].set_xticklabels([f"{t:.1f}" for t in times[::2]], fontsize=6)

                    y_tensor[i * ncols + j, :, :] = torch.tensor(y, device=args.device)
                    
            # if y tensor has any nan values, print a warning
            if torch.isnan(y_tensor).any():
                print("Warning: y_tensor has NaN values at index {}!".format(torch.isnan(y_tensor).nonzero(as_tuple=True)[0]))

            ncol = len(_t)
            fig.legend(
                handles=_t,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=ncol,
                frameon=False,
            )
            fig.legend(
                handles=_p,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=ncol,
                frameon=False,
            )

            plt.savefig(os.path.join(args.log_dir, "go_plot.png"))

        return y_tensor

class TwoTankTrajectoryDataset(BaseTrajectoryDataset):


    def __init__(self, dt, coefficients, horizon, batches_per_epoch=3_200, device="cuda", name="train", rk4_nsteps=RK4_NSTEPS):
        super().__init__(
            state_size=2,
            action_size=2,
            state_bounds = torch.tensor([[0.01, 0.01], [1.0, 1.0]]),
            action_bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            initial_state_bounds = torch.tensor([[0.01, 0.01], [1.0, 1.0]]),
            dt=dt,
        )
        self.coefficients = coefficients.to(device)
        self.batches_per_epoch = batches_per_epoch
        self.horizon = horizon
        self.device = device
        self.name = name
        self.dt = dt
        self.rk4_nsteps = rk4_nsteps  # Number of RK4 steps to take for each integration

    def __getitem__(self, i):
        data = self.get_policy_training_data(self.horizon, self.device)

        # randomly select a row of coefficients
        random_index = torch.randint(0, self.coefficients.shape[0], (1,)).item()
        coefficients = self.coefficients[random_index]

        # add coefficients to the data
        data['c'] = coefficients.expand(self.horizon + 1, -1)

        return data

    def __len__(self):
        return self.batches_per_epoch

    def get_policy_training_data(self, horizon, device):
        # initial states
        state_bounds = self.initial_state_bounds.to(device)
        initial_states = torch.rand(1, self.state_size, device=device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]

        # target locations
        reference_location = torch.rand(1, 1, device=device) * torch.ones(horizon + 1, self.state_size, device=device)
        # reference_location[:, 1] =  1.0
        # Training dataset generation
        data = {'x': initial_states, 'r': reference_location,}
        return data

    def get_constraints_objectives(self, device):
        # state and reference variables
        x, ref, u = variable('x'), variable("r"), variable('u')

        # objectives
        regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion #5 20
        control_loss = 0.1 * ((u == torch.zeros_like(u)) ^ 2) # control effort 

        # state terminal penalties
        terminal_lower_bound_penalty = 10.0 * (x[:, [-1], :] > ref - 0.01) # 200.0
        terminal_upper_bound_penalty = 10.0 * (x[:, [-1], :] < ref + 0.01) # 200.0

        # objectives and constraints names for nicer plot
        regulation_loss.name = 'state_loss'
        terminal_lower_bound_penalty.name = 'y_N_min'
        terminal_upper_bound_penalty.name = 'y_N_max'

        # list of constraints and objectives
        objectives = [regulation_loss, control_loss]
        constraints = [
            terminal_lower_bound_penalty,
            terminal_upper_bound_penalty,
        ]
        return objectives, constraints

    def plot_trajectory(self, coefficients, cl_system, save_dir, wb_cl_system=None, hp=None, casadi_plot=False):
        print('Testing Closed Loop System...')
        nsteps = 700
        
        """
        coefficients = coefficients[0:1].unsqueeze(0).repeat(1, nsteps + 1, 1)  # repeat coefficients for each time step
        c1 = c1[0:1].unsqueeze(0).repeat(1, nsteps + 1, 1)  if c1 is not None else None
        c2 = c2[0:1].unsqueeze(0).repeat(1, nsteps + 1, 1)  if c2 is not None else None
        split_point=None
        """
        
        casadi_comp_time = 0 
        wb_dpc_comp_time = 0
        fe_dpc_comp_time = 0
        
        c1 = hp["c1"].cpu().numpy()
        c2 = hp["c2"].cpu().numpy()

        default_index = int(np.argmin(c1))
        hp_index = int(np.argmax(c1)) 

        split_point1 = nsteps // 2 - nsteps // 10
        split_point2 = 4 * nsteps // 5 - nsteps // 10
        max_c2_index = int(np.argmax(c2))
        indices = torch.zeros(nsteps + 1, dtype=torch.long)
        indices[:split_point1] = default_index  # use default (lowest c1) for first part
        indices[split_point1:split_point2] = hp_index      # use hp_index (largest c1) for the middle part
        indices[split_point2:] = max_c2_index      # use max_c2_index (largest c2) for the rest

        # # Prepare coefficients and hidden parameters for each time step
        coefficients = coefficients[indices].unsqueeze(0)  # shape [1, nsteps+1, ...]
        c1 = hp["c1"].to(coefficients.device)[indices].unsqueeze(0) if hp.get("c1") is not None else None
        c2 = hp["c2"].to(coefficients.device)[indices].unsqueeze(0) if hp.get("c2") is not None else None
        
        hp_trajectory = {
            'c1': c1,
            'c2': c2,
        }

        # generate initial data for closed loop simulation
        state_bounds = self.initial_state_bounds.to(coefficients.device)
        # R_orig = self.generate_step_reference(nsteps, batch_size=1, state_size = self.state_size, xmin=0.0, xmax=1.0, device=coefficients.device)
        # R_orig = torch.rand(1, 1, device=coefficients.device) * torch.ones(1, nsteps + 1, self.state_size, device=coefficients.device)
        change_points = torch.tensor([0, nsteps//4, nsteps//2, 3*nsteps//4, nsteps], dtype=torch.long)
        step_vals = torch.tensor([0.65, 0.85, 0.45, 0.25])
        ref = torch.zeros(nsteps+1, 2)
        for i in range(len(change_points)-1):
            start = change_points[i]
            end = change_points[i+1] if i < len(change_points)-2 else nsteps+1
            ref[start:end] = step_vals[i]
        R_orig = ref.unsqueeze(0).to(coefficients.device)
        X_orig = torch.rand(1, 1, self.state_size, device=coefficients.device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]

        # constraints bounds
        Umin = self.action_bounds[0].unsqueeze(0).expand(nsteps, 2).cpu()
        Umax = self.action_bounds[1].unsqueeze(0).expand(nsteps, 2).cpu()

        fig, axes = plt.subplots(2, 3, figsize=(7, 3.5), sharex=True)

        if wb_cl_system is not None:
            dev_dict = {'x': X_orig,
                        'r': R_orig,
                        'c': coefficients,
                        'c1': c1,
                        'c2': c2}

            # perform closed-loop simulation with white box model
            wb_cl_system.nsteps = nsteps
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            trajectories = wb_cl_system(dev_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wb_dpc_comp_time = time.perf_counter() - t0

            # plot closed loop trajectory
            Y_wb_dpc = trajectories['x'].detach().cpu().reshape(nsteps + 1, 2)
            R_wb_dpc = trajectories['r'].detach().cpu().reshape(nsteps + 1, 2)
            U_wb_dpc = trajectories['u'].detach().cpu().reshape(nsteps, 2)

            print(f"Plotting trajectory for closed loop system with white box model...")
            self._plot_trajectory(Y_wb_dpc, R_wb_dpc, U_wb_dpc, Umin, Umax, ax=axes[:,0], title="WB + DPC")

            if casadi_plot:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                Y_wb_casadi, U_wb_casadi = run_mpc_simulation(
                    c1=c1[0,0,0].cpu().numpy(), 
                    c2=c2[0,0,0].cpu().numpy(), 
                    N=self.horizon, 
                    dt=self.dt, 
                    Ulim=1, 
                    Q=np.diag([5, 5]), 
                    R=0,  # 0.1 * np.eye(2), 
                    T_sim=nsteps, 
                    ref_traj=R_orig[0].T.cpu().numpy(), 
                    init_state=X_orig[0,0].cpu().numpy()
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                casadi_comp_time = time.perf_counter() - t0
                print(f"Plotting trajectory for closed loop system with casadi model...")
                self._plot_trajectory(Y_wb_casadi.T, R_wb_dpc, U_wb_casadi.T, Umin, Umax, ax=axes[:,1], title="WB + Casadi")
            else:
                Y_wb_casadi = None
                U_wb_casadi = None

        dev_dict = {
                'x': X_orig,
                'r': R_orig,
                'c': coefficients
            }
       
        # perform closed-loop simulation
        cl_system.nsteps = nsteps
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        trajectories = cl_system(dev_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fe_dpc_comp_time = time.perf_counter() - t0

        # plot closed loop trajectory
        Y_fe_dpc = trajectories['x'].detach().cpu().reshape(nsteps + 1, 2)
        R_fe_dpc = trajectories['r'].detach().cpu().reshape(nsteps + 1, 2)
        U_fe_dpc = trajectories['u'].detach().cpu().reshape(nsteps, 2)

        print(f"Plotting trajectory for closed loop system with function encoder model...")
        self._plot_trajectory(Y_fe_dpc, R_fe_dpc, U_fe_dpc, Umin, Umax, ax=axes[:,2], title="FE + DPC")

        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"cl_trajectory_plot.eps"), format='eps', dpi=300)
        plt.show()
        
        print(f"Function Encoder DPC computation time: {fe_dpc_comp_time:.6f} s")
        print(f"White Box DPC computation time: {wb_dpc_comp_time:.6f} s")
        if casadi_plot:
            print(f"Casadi MPC computation time: {casadi_comp_time:.6f} s")

        if wb_cl_system is not None:
            if casadi_plot:
                return R_fe_dpc, Y_fe_dpc, U_fe_dpc, Y_wb_dpc, U_wb_dpc, Y_wb_casadi, U_wb_casadi, hp_trajectory, split_point1, split_point2
            else:
                return R_fe_dpc, Y_fe_dpc, U_fe_dpc, Y_wb_dpc, U_wb_dpc, hp_trajectory, split_point1, split_point2
        else:
            return R_fe_dpc, Y_fe_dpc, U_fe_dpc, hp_trajectory, split_point1, split_point2

    def _plot_trajectory(self, Y, R, U, Umin, Umax, ax=None, title="Closed Loop Trajectories"):
        state_labels = [f"State {i+1}" for i in range(Y.shape[1])]
        colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 colors

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        for i in range(Y.shape[1]):
            ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(Y.shape[0]),Y[:, i], label=state_labels[i], color=colors[i % len(colors)])
        
        # ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), Y[1:,0].unsqueeze(-1) - U, linestyle=':', color=colors[0], alpha=0.5, label='State 1 - Control Input')
        
        ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(R.shape[0]), R[:,0], linestyle='--', color=colors[0], alpha=0.5, label='Reference 0')
        ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(R.shape[0]), R[:,1], linestyle='--', color=colors[1], alpha=0.5, label='Reference 1')
        # ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(Y.shape[0]), R_orig[0,:,0].detach().cpu(), linestyle='--', color=colors[0], alpha=0.5, label='Reference Orig 1')
        # ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(Y.shape[0]), R_orig[0,:,1].detach().cpu(), linestyle='--', color=colors[1], alpha=0.5, label='Reference Orig 2')
        
        ax[1].plot(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), U[:,0], linestyle='--', color=colors[0], alpha=0.5, label='Control Input 1')
        ax[1].plot(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), U[:,1], linestyle='--', color=colors[1], alpha=0.5, label='Control Input 2')
        ax[1].fill_between(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), Umin[:,0].squeeze(), Umax[:,0].squeeze(), color=colors[0], alpha=0.2, label='Control Bound 1')
        ax[1].fill_between(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), Umin[:,1].squeeze(), Umax[:,1].squeeze(), color=colors[1], alpha=0.2, label='Control Bound 1')

        ax[0].set_title(title)

        ax[1].set_xlabel("Time steps")
        ax[0].set_ylabel("States")
        ax[1].set_ylabel("Control Input")
       
        if ax is None:
            ax[0].legend(ncol=4)
            ax[1].legend()
            plt.show()

    def rollout_real_trajectory(self, hidden_parameter, coefficients, policy, save_dir=None):
        c1 = hidden_parameter["c1"].to(coefficients.device)
        c2 = hidden_parameter["c2"].to(coefficients.device)
        nsteps = 1000

        # sample an initial state
        batch_size = c1.shape[0]
        state_bounds = self.initial_state_bounds.to(coefficients.device)
        x = torch.rand(batch_size, self.state_size, device=coefficients.device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]

        #  rollout an episode
        states = [x]
        actions = []
        r = torch.rand(1, 1, device=coefficients.device) * torch.zeros(batch_size, self.state_size, device=coefficients.device)
        for i in range(nsteps):
            # compute control input
            ins = {"x": x,
                   "r": r,
                   "c": coefficients,
                   }
            u = policy(ins)['u']

            # integrate the system
            dt = torch.tensor([self.dt], device=coefficients.device)  # expand dt for batch size
            for _ in range(self.rk4_nsteps):
                change_in_state = rk4_step(two_tank_dynamics, x, u, dt, c1=c1, c2=c2)
                x = x + change_in_state

            # log
            states.append(x)
            actions.append(u)

        # plot the phase portraits of the real trajectory
        states = torch.stack(states, dim=0).detach()
        actions = torch.stack(actions, dim=0).detach()
        if save_dir:
            states = states[:, 0]
            actions = actions[:, 0]
            pltPhase(X=states.cpu().numpy(), figname=os.path.join(save_dir, 'real_phase.png'))
            pltCL(Y=states.cpu().numpy(), R=torch.zeros_like(states).cpu().numpy(), U=actions.cpu().numpy(),
                  Umin=self.action_bounds[0].unsqueeze(0).expand(nsteps, 2).cpu(),
                  Umax=self.action_bounds[1].unsqueeze(0).expand(nsteps, 2).cpu(),
                  figname=os.path.join(save_dir, 'real_trajectory.png'))

        # measure objective error
        # objectives
        regulation_loss = 10 * (states - r).square().mean() #torch.zeros_like(states)
        control_loss = 0.1 * actions.square().mean()
        total_loss = regulation_loss#+ control_loss
        return total_loss

    def generate_step_reference(self, nsteps, batch_size, state_size, xmin=0.0, xmax=1.0, randsteps=1, device="cpu"):
        change_points = torch.linspace(0, nsteps, randsteps+1, dtype=torch.long)
        change_points += torch.randint(-nsteps//(2*randsteps), nsteps//(2*randsteps), (randsteps+1,))
        change_points = torch.clamp(change_points, 0, nsteps).sort().values

        step_vals = xmin + (xmax - xmin) * torch.rand(randsteps+1)

        ref = torch.zeros(nsteps+1)
        for i in range(randsteps+1):
            start = change_points[i]
            end = change_points[i+1] if i+1 < len(change_points) else nsteps+1
            ref[start:end] = step_vals[i]

        ref = ref.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, nsteps+1)
        ref = ref.unsqueeze(-1).repeat(1, 1, state_size)  # (batch_size, nsteps+1, state_size)

        return ref.to(device)
