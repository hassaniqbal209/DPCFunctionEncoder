import os.path

import torch
import matplotlib.pyplot as plt
import numpy as np

from Datasets.BaseDatasets import *
from Integrator import rk4_step
from neuromancer import variable, pltCL, pltPhase
from torch.utils.data import DataLoader, default_collate
from VDP_casadi_solver import run_mpc_simulation
import time 

def van_der_pol(t, x, u, mu, d):
    x0 = d * x[..., 0:1]
    x1 = x[..., 1:2]
    dx0 = d * x1
    dx1 = mu * (1 - x0 ** 2) * x1 - x0 + u[..., 0:1]
    return torch.cat([dx0, dx1], dim=-1,)

class VanDerPolDataset(BaseFunctionEncoderDataset):
    def __init__(
        self,
        mu_range=(0.1, 3.0),
        dt_range=(0.1, 0.1),
        **base_dataset_kwargs,
    ):
        state_bounds = torch.tensor([[-2, -5], [2, 5]])
        action_bounds = torch.tensor([[-3.0], [3.0]])
        initial_state_bounds = torch.tensor([[-1, -2], [1, 2]])

        super().__init__(state_size=2,
                         action_size=1,
                         reference_size=2,
                         state_bounds=state_bounds,
                         action_bounds=action_bounds,
                         dt=dt_range[1],
                         **base_dataset_kwargs)
        self.mu_range = mu_range
        self.dt_range = dt_range
        self.initial_state_bounds = initial_state_bounds

    def __iter__(self):
        while True:
            total_points = self.n_example_points + self.n_points
            # Generate a single mu
            mu = torch.empty(1).uniform_(*self.mu_range)

            # generate a single d ~ {1, -1}
            d = torch.randint(0, 2, (1,)).float() * 2 - 1  # Randomly choose between -1 and 1
                        
            # Generate random initial conditions
            _y0 = torch.rand(total_points, 2) * (self.state_bounds[1] - self.state_bounds[0]) + self.state_bounds[0]

            # Generate random control inputs
            _u0 = torch.rand(total_points, 1) * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]
            
            # Generate random time steps
            _dt = torch.empty(total_points).uniform_(*self.dt_range)

            # Integrate one step
            _y_change = rk4_step(van_der_pol, _y0, _u0, _dt, mu=mu, d=d)

            # Split the data
            y0_example = _y0[: self.n_example_points]
            u0_example = _u0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points]
            ychange_example = _y_change[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            u0 = _u0[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            ychange = _y_change[self.n_example_points :]

            yield {"mu":mu, "d":d}, y0, u0, dt, ychange, y0_example, u0_example, dt_example, ychange_example

    def plot(self, model, args):

        model.eval()
        with torch.no_grad():
            # Generate a single batch of functions for plotting
            dataloader = DataLoader(self, batch_size=9)
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

            hp, y0, u0, dt, y1, y0_example, u0_example, dt_example, y1_example = batch
            mu = hp["mu"].to(args.device)
            d = hp["d"].to(args.device)
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

            fig, ax = plt.subplots(3, 3, figsize=(10, 10))

            for i in range(3):
                for j in range(3):
                    
                    # Plot a single trajectory
                    # _mu = mu[i * 3 + j]
                    # _d = d[i * 3 + j]
                    starting_mu = mu[i * 3 + j]
                    starting_d = d[i * 3 + j]
                    
                    #  ending_mp is one shifted version of starting_mp
                    ending_mu = mu[i * 3 + j + 1] if (i * 3 + j + 1) < mu.shape[0] else mu[i * 3 + j]
                    ending_d = d[i * 3 + j + 1] if (i * 3 + j + 1) < d.shape[0] else d[i * 3 + j]
                    
                    _y0 = torch.rand(1, 2, device=args.device) * (self.initial_state_bounds[1].to(args.device) - self.initial_state_bounds[0].to(args.device)) + self.initial_state_bounds[0].to(args.device)

                    # We use the coefficients that we computed before
                    # _c = coefficients[i * 3 + j].unsqueeze(0)
                    starting_c = coefficients[i * 3 + j].unsqueeze(0)
                    ending_c = coefficients[i * 3 + j + 1].unsqueeze(0) if (i * 3 + j + 1) < coefficients.shape[0] else coefficients[i * 3 + j].unsqueeze(0)
                    
                    s = 0.1  # Time step for simulation
                    n = int(10 / s)
                    _dt = torch.tensor([s], device=args.device)

                    # Integrate the true trajectory
                    x = _y0.clone()
                    u = torch.zeros((1, 1), device=args.device)
                    y = [x]
                    for k in range(n):
                        _mu = starting_mu if k < n // 2 else ending_mu
                        _d = starting_d if k < n // 2 else ending_d
                        _c = starting_c if k < n // 2 else ending_c
                        x = rk4_step(van_der_pol, x, u, _dt, mu=_mu, d=_d) + x
                        y.append(x)
                    y = torch.cat(y, dim=0)
                    y = y.detach().cpu().numpy()

                    # Integrate the predicted trajectory
                    x = _y0.clone()
                    u = torch.zeros((1, 1, 1), device=args.device)
                    x = x.unsqueeze(1)
                    _dt = _dt.unsqueeze(0)
                    pred = [x]
                    for k in range(n):
                        _mu = starting_mu if k < n // 2 else ending_mu
                        _d = starting_d if k < n // 2 else ending_d
                        _c = starting_c if k < n // 2 else ending_c
                        x = model((x, u, _dt), coefficients=_c) + x
                        pred.append(x)
                    pred = torch.cat(pred, dim=1)
                    pred = pred.detach().cpu().numpy()

                    ax[i, j].set_xlim(-5, 5)
                    ax[i, j].set_ylim(-5, 5)
                    # ax[i, j].set_title(f"mu={_mu.item():.2f}, d={_d.item():.0f}")
                    ax[i, j].set_title(f"mu={starting_mu.item():.2f}->{ending_mu.item():.2f}, d={starting_d.item():.0f}->{ending_d.item():.0f}")
                    # (_t,) = ax[i, j].plot(y[:, 0], y[:, 1], label="True")
                    # (_p,) = ax[i, j].plot(pred[0, :, 0], pred[0, :, 1], label="Predicted")
                    
                    split_idx = n // 2
                    s0, s1 = slice(0, split_idx + 1), slice(split_idx, None)
                    colors = ("blue", "red")

                    (_p,) = ax[i, j].plot(
                        pred[0, s0, 0],
                        pred[0, s0, 1],
                        color=colors[0],
                        label="Predicted",
                        linewidth=2.0,
                        alpha=0.7,
                        zorder=1,
                    )
                    ax[i, j].plot(
                        pred[0, s1, 0],
                        pred[0, s1, 1],
                        color=colors[1],
                        linewidth=2.0,
                        alpha=0.7,
                        zorder=1,
                    )

                    (_t,) = ax[i, j].plot(
                        y[s0, 0],
                        y[s0, 1],
                        linestyle="--",
                        color=colors[0],
                        label="True",
                        linewidth=2.8,   
                        alpha=1.0,
                        zorder=2,       
                    )
                    ax[i, j].plot(
                        y[s1, 0],
                        y[s1, 1],
                        linestyle="--",
                        color=colors[1],
                        linewidth=2.8,
                        alpha=1.0,
                        zorder=2,
                    )

            fig.legend(
                handles=[_t, _p],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=2,
                frameon=False,
            )

            plt.savefig(os.path.join(args.log_dir, "vanderpol_plot.png"))








class VanDerPolTrajectoryDataset(BaseTrajectoryDataset):

    def __init__(self, dt, coefficients, horizon, batches_per_epoch=3_200, device="cuda", name="train"):
        super().__init__(
            state_size=2,
            action_size=1,
            state_bounds=torch.tensor([[-2, -5], [2, 5]]),
            action_bounds=torch.tensor([[-3.0], [3.0]]),
            initial_state_bounds=torch.tensor([[-1, -2], [1, 2]]),
            dt=dt,
        )
        self.coefficients = coefficients.to(device)
        self.batches_per_epoch = batches_per_epoch
        self.horizon = horizon
        self.device = device
        self.name = name


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
        reference_location = torch.zeros(horizon + 1, self.state_size, device=device)

        # Training dataset generation
        data = {'x': initial_states, 'r': reference_location,}
        return data

    def get_constraints_objectives(self, device):
        # state and reference variables
        x, ref, u = variable('x'), variable("r"), variable('u')

        # objectives
        control_loss = 0.1 * ((u == torch.zeros_like(u)) ^ 2) # control effort

        # state bound constraints
        state_bounds = self.state_bounds.to(device)
        state_lower_bound_penalty = 10.0 * (x > state_bounds[0])
        state_upper_bound_penalty = 10.0 * (x < state_bounds[1])

        # state terminal penalties
        terminal_lower_bound_penalty = 20.0 * (x[:, [-1], :] > ref - 0.01)
        terminal_upper_bound_penalty = 20.0 * (x[:, [-1], :] < ref + 0.01)
        # objectives and constraints names for nicer plot
        state_lower_bound_penalty.name = 'x_min'
        state_upper_bound_penalty.name = 'x_max'
        terminal_lower_bound_penalty.name = 'y_N_min'
        terminal_upper_bound_penalty.name = 'y_N_max'

        # list of constraints and objectives
        objectives = [ control_loss] 
        constraints = [
            state_lower_bound_penalty,
            state_upper_bound_penalty,
            terminal_lower_bound_penalty,
            terminal_upper_bound_penalty,
        ]
        return objectives, constraints

    def plot_trajectory(self, coefficients, cl_system, save_dir, hp=None, wb_cl_system=None, casadi_plot=False):
        print('Testing Closed Loop System...')
        nsteps = 100
        
        casadi_comp_time = 0 
        wb_dpc_comp_time = 0
        fe_dpc_comp_time = 0
        # Choose hp_index with largest mu, default with lowest mu while ensuring d is diffrent where d has only two values {1, -1}
        
        mus = hp["mu"].cpu().numpy()
        ds = hp["d"].cpu().numpy()

        default_index = int(np.argmin(ds))
        hp_index = int(np.argmax(ds)) 

        split_point = 25  # nsteps // 4 # 20 time step = 2 seconds
        indices = torch.zeros(nsteps + 1, dtype=torch.long)
        indices[:split_point] = default_index  # use default (lowest mu) for first part
        indices[split_point:] = hp_index      # use hp_index (largest mu) for the rest

        # # Prepare coefficients and hidden parameters for each time step
        coefficients = coefficients[indices].unsqueeze(0)  # shape [1, nsteps+1, ...]
        mu = hp["mu"].to(coefficients.device)[indices].unsqueeze(0) if hp.get("mu") is not None else None
        d = hp["d"].to(coefficients.device)[indices].unsqueeze(0) if hp.get("d") is not None else None
        
        hp_trajectory = {
            'mu': mu,
            'd': d,
        }
            
        # generate initial data for closed loop simulation
        state_bounds = self.initial_state_bounds.to(coefficients.device)
        initial_state = torch.rand(1, 1, self.state_size, device=coefficients.device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]
        R_orig = torch.zeros(1, nsteps + 1, self.state_size, device=coefficients.device)

        # constraints bounds
        Umin = self.action_bounds[0].unsqueeze(0).expand(nsteps, 1).cpu()
        Umax = self.action_bounds[1].unsqueeze(0).expand(nsteps, 1).cpu()
        
        fig_states, axes_states = plt.subplots(3, 3, figsize=(15, 10))
        
        if wb_cl_system is not None:
            dev_dict = {
                'x': initial_state,
                'r': R_orig,
                'c': coefficients,
                'mu': mu,
                'd': d,
            }

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
            Y_wb_dpc = trajectories['x'].detach().cpu().reshape(nsteps + 1, self.state_size)
            R_wb_dpc = trajectories['r'].detach().cpu().reshape(nsteps + 1, self.state_size)
            U_wb_dpc = trajectories['u'].detach().cpu().reshape(nsteps, self.action_size)

            print(f"Plotting trajectory for closed loop system with white box model...")
            self._plot_trajectory(Y_wb_dpc, R_wb_dpc, U_wb_dpc, Umin, Umax, nsteps, ax=axes_states[:,0], split_point=split_point, title="WB + DPC")

            if casadi_plot:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                Y_wb_casadi, U_wb_casadi = run_mpc_simulation(
                    mu=mu[0,0,0].cpu().numpy(), 
                    d=d[0,0,0].cpu().numpy(),
                    N=self.horizon, 
                    dt=self.dt, # check Ulim within the file 
                    Q=np.diag([5.0, 5.0]),
                    R=0.1 * np.eye(1), 
                    N_sim=nsteps, 
                    ref_traj=R_orig[0].T.cpu().numpy(), 
                    init_state=initial_state[0,0].cpu().numpy()
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                casadi_comp_time = time.perf_counter() - t0
                print(f"Plotting trajectory for closed loop system with casadi model...")
                self._plot_trajectory(Y_wb_casadi.T, R_wb_dpc, U_wb_casadi.T, Umin, Umax, nsteps, ax=axes_states[:,1], split_point=split_point, title="WB + Casadi")

        dev_dict = {
            "x": initial_state,
            "r": R_orig,
            "c": coefficients,
            'mu': mu,
            'd': d
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
        Y_fe_dpc = trajectories['x'].detach().cpu().reshape(nsteps + 1, self.state_size)
        R_fe_dpc = trajectories['r'].detach().cpu().reshape(nsteps + 1, self.state_size)
        U_fe_dpc = trajectories['u'].detach().cpu().reshape(nsteps, self.action_size)

        self._plot_trajectory(Y_fe_dpc, R_fe_dpc, U_fe_dpc, Umin, Umax, nsteps, ax=axes_states[:, 2], split_point=split_point, title="FE + DPC")

        plt.tight_layout()
        plt.show()
        
        print(f"Function Encoder DPC computation time: {fe_dpc_comp_time:.6f} s")
        print(f"White Box DPC computation time: {wb_dpc_comp_time:.6f} s")
        if casadi_plot:
            print(f"Casadi MPC computation time: {casadi_comp_time:.6f} s")

        if wb_cl_system is not None:
            if casadi_plot:
                return R_fe_dpc, Y_fe_dpc, U_fe_dpc, Y_wb_dpc, U_wb_dpc, Y_wb_casadi, U_wb_casadi, hp_trajectory, split_point
            else:
                return R_fe_dpc, Y_fe_dpc, U_fe_dpc, Y_wb_dpc, U_wb_dpc, hp_trajectory, split_point
        else:
            return R_fe_dpc, Y_fe_dpc, U_fe_dpc, hp_trajectory, split_point
        
    def _plot_trajectory(self, Y, R, U, Umin, Umax, nsteps, ax, split_point, title=""):
        # plot phase portrait
        ax[0].set_title(f"{title} Phase Portrait")
        ax[0].set_xlabel("x1")
        ax[0].set_ylabel("x2")
        if split_point is not None:
            ax[0].plot(Y[:split_point+1, 0], Y[:split_point+1, 1], label="Trajectory")
            ax[0].plot(Y[split_point:, 0], Y[split_point:, 1], label="Trajectory")
        else:
            ax[0].plot(Y[:, 0], Y[:, 1], label="Trajectory")
        ax[0].scatter(Y[0, 0], Y[0, 1], color='black', label="Start", marker='o')
        ax[0].scatter(R[0, 0], R[0, 1], color='red', label="Target", marker='x')
        ax[0].legend()

        # plot state over time
        ax[1].set_title(f"{title} States Over Time")
        ax[1].plot(Y[:, 0], label="x1")
        ax[1].plot(Y[:, 1], label="x2")
        ax[1].plot(R[:, 0], '--', label="r1")
        ax[1].plot(R[:, 1], '--', label="r2")
        ax[1].set_xlabel("Time Step")
        ax[1].legend()

        # plot control input over time
        ax[2].set_title(f"{title} Control Input Over Time")
        ax[2].set_xlabel("Time Step")
        ax[2].set_ylabel("Control Input")
        ax[2].plot(U, label="u")
        if split_point is not None:
            ax[2].vlines(split_point, Umin.min(), Umax.max(), colors='gray', linestyles='dashed', label='Change Point')
        ax[2].hlines(Umin, 0, nsteps, colors='red', linestyles='dashed', label='U bounds')
        ax[2].hlines(Umax, 0, nsteps, colors='red', linestyles='dashed')
        ax[2].legend()

    def rollout_real_trajectory(self, hidden_parameter, coefficients, policy, save_dir=None):
        mu = hidden_parameter["mu"].to(coefficients.device)
        d = hidden_parameter["d"].to(coefficients.device)
        nsteps = 100

        # sample an initial state
        batch_size = mu.shape[0]
        state_bounds = self.initial_state_bounds.to(coefficients.device)
        x = torch.rand(batch_size, self.state_size, device=coefficients.device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]

        #  rollout an episode
        states = [x]
        actions = []
        for i in range(nsteps):
            # compute control input
            ins = {"x": x,
                   "r": torch.zeros(batch_size, self.state_size, device=coefficients.device),
                   "c": coefficients,
                   }
            u = policy(ins)['u']

            # integrate the system
            dt = torch.tensor([self.dt], device=coefficients.device)  # expand dt for batch size
            change_in_state = rk4_step(van_der_pol, x, u, dt, mu=mu, d=d)
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
                  Umin=self.action_bounds[0].unsqueeze(0).expand(nsteps, 1).cpu(),
                  Umax=self.action_bounds[1].unsqueeze(0).expand(nsteps, 1).cpu(),
                  figname=os.path.join(save_dir, 'real_trajectory.png'))

        # measure objective error
        # objectives
        regulation_loss = 100 * (states - torch.zeros_like(states)).square().mean()
        control_loss = 0.1 * actions.square().mean()
        total_loss = regulation_loss + control_loss
        return total_loss
