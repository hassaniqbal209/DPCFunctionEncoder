import os.path

import torch
import matplotlib.pyplot as plt

from Datasets.BaseDatasets import *
from Integrator import rk4_step
from neuromancer import variable, pltCL, pltPhase
from torch.utils.data import DataLoader, default_collate
from GO_casadi_solver import run_mpc_simulation
import numpy as np 

from tqdm import trange
import time

RK4_NSTEPS = 1
ULIM = 4

def glycolytic_oscillator_dynamics(t, x, u, k1s, K1s):

    J0 = torch.tensor(3.0)
    k2 = torch.tensor(6.0)
    k3 = torch.tensor(16.0)
    k4 = torch.tensor(100.0)
    k5 = torch.tensor(1.28)
    k6 = torch.tensor(12.0)
    q = torch.tensor(4.0)
    N = torch.tensor(1.0)
    A = torch.tensor(4.0)
    kappa = torch.tensor(13.0)
    psi = torch.tensor(0.1)
    k = torch.tensor(1.8)

    x1, x2, x3, x4, x5, x6, x7 = [x[..., i:i+1] for i in range(7)]
    k1s1s6 = k1s * x1 * x6 / (1 + (x6 / (K1s)) ** q)
    dx1 = J0 - k1s1s6 + u[..., 0:1]
    dx2 = 2 * k1s1s6 - k2 * x2 * (N - x5) - k6 * x2 * x5
    dx3 = k2 * x2 * (N - x5) - k3 * x3 * (A - x6) 
    dx4 = k3 * x3 * (A - x6) - k4 * x4 * x5 - kappa * (x4 - x7)
    dx5 = k2 * x2 * (N - x5) - k4 * x4 * x5 - k6 * x2 * x5
    dx6 = -2 * k1s1s6 + 2 * k3 * x3 * (A - x6) - k5 * x6
    dx7 = psi * kappa * (x4 - x7) - k * x7

    return torch.cat([dx1, dx2, dx3, dx4, dx5, dx6, dx7], dim=-1,)

class GlycolyticOscillatorDataset(BaseFunctionEncoderDataset):
    def __init__(
        self,
        k1_range: float = [90, 100],
        K1_range: float = [0.5, 1],
        dt_range: tuple = (0.01, 0.01),
        rk4_nsteps: int = RK4_NSTEPS,
        **base_dataset_kwargs,
    ):
        
        state_bounds = torch.tensor([[0.15, 0.19, 0.04, 0.10, 0.08, 0.14, 0.05], [1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10]])
        action_bounds = torch.tensor([[-ULIM], [ULIM]])
        initial_state_bounds = torch.tensor([[0.15, 0.19, 0.04, 0.10, 0.08, 0.14, 0.05], [1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10]])

        super().__init__(state_size=7,
                         action_size=1,
                         reference_size=7,
                         state_bounds=state_bounds,
                         action_bounds=action_bounds,
                         dt=dt_range[0],
                         **base_dataset_kwargs)
        
        self.k1_range = k1_range
        self.K1_range = K1_range
        self.dt_range = dt_range
        self.initial_state_bounds = initial_state_bounds

        self.J0 = torch.tensor(3.0)
        self.k2 = torch.tensor(6.0)
        self.k3 = torch.tensor(16.0)
        self.k4 = torch.tensor(100.0)
        self.k5 = torch.tensor(1.28)
        self.k6 = torch.tensor(12.0)
        self.q = torch.tensor(4.0)
        self.N = torch.tensor(1.0)
        self.A = torch.tensor(4.0)
        self.kappa = torch.tensor(13.0)
        self.psi = torch.tensor(0.1)
        self.k = torch.tensor(1.8)
        self.rk4_nsteps = rk4_nsteps  # Number of RK4 steps to take for each integration
        self.y1_std = torch.tensor([
                    0.1176693035127136, 0.2233273544794025, 0.0287942333318044,
                    0.0279372443365961, 0.0268644750222121, 0.2102727574801566,
                    0.0009169701841551])
        
        self.weights = 1.0 / (torch.clamp(self.y1_std, min=0.1, max=3.0)**2)

    def __iter__(self):
        k1_choices = torch.tensor([80.0, 90.0, 100.0])
        K1_choices = torch.tensor([0.5, 0.75, 1.0])
        combinations = torch.cartesian_prod(K1_choices, k1_choices)
        
        while True:
            
            for combination in combinations:
                total_points = self.n_example_points + self.n_points
                # select K1 and K1 
                k1s = combination[1].unsqueeze(0)
                K1s = combination[0].unsqueeze(0)
                
                # Generate random initial conditions
                _y0 = torch.rand(total_points, 7) * (self.state_bounds[1] - self.state_bounds[0]) + self.state_bounds[0]

                # Generate random control inputs
                _u0 = torch.rand(total_points, 1) * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]

                # Generate random time steps
                _dt = torch.empty(total_points).uniform_(*self.dt_range)

                # Integrate multiple steps
                _ym = _y0.clone()
                for _ in range(self.rk4_nsteps):
                    _ym = rk4_step(glycolytic_oscillator_dynamics, _ym, _u0, _dt, k1s=k1s, K1s=K1s) + _ym
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

                yield {"k1s":k1s, "K1s":K1s}, y0, u0, dt, ychange, y0_example, u0_example, dt_example, ychange_example

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
            k1s = hp["k1s"].to(args.device)
            K1s = hp["K1s"].to(args.device)
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

            fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            # save all plots in y_array
            total_time = 1
            s = self.dt_range[0] #self.dt  # Time step for simulation
            n = int(total_time / (s*self.rk4_nsteps))  # Number of steps to simulate, e.g., 10 seconds
            
            y_array = np.zeros((nrows*ncols,n+1,7))
            # compute sqrt of num_plots and make it int
            _y0 = torch.rand(1, 7, device=args.device) * (self.initial_state_bounds[1].to(args.device) - self.initial_state_bounds[0].to(args.device)) + self.initial_state_bounds[0].to(args.device)
            for i in range(nrows):
                for j in range(ncols):

                    # Plot a single trajectory
                    _k1s = k1s[i * ncols + j]
                    _K1s = K1s[i * ncols + j]
                    
                    # We use the coefficients that we computed before
                    _c = coefficients[i * ncols + j].unsqueeze(0)
                    _dt = torch.tensor([s], device=args.device)

                    # Integrate the true trajectory
                    x = _y0.clone()
                    u = torch.zeros((1, 1), device=args.device)
                    # u = torch.rand(1, 1, device=args.device) * (self.action_bounds[1].to(args.device) - self.action_bounds[0].to(args.device)) + self.action_bounds[0].to(args.device)
                    y = [x]
                    for k in trange(n, desc="Integrating true trajectory {} out of {}".format(i * ncols + j + 1, num_plots), leave=True):
                        for _ in range(self.rk4_nsteps):
                            x = rk4_step(glycolytic_oscillator_dynamics, x, u, _dt, k1s=_k1s, K1s=_K1s) + x
                        y.append(x)
                    y = torch.cat(y, dim=0)
                    y = y.detach().cpu().numpy()
                    y_array[i * ncols + j,:,:] =  y

                    # Integrate the predicted trajectory
                    x = _y0.clone()
                    x = x.unsqueeze(1)
                    _dt = _dt.unsqueeze(0)

                    # pred = [x]
                    # for k in trange(n, desc="Integrating predicted trajectory {} out of {}".format(i * ncols + j + 1, num_plots), leave=True):
                    #     x = model((x, u.unsqueeze(0), _dt * self.rk4_nsteps), coefficients=_c) + x
                    #     pred.append(x)
                    # pred = torch.cat(pred, dim=1)
                    # pred = pred.detach().cpu().numpy()

                    _t = []
                    _p = []
                    times = torch.arange(0, n + 1) * s * self.rk4_nsteps  
                    colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 colors
                    for k in range(7):
                        color = colors[k % len(colors)]
                        (t_line,) = ax[i, j].plot(times, y[:, k], color=color, linestyle='-', label=f"True x{k+1}")
                        # (p_line,) = ax[i, j].plot(times, pred[0, :, k], color=color, linestyle='--', label=f"Pred x{k+1}")
                        ax[i, j].set_title(f"k1={_k1s.item()}, K1={_K1s.item()}, u={u.item()}", fontsize=6)
                        ax[i, j].set_xlabel("Time", fontsize=6)
                        _t.append(t_line)
                        # _p.append(p_line)

                    ax[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)
                    # xlabel every 2 seconds i.e. every 2/s steps
                    ax[i, j].set_xticks(times[::int(2/(s * self.rk4_nsteps))])
                    # ax[i, j].set_xticklabels([f"{t:.1f}" for t in times[::2]], fontsize=6)


            ncol = len(_t)
            fig.legend(
                handles=_t,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=ncol,
                frameon=False,
            )
            # fig.legend(
            #     handles=_p,
            #     loc="upper center",
            #     bbox_to_anchor=(0.5, 0.94),
            #     ncol=ncol,
            #     frameon=False,
            # )

            plt.savefig(os.path.join(args.log_dir, "go_plot.png"))
            
            # create another figure which has 3000 time steps, and does coefficient switching after every 1000 time steps 
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 2.25))
            total_time = 15
            s = self.dt_range[0] #self.dt  # Time step for simulation
            n = int(total_time / (s*self.rk4_nsteps))  # Number of steps to simulate, e.g., 10 seconds
            y_array = np.zeros((n+1,7))
            y_array_pred = np.zeros((n+1,7))
            _y0 = torch.rand(1, 7, device=args.device) * (self.initial_state_bounds[1].to(args.device) - self.initial_state_bounds[0].to(args.device)) + self.initial_state_bounds[0].to(args.device)       
            split_point = n // 3  # 10 seconds
            indices = torch.zeros(n + 1, dtype=torch.long)
            indices[:split_point] = 0  # use first combination for first part
            indices[split_point:2*split_point] = 4      # use fifth combination for the rest
            indices[2*split_point:] = 0
            hp_index = indices[0].item()
            for i in range(n + 1):
                _k1s = k1s[hp_index]
                _K1s = K1s[hp_index]
                
                # We use the coefficients that we computed before
                _c = coefficients[hp_index].unsqueeze(0)
                _dt = torch.tensor([s], device=args.device)

                # Integrate the true trajectory
                if i == 0:
                    x = _y0.clone()
                    x_pred = _y0.clone().unsqueeze(1)
                else:
                    for _ in range(self.rk4_nsteps):
                        x = rk4_step(glycolytic_oscillator_dynamics, x, u, _dt, k1s=_k1s, K1s=_K1s) + x
                y_array[i,:] = x.detach().cpu().numpy().squeeze()
                
                # every 50 time step, make x_pred equal to x
                if i % 50 == 0:
                    x_pred = x.clone().unsqueeze(1)
                    
                # also do pred
                _dt = _dt.unsqueeze(0)
                for _ in range(self.rk4_nsteps):
                    x_pred = model((x_pred, u.unsqueeze(0), _dt * self.rk4_nsteps), coefficients=_c) + x_pred
                y_array_pred[i,:] = x_pred[0].detach().cpu().numpy().squeeze()

                if i == split_point or i == 2*split_point:
                    hp_index = indices[i+1].item()
            
            times = torch.arange(0, n + 1)# * s * self.rk4_nsteps
            colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 colors
            all_handles = []
            for k in range(7):
                color = colors[k % len(colors)]
                (t_line,) = ax2.plot(times, y_array[:, k], color=color, linestyle='-', label=fr"$x_{k+1}$")
                (p_line,) = ax2.plot(times, y_array_pred[:, k], color=color, linestyle='--', label=fr"Pred $x_{{{k+1}}}$")
                # ax2.set_title(f"Glycolytic Oscillator Trajectory with Coefficient Switching", fontsize=10)
                ax2.set_xlabel("Time steps", fontsize=10) 
                ax2.xaxis.set_label_coords(0.5, -0.03)
                # make y axis log scale
                all_handles.extend([t_line, p_line])
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax2.set_yscale('log')

            # Sort them so that for each i: x_i, Pred x_i
            # xlabel every 5 seconds i.e. every 5/s steps
            ax2.set_xticks(times[::int(5/(s * self.rk4_nsteps))])
            ax2.margins(x=0)
            fig2.legend(
                handles=all_handles,
                loc="upper center",           # anchor legend's left center
                bbox_to_anchor=(0.5, 1.2),  # place it just outside the right edge of the plot
                ncol=7,                      # vertical legend
                frameon=False,
                columnspacing=1.0, 
                handleheight=0.25,
                handletextpad=0.5, 
            )
            plt.tight_layout()
            plt.savefig(os.path.join(args.log_dir, "go_plot_coefficient_switching_2.pdf"), bbox_inches='tight',format='pdf', dpi=300)
            
            return y_array



class GlycolyticOscillatorTrajectoryDataset(BaseTrajectoryDataset):


    def __init__(self, dt, coefficients, horizon, batches_per_epoch=3_200, device="cpu", name="train", rk4_nsteps=RK4_NSTEPS):
        super().__init__(
            state_size=7,
            action_size=1,
            state_bounds = torch.tensor([[0.15, 0.19, 0.04, 0.10, 0.08, 0.14, 0.05], [1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10]]),
            action_bounds = torch.tensor([[-ULIM], [ULIM]]),
            initial_state_bounds = torch.tensor([[0.15, 0.19, 0.04, 0.10, 0.08, 0.14, 0.05], [1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10]]),
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
        reference_location = (torch.rand(1, self.state_size, device=device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]) * torch.ones(horizon + 1, self.state_size, device=device)
        # reference_location = torch.ones(horizon + 1, self.state_size, device=device)
        
        # Training dataset generation
        data = {'x': initial_states, 'r': reference_location,}
        return data

    def get_constraints_objectives(self, device):
        # state and reference variables
        x, ref, u = variable('x'), variable("r"), variable('u')

        # objectives
        regulation_loss = 2. * ((x[:,:,0] == ref[:,:,0]) ^ 2)  # target posistion #5
        
        # state terminal penalties
        terminal_lower_bound_penalty = 200.0 * (x[:, [-1], 0] > ref[:, [-1], 0] - 0.001) #20
        terminal_upper_bound_penalty = 200.0 * (x[:, [-1], 0] < ref[:, [-1], 0] + 0.001) #20

        # objectives and constraints names for nicer plot
        regulation_loss.name = 'state_loss'
        terminal_lower_bound_penalty.name = 'y_N_min'
        terminal_upper_bound_penalty.name = 'y_N_max'

        # list of constraints and objectives
        objectives = [] 
        constraints = [
            terminal_lower_bound_penalty,
            terminal_upper_bound_penalty,
        ]
        return objectives, constraints

    def plot_trajectory(self, coefficients, cl_system, save_dir, fe_model=None, dt=None, wb_cl_system=None, hp=None, casadi_plot=False):
        print('Testing Closed Loop System...')
        nsteps = 500
        casadi_comp_time = 0 
        wb_dpc_comp_time = 0
        fe_dpc_comp_time = 0
        
        # select idx_coefficient which has k1 = 100 and K1 = 0.5000
        
        k1s = hp["k1s"].to(coefficients.device)
        K1s = hp["K1s"].to(coefficients.device)
        
        for k in range(len(k1s)):
            if k1s[k] == 100 and K1s[k] == 0.5:
                idx_coeffcient = k
                break
        coefficients = coefficients[idx_coeffcient:idx_coeffcient+1].unsqueeze(0).repeat(1, nsteps + 1, 1)  # repeat coefficients for each time step
        k1s = k1s[idx_coeffcient:idx_coeffcient+1].unsqueeze(0).repeat(1, nsteps + 1, 1)  if k1s is not None else None
        K1s = K1s[idx_coeffcient:idx_coeffcient+1].unsqueeze(0).repeat(1, nsteps + 1, 1)  if K1s is not None else None
        
        """
        k1s = hp["k1s"].cpu().numpy()
        K1s = hp["K1s"].cpu().numpy()

        # default_index = int(np.argmin(K1s))
        # hp_index = int(np.argmax(K1s)) 
        
        for k in range(len(k1s)):
            if k1s[k] == 80 and K1s[k] == 0.75:
                default_index = k
                break
            
        for k in range(len(k1s)):
            if k1s[k] == 100 and K1s[k] == 0.5:
                hp_index = k
                break

        split_point = 500  # nsteps // 4 # 20 time step = 2 seconds
        indices = torch.zeros(nsteps + 1, dtype=torch.long)
        indices[:split_point] = default_index  # use default (lowest mu) for first part
        indices[split_point:2*split_point] = hp_index      # use hp_index (largest mu) for the rest
        indices[2*split_point:3*split_point] = default_index

        # # Prepare coefficients and hidden parameters for each time step
        coefficients = coefficients[indices].unsqueeze(0)  # shape [1, nsteps+1, ...]
        k1s = hp["k1s"].to(coefficients.device)[indices].unsqueeze(0) if hp.get("k1s") is not None else None
        K1s = hp["K1s"].to(coefficients.device)[indices].unsqueeze(0) if hp.get("K1s") is not None else None
        """
        hp_trajectory = {
            'k1': k1s,
            'K1': K1s,
        }

        # generate initial data for closed loop simulation
        state_bounds = self.initial_state_bounds.to(coefficients.device)

        X = torch.rand(1, 1, self.state_size, device=coefficients.device,) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]
        X = torch.tensor([[0.7286, 1.2079, 0.0440, 0.3350, 0.2881, 2.1557, 0.0708]], device=coefficients.device).unsqueeze(0)
        R = (torch.rand(1, self.state_size, device=coefficients.device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]) * torch.ones(1, nsteps + 1, self.state_size, device=coefficients.device,)
        R = 1.5597 * torch.ones(1, nsteps + 1, self.state_size, device=coefficients.device)
       
        # fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6.5, 2.5))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
        plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.25, hspace=0.25)

        if fe_model is not None:
            # predict open loop trajectory
            pred = []
            _xm = X.clone()
            pred.append(_xm)
            for i in trange(nsteps, desc="Integrating open loop trajectory", leave=True):
                _xm = fe_model((_xm, torch.zeros(1,1,1,device=coefficients.device), dt * self.rk4_nsteps), coefficients=coefficients[:,0,:]) + _xm
                pred.append(_xm)
            pred = torch.cat(pred, dim=1)
            pred = pred.detach().cpu().numpy()

        if wb_cl_system is not None:
            dev_dict = {'x': X,
                        'r': R,
                        'c': coefficients,
                        'k1s': k1s,
                        'K1s': K1s}
        else:
            dev_dict = {'x': X,
                        'r': R,
                        'c': coefficients}

        # constraints bounds
        Umin = self.action_bounds[0].unsqueeze(0).expand(nsteps, 1).cpu()
        Umax = self.action_bounds[1].unsqueeze(0).expand(nsteps, 1).cpu()

        if wb_cl_system is not None:
            # perform closed-loop simulation
            wb_cl_system.nsteps = nsteps
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            trajectories = wb_cl_system(dev_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wb_dpc_comp_time = time.perf_counter() - t0

            # plot closed loop trajectory
            Y_wb_dpc=trajectories['x'].detach().cpu().reshape(nsteps + 1, 7)
            R_wb_dpc=trajectories['r'].detach().cpu().reshape(nsteps + 1, 7)
            U_wb_dpc=trajectories['u'].detach().cpu().reshape(nsteps, 1)

            print(f"Plotting trajectory for closed loop system with white box model...")
            # if fe_model is not None:
            #     self._plot_trajectory(Y_wb_dpc, R_wb_dpc, U_wb_dpc, Umin, Umax, fe_model=fe_model, pred=pred, dt=dt, ax=axes[:,0], title="WB + FE-DPC")
            # else:
            #     self._plot_trajectory(Y_wb_dpc, R_wb_dpc, U_wb_dpc, Umin, Umax, ax=axes[:,0], title="WB + DPC")

            if casadi_plot:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                Y_wb_casadi, U_wb_casadi = run_mpc_simulation(
                    k1s=k1s[0,0,0].cpu().numpy(), 
                    K1s=K1s[0,0,0].cpu().numpy(), 
                    N=self.horizon, 
                    dt=self.dt, 
                    Ulim=4, 
                    Q=np.diag([2,0,0,0,0,0,0]), 
                    R=0,  # 0.1 * np.eye(2), 
                    N_sim=nsteps, 
                    ref_traj= R[0].T.cpu().numpy(), # R[0][0,:].T.cpu().numpy(), # NOTE: if reference changes over time, remove [0,:]
                    init_state=X[0,0].cpu().numpy()
                )
                # mismatch
                # Y_wb_casadi, U_wb_casadi = run_mpc_simulation(
                #     k1s=k1s[0].T.cpu().numpy(), 
                #     K1s=K1s[0].T.cpu().numpy(), 
                #     N=self.horizon, 
                #     dt=self.dt, 
                #     Ulim=4, 
                #     Q=np.diag([2,0,0,0,0,0,0]), 
                #     R=0,  # 0.1 * np.eye(2), 
                #     N_sim=nsteps, 
                #     ref_traj= R[0].T.cpu().numpy(), # R[0][0,:].T.cpu().numpy(), # NOTE: if reference changes over time, remove [0,:]
                #     init_state=X[0,0].cpu().numpy()
                # )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                casadi_comp_time = time.perf_counter() - t0
                print(f"Casadi simulation time: {casadi_comp_time:.6f} s")
                print(f"Plotting trajectory for closed loop system with casadi model...")
                self._plot_trajectory(Y_wb_casadi.T, R_wb_dpc, U_wb_casadi, Umin, Umax, ax=axes[:,0], title="WB-MPC") #np.expand_dims(U_wb_casadi,1).T
            else:
                Y_wb_casadi = None
                U_wb_casadi = None

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
        Y_fe_dpc=trajectories['x'].detach().cpu().reshape(nsteps + 1, 7)
        R_fe_dpc=trajectories['r'].detach().cpu().reshape(nsteps + 1, 7)
        U_fe_dpc=trajectories['u'].detach().cpu().reshape(nsteps, 1)

        print(f"Plotting trajectory for closed loop system with function encoder model...")
        if fe_model is not None:
            self._plot_trajectory(Y_fe_dpc, R_fe_dpc, U_fe_dpc, Umin, Umax, fe_model=fe_model, pred=pred, dt=dt, ax=axes[:,1], title="FE-DPC")
        else:
            self._plot_trajectory(Y_fe_dpc, R_fe_dpc, U_fe_dpc, Umin, Umax, ax=axes[:,1], title="FE-DPC")

        if axes is not None:
            # plt.legend()
            axes[0,1].set_ylabel('')
            axes[0,1].set_yscale('log')
            axes[0,0].set_yscale('log')
            # axes[0,2].set_ylabel('')
            axes[1,1].set_ylabel('')
            # axes[1,2].set_ylabel('')
            axes[1,0].set_xlabel('Time steps')
            # axes[1,2].set_xlabel('')
            axes[1,1].set_xlabel('Time steps')
            axes[1,0].set_yticks([-3, 0, 3])
            axes[1,0].yaxis.set_label_coords(-0.1, 0.5)
            
            axes[0,1].get_shared_y_axes().join(axes[0,0], axes[0,1])
            axes[1,1].get_shared_y_axes().join(axes[1,0], axes[1,1])
            axes[0,1].set_yticks([])
            axes[1,1].set_yticks([])
            
            ylims = axes[0,1].get_ylim()
            axes[0,0].set_ylim(ylims)
            ylims = axes[1,1].get_ylim()
            axes[1,0].set_ylim(ylims)
            
            plt.subplots_adjust(wspace=0.1, hspace=0.2)
            # plt.tight_layout()
            # plt.savefig(os.path.join(save_dir, f"cl_trajectory.eps"), format='eps', dpi=300)
            plt.savefig(os.path.join(save_dir, f"go_cl_trajectory.pdf"), format='pdf', dpi=300)
            plt.show()

        print(f"Function Encoder DPC computation time: {fe_dpc_comp_time:.6f} s")
        print(f"White Box DPC computation time: {wb_dpc_comp_time:.6f} s")
        if casadi_plot:
            print(f"Casadi MPC computation time: {casadi_comp_time:.6f} s")
            
        if wb_cl_system is not None:
            if casadi_plot:
                return R_fe_dpc, Y_fe_dpc, U_fe_dpc, Y_wb_dpc, U_wb_dpc, Y_wb_casadi, U_wb_casadi, hp_trajectory
            else:
                return R_fe_dpc, Y_fe_dpc, U_fe_dpc, Y_wb_dpc, U_wb_dpc, Y_wb_casadi, U_wb_casadi, hp_trajectory
        else:
            return R_fe_dpc, Y_fe_dpc, U_fe_dpc, hp_trajectory

    def _plot_trajectory(self, Y, R, U, Umin, Umax, fe_model=None, pred=None, dt=None, ax=None, title="Closed Loop Trajectories"):
        state_labels = [f"$x_{{{i+1}}}$" for i in range(Y.shape[1])]
        colors = plt.cm.tab10.colors  # Use tab10 colormap for up to 10 colors
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
        for i in range(Y.shape[1]):
            # self.dt*self.rk4_nsteps*
            ax[0].plot(torch.arange(Y.shape[0]),Y[:, i], label=state_labels[i], color=colors[i % len(colors)], linewidth=1)
            if fe_model is not None:
                # if i ==0:
                #     U_np = U.squeeze().numpy()
                #     ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(pred.shape[1]-1), pred[0, 1:, i] + U_np, linestyle=':', color=colors[i % len(colors)], alpha=0.5, label='Controlled State 1')
                ax[0].plot(torch.arange(pred.shape[1]), pred[0, :, i], linestyle='-', color=colors[i % len(colors)], alpha=0.5) #self.dt*self.rk4_nsteps*
        
        # ax[0].plot(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), Y[1:,0].unsqueeze(-1) - U, linestyle=':', color=colors[0], alpha=0.5, label='State 1 - Control Input')
        ax[0].plot(torch.arange(R.shape[0]), R[:,0], linestyle='--', color='black', alpha=0.5, label='Reference')
        # self.dt*self.rk4_nsteps*
        ax[1].plot(torch.arange(U.shape[0]), U, color="orange")
        # ax[1].fill_between(self.dt*self.rk4_nsteps*torch.arange(U.shape[0]), Umin.squeeze(), Umax.squeeze(), color='gray', alpha=0.2, label='Control Bounds')
        final_time = torch.arange(R.shape[0])[-1].numpy()
        ax[1].hlines(Umin.squeeze()[0], 0, final_time, colors='red', linestyles='dashed', label='Bounds')
        ax[1].hlines(Umax.squeeze()[0], 0, final_time, colors='red', linestyles='dashed')

        ax[1].set_xlabel("Time")
        ax[0].set_ylabel("States")
        ax[0].set_xticklabels([])
        # ax[1].set_yticks([-3, 0, 3])
        
        
        
        ax[0].set_title(title, fontsize=10)
        ax[1].set_ylabel("Control input")

        if ax is None:
            ax[0].legend(ncol=4)
            ax[1].legend(ncol=9)
            plt.show()
        
    def rollout_real_trajectory(self, hidden_parameter, coefficients, policy, save_dir=None):
        k1s = hidden_parameter["k1s"].to(coefficients.device)
        K1s = hidden_parameter["K1s"].to(coefficients.device)
        nsteps = 1000

        # sample an initial state
        batch_size = k1s.shape[0]
        state_bounds = self.initial_state_bounds.to(coefficients.device)
        x = torch.rand(batch_size, self.state_size, device=coefficients.device) * (state_bounds[1] - state_bounds[0]) + state_bounds[0]

        #  rollout an episode
        states = [x]
        actions = []
        r = torch.rand(1, self.state_size, device=coefficients.device) * torch.ones(batch_size, self.state_size, device=coefficients.device)
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
                change_in_state = rk4_step(glycolytic_oscillator_dynamics, x, u, dt, k1s=k1s, K1s=K1s)
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
        regulation_loss = 100 * (states[:,:,0] - r[:,0]).square().mean()
        control_loss = 0.1 * actions.square().mean()
        total_loss = regulation_loss #+ control_loss
        return total_loss