import casadi;
import matplotlib.pyplot as plt;
import numpy as np
import time

K1s = 0.5 #0.75
k1s = 100 #90.0

N = 50; 
dt = 0.01; 
Ulim = 4; 

Q = np.diag([2,0,0,0,0,0,0]) 
R = 0 #0.1 * np.eye(1) 

T_sim = 1000       # total simulation time
ref_dt = dt        # time step for reference tracking
N_sim = int(T_sim // ref_dt)

ref_traj = np.ones((7,1)) #,N_sim
ref_traj = np.tile(ref_traj.reshape(-1, 1), (1, N_sim))

initial_state_bounds = np.array([[0.15, 0.19, 0.04, 0.10, 0.08, 0.14, 0.05], [1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10]])
init_state = initial_state_bounds[0] + (initial_state_bounds[1] - initial_state_bounds[0]) * np.random.rand(7)

plot = False

def run_mpc_simulation(k1s, K1s, N, dt, Ulim, Q, R, N_sim, ref_traj, init_state):


    def dynamic(x_state, u):
        # Constants
        J0 = 3.0
        k2 = 6.0
        k3 = 16.0
        k4 = 100.0
        k5 = 1.28
        k6 = 12.0
        q = 4.0
        N = 1.0
        A = 4.0
        kappa = 13.0
        psi = 0.1
        k = 1.8
        
        # States
        x1 = x_state[0]
        x2 = x_state[1]
        x3 = x_state[2]
        x4 = x_state[3]
        x5 = x_state[4]
        x6 = x_state[5]
        x7 = x_state[6]

        # Input
        u1 = u[0]

        # Dynamics
        k1s1s6 = k1s * x1 * x6 / (1 + casadi.power(x6 / K1s, q))
        dx1 = J0 - k1s1s6 + u1
        dx2 = 2 * k1s1s6 - k2 * x2 * (N - x5) - k6 * x2 * x5
        dx3 = k2 * x2 * (N - x5) - k3 * x3 * (A - x6)
        dx4 = k3 * x3 * (A - x6) - k4 * x4 * x5 - kappa * (x4 - x7)
        dx5 = k2 * x2 * (N - x5) - k4 * x4 * x5 - k6 * x2 * x5
        dx6 = -2 * k1s1s6 + 2 * k3 * x3 * (A - x6) - k5 * x6
        dx7 = psi * kappa * (x4 - x7) - k * x7

        return casadi.vertcat(dx1, dx2, dx3, dx4, dx5, dx6, dx7)

    # Numeric Integrator RK4
    def rk4(ode, h, xs, u):
        k1 = ode( xs           , u)
        k2 = ode( xs + h/2 * k1, u)
        k3 = ode( xs + h/2 * k2, u)
        k4 = ode( xs +  h  * k3, u)

        return xs + h/6*(k1 + 2*k2 + 2*k3 + k4)


    opti = casadi.Opti();
    opti.debug.value = True; # Enable debug value

    # Parameters
    opt_x0 = opti.parameter(7) # initial state
    ref_traj_param = opti.parameter(7, N) # reference state

    # Variables
    X = opti.variable(7,N+1);

    # Control 
    U = opti.variable(1,N); 

    # Input Constraints
    opti.subject_to(opti.bounded(-Ulim,U,Ulim));

    # Initial Conditions
    opti.subject_to(X[:,0] == opt_x0);
    for k in range(0,N):
        k1 = dynamic( X[:,k], U[0,k])
        k2 = dynamic( X[:,k] + dt/2 * k1, U[0,k])
        k3 = dynamic( X[:,k] + dt/2 * k2, U[0,k])
        k4 = dynamic( X[:,k] +  dt  * k3, U[0,k])
        x_next =  X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:,k+1] == x_next)



    # Cost function
    obj = 0  # cost J = x'Qx - u'Ru 
    for i in range(N):
        err = X[:,i] - ref_traj_param[:,i]; # Error for the first state
        obj = obj + casadi.mtimes(casadi.mtimes(err.T,Q),err) + U[0,i]*R*U[0,i]

    opti.minimize(obj);
    opts_setting = {'ipopt.print_level': 0, 'print_time': 0,'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', opts_setting);


    

    current_state = init_state.copy();
    u0 = np.zeros((1,N));
    next_states = np.zeros((7,N+1)) #np.tile(init_state.reshape(-1, 1), (1, N+1))
    # Sample next_states uniformly within initial_state_bounds for each state and time step
    next_states = np.tile(current_state.reshape(-1, 1), (1, N+1)) 
    # next_states = initial_state_bounds[0].reshape(-1, 1) + (initial_state_bounds[1] - initial_state_bounds[0]).reshape(-1, 1) * np.random.rand(7, N+1)
    # FOR LOGGING
    U_log = np.zeros((1,N_sim));
    X_log = np.zeros((7,N_sim+1));
    X_log[:,0] = current_state;
    mpciter = 0;


    while(mpciter < N_sim):
        
        opti.set_value(opt_x0, current_state); # Set the constraint again
        
        if mpciter + N >= N_sim:
            ref_slice = ref_traj[:, -N:]  # end of trajectory
        else:
            ref_slice = ref_traj[:, mpciter:mpciter+N]

        # If ref_slice has less than N columns (e.g. at the end), pad with last column
        if ref_slice.shape[1] < N:
            pad_width = N - ref_slice.shape[1]
            last_col = ref_slice[:, -1].reshape(-1, 1)
            ref_slice = np.hstack([ref_slice, np.repeat(last_col, pad_width, axis=1)])
        opti.set_value(ref_traj_param, ref_slice);
        
        opti.set_initial(U,u0); # RESET the U variable (INPUTS)
        opti.set_initial(X,next_states) # RESET the X variable(STATES)

        sol = opti.solve();

        u_solved = sol.value(U);
        x_solved = sol.value(X);

        current_state = rk4(dynamic,dt,current_state,u_solved);
        u0 = u_solved[0];

        print(mpciter,np.linalg.norm(current_state - ref_traj[:, min(mpciter, ref_traj.shape[1]-1)]),x_solved[:,0],u_solved[0]);
        U_log[0,mpciter] = u_solved[0];
        X_log[:,mpciter+1] = x_solved[:,0];
        mpciter = mpciter + 1;

    X_log = X_log[:,0:mpciter+1];
    U_log = U_log[0,0:mpciter];

    if plot:

        timestr = time.strftime("%Y%m%d-%H%M%S")
        time_steps = np.arange(0, mpciter * dt, dt)

        # Define a list of colors for the subplots
        colors = plt.cm.tab10.colors[:8]  # Using the first 8 colors from the tab10 colormap

        # Plotting each state in different subplots with y-axis labels
        fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
        states = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$", r"$x_6$", r"$x_7$", r"$u$"]
        y_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$", r"$x_6$", r"$x_7$", r"$u$"]

        for i in range(7):
            axs[0].plot(time_steps, X_log[i, :], label=states[i], color=colors[i], linewidth=1.5)
            axs[0].grid(True)
            axs[0].legend(loc='upper right', fontsize='small', ncol=8)
            

        # Plot the input command on the fifth subplot
        axs[1].plot(time_steps, U_log.T, label='Input Command U', color=colors[7], linewidth=1.5)
        axs[1].axhline(y=Ulim, color='r', linestyle='--', label='Upper Limit ({})'.format(Ulim))
        axs[1].axhline(y=-Ulim, color='r', linestyle='--', label='Lower Limit (-{})'.format(Ulim))
        axs[1].grid(True)
        axs[1].legend(loc='upper right', fontsize='small')

        axs[0].set_title('States and Input Command vs. Time')
        axs[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        # plt.savefig("GOLog-{}.png".format(timestr), dpi=300)

    return X_log, U_log

if __name__ == "__main__":
    run_mpc_simulation(k1s, K1s, N, dt, Ulim, Q, R, T_sim, ref_traj, init_state)