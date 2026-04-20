import casadi;
import matplotlib.pyplot as plt;
import numpy as np
import time

## default values
mu = 1.0
d = -1

N = 50; 
dt = 0.1; 
Ulim = 3; 

Q = np.zeros((5,5)) #np.diag([5,5]) 
R = 0.1 
Q_terminal = np.diag((20,20)) 
terminal_tolerance = 0.01
target_state = np.array([0.0, 0.0])

T_sim = 1500       # total simulation time
ref_dt = dt        # time step for reference tracking
N_sim = int(T_sim // ref_dt)

# ref_traj = np.array([[0.5,0.5]] * (N_sim+1)); 
ref_traj = 0.5 * np.ones((2, N_sim))  # reference trajectory for both states
initial_state_bounds = np.array([[-1, -2], [1, 2]])
init_state = initial_state_bounds[0] + (initial_state_bounds[1] - initial_state_bounds[0]) * np.random.rand(2)


plot = False

def run_mpc_simulation(mu, d, N, dt, Q, R, N_sim, ref_traj, init_state):


    def dynamic(x_state, u):

        x0 = d * x_state[0] 
        x1 = x_state[1]  

        dx0 = d * x1
        dx1 = mu * (1 - x0**2) * x1 - x1 + u

        return casadi.vertcat(dx0, dx1)

    def rk4(ode, h, xs, u):
        k1 = ode( xs           , u)
        k2 = ode( xs + h/2 * k1, u)
        k3 = ode( xs + h/2 * k2, u)
        k4 = ode( xs +  h  * k3, u)

        return xs + h/6*(k1 + 2*k2 + 2*k3 + k4)


    opti = casadi.Opti();
    opti.debug.value = True; # Enable debug value

    # Parameters
    opt_x0 = opti.parameter(2) # initial state
    # ref = opti.parameter(2) # reference state
    ref_traj_param = opti.parameter(2,N) # reference trajectory
    ref_terminal_param = opti.parameter(2) # The final target state

    # Variables
    X = opti.variable(2,N+1);

    # Control
    U = opti.variable(1,N); 

    # Input Constraints
    opti.subject_to(opti.bounded(-Ulim,U,Ulim));

    # Initial Conditions
    opti.subject_to(X[:,0] == opt_x0);
    for k in range(0,N):
        k1 = dynamic( X[:,k], U[:,k])
        k2 = dynamic( X[:,k] + dt/2 * k1, U[:,k])
        k3 = dynamic( X[:,k] + dt/2 * k2, U[:,k])
        k4 = dynamic( X[:,k] +  dt  * k3, U[:,k])
        x_next =  X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:,k+1] == x_next)

    # Cost function
    obj = 0  # cost J = x'Qx - u'Ru 
    for i in range(N):
        err = X[:,i] - ref_traj_param[:,i]; # Error for the first state
        obj = obj + casadi.mtimes(casadi.mtimes(err.T,Q),err) + U[0,i]*R*U[0,i]
        
    # 2. Add the terminal cost
    # terminal_err = X[:, N] - ref_terminal_param
    # terminal_cost = casadi.mtimes([terminal_err.T, Q_terminal, terminal_err])
    # obj += terminal_cost
    
    opti.subject_to(X[:, N] <= target_state + terminal_tolerance)
    opti.subject_to(X[:, N] >= target_state - terminal_tolerance)

    opti.minimize(obj);
    opts_setting = {'ipopt.print_level': 0, 'print_time': 0,'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
    opti.solver('ipopt', opts_setting);


    current_state = init_state.copy();
    u0 = np.zeros((1,N));
    next_states = np.zeros((1,N+1)) #np.tile(init_state.reshape(-1, 1), (1, N+1))
    # Sample next_states uniformly within initial_state_bounds for each state and time step
    next_states = np.tile(current_state.reshape(-1, 1), (1, N+1)) 
    # next_states = initial_state_bounds[0].reshape(-1, 1) + (initial_state_bounds[1] - initial_state_bounds[0]).reshape(-1, 1) * np.random.rand(7, N+1)

    # FOR LOGGING
    U_log = np.zeros((1,N_sim));
    X_log = np.zeros((2,N_sim+1));
    X_log[:,0] = current_state; # Set the initial state in the log
    mpciter = 0;


    while mpciter < N_sim: # and np.linalg.norm(current_state - ref_traj[:, min(mpciter, ref_traj.shape[1]-1)]) > 1e-3

        opti.set_value(opt_x0, current_state); # Set the constraint again

        if mpciter + N >= N_sim:
            ref_slice = ref_traj[:, -N:]  # end of trajectory
        else:
            ref_slice = ref_traj[:, mpciter:mpciter+N]

        opti.set_value(ref_traj_param, ref_slice)

        opti.set_initial(U,u0); # RESET the U variable (INPUTS)
        opti.set_initial(X,next_states) # RESET the X variable(STATES)

        sol = opti.solve();

        u_solved = sol.value(U);
        x_solved = sol.value(X);

        current_state = rk4(dynamic,dt,current_state,u_solved[0]);
        u0 = u_solved;

        print(mpciter,np.linalg.norm(current_state - ref_traj[:, min(mpciter, ref_traj.shape[1]-1)]),x_solved[:,0],u_solved[0]);
        U_log[:,mpciter] = u_solved[0];
        X_log[:,mpciter+1] = x_solved[:,0];
        mpciter = mpciter + 1;

    X_log = X_log[:,0:mpciter+1];
    U_log = U_log[:,0:mpciter];


    if plot:
        time_steps = np.arange(0, mpciter * dt, dt)

        # Define a list of colors for the subplots
        colors = plt.cm.tab10.colors[:4]  # Using the first 8 colors from the tab10 colormap

        # Plotting each state in different subplots with y-axis labels
        fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
        states = [r"$x_1$", r"$x_2$", r"$u_1$", r"$u_2$"]
        y_labels = [r"$x_1$", r"$x_2$", r"$u_1$", r"$u_2$"]

        for i in range(2):
            axs[0].plot(time_steps, X_log[i, :], label=states[i], color=colors[i], linewidth=1.5)
            axs[0].grid(True)
            axs[0].legend(loc='upper right', fontsize='small', ncol=8)
            

        # Plot the input command on the fifth subplot
        axs[1].plot(time_steps, U_log[0,:].T, label='Input Command U1', color=colors[2], linewidth=1.5)
        axs[1].plot(time_steps, U_log[1,:].T, label='Input Command U2', color=colors[3], linewidth=1.5)
        axs[1].axhline(y=Ulim, color='r', linestyle='--', label='Upper Limit ({})'.format(Ulim))
        axs[1].axhline(y=0, color='r', linestyle='--', label='Lower Limit (-{})'.format(Ulim))
        axs[1].grid(True)
        axs[1].legend(loc='upper right', fontsize='small')

        axs[0].set_title('States and Input Command vs. Time')
        axs[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig("TTLog-{}.png".format(timestr), dpi=300)

    return X_log, U_log