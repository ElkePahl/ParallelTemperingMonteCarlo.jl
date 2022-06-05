using ParallelTemperingMonteCarlo

#temperature grid
ti = 2.
tf = 16.
n_traj = 32

#Monte Carlo Cycles
mc_cycles = 10000
mc_sample = 1

#define boundary conditions 
bc_ne13 = SphericalBC(radius=5.32)   #Angstrom

#starting configuration
conf_ne13 = Config(pos_ne13, bc_ne13)
# max. displacement
max_displ = 0.1 # Angstrom

#histograms
Ebins=100
Emin=-0.006
Emax=-0.001
dE=(Emax-Emin)/Ebins
Ehistogram=Array{Array}(undef,n_traj)      #initialization
for i=1:n_traj
    Ehistogram[i]=zeros(Ebins)
end


temp = TempGrid{n_traj}(ti,tf) # move to input file at a later stage ...

mc_params = MCParams(mc_cycles)
#mc_params = MCParams(mc_cycles;eq_percentage=0.2)

count_acc = zeros(n_traj)
count_acc_adj = zeros(n_traj)

count_acc_adj = zeros(n_traj)    #acceptance used for stepsize adjustment, will be reset to 0 after each adjustment
count_exc = zeros(n_traj)        #number of proposed exchanges
count_exc_acc = zeros(n_traj)    #number of accepted exchanges

count_v_acc = zeros(n_traj)        #total count of acceptance
count_v_acc_adj = zeros(n_traj)    #acceptance used for stepsize adjustment, will be reset to 0 after each adjustment

displ_param = DisplacementParamsAtomMove(max_displ, temp.t_grid; update_stepsize=100)