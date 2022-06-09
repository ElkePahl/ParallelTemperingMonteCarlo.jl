using ParallelTemperingMonteCarlo

ti = 2.
tf = 40.
n_traj = 30

mc_cycles = 10000

max_displ = 0.1 # Angstrom

#ELJpotential for neon
#check units!!!
c=[-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765]
elj_ne = ELJPotential{11}(c)

temp = TempGrid{n_traj}(ti,tf) # move to input file at a later stage ...

mc_params = MCParams(mc_cycles)


#mc_params = MCParams(mc_cycles;eq_percentage=0.2)

count_acc = zeros(n_traj)
count_acc_adj = zeros(n_traj)

displ_param = DisplacementParamsAtomMove(max_displ, temp.t_grid; update_stepsize=100)