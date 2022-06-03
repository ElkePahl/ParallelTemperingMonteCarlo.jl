using ParallelTemperingMonteCarlo

ti = 2.
tf = 40.
n_traj = 30

mc_cycles = 10000

max_displ = 0.1 # Angstrom


temp = TempGrid{n_traj}(ti,tf) # move to input file at a later stage ...

mc_params = MCParams(mc_cycles)
#mc_params = MCParams(mc_cycles;eq_percentage=0.2)

count_acc = zeros(n_traj)

displ_param = DisplacementParamsAtomMove(max_displ, temp.t_grid; update_stepsize=100)