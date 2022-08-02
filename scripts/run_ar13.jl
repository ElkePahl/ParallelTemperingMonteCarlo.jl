using ParallelTemperingMonteCarlo
using Random,Plots

#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 13

# temperature grid
ti = 30.
tf = 50.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details
mc_cycles = 10000 #default 20% equilibration cycles on top
mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

#moves - allowed at present: atom, volume and rotation moves (volume,rotation not yet implemented)
move_strat = MoveStrategy(atom_moves = n_atoms)  

#ensemble
ensemble = NVT(n_atoms)

#ELJpotential (even c) for argon
c=[-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]
pot = ELJPotentialEven{6}(c)
