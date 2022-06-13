using ParallelTemperingMonteCarlo

# number of atoms
n_atoms = 13

# temperature grid
ti = 2.
tf = 16.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC details
mc_cycles = 10000
mc_sample = 1

mc_params = MCParams(mc_cycles) #20% equilibration is default

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
max_displ = 0.1 # Angstrom
moves = [AtomMove(n_atoms, max_displ, temp.t_grid)]

#displ_param = DisplacementParamsAtomMove(max_displ, temp.t_grid; update_stepsize=100)

#ensemble
ensemble = NVT(n_atoms)

#ELJpotential for neon
#check units!!!
c1=[-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765]
elj_ne1 = ELJPotential{11}(c1)

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
elj_ne = ELJPotentialEven{6}(c)

#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne13 = [[2.825384495892464, 0.928562467914040, 0.505520149314310],
[2.023342172678102,	-2.136126268595355, 0.666071287554958],
[2.033761811732818,	-0.643989413759464, -2.133000349161121],
[0.979777205108572,	2.312002562803556, -1.671909307631893],
[0.962914279874254,	-0.102326586625353, 2.857083360096907],
[0.317957619634043,	2.646768968413408, 1.412132053672896],
[-2.825388342924982, -0.928563755928189, -0.505520471387560],
[-0.317955944853142, -2.646769840660271, -1.412131825293682],
[-0.979776174195320, -2.312003751825495, 1.671909138648006],
[-0.962916072888105, 0.102326392265998,	-2.857083272537599],
[-2.023340541398004, 2.136128558801072,	-0.666071089291685],
[-2.033762834001679, 0.643989905095452, 2.132999911364582],
[0.000002325340981,	0.000000762100600, 0.000000414930733]]

length(pos_ne13) == n_atoms || error("number of atoms and positions not the same - check starting config")

#define boundary conditions starting configuration
bc_ne13 = SphericalBC(radius=5.32)   #Angstrom

#starting configuration
conf_ne13 = Config(pos_ne13, bc_ne13)

count = StatMoves(0,0,0,0,0,0)
status_count = [count for i=1:n_traj]

ptmc_run!(temp, mc_params, conf_ne13, elj_ne, moves, ensemble, status_count)