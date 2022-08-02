using ParallelTemperingMonteCarlo
using Random,Plots

#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 13

# temperature grid
ti = 30.
tf = 60.
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

#starting configurations
#icosahedral ground state of Ar13 (from Cambridge cluster database) in Angstrom
pos_ar13 = [[1.0146454029, 0.3334631631, 0.1815411345],
[0.7266170583, -0.7671194819, 0.2391976365],
[0.7303594101, -0.2312674048, -0.7659965259],
[0.3518552267, 0.8302799340, -0.6004109333],
[0.3457999742, -0.0367469397, 1.0260273868],
[0.1141844246, 0.9505001930, 0.5071207534],
[-1.0146454029, -0.3334631631, -0.1815411345],
[-0.1141844246, -0.9505001930, -0.5071207534],
[-0.3518552267, -0.8302799340, 0.6004109333],
[-0.3457999742, 0.0367469397, -1.0260273868],
[-0.7266170583, 0.7671194819, -0.2391976365],
[-0.7303594101, 0.2312674048, 0.7659965259],
[0.0000000000, 0.0000000000, 0.0000000000]
]

#convert to Bohr
AtoBohr = 1.8897259886
pos_ar13 = pos_ar13 * AtoBohr

length(pos_ar13) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ar13 = SphericalBC(radius=5.6673*AtoBohr)   #5.6673 Angstrom, 150% of equilibrium distance

#starting configuration
start_config = Config(pos_ar13, bc_ar13)

#histogram information
n_bin = 100

#construct array of MCState (for each temperature)
mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot; max_displ=[max_displ_atom[i],0.01,1.]) for i in 1:n_traj]

#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results)

#plot script
plot(temp.t_grid, results.heat_cap)


