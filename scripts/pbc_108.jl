using ParallelTemperingMonteCarlo

using Random


#set random seed - for reproducibility
#Random.seed!(1234)

# number of atoms
n_atoms = 108
pressure=101325

# temperature grid
ti = 15.
tf = 35.
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details



mc_cycles = 100000 #default 20% equilibration cycles on top



mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 1.0 # Angstrom
max_vchange = 0.02
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

#moves - allowed at present: atom, volume and rotation moves (volume,rotation not yet implemented)
move_strat = MoveStrategy(atom_moves = n_atoms, vol_moves = 1)  
#move_strat = MoveStrategy(atom_moves = n_atoms) 

#ensemble
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14)
#ensemble = NVT(n_atoms)

#ELJpotential for neon
#c1=[-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765]
#elj_ne1 = ELJPotential{11}(c1)

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)

#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne108 =  [[-6.58,       -6.58,       -6.58],
[-4.39,       -4.39,       -6.58],
[-4.39,       -6.58,       -4.39],
[-6.58,       -4.39,       -4.39],
[-6.58,       -6.58,       -2.19],
[-4.39,       -4.39,       -2.19],
[-4.39,       -6.58,        0.00],
 [-6.58,       -4.39,        0.00],
 [-6.58,       -6.58,        2.19],
 [-4.39,       -4.39,        2.19],
 [-4.39,       -6.58,        4.39],
 [-6.58,       -4.39,        4.39],
 [-6.58,       -2.19,       -6.58],
 [-4.39,        0.00,       -6.58],
 [-4.39,       -2.19,       -4.39],
 [-6.58,        0.00,       -4.39],
 [-6.58,       -2.19,       -2.19],
 [-4.39,        0.00,       -2.19],
 [-4.39,       -2.19,        0.00],
 [-6.58,        0.00,        0.00],
 [-6.58,       -2.19,        2.19],
 [-4.39,       0.00,        2.19],
 [-4.39,       -2.19,        4.39],
 [-6.58,        0.00,       4.39],
 [-6.58,        2.19,       -6.58],
 [-4.39,        4.39,       -6.58],
 [-4.39,        2.19,       -4.39],
 [-6.58,        4.39,       -4.39],
 [-6.58,        2.19,       -2.19],
 [-4.39,        4.39,       -2.19],
 [-4.39,        2.19,        0.00],
 [-6.58,        4.39,        0.00],
 [-6.58,        2.19,        2.19],
 [-4.39,        4.39,        2.19],
 [-4.39,        2.19,        4.39],
 [-6.58,        4.39,        4.39],
 [-2.19,       -6.58,       -6.58],
[0.00,       -4.39,       -6.58],
[0.00,       -6.58,       -4.39],
[-2.19,       -4.39,       -4.39],
[-2.19,       -6.58,       -2.19],
[0.00,       -4.39,       -2.19],
[0.00,       -6.58,        0.00],
[-2.19,       -4.39,        0.00],
[-2.19,       -6.58,        2.19],
[0.00,       -4.39,        2.19],
[0.00,       -6.58,        4.39],
[-2.19,       -4.39,        4.39],
[-2.19,       -2.19,       -6.58],
[0.00,        0.00,       -6.58],
[0.00,       -2.19,       -4.39],
[-2.19,        0.00,       -4.39],
[-2.19,       -2.19,       -2.19],
[0.00,        0.00,       -2.19],
[0.00,       -2.19,        0.00],
[-2.19,        0.00,        0.00],
[-2.19,       -2.19,        2.19],
[0.00,        0.00,        2.19],
[0.00,       -2.19,        4.39],
[-2.19,        0.00,        4.39],
[-2.19,        2.19,       -6.58],
[0.00,        4.39,       -6.58],
[0.00,        2.19,       -4.39],
[-2.19,        4.39,       -4.39],
[-2.19,        2.19,       -2.19],
[0.00,        4.39,       -2.19],
[0.00,        2.19,        0.00],
[-2.19,        4.39,        0.00],
[-2.19,        2.19,        2.19],
[0.00,        4.39,        2.19],
[0.00,        2.19,        4.39],
[-2.19,        4.39,        4.39],
[2.19,       -6.58,       -6.58],
[4.39,       -4.39,       -6.58],
[4.39,       -6.58,       -4.39],
[2.19,       -4.39,       -4.39],
[2.19,       -6.58,       -2.19],
[4.39,       -4.39,       -2.19],
[4.39,       -6.58,        0.00],
[2.19,       -4.39,        0.00],
[2.19,       -6.58,        2.19],
[4.39,       -4.39,        2.19],
[4.39,       -6.58,        4.39],
[2.19,       -4.39,        4.39],
[2.19,       -2.19,       -6.58],
[4.39,        0.00,       -6.58],
[4.39,       -2.19,       -4.39],
[2.19,        0.00,       -4.39],
[2.19,       -2.19,       -2.19],
[4.39,        0.00,       -2.19],
[4.39,       -2.19,        0.00],
[2.19,        0.00,        0.00],
[2.19,       -2.19,        2.19],
[4.39,        0.00,        2.19],
[4.39,       -2.19,        4.39],
[2.19,        0.00,        4.39],
[2.19,        2.19,       -6.58],
[4.39,        4.39,       -6.58],
[4.39,        2.19,       -4.39],
[2.19,        4.39,       -4.39],
[2.19,        2.19,       -2.19],
[4.39,        4.39,       -2.19],
[4.39,        2.19,        0.00],
[2.19,        4.39,        0.00],
[2.19,        2.19,        2.19],
[4.39,        4.39,        2.19],
[4.39,        2.19,        4.39],
[2.19,        4.39,        4.39]]


#convert to Bohr
AtoBohr = 1.8897259886
pos_ne108 = pos_ne108 * AtoBohr

length(pos_ne108) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ne108 = PeriodicBC(13.16 * AtoBohr)   

#starting configuration
start_config = Config(pos_ne108, bc_ne108)

#histogram information
n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot) for i in 1:n_traj]

println(mc_states[1].en_tot)
println(mc_states[1].en_tot+ensemble.pressure*mc_states[1].config.bc.box_length^3)


#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

Random.seed!(1234)
@time ptmc_run!((mc_states, move_strat, mc_params, pot, ensemble, results); save=false)





# plot(temp.t_grid,results.heat_cap)

# data = [results.en_histogram[i] for i in 1:n_traj]
# plot(data)]