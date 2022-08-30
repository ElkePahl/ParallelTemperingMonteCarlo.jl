using ParallelTemperingMonteCarlo 
using Random

#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 12

# temperature grid, which is in Kelvin
ti = 150
tf = 500
n_traj = 32

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details
mc_cycles = 100000  #default 20% equilibration cycles on top
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

#potential 
#c = [265682.2899854795, -4.441948849660487e6, 2.954956463280794e7, -1.019551994374658e8, 1.940867564608051e8, -1.9409753310956642e8, 7.98420805167911e7]   
a = 14.44 #box length in Bohrs
pot = DFTPotential(a, n_atoms) 

#initial configration 
#icoshedral structure (with central atom removed) for Ga12, from Krista Steenbergen
pos_ga12 = [[0.000968, -0.735658, -2.448964], 
[-2.012798, -1.522479, -0.814074], 
[-1.732366, 1.112479, -1.641393], 
[0.946136, 1.733948, -1.739984], 
[2.293939, -0.475975, -0.933625], 
[0.482849, -2.482840, -0.374164], 
[-0.000472, 0.735892, 2.449197], 
[2.012866, 1.522774, 0.813977], 
[1.732373, -1.112540, 1.640889], 
[-0.945902, -1.733528, 1.740266], 
[-2.294086, 0.475728, 0.933702], 
[-0.483507, 2.482200, 0.374172]] 

AtoBohr = 1.8897259886
pos_ga12 = pos_ga12 * AtoBohr
            
length(pos_ga12) == n_atoms || error("number of atoms and positions not the same - check starting config") 

bc_ga12 = SphericalBC(radius=6*AtoBohr)   #5.32 Angstrom

#starting configuration
start_config = Config(pos_ga12, bc_ga12)

#histogram information
n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, pot; max_displ=[max_displ_atom[i],0.01,1.]) for i in 1:n_traj]

#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

@time ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results)
##
#plot(temp.t_grid,results.heat_cap)
##
#data = [results.en_histogram[i] for i in 1:n_traj]
#plot(data)