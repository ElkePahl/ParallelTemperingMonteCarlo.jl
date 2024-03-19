using ParallelTemperingMonteCarlo

using Random


#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms
n_atoms = 13

# temperature grid
ti = 9.
tf = 16.
n_traj = 16

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 200000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


#ensemble
ensemble = NVT(n_atoms)

#moves - allowed at present: atom, volume and rotation moves (volume,rotation not yet implemented)
move_strat = MoveStrategy(ensemble)  

#c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
#pot = ELJPotentialEven{6}(c)

a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
potB = ELJPotentialB{6}(a,b,c1)

#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne13 = [[2.64403563493521, 0.7912322223900569, -0.565831477176502],
[2.0057915057940128, -1.896308082161984, 0.5635072131266715],
[1.221650294897135, -1.1467297261033085, -2.440332914948713],
[0.38904100625441285, 1.6411500593310273, -2.4314711575317376],
[1.6088717738293812, 0.4740614140622402, 2.4389609753948887],
[0.6378263272466854, 2.6828742459155532, 0.5785213023053898],
[-2.644036039215235, -0.7912313971548159, 0.565830689296014],
[-0.6378264407588382, -2.682874140057697, -0.5785193771824043],
[-0.38904118110730923, -1.6411499639142912, 2.4314716575935194],
[-1.6088712676049575, -0.47406301895270003, -2.438960644598901],
[-2.005790999868592, 1.8963075827307063, -0.5635094256330331],
[-1.2216508572751321, 1.1467309129460042, 2.440332017842955],
[2.428729943900279e-7, -1.0903057930769165e-7, 1.1423484379143698e-6]]


#convert to Bohr
#AtoBohr = 1.8897259886
#pos_ne13 = pos_ne13 * AtoBohr

length(pos_ne13) == n_atoms || error("number of atoms and positions not the same - check starting config")

#boundary conditions 
bc_ne13 = SphericalBC(radius=5.32)   #5.32 Angstrom

#starting configuration
start_config = Config(pos_ne13, bc_ne13)

#histogram information
#n_bin = 100
#en_min = -0.006    #might want to update after equilibration run if generated on the fly
#en_max = -0.001    #otherwise will be determined after run as min/max of sampled energies (ham vector)

#construct array of MCState (for each temperature)
mc_states, results = ptmc_run!(mc_params,temp,start_config,potB,ensemble)

#to check code in REPL
@profview ptmc_run!(mc_params,temp,start_config,potB,ensemble)
#@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)




# plot(temp.t_grid,results.heat_cap)

# data = [results.en_histogram[i] for i in 1:n_traj]
# plot(data)
