using Revise
using ParallelTemperingMonteCarlo
using Random, Profile

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#


Random.seed!(1234)

# number of atoms
n_atoms = 13

# temperature grid
ti = 4.
tf = 16.
n_traj = 25

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 10000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
ensemble = NVT(n_atoms)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
bc_ne13 = SphericalBC(radius=7.)   #5.32 Angstrom
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
AtoBohr = 1.8897259886
pos_ne13 = pos_ne13 * AtoBohr

length(pos_ne13) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne13, bc_ne13)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#

@profview  ptmc_run!(mc_params,temp,start_config,pot,ensemble)
#@time states,results = ptmc_run!(mc_params,temp,start_config,pot,ensemble)

## 