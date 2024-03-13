using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

# number of atoms
n_atoms = 27
pressure = 101325

# temperature grid
ti = 10.
tf = 40.
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
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations

#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne27 = [[ 1.56624152,  0.90426996,  0.        ],
       [ 4.69872456,  0.90426996,  0.        ],
       [ 7.8312076 ,  0.90426996,  0.        ],
       [ 3.13248304,  3.61707985,  0.        ],
       [ 6.26496608,  3.61707985,  0.        ],
       [ 9.39744912,  3.61707985,  0.        ],
       [ 4.69872456,  6.32988974,  0.        ],
       [ 7.8312076 ,  6.32988974,  0.        ],
       [10.96369064,  6.32988974,  0.        ],
       [ 9.39744912,  1.80853993,  2.55766169],
       [ 3.13248304,  1.80853993,  2.55766169],
       [ 6.26496608,  1.80853993,  2.55766169],
       [10.96369064,  4.52134982,  2.55766169],
       [ 4.69872456,  4.52134982,  2.55766169],
       [ 7.8312076 ,  4.52134982,  2.55766169],
       [12.52993216,  7.23415971,  2.55766169],
       [ 6.26496608,  7.23415971,  2.55766169],
       [ 9.39744912,  7.23415971,  2.55766169],
       [ 0.        ,  0.        ,  5.11532339],
       [ 3.13248304,  0.        ,  5.11532339],
       [ 6.26496608,  0.        ,  5.11532339],
       [ 1.56624152,  2.71280989,  5.11532339],
       [ 4.69872456,  2.71280989,  5.11532339],
       [ 7.8312076 ,  2.71280989,  5.11532339],
       [ 3.13248304,  5.42561978,  5.11532339],
       [ 6.26496608,  5.42561978,  5.11532339],
       [ 9.39744912,  5.42561978,  5.11532339]]

#convert to Bohr
AtoBohr = 1.8897259886 * 0.98
pos_ne27 = pos_ne27 * AtoBohr


#binding sphere
box_length = 9.3974 * AtoBohr
box_height = 7.673 * AtoBohr
bc_ne27 = RhombicBC(box_length, box_height)   

length(pos_ne27) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne27, bc_ne27)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
#mc_states, results = ptmc_run!(mc_params,temp,start_config,pot,ensemble)

#to check code in REPL
@profview ptmc_run!(mc_params,temp,start_config,pot,ensemble)
#@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)

## 