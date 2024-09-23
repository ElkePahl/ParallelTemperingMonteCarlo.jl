using ParallelTemperingMonteCarlo
using Random
using Plots

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

# Random.seed!(1234)

# number of atoms
n_atoms = 27
pressure = 101325

# temperature grid
ti = 80
tf = 150
n_traj = 25

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 20000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data"

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

#c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
c=[-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]
pot = ELJPotentialEven{6}(c)


# link="/Users/tiantianyu/Downloads/look-up_table_he.txt"
# potlut=LookuptablePotential(link)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
separated_volume=false
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14, separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations
r_start = 3.7782 #r_start is the desired min. radius between atoms in the starting config.
L_start = 2*(r_start^2/2)^.5  #L_start refers to the distance between adjacent atoms which are parallel to the x or y axis
scaling_factor_start = L_start/(4.3837 - 0)
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
pos_ne27 = pos_ne27 * AtoBohr * scaling_factor_start


#Box length
box_length = 9.3974 * AtoBohr * scaling_factor_start
box_height = 7.673 * AtoBohr * scaling_factor_start
bc_ne27 = RhombicBC(box_length, box_height)   

length(pos_ne27) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne27, bc_ne27)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
mc_states, results = ptmc_run!(save_directory, mc_params,temp,start_config,pot,ensemble)

temp_result, cp = multihistogram_NPT(ensemble, temp, results, 10^(-9), false)
plot(temp_result,cp)

max_value, index = findmax(cp)
t_max = temp_result[index]
println(t_max)

#to check code in REPL
#@profview ptmc_run!(mc_params,temp,start_config,pot,ensemble)
#@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)



## 