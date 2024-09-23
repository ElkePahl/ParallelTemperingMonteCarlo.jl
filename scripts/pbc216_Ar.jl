using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

# number of atoms
n_atoms = 216
pressure = 101325 

# temperature grid
ti = 90.
tf = 150.
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

save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#
c_argon = [-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]
pot = ELJPotentialEven{6}(c_argon)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
separated_volume = false
ensemble = NPT(n_atoms, pressure * 3.398928944382626e-14, separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations
r_start = 3.7782 # r_start is the desired min. radius between atoms in the starting config.
L_start = 2 * (r_start^2 / 2)^0.5  # L_start refers to the distance between adjacent atoms parallel to x or y axis
scaling_factor_start = L_start / (4.3837 - 0)

#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ar216 = [[ 1.56624152,  0.90426996,  0.        ],
[ 4.69872456,  0.90426996,  0.        ],
[ 7.8312076 ,  0.90426996,  0.        ],
[10.96369064,  0.90426996,  0.        ],
[14.09617368,  0.90426996,  0.        ],
[17.22865672,  0.90426996,  0.        ],
[ 3.13248304,  3.61707985,  0.        ],
[ 6.26496608,  3.61707985,  0.        ],
[ 9.39744912,  3.61707985,  0.        ],
[12.52993216,  3.61707985,  0.        ],
[15.6624152 ,  3.61707985,  0.        ],
[18.79489824,  3.61707985,  0.        ],
[ 4.69872456,  6.32988974,  0.        ],
[ 7.8312076 ,  6.32988974,  0.        ],
[10.96369064,  6.32988974,  0.        ],
[14.09617368,  6.32988974,  0.        ],
[17.22865672,  6.32988974,  0.        ],
[20.36113976,  6.32988974,  0.        ],
[ 6.26496608,  9.04269963,  0.        ],
[ 9.39744912,  9.04269963,  0.        ],
[12.52993216,  9.04269963,  0.        ],
[15.6624152 ,  9.04269963,  0.        ],
[18.79489824,  9.04269963,  0.        ],
[21.92738128,  9.04269963,  0.        ],
[ 7.8312076 , 11.75550952,  0.        ],
[10.96369064, 11.75550952,  0.        ],
[14.09617368, 11.75550952,  0.        ],
[17.22865672, 11.75550952,  0.        ],
[20.36113976, 11.75550952,  0.        ],
[23.4936228 , 11.75550952,  0.        ],
[ 9.39744912, 14.46831941,  0.        ],
[12.52993216, 14.46831941,  0.        ],
[15.6624152 , 14.46831941,  0.        ],
[18.79489824, 14.46831941,  0.        ],
[21.92738128, 14.46831941,  0.        ],
[25.05986433, 14.46831941,  0.        ],
[ 0.        ,  1.80853993,  2.55766169],
[ 3.13248304,  1.80853993,  2.55766169],
[ 6.26496608,  1.80853993,  2.55766169],
[ 9.39744912,  1.80853993,  2.55766169],
[12.52993216,  1.80853993,  2.55766169],
[15.6624152 ,  1.80853993,  2.55766169],
[ 1.56624152,  4.52134982,  2.55766169],
[ 4.69872456,  4.52134982,  2.55766169],
[ 7.8312076 ,  4.52134982,  2.55766169],
[10.96369064,  4.52134982,  2.55766169],
[14.09617368,  4.52134982,  2.55766169],
[17.22865672,  4.52134982,  2.55766169],
[ 3.13248304,  7.23415971,  2.55766169],
[ 6.26496608,  7.23415971,  2.55766169],
[ 9.39744912,  7.23415971,  2.55766169],
[12.52993216,  7.23415971,  2.55766169],
[15.6624152 ,  7.23415971,  2.55766169],
[18.79489824,  7.23415971,  2.55766169],
[ 4.69872456,  9.9469696 ,  2.55766169],
[ 7.8312076 ,  9.9469696 ,  2.55766169],
[10.96369064,  9.9469696 ,  2.55766169],
[14.09617368,  9.9469696 ,  2.55766169],
[17.22865672,  9.9469696 ,  2.55766169],
[20.36113976,  9.9469696 ,  2.55766169],
[ 6.26496608, 12.65977949,  2.55766169],
[ 9.39744912, 12.65977949,  2.55766169],
[12.52993216, 12.65977949,  2.55766169],
[15.6624152 , 12.65977949,  2.55766169],
[18.79489824, 12.65977949,  2.55766169],
[21.92738128, 12.65977949,  2.55766169],
[ 7.8312076 , 15.37258938,  2.55766169],
[10.96369064, 15.37258938,  2.55766169],
[14.09617368, 15.37258938,  2.55766169],
[17.22865672, 15.37258938,  2.55766169],
[20.36113976, 15.37258938,  2.55766169],
[23.4936228 , 15.37258938,  2.55766169],
[ 0.        ,  0.        ,  5.11532339],
[ 3.13248304,  0.        ,  5.11532339],
[ 6.26496608,  0.        ,  5.11532339],
[ 9.39744912,  0.        ,  5.11532339],
[12.52993216,  0.        ,  5.11532339],
[15.6624152 ,  0.        ,  5.11532339],
[ 1.56624152,  2.71280989,  5.11532339],
[ 4.69872456,  2.71280989,  5.11532339],
[ 7.8312076 ,  2.71280989,  5.11532339],
[10.96369064,  2.71280989,  5.11532339],
[14.09617368,  2.71280989,  5.11532339],
[17.22865672,  2.71280989,  5.11532339],
[ 3.13248304,  5.42561978,  5.11532339],
[ 6.26496608,  5.42561978,  5.11532339],
[ 9.39744912,  5.42561978,  5.11532339],
[12.52993216,  5.42561978,  5.11532339],
[15.6624152 ,  5.42561978,  5.11532339],
[18.79489824,  5.42561978,  5.11532339],
[ 4.69872456,  8.13842967,  5.11532339],
[ 7.8312076 ,  8.13842967,  5.11532339],
[10.96369064,  8.13842967,  5.11532339],
[14.09617368,  8.13842967,  5.11532339],
[17.22865672,  8.13842967,  5.11532339],
[20.36113976,  8.13842967,  5.11532339],
[ 6.26496608, 10.85123956,  5.11532339],
[ 9.39744912, 10.85123956,  5.11532339],
[12.52993216, 10.85123956,  5.11532339],
[15.6624152 , 10.85123956,  5.11532339],
[18.79489824, 10.85123956,  5.11532339],
[21.92738128, 10.85123956,  5.11532339],
[ 7.8312076 , 13.56404945,  5.11532339],
[10.96369064, 13.56404945,  5.11532339],
[14.09617368, 13.56404945,  5.11532339],
[17.22865672, 13.56404945,  5.11532339],
[20.36113976, 13.56404945,  5.11532339],
[23.4936228 , 13.56404945,  5.11532339],
[ 1.56624152,  0.90426996,  7.67298508],
[ 4.69872456,  0.90426996,  7.67298508],
[ 7.8312076 ,  0.90426996,  7.67298508],
[10.96369064,  0.90426996,  7.67298508],
[14.09617368,  0.90426996,  7.67298508],
[17.22865672,  0.90426996,  7.67298508],
[ 3.13248304,  3.61707985,  7.67298508],
[ 6.26496608,  3.61707985,  7.67298508],
[ 9.39744912,  3.61707985,  7.67298508],
[12.52993216,  3.61707985,  7.67298508],
[15.6624152 ,  3.61707985,  7.67298508],
[18.79489824,  3.61707985,  7.67298508],
[ 4.69872456,  6.32988974,  7.67298508],
[ 7.8312076 ,  6.32988974,  7.67298508],
[10.96369064,  6.32988974,  7.67298508],
[14.09617368,  6.32988974,  7.67298508],
[17.22865672,  6.32988974,  7.67298508],
[20.36113976,  6.32988974,  7.67298508],
[ 6.26496608,  9.04269963,  7.67298508],
[ 9.39744912,  9.04269963,  7.67298508],
[12.52993216,  9.04269963,  7.67298508],
[15.6624152 ,  9.04269963,  7.67298508],
[18.79489824,  9.04269963,  7.67298508],
[21.92738128,  9.04269963,  7.67298508],
[ 7.8312076 , 11.75550952,  7.67298508],
[10.96369064, 11.75550952,  7.67298508],
[14.09617368, 11.75550952,  7.67298508],
[17.22865672, 11.75550952,  7.67298508],
[20.36113976, 11.75550952,  7.67298508],
[23.4936228 , 11.75550952,  7.67298508],
[ 9.39744912, 14.46831941,  7.67298508],
[12.52993216, 14.46831941,  7.67298508],
[15.6624152 , 14.46831941,  7.67298508],
[18.79489824, 14.46831941,  7.67298508],
[21.92738128, 14.46831941,  7.67298508],
[25.05986433, 14.46831941,  7.67298508],
[ 0.        ,  1.80853993, 10.23064677],
[ 3.13248304,  1.80853993, 10.23064677],
[ 6.26496608,  1.80853993, 10.23064677],
[ 9.39744912,  1.80853993, 10.23064677],
[12.52993216,  1.80853993, 10.23064677],
[15.6624152 ,  1.80853993, 10.23064677],
[ 1.56624152,  4.52134982, 10.23064677],
[ 4.69872456,  4.52134982, 10.23064677],
[ 7.8312076 ,  4.52134982, 10.23064677],
[10.96369064,  4.52134982, 10.23064677],
[14.09617368,  4.52134982, 10.23064677],
[17.22865672,  4.52134982, 10.23064677],
[ 3.13248304,  7.23415971, 10.23064677],
[ 6.26496608,  7.23415971, 10.23064677],
[ 9.39744912,  7.23415971, 10.23064677],
[12.52993216,  7.23415971, 10.23064677],
[15.6624152 ,  7.23415971, 10.23064677],
[18.79489824,  7.23415971, 10.23064677],
[ 4.69872456,  9.9469696 , 10.23064677],
[ 7.8312076 ,  9.9469696 , 10.23064677],
[10.96369064,  9.9469696 , 10.23064677],
[14.09617368,  9.9469696 , 10.23064677],
[17.22865672,  9.9469696 , 10.23064677],
[20.36113976,  9.9469696 , 10.23064677],
[ 6.26496608, 12.65977949, 10.23064677],
[ 9.39744912, 12.65977949, 10.23064677],
[12.52993216, 12.65977949, 10.23064677],
[15.6624152 , 12.65977949, 10.23064677],
[18.79489824, 12.65977949, 10.23064677],
[21.92738128, 12.65977949, 10.23064677],
[ 7.8312076 , 15.37258938, 10.23064677],
[10.96369064, 15.37258938, 10.23064677],
[14.09617368, 15.37258938, 10.23064677],
[17.22865672, 15.37258938, 10.23064677],
[20.36113976, 15.37258938, 10.23064677],
[23.4936228 , 15.37258938, 10.23064677],
[ 0.        ,  0.        , 12.78830846],
[ 3.13248304,  0.        , 12.78830846],
[ 6.26496608,  0.        , 12.78830846],
[ 9.39744912,  0.        , 12.78830846],
[12.52993216,  0.        , 12.78830846],
[15.6624152 ,  0.        , 12.78830846],
[ 1.56624152,  2.71280989, 12.78830846],
[ 4.69872456,  2.71280989, 12.78830846],
[ 7.8312076 ,  2.71280989, 12.78830846],
[10.96369064,  2.71280989, 12.78830846],
[14.09617368,  2.71280989, 12.78830846],
[17.22865672,  2.71280989, 12.78830846],
[ 3.13248304,  5.42561978, 12.78830846],
[ 6.26496608,  5.42561978, 12.78830846],
[ 9.39744912,  5.42561978, 12.78830846],
[12.52993216,  5.42561978, 12.78830846],
[15.6624152 ,  5.42561978, 12.78830846],
[18.79489824,  5.42561978, 12.78830846],
[ 4.69872456,  8.13842967, 12.78830846],
[ 7.8312076 ,  8.13842967, 12.78830846],
[10.96369064,  8.13842967, 12.78830846],
[14.09617368,  8.13842967, 12.78830846],
[17.22865672,  8.13842967, 12.78830846],
[20.36113976,  8.13842967, 12.78830846],
[ 6.26496608, 10.85123956, 12.78830846],
[ 9.39744912, 10.85123956, 12.78830846],
[12.52993216, 10.85123956, 12.78830846],
[15.6624152 , 10.85123956, 12.78830846],
[18.79489824, 10.85123956, 12.78830846],
[21.92738128, 10.85123956, 12.78830846],
[ 7.8312076 , 13.56404945, 12.78830846],
[10.96369064, 13.56404945, 12.78830846],
[14.09617368, 13.56404945, 12.78830846],
[17.22865672, 13.56404945, 12.78830846],
[20.36113976, 13.56404945, 12.78830846],
[23.4936228 , 13.56404945, 12.78830846]]


# Apply scaling factor to atomic positions
pos_ar216 = pos_ar216 * scaling_factor_start

# Convert to Bohr for Argon
AtoBohr = 1.8897259886 * 0.98 
pos_ar216 = pos_ar216 * AtoBohr

# Box dimensions - ensure equal x and y lengths
box_length = 18.79489824 * AtoBohr * scaling_factor_start
box_height = 15.34597014 * AtoBohr * scaling_factor_start

# Use equal x and y dimensions in RhombicBC
bc_ar216 = RhombicBC(box_length, box_height)

# Check that the number of atoms and positions match
length(pos_ar216) == n_atoms || error("Number of atoms and positions do not match - check starting config")

# Create starting configuration
start_config = Config(pos_ar216, bc_ar216)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
mc_states, results = ptmc_run!(save_directory, mc_params, temp, start_config, pot, ensemble)

temp_result, cp = multihistogram_NPT(ensemble, temp, results, 10^(-9), false)
plot(temp_result, cp)

max_value, index = findmax(cp)
t_max = temp_result[index]
println(t_max)

