using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

# number of atoms
n_atoms = 150
pressure = 101325

# temperature grid
ti = 10.
tf = 40.
n_traj = 16

temp = TempGrid{n_traj}(ti,tf) 

# MC simulation details

mc_cycles = 1000 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)


#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765] #ne
#c=[-123.63510161951,21262.8963716972,-3239750.64086661,189367623.844691,-4304257347.72069,35314085074.72069] #ar
pot = ELJPotentialEven{6}(c)

a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
potB = ELJPotentialB{6}(a,b,c1)


link="/Users/tiantianyu/Downloads/look-up_table.txt"
potlut=LookuptablePotential(link)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
separated_volume=false
ensemble = NPT(n_atoms,pressure*3.398928944382626e-14,separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations

#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne150 = [[ 1.56624152,  0.90426996,  0.        ],
[ 4.69872456,  0.90426996,  0.        ],
[ 7.8312076 ,  0.90426996,  0.        ],
[10.96369064,  0.90426996,  0.        ],
[14.09617368,  0.90426996,  0.        ],
[ 3.13248304,  3.61707985,  0.        ],
[ 6.26496608,  3.61707985,  0.        ],
[ 9.39744912,  3.61707985,  0.        ],
[12.52993216,  3.61707985,  0.        ],
[15.6624152 ,  3.61707985,  0.        ],
[ 4.69872456,  6.32988974,  0.        ],
[ 7.8312076 ,  6.32988974,  0.        ],
[10.96369064,  6.32988974,  0.        ],
[14.09617368,  6.32988974,  0.        ],
[17.22865672,  6.32988974,  0.        ],
[ 6.26496608,  9.04269963,  0.        ],
[ 9.39744912,  9.04269963,  0.        ],
[12.52993216,  9.04269963,  0.        ],
[15.6624152 ,  9.04269963,  0.        ],
[18.79489824,  9.04269963,  0.        ],
[ 7.8312076 , 11.75550952,  0.        ],
[10.96369064, 11.75550952,  0.        ],
[14.09617368, 11.75550952,  0.        ],
[17.22865672, 11.75550952,  0.        ],
[20.36113976, 11.75550952,  0.        ],
[ 0.        ,  1.80853993,  2.55766169],
[ 3.13248304,  1.80853993,  2.55766169],
[ 6.26496608,  1.80853993,  2.55766169],
[ 9.39744912,  1.80853993,  2.55766169],
[12.52993216,  1.80853993,  2.55766169],
[ 1.56624152,  4.52134982,  2.55766169],
[ 4.69872456,  4.52134982,  2.55766169],
[ 7.8312076 ,  4.52134982,  2.55766169],
[10.96369064,  4.52134982,  2.55766169],
[14.09617368,  4.52134982,  2.55766169],
[ 3.13248304,  7.23415971,  2.55766169],
[ 6.26496608,  7.23415971,  2.55766169],
[ 9.39744912,  7.23415971,  2.55766169],
[12.52993216,  7.23415971,  2.55766169],
[15.6624152 ,  7.23415971,  2.55766169],
[ 4.69872456,  9.9469696 ,  2.55766169],
[ 7.8312076 ,  9.9469696 ,  2.55766169],
[10.96369064,  9.9469696 ,  2.55766169],
[14.09617368,  9.9469696 ,  2.55766169],
[17.22865672,  9.9469696 ,  2.55766169],
[ 6.26496608, 12.65977949,  2.55766169],
[ 9.39744912, 12.65977949,  2.55766169],
[12.52993216, 12.65977949,  2.55766169],
[15.6624152 , 12.65977949,  2.55766169],
[18.79489824, 12.65977949,  2.55766169],
[ 0.        ,  0.        ,  5.11532339],
[ 3.13248304,  0.        ,  5.11532339],
[ 6.26496608,  0.        ,  5.11532339],
[ 9.39744912,  0.        ,  5.11532339],
[12.52993216,  0.        ,  5.11532339],
[ 1.56624152,  2.71280989,  5.11532339],
[ 4.69872456,  2.71280989,  5.11532339],
[ 7.8312076 ,  2.71280989,  5.11532339],
[10.96369064,  2.71280989,  5.11532339],
[14.09617368,  2.71280989,  5.11532339],
[ 3.13248304,  5.42561978,  5.11532339],
[ 6.26496608,  5.42561978,  5.11532339],
[ 9.39744912,  5.42561978,  5.11532339],
[12.52993216,  5.42561978,  5.11532339],
[15.6624152 ,  5.42561978,  5.11532339],
[ 4.69872456,  8.13842967,  5.11532339],
[ 7.8312076 ,  8.13842967,  5.11532339],
[10.96369064,  8.13842967,  5.11532339],
[14.09617368,  8.13842967,  5.11532339],
[17.22865672,  8.13842967,  5.11532339],
[ 6.26496608, 10.85123956,  5.11532339],
[ 9.39744912, 10.85123956,  5.11532339],
[12.52993216, 10.85123956,  5.11532339],
[15.6624152 , 10.85123956,  5.11532339],
[18.79489824, 10.85123956,  5.11532339],
[ 1.56624152,  0.90426996,  7.67298508],
[ 4.69872456,  0.90426996,  7.67298508],
[ 7.8312076 ,  0.90426996,  7.67298508],
[10.96369064,  0.90426996,  7.67298508],
[14.09617368,  0.90426996,  7.67298508],
[ 3.13248304,  3.61707985,  7.67298508],
[ 6.26496608,  3.61707985,  7.67298508],
[ 9.39744912,  3.61707985,  7.67298508],
[12.52993216,  3.61707985,  7.67298508],
[15.6624152 ,  3.61707985,  7.67298508],
[ 4.69872456,  6.32988974,  7.67298508],
[ 7.8312076 ,  6.32988974,  7.67298508],
[10.96369064,  6.32988974,  7.67298508],
[14.09617368,  6.32988974,  7.67298508],
[17.22865672,  6.32988974,  7.67298508],
[ 6.26496608,  9.04269963,  7.67298508],
[ 9.39744912,  9.04269963,  7.67298508],
[12.52993216,  9.04269963,  7.67298508],
[15.6624152 ,  9.04269963,  7.67298508],
[18.79489824,  9.04269963,  7.67298508],
[ 7.8312076 , 11.75550952,  7.67298508],
[10.96369064, 11.75550952,  7.67298508],
[14.09617368, 11.75550952,  7.67298508],
[17.22865672, 11.75550952,  7.67298508],
[20.36113976, 11.75550952,  7.67298508],
[ 0.        ,  1.80853993, 10.23064677],
[ 3.13248304,  1.80853993, 10.23064677],
[ 6.26496608,  1.80853993, 10.23064677],
[ 9.39744912,  1.80853993, 10.23064677],
[12.52993216,  1.80853993, 10.23064677],
[ 1.56624152,  4.52134982, 10.23064677],
[ 4.69872456,  4.52134982, 10.23064677],
[ 7.8312076 ,  4.52134982, 10.23064677],
[10.96369064,  4.52134982, 10.23064677],
[14.09617368,  4.52134982, 10.23064677],
[ 3.13248304,  7.23415971, 10.23064677],
[ 6.26496608,  7.23415971, 10.23064677],
[ 9.39744912,  7.23415971, 10.23064677],
[12.52993216,  7.23415971, 10.23064677],
[15.6624152 ,  7.23415971, 10.23064677],
[ 4.69872456,  9.9469696 , 10.23064677],
[ 7.8312076 ,  9.9469696 , 10.23064677],
[10.96369064,  9.9469696 , 10.23064677],
[14.09617368,  9.9469696 , 10.23064677],
[17.22865672,  9.9469696 , 10.23064677],
[ 6.26496608, 12.65977949, 10.23064677],
[ 9.39744912, 12.65977949, 10.23064677],
[12.52993216, 12.65977949, 10.23064677],
[15.6624152 , 12.65977949, 10.23064677],
[18.79489824, 12.65977949, 10.23064677],
[ 0.        ,  0.        , 12.78830846],
[ 3.13248304,  0.        , 12.78830846],
[ 6.26496608,  0.        , 12.78830846],
[ 9.39744912,  0.        , 12.78830846],
[12.52993216,  0.        , 12.78830846],
[ 1.56624152,  2.71280989, 12.78830846],
[ 4.69872456,  2.71280989, 12.78830846],
[ 7.8312076 ,  2.71280989, 12.78830846],
[10.96369064,  2.71280989, 12.78830846],
[14.09617368,  2.71280989, 12.78830846],
[ 3.13248304,  5.42561978, 12.78830846],
[ 6.26496608,  5.42561978, 12.78830846],
[ 9.39744912,  5.42561978, 12.78830846],
[12.52993216,  5.42561978, 12.78830846],
[15.6624152 ,  5.42561978, 12.78830846],
[ 4.69872456,  8.13842967, 12.78830846],
[ 7.8312076 ,  8.13842967, 12.78830846],
[10.96369064,  8.13842967, 12.78830846],
[14.09617368,  8.13842967, 12.78830846],
[17.22865672,  8.13842967, 12.78830846],
[ 6.26496608, 10.85123956, 12.78830846],
[ 9.39744912, 10.85123956, 12.78830846],
[12.52993216, 10.85123956, 12.78830846],
[15.6624152 , 10.85123956, 12.78830846],
[18.79489824, 10.85123956, 12.78830846]]

#convert to Bohr
#AtoBohr = 1.8897259886 * 0.98 * 1.25
AtoBohr = 1.0
pos_ne150 = pos_ne150 * AtoBohr


#binding sphere
box_length = 15.6624152 * AtoBohr
box_height = 15.34597014 * AtoBohr
bc_ne150 = RhombicBC(box_length, box_height)   

length(pos_ne150) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne150, bc_ne150)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
mc_states, results = ptmc_run!(mc_params,temp,start_config,potB,ensemble)

#to check code in REPL
#@profview ptmc_run!(mc_params,temp,start_config,pot,ensemble)
#@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)

#multihistogram_NPT(ensemble, temp, results, 10^(-9), false)

## 