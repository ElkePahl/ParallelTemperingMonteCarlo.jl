using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the new code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

Random.seed!(1234)

# number of atoms
n_atoms = 32
pressure = 10e9

# temperature grid
ti = 900
tf = 1400
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
pos_ne32 =  [[ -4.3837,       -4.3837,       -4.3837],
  [-2.1918,       -2.1918,       -4.3837],
  [-2.1918,       -4.3837,       -2.1918],
  [-4.3837,       -2.1918,       -2.1918],
  [-4.3837,       -4.3837,        0.0000],
  [-2.1918,       -2.1918,        0.0000],
  [-2.1918,       -4.3837,        2.1918],
  [-4.3837,       -2.1918,        2.1918],
  [-4.3837,        0.0000,       -4.3837],
  [-2.1918,        2.1918,       -4.3837],
  [-2.1918,        0.0000,       -2.1918],
  [-4.3837,        2.1918,       -2.1918],
  [-4.3837,        0.0000,        0.0000],
  [-2.1918,        2.1918,        0.0000],
  [-2.1918,        0.0000,        2.1918],
  [-4.3837,        2.1918,        2.1918],
 [0.0000,       -4.3837,       -4.3837],
 [2.1918,       -2.1918,       -4.3837],
 [2.1918,       -4.3837,       -2.1918],
 [0.0000,       -2.1918,       -2.1918],
 [0.0000,       -4.3837,        0.0000],
 [2.1918,       -2.1918,        0.0000],
 [2.1918,       -4.3837,        2.1918],
 [0.0000,       -2.1918,        2.1918],
 [0.0000,        0.0000,       -4.3837],
 [2.1918,        2.1918,       -4.3837],
 [2.1918,        0.0000,       -2.1918],
 [0.0000,        2.1918,       -2.1918],
 [0.0000,        0.0000,        0.0000],
 [2.1918,        2.1918,       0.0000],
 [2.1918,        0.0000,        2.1918],
 [0.0000,        2.1918,        2.1918]]

#convert to Bohr
AtoBohr = 1.8897259886
pos_ne32 = pos_ne32 * AtoBohr * 0.816901409

#binding sphere
box_length = 8.7674 * AtoBohr * 0.816901409
bc_ne32 = CubicBC(box_length)   

length(pos_ne32) == n_atoms || error("number of atoms and positions not the same - check starting config")

start_config = Config(pos_ne32, bc_ne32)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
mc_states, results = ptmc_run!(mc_params,temp,start_config,pot,ensemble)

#to check code in REPL
#@profview ptmc_run!(mc_params,temp,start_config,pot,ensemble)
#@benchmark ptmc_run!(mc_params,temp,start_config,pot,ensemble)

## 