###
### This script is equivalent to pbc32.jl, but uses fewer MC cycles. It's meant to be used
### during testing to check if it runs through successfully.
###
using ParallelTemperingMonteCarlo
using Random

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#
Random.seed!(1234)

n_atoms = 32
pressure = 101325
AtoBohr = 1.0

# temperature grid
ti = 30
tf = 50
n_traj = 24

temp = TempGrid{n_traj}(ti, tf)

mc_cycles = 1000
mc_sample = 1

displ_atom = 0.05 # Angstrom
n_adjust = 100

max_displ_atom = [0.1sqrt(displ_atom * temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms; mc_sample, n_adjust)

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

c = [-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)


link = joinpath(@__DIR__, "lookup-tables", "LookupTable_Neon_B0.0_MP2.txt")
potlut = LookuptablePotential(link)
#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
separated_volume = true
pressure_scale = 3.398928944382626e-14
ensemble = NPT(n_atoms,pressure * 2.2937122783969076e-13 / AtoBohr^3, separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
#starting configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne32 =  [
    [-4.3837,       -4.3837,       -4.3837],
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
    [0.0000,        -4.3837,       -4.3837],
    [2.1918,        -2.1918,       -4.3837],
    [2.1918,        -4.3837,       -2.1918],
    [0.0000,        -2.1918,       -2.1918],
    [0.0000,        -4.3837,        0.0000],
    [2.1918,        -2.1918,        0.0000],
    [2.1918,        -4.3837,        2.1918],
    [0.0000,        -2.1918,        2.1918],
    [0.0000,         0.0000,       -4.3837],
    [2.1918,         2.1918,       -4.3837],
    [2.1918,         0.0000,       -2.1918],
    [0.0000,         2.1918,       -2.1918],
    [0.0000,         0.0000,        0.0000],
    [2.1918,         2.1918,        0.0000],
    [2.1918,         0.0000,        2.1918],
    [0.0000,         2.1918,        2.1918],
]

pos_ne32 = pos_ne32 * AtoBohr

box_length = 8.7674 * AtoBohr
bc_ne32 = RectangularBC(box_length, box_length)

start_config_1 = Config(pos_ne32, bc_ne32)
start_config_2 = Config(pos_ne32, bc_ne32)
start_configs = [start_config_1, start_config_2]

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
mc_states, results = ptmc_run!(mc_params, temp, start_configs, potlut, ensemble)
