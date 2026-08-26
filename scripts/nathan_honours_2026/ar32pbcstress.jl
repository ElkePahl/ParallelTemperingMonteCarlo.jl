# Melting of argon under constant stress
# ======================================
#= This is a calculation for the high-pressure melting of crystalline argon, using a
rectangular simulation box, constant pressure, and constant tensorial stress =#

using ParallelTemperingMonteCarlo
using Random 
using DelimitedFiles
# ## Setting up the model 

# for testing we use a set random seed

#Random.seed!(1234)

n_atoms = 32
AtoBohr = 1.8897261259077824

# temperature grid

ti = 4000
tf = 5000
n_traj = 24
temp = TempGrid{n_traj}(ti, tf)

# MC Simulation details

mc_cycles = 40000
displ_atom = 0.05 # in Angstrom
mc_sample = 1
n_adjust = 100
max_displ_atom = [0.1 * √(displ_atom * temp.t_grid[i]) for i in 1:n_traj]
mc_params = MCParams(mc_cycles, n_traj, n_atoms; mc_sample=mc_sample, n_adjust=n_adjust)

# Lennard Jones coefficients

c = [
    −123.635101619510,
    21262.8963716972,
    −3239750.64086661,
    189367623.844691,
    −4304257347.72069,
    35315085074.3605,
]

pot = ELJPotentialEven{6}(c)
separated_volume = true
pos_ne32 = [
    [-4.3837, -4.3837, -4.3837],
    [-2.1918, -2.1918, -4.3837],
    [-2.1918, -4.3837, -2.1918],
    [-4.3837, -2.1918, -2.1918],
    [-4.3837, -4.3837, 0.0000],
    [-2.1918, -2.1918, 0.0000],
    [-2.1918, -4.3837, 2.1918],
    [-4.3837, -2.1918, 2.1918],
    [-4.3837, 0.0000, -4.3837],
    [-2.1918, 2.1918, -4.3837],
    [-2.1918, 0.0000, -2.1918],
    [-4.3837, 2.1918, -2.1918],
    [-4.3837, 0.0000, 0.0000],
    [-2.1918, 2.1918, 0.0000],
    [-2.1918, 0.0000, 2.1918],
    [-4.3837, 2.1918, 2.1918],
    [0.0000, -4.3837, -4.3837],
    [2.1918, -2.1918, -4.3837],
    [2.1918, -4.3837, -2.1918],
    [0.0000, -2.1918, -2.1918],
    [0.0000, -4.3837, 0.0000],
    [2.1918, -2.1918, 0.0000],
    [2.1918, -4.3837, 2.1918],
    [0.0000, -2.1918, 2.1918],
    [0.0000, 0.0000, -4.3837],
    [2.1918, 2.1918, -4.3837],
    [2.1918, 0.0000, -2.1918],
    [0.0000, 2.1918, -2.1918],
    [0.0000, 0.0000, 0.0000],
    [2.1918, 2.1918, 0.0000],
    [2.1918, 0.0000, 2.1918],
    [0.0000, 2.1918, 2.1918],
]

positions = pos_ne32 * AtoBohr * 3.7782/3.0985
box_length = 8.7674 * 3.7782/3.0985 * AtoBohr
boundary_condition = RectangularBC(box_length, box_length)

start_config = Config(positions, boundary_condition)

#----------------------------------------------------------------#
#-------------------------Run Simulation-------------------------#
#----------------------------------------------------------------#
pascal_pressure = 50e9
pressure = pascal_pressure * 3.3989e-14
relative_stress = 0
stress = relative_stress * pressure
ensemble = NPT(
    n_atoms, 
    pressure,
    separated_volume,
    [-stress/2, stress]
    )
mc_states, results = ptmc_run!(mc_params, temp, start_config, pot, ensemble; save=1000)
T, Cp = multihistogram_NPT(ensemble, temp, results, 1e-10, false; debug=false)