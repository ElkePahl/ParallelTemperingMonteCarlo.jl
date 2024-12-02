using ParallelTemperingMonteCarlo
using Random
using Plots

# demonstration of the new version of the code   

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

# Random.seed!(1234)

# number of atoms
n_atoms = 216
pressure = 10e9

# temperature grid
ti = 600
tf = 1500
n_traj = 25

temp = TempGrid{n_traj}(ti, tf)

# MC simulation details

mc_cycles = 1000 # default 20% equilibration cycles on top
mc_sample = 1  # sample every mc_sample MC cycles

displ_atom = 0.1 # Angstrom 
n_adjust = 100

max_displ_atom = [0.1 * sqrt(displ_atom * temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Hmin"

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

# Include all 216 atom positions
pos_ar216 = [
    [ 1.56624152,  0.90426996,  0.00000000],
    [ 4.69872456,  0.90426996,  0.00000000],
    [ 7.83120760,  0.90426996,  0.00000000],
    [ 10.96369064,  0.90426996,  0.00000000],
    [ 14.09617368,  0.90426996,  0.00000000],
    [ 17.22865672,  0.90426996,  0.00000000],
    [ 3.13248304,  3.61707985,  0.00000000],
    [ 6.26496608,  3.61707985,  0.00000000],
    [ 9.39744912,  3.61707985,  0.00000000],
    [ 12.52993216,  3.61707985,  0.00000000],
    [ 15.66241520,  3.61707985,  0.00000000],
    [ 18.79489824,  3.61707985,  0.00000000],
    [ 4.69872456,  6.32988974,  0.00000000],
    [ 7.83120760,  6.32988974,  0.00000000],
    [ 10.96369064,  6.32988974,  0.00000000],
    [ 14.09617368,  6.32988974,  0.00000000],
    [ 17.22865672,  6.32988974,  0.00000000],
    [ 20.36113976,  6.32988974,  0.00000000],
    [ 6.26496608,  9.04269963,  0.00000000],
    [ 9.39744912,  9.04269963,  0.00000000],
    [ 12.52993216,  9.04269963,  0.00000000],
    [ 15.66241520,  9.04269963,  0.00000000],
    [ 18.79489824,  9.04269963,  0.00000000],
    [ 21.92738128,  9.04269963,  0.00000000],
    [ 7.83120760,  11.75550952,  0.00000000],
    [ 10.96369064,  11.75550952,  0.00000000],
    [ 14.09617368,  11.75550952,  0.00000000],
    [ 17.22865672,  11.75550952,  0.00000000],
    [ 20.36113976,  11.75550952,  0.00000000],
    [ 23.49362280,  11.75550952,  0.00000000],
    [ 9.39744912,  14.46831941,  0.00000000],
    [ 12.52993216,  14.46831941,  0.00000000],
    [ 15.66241520,  14.46831941,  0.00000000],
    [ 18.79489824,  14.46831941,  0.00000000],
    [ 21.92738128,  14.46831941,  0.00000000],
    [ 25.05986433,  14.46831941,  0.00000000],
    [ 0.00000000,  1.80853993,  2.55766169],
    [ 3.13248304,  1.80853993,  2.55766169],
    [ 6.26496608,  1.80853993,  2.55766169],
    [ 9.39744912,  1.80853993,  2.55766169],
    [ 12.52993216,  1.80853993,  2.55766169],
    [ 15.66241520,  1.80853993,  2.55766169],
    [ 1.56624152,  4.52134982,  2.55766169],
    [ 4.69872456,  4.52134982,  2.55766169],
    [ 7.83120760,  4.52134982,  2.55766169],
    [ 10.96369064,  4.52134982,  2.55766169],
    [ 14.09617368,  4.52134982,  2.55766169],
    [ 17.22865672,  4.52134982,  2.55766169],
    [ 3.13248304,  7.23415971,  2.55766169],
    [ 6.26496608,  7.23415971,  2.55766169],
    [ 9.39744912,  7.23415971,  2.55766169],
    [ 12.52993216,  7.23415971,  2.55766169],
    [ 15.66241520,  7.23415971,  2.55766169],
    [ 18.79489824,  7.23415971,  2.55766169],
    [ 4.69872456,  9.94696960,  2.55766169],
    [ 7.83120760,  9.94696960,  2.55766169],
    [ 10.96369064,  9.94696960,  2.55766169],
    [ 14.09617368,  9.94696960,  2.55766169],
    [ 17.22865672,  9.94696960,  2.55766169],
    [ 20.36113976,  9.94696960,  2.55766169],
    [ 6.26496608,  12.65977949,  2.55766169],
    [ 9.39744912,  12.65977949,  2.55766169],
    [ 12.52993216,  12.65977949,  2.55766169],
    [ 15.66241520,  12.65977949,  2.55766169],
    [ 18.79489824,  12.65977949,  2.55766169],
    [ 21.92738128,  12.65977949,  2.55766169],
    [ 7.83120760,  15.37258938,  2.55766169],
    [ 10.96369064,  15.37258938,  2.55766169],
    [ 14.09617368,  15.37258938,  2.55766169],
    [ 17.22865672,  15.37258938,  2.55766169],
    [ 20.36113976,  15.37258938,  2.55766169],
    [ 23.49362280,  15.37258938,  2.55766169],
    [ 1.56624152,  0.90426996,  5.11532339],
    [ 4.69872456,  0.90426996,  5.11532339],
    [ 7.83120760,  0.90426996,  5.11532339],
    [ 10.96369064,  0.90426996,  5.11532339],
    [ 14.09617368,  0.90426996,  5.11532339],
    [ 17.22865672,  0.90426996,  5.11532339],
    [ 3.13248304,  3.61707985,  5.11532339],
    [ 6.26496608,  3.61707985,  5.11532339],
    [ 9.39744912,  3.61707985,  5.11532339],
    [ 12.52993216,  3.61707985,  5.11532339],
    [ 15.66241520,  3.61707985,  5.11532339],
    [ 18.79489824,  3.61707985,  5.11532339],
    [ 4.69872456,  6.32988974,  5.11532339],
    [ 7.83120760,  6.32988974,  5.11532339],
    [ 10.96369064,  6.32988974,  5.11532339],
    [ 14.09617368,  6.32988974,  5.11532339],
    [ 17.22865672,  6.32988974,  5.11532339],
    [ 20.36113976,  6.32988974,  5.11532339],
    [ 6.26496608,  9.04269963,  5.11532339],
    [ 9.39744912,  9.04269963,  5.11532339],
    [ 12.52993216,  9.04269963,  5.11532339],
    [ 15.66241520,  9.04269963,  5.11532339],
    [ 18.79489824,  9.04269963,  5.11532339],
    [ 21.92738128,  9.04269963,  5.11532339],
    [ 7.83120760,  11.75550952,  5.11532339],
    [ 10.96369064,  11.75550952,  5.11532339],
    [ 14.09617368,  11.75550952,  5.11532339],
    [ 17.22865672,  11.75550952,  5.11532339],
    [ 20.36113976,  11.75550952,  5.11532339],
    [ 23.49362280,  11.75550952,  5.11532339],
    [ 9.39744912,  14.46831941,  5.11532339],
    [ 12.52993216,  14.46831941,  5.11532339],
    [ 15.66241520,  14.46831941,  5.11532339],
    [ 18.79489824,  14.46831941,  5.11532339],
    [ 21.92738128,  14.46831941,  5.11532339],
    [ 25.05986433,  14.46831941,  5.11532339],
    [ 0.00000000,  1.80853993,  7.67298508],
    [ 3.13248304,  1.80853993,  7.67298508],
    [ 6.26496608,  1.80853993,  7.67298508],
    [ 9.39744912,  1.80853993,  7.67298508],
    [ 12.52993216,  1.80853993,  7.67298508],
    [ 15.66241520,  1.80853993,  7.67298508],
    [ 1.56624152,  4.52134982,  7.67298508],
    [ 4.69872456,  4.52134982,  7.67298508],
    [ 7.83120760,  4.52134982,  7.67298508],
    [ 10.96369064,  4.52134982,  7.67298508],
    [ 14.09617368,  4.52134982,  7.67298508],
    [ 17.22865672,  4.52134982,  7.67298508],
    [ 3.13248304,  7.23415971,  7.67298508],
    [ 6.26496608,  7.23415971,  7.67298508],
    [ 9.39744912,  7.23415971,  7.67298508],
    [ 12.52993216,  7.23415971,  7.67298508],
    [ 15.66241520,  7.23415971,  7.67298508],
    [ 18.79489824,  7.23415971,  7.67298508],
    [ 4.69872456,  9.94696960,  7.67298508],
    [ 7.83120760,  9.94696960,  7.67298508],
    [ 10.96369064,  9.94696960,  7.67298508],
    [ 14.09617368,  9.94696960,  7.67298508],
    [ 17.22865672,  9.94696960,  7.67298508],
    [ 20.36113976,  9.94696960,  7.67298508],
    [ 6.26496608,  12.65977949,  7.67298508],
    [ 9.39744912,  12.65977949,  7.67298508],
    [ 12.52993216,  12.65977949,  7.67298508],
    [ 15.66241520,  12.65977949,  7.67298508],
    [ 18.79489824,  12.65977949,  7.67298508],
    [ 21.92738128,  12.65977949,  7.67298508],
    [ 7.83120760,  15.37258938,  7.67298508],
    [ 10.96369064,  15.37258938,  7.67298508],
    [ 14.09617368,  15.37258938,  7.67298508],
    [ 17.22865672,  15.37258938,  7.67298508],
    [ 20.36113976,  15.37258938,  7.67298508],
    [ 23.49362280,  15.37258938,  7.67298508],
    [ 1.56624152,  0.90426996,  10.23064677],
    [ 4.69872456,  0.90426996,  10.23064677],
    [ 7.83120760,  0.90426996,  10.23064677],
    [ 10.96369064,  0.90426996,  10.23064677],
    [ 14.09617368,  0.90426996,  10.23064677],
    [ 17.22865672,  0.90426996,  10.23064677],
    [ 3.13248304,  3.61707985,  10.23064677],
    [ 6.26496608,  3.61707985,  10.23064677],
    [ 9.39744912,  3.61707985,  10.23064677],
    [ 12.52993216,  3.61707985,  10.23064677],
    [ 15.66241520,  3.61707985,  10.23064677],
    [ 18.79489824,  3.61707985,  10.23064677],
    [ 4.69872456,  6.32988974,  10.23064677],
    [ 7.83120760,  6.32988974,  10.23064677],
    [ 10.96369064,  6.32988974,  10.23064677],
    [ 14.09617368,  6.32988974,  10.23064677],
    [ 17.22865672,  6.32988974,  10.23064677],
    [ 20.36113976,  6.32988974,  10.23064677],
    [ 6.26496608,  9.04269963,  10.23064677],
    [ 9.39744912,  9.04269963,  10.23064677],
    [ 12.52993216,  9.04269963,  10.23064677],
    [ 15.66241520,  9.04269963,  10.23064677],
    [ 18.79489824,  9.04269963,  10.23064677],
    [ 21.92738128,  9.04269963,  10.23064677],
    [ 7.83120760,  11.75550952,  10.23064677],
    [ 10.96369064,  11.75550952,  10.23064677],
    [ 14.09617368,  11.75550952,  10.23064677],
    [ 17.22865672,  11.75550952,  10.23064677],
    [ 20.36113976,  11.75550952,  10.23064677],
    [ 23.49362280,  11.75550952,  10.23064677],
    [ 9.39744912,  14.46831941,  10.23064677],
    [ 12.52993216,  14.46831941,  10.23064677],
    [ 15.66241520,  14.46831941,  10.23064677],
    [ 18.79489824,  14.46831941,  10.23064677],
    [ 21.92738128,  14.46831941,  10.23064677],
    [ 25.05986433,  14.46831941,  10.23064677],
    [ 0.00000000,  1.80853993,  12.78830846],
    [ 3.13248304,  1.80853993,  12.78830846],
    [ 6.26496608,  1.80853993,  12.78830846],
    [ 9.39744912,  1.80853993,  12.78830846],
    [ 12.52993216,  1.80853993,  12.78830846],
    [ 15.66241520,  1.80853993,  12.78830846],
    [ 1.56624152,  4.52134982,  12.78830846],
    [ 4.69872456,  4.52134982,  12.78830846],
    [ 7.83120760,  4.52134982,  12.78830846],
    [ 10.96369064,  4.52134982,  12.78830846],
    [ 14.09617368,  4.52134982,  12.78830846],
    [ 17.22865672,  4.52134982,  12.78830846],
    [ 3.13248304,  7.23415971,  12.78830846],
    [ 6.26496608,  7.23415971,  12.78830846],
    [ 9.39744912,  7.23415971,  12.78830846],
    [ 12.52993216,  7.23415971,  12.78830846],
    [ 15.66241520,  7.23415971,  12.78830846],
    [ 18.79489824,  7.23415971,  12.78830846],
    [ 4.69872456,  9.94696960,  12.78830846],
    [ 7.83120760,  9.94696960,  12.78830846],
    [ 10.96369064,  9.94696960,  12.78830846],
    [ 14.09617368,  9.94696960,  12.78830846],
    [ 17.22865672,  9.94696960,  12.78830846],
    [ 20.36113976,  9.94696960,  12.78830846],
    [ 6.26496608,  12.65977949,  12.78830846],
    [ 9.39744912,  12.65977949,  12.78830846],
    [ 12.52993216,  12.65977949,  12.78830846],
    [ 15.66241520,  12.65977949,  12.78830846],
    [ 18.79489824,  12.65977949,  12.78830846],
    [ 21.92738128,  12.65977949,  12.78830846],
    [ 7.83120760,  15.37258938,  12.78830846],
    [ 10.96369064,  15.37258938,  12.78830846],
    [ 14.09617368,  15.37258938,  12.78830846],
    [ 17.22865672,  15.37258938,  12.78830846],
    [ 20.36113976,  15.37258938,  12.78830846],
    [ 23.49362280,  15.37258938,  12.78830846],
]


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

# temp_result, cp = multihistogram_NPT(ensemble, temp, results, 10^(-9), false)
# plot(temp_result, cp)

filename = "all_rdfs.csv"
save_rdfs_concatenated(results.rdf, save_directory, filename)


# max_value, index = findmax(cp)
# t_max = temp_result[index]
# println(t_max)

# For REPL check
# @profview ptmc_run!(mc_params, temp, start_config, pot, ensemble)
# @benchmark ptmc_run!(mc_params, temp, start_config, pot, ensemble)
