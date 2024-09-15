using ParallelTemperingMonteCarlo
using Random
using Plots

#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#

# number of atoms
n_atoms = 32
pressure = 101325

# temperature grid
ti = 90
tf = 500
n_traj = 25

temp = TempGrid{n_traj}(ti, tf)

# MC simulation details
total_mc_cycles = 10000  # Total MC cycles
cycle_interval =  5000     # Interval for performing multi-histogram analysis
mc_sample = 1              # Sample every mc_sample MC cycles

displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1 * sqrt(displ_atom * temp.t_grid[i]) for i in 1:n_traj]

save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"
output_file = "melting_temperatures.txt"

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#
c = [-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]
pot = ELJPotentialEven{6}(c)

#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
separated_volume = false
ensemble = NPT(n_atoms, pressure * 3.398928944382626e-14, separated_volume)
move_strat = MoveStrategy(ensemble)

#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#
# Starting configuration (unchanged from your original script)
r_start = 3.7782
L_start = 2 * (r_start^2 / 2)^0.5
Cell_Repeats = cbrt(n_atoms / 4)

if isinteger(Cell_Repeats) == false error("Number of atoms not correct for FCC") end

# Generate simple cubic atoms
pos = Vector{Float64}[]
for i in 0:(Cell_Repeats-1)
    x = i * L_start
    for j in 0:(Cell_Repeats-1)
        y = j * L_start
        for k in 0:(Cell_Repeats-1)
            z = k * L_start
            pos_entry = [x, y, z]
            push!(pos, pos_entry)
        end
    end
end 

# Generate atoms in the face of the unit cell
for i in 0:(Cell_Repeats-1)
    x = i * L_start + L_start / 2
    for j in 0:(Cell_Repeats-1)
        y = j * L_start + L_start / 2
        for k in 0:(Cell_Repeats-1)
            z = k * L_start
            pos_entry = [x, y, z]
            push!(pos, pos_entry)
        end
    end
end 

for i in 0:(Cell_Repeats-1)
    x = i * L_start
    for j in 0:(Cell_Repeats-1)
        y = j * L_start + L_start / 2
        for k in 0:(Cell_Repeats-1)
            z = k * L_start + L_start / 2
            pos_entry = [x, y, z]
            push!(pos, pos_entry)
        end
    end
end 

for i in 0:(Cell_Repeats-1)
    x = i * L_start + L_start / 2
    for j in 0:(Cell_Repeats-1)
        y = j * L_start
        for k in 0:(Cell_Repeats-1)
            z = k * L_start + L_start / 2
            pos_entry = [x, y, z]
            push!(pos, pos_entry)
        end
    end
end 

# Center the starting configuration at the origin
center = Vector{Float64}([Cell_Repeats * L_start / 2, Cell_Repeats * L_start / 2, Cell_Repeats * L_start / 2])
for l in 1:n_atoms
    pos[l] = pos[l] - center
end

pos_ne32 = pos
AtoBohr = 1.8897259886
pos_ne32 = pos_ne32 * AtoBohr
box_length = Cell_Repeats * L_start * AtoBohr
bc_ne32 = CubicBC(box_length)
start_config = Config(pos_ne32, bc_ne32)

#-------------------------------------------------------------#
#-----------------------Run Simulation------------------------#
#-------------------------------------------------------------#

# Open file to save results
open(output_file, "w") do io
    for mc_cycle in cycle_interval:cycle_interval:total_mc_cycles
        # Create a new instance of MCParams for each interval
        mc_params = MCParams(mc_cycle, n_traj, n_atoms, mc_sample=mc_sample, n_adjust=n_adjust)

        # Run PTMC simulation for this interval
        mc_states, results = ptmc_run!(save_directory, mc_params, temp, start_config, pot, ensemble)

        # Perform multi-histogram analysis after 50000 cycles
        if mc_cycle >= cycle_interval
            temp_result, cp = multihistogram_NPT(ensemble, temp, results, 10^(-9), false)
            plot(temp_result, cp)

            # Find the melting temperature (temperature corresponding to maximum Cp)
            max_value, index = findmax(cp)
            t_max = temp_result[index]

            # Save the cycle number and melting temperature
            println(io, "$mc_cycle $t_max")

            # Log to console
            println("Cycle: $mc_cycle, Melting Temperature: $t_max")
        end
    end
end
