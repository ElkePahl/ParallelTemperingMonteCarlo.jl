using ParallelTemperingMonteCarlo
using Random
using Plots

function main()
    # -------------------------------------------------------#
    # -----------------------MC Params-----------------------#
    # -------------------------------------------------------#

    # Number of atoms
    n_atoms = 108
    pressure = 101325  # Pressure in Pa

    # Temperature grid
    ti = 100.0
    tf = 150.0
    n_traj = 25

    temp = TempGrid{n_traj}(ti, tf)

    # MC simulation details
    total_cycles = 1_000_000
    cycle_increment = 50_000
    n_increments = total_cycles ÷ cycle_increment  # Integer division

    mc_sample = 1  # Sample every mc_sample MC cycles

    displ_atom = 0.1  # Angstrom
    n_adjust = 100

    max_displ_atom = [0.1 * sqrt(displ_atom * temp.t_grid[i]) for i in 1:n_traj]

    mc_params = MCParams(cycle_increment, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

    save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data"

    # -------------------------------------------------------------#
    # ----------------------Potential------------------------------#
    # -------------------------------------------------------------#

    # Potential coefficients
    c = [-123.635101619510, 21262.8963716972, -3239750.64086661,
         189367623.844691, -4304257347.72069, 35315085074.3605]
    pot = ELJPotentialEven{6}(c)

    # -------------------------------------------------------------#
    # ------------------------Move Strategy------------------------#
    # -------------------------------------------------------------#

    separated_volume = false
    ensemble = NPT(n_atoms, pressure * 3.398928944382626e-14, separated_volume)
    move_strat = MoveStrategy(ensemble)

    # -------------------------------------------------------------#
    # -----------------------Starting Config-----------------------#
    # -------------------------------------------------------------#

    # Starting configurations
    r_start = 3.7782  # Desired min. radius between atoms in the starting config.
    L_start = 2 * sqrt(r_start^2 / 2)  # Distance between adjacent atoms along x or y axis
    Cell_Repeats = cbrt(n_atoms / 4)  # Number of times the unit cell is repeated in one direction

    if !isinteger(Cell_Repeats)
        error("Number of atoms not correct for fcc")
    end

    # Generate fcc starting arrangement
    pos = Vector{Vector{Float64}}()
    for i in 0:(Cell_Repeats - 1)
        x = i * L_start
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start
                push!(pos, [x, y, z])
            end
        end
    end

    # Generate atoms in the faces of the unit cell
    for i in 0:(Cell_Repeats - 1)
        x = i * L_start + L_start / 2
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start + L_start / 2
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start
                push!(pos, [x, y, z])
            end
        end
    end

    for i in 0:(Cell_Repeats - 1)
        x = i * L_start
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start + L_start / 2
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start + L_start / 2
                push!(pos, [x, y, z])
            end
        end
    end

    for i in 0:(Cell_Repeats - 1)
        x = i * L_start + L_start / 2
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start + L_start / 2
                push!(pos, [x, y, z])
            end
        end
    end

    # Center starting configuration at the origin
    center = [Cell_Repeats * L_start / 2, Cell_Repeats * L_start / 2, Cell_Repeats * L_start / 2]
    for l in 1:n_atoms
        pos[l] .-= center
    end

    # Convert to Bohr
    AtoBohr = 1.8897259886
    pos_ne32 = [p .* AtoBohr for p in pos]

    length(pos_ne32) == n_atoms || error("Number of atoms and positions not the same - check starting config")

    # Boundary conditions
    box_length = Cell_Repeats * L_start * AtoBohr
    bc_ne32 = CubicBC(box_length)

    # Starting configuration
    start_config = Config(pos_ne32, bc_ne32)

    # ----------------------------------------------------------------#
    # -------------------------Run Simulation-------------------------#
    # ----------------------------------------------------------------#

    # Initialize arrays to store results
    melting_temperatures = zeros(Float64, n_increments)
    mc_cycles_array = zeros(Int, n_increments)

    # Initialize current configuration
    current_config = deepcopy(start_config)

    # Initialize accumulated histograms
    # Run a small test to get histogram size
    mc_params_test = MCParams(10, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)
    _, results_test = ptmc_run!(save_directory, mc_params_test, temp, start_config, pot, ensemble)
    histogram_size = length(results_test.en_histogram[1])
    accumulated_histograms = [zeros(Float64, histogram_size) for _ in 1:n_traj]

    # Main simulation loop
    for i in 1:n_increments
        # Run ptmc_run! for cycle_increment cycles
        mc_states, results = ptmc_run!(save_directory, mc_params, temp, current_config, pot, ensemble)

        # Update current configuration with the last configuration from mc_states
        # Assuming mc_states is an array where mc_states[end] contains the final state
        current_config = deepcopy(mc_states[end].config)

        # Accumulate histograms
        for j in 1:n_traj
            accumulated_histograms[j] += results.en_histogram[j]
        end

        # Create a results struct with accumulated histograms
        accumulated_results = deepcopy(results)
        accumulated_results.en_histogram = accumulated_histograms

        # Run multihistogram analysis
        temp_result, cp = multihistogram_NPT(ensemble, temp, accumulated_results, 1e-9, false)

        # Find melting temperature
        max_value, index = findmax(cp)
        t_max = temp_result[index]

        # Record melting temperature and MC cycles
        melting_temperatures[i] = t_max
        mc_cycles_array[i] = i * cycle_increment

        # Print progress
        println("Completed ", mc_cycles_array[i], " cycles, melting temperature: ", t_max)
    end

    # Plot melting temperature vs MC cycles
    plot(mc_cycles_array, melting_temperatures,
         xlabel = "MC Cycles",
         ylabel = "Melting Temperature (K)",
         title = "Melting Temperature vs MC Cycles",
         legend = false)

    # Save the plot if needed
    # savefig("melting_temperature_convergence.png")
end

# Call the main function
main()
