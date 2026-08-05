module MCRun

export metropolis_condition, mc_step!, mc_cycle!, ptmc_run!, get_energy!
export exc_acceptance, exc_trajectories!
export acc_test!, check_e_bounds, reset_counters, equilibration_cycle!, equilibration
export mc_move!
export ptmc_run_neigh!

using StaticArrays, DelimitedFiles
using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..Exchange
using ..ReadSave

using ..MCSampling

using ..Initialization
using ..CustomTypes

include("swap_config.jl")

#TODO update energy documentation
"""
    get_energy!(mc_state::MCState, pot, movetype::String)
    get_energy!(mc_state::MCState, pot, movetype::String)
    get_energy!(mc_state::MCState, pot, movetype::String)

Calculates energy for different ensembles and move types.
Currently implemented for:
        - NVT ensemble without r_cut
        - NPT ensemble with r_cut
        - NNVT ensemble for multiple-species atoms
"""
function get_energy!(
    mc_state::MCState{<:Any,<:Any,<:Any,E}, pot::AbstractPotential, movetype::String
) where {E<:NVTVariables}
    if movetype == "atommove"
        mc_state.potential_variables, mc_state.new_en = energy_update!(
            mc_state.ensemble_variables,
            mc_state.config,
            mc_state.potential_variables,
            mc_state.dist2_mat,
            mc_state.new_dist2_vec,
            mc_state.en_tot,
            pot,
        )
    end
    return mc_state
end
function get_energy!(
    mc_state::MCState{<:Any,<:Any,<:Any,E}, pot::AbstractDimerPotential, movetype::String
) where {E<:NPTVariables}
    if movetype == "atommove"
        mc_state.potential_variables, mc_state.new_en = energy_update!(
            mc_state.ensemble_variables,
            mc_state.config,
            mc_state.potential_variables,
            mc_state.dist2_mat,
            mc_state.new_dist2_vec,
            mc_state.en_tot,
            pot,
        )
    else
        mc_state.new_en = dimer_energy_config(
            mc_state.ensemble_variables.trial_config,
            mc_state.ensemble_variables.new_dist2_mat,
            mc_state.potential_variables,
            pot;
            new=true,
        )
    end
    return mc_state
end
function get_energy!(
    mc_state::MCState{<:Any,<:Any,<:Any,E}, pot::AbstractPotential, movetype::String
) where {E<:NNVTVariables}
    if movetype == "atommove"
        mc_state.potential_variables, mc_state.new_en = energy_update!(
            mc_state.ensemble_variables,
            mc_state.config,
            mc_state.potential_variables,
            mc_state.dist2_mat,
            mc_state.new_dist2_vec,
            mc_state.en_tot,
            pot,
        )
    else
        mc_state.potential_variables, mc_state.new_en = swap_energy_update(
            mc_state.ensemble_variables,
            mc_state.config,
            mc_state.potential_variables,
            mc_state.dist2_mat,
            mc_state.en_tot,
            pot,
        )
    end
    return mc_state
end

"""
    get_energy_neigh!(mc_state, pot, movetype::String)

Neighbour-restricted counterpart of [`get_energy!`](@ref).

Dispatches the energy calculation for the proposed move stored in `mc_state`.
For an `"atommove"`, calls [`energy_update_neigh!`](@ref), which updates the
RuNNer neural-network potential using only symmetry-function values affected by
the moved atom and its old/new neighbourhood.

This is only for atom moves using the neighbour-restricted
RuNNer potential path. An error is thrown for unsupported move types.

Returns the updated `mc_state`.
"""
function get_energy_neigh!(mc_state, pot, movetype::String)
    if movetype == "atommove"
        mc_state.potential_variables, mc_state.new_en = energy_update_neigh!(
            mc_state.ensemble_variables,
            mc_state.config,
            mc_state.potential_variables,
            mc_state.dist2_mat,
            mc_state.new_dist2_vec,
            mc_state.en_tot,
            pot
        )
    else
        error("get_energy_neigh! currently only supports atommove")
    end

    return mc_state
end


"""
    acc_test!(mc_state::MCState,ensemble,movetype::String)

Checks if metropolis condition is fulfilled, comparing it to a random variable in [0,1].
If the condition is met, the new variables become the current `mc_state` using [`swap_config!`](@ref).
`ensemble` and `movetype` dictate the exact calculation of the metropolis condition,
and the internal `potential_variables` within the mc_states dictate how [`swap_config!`](@ref) operates.
"""
function acc_test!(mc_state::MCState, ensemble, movetype::String)
    if metropolis_condition(movetype, mc_state, ensemble) >= rand()
        swap_config!(mc_state, movetype)
    end
end
"""
    mc_move!(mc_state::MCState,move_strat::MoveStrategy, potential, ensemble)

Basic move for one `mc_state` according to a `move_strat` dictating the types of moves allowed within the `ensemble` when moving across a `potential` defining the PES.
-   Calculates an index for the move
-   Generates either a volume or atom move depending on `movestrat[index]`
-   Calculates energy based on the pot and new move
-   Tests acc and swaps if relevant
"""
function mc_move!(
    mc_state::MCState, move_strat::MoveStrategy{N,E}, pot, ensemble
) where {N,E}
    mc_state.ensemble_variables.index = rand(1:N)

    mc_state = generate_move!(
        mc_state, move_strat.movestrat[mc_state.ensemble_variables.index], ensemble
    )

    mc_state = get_energy!(
        mc_state, pot, move_strat.movestrat[mc_state.ensemble_variables.index]
    )

    acc_test!(
        mc_state,
        move_strat.ensemble,
        move_strat.movestrat[mc_state.ensemble_variables.index],
    )

    return mc_state
end

"""
    mc_move_neigh!(mc_state, move_strat, pot, ensemble)

Neighbour-restricted counterpart of [`mc_move!`](@ref).

Performs one Monte Carlo move for a single `mc_state` according to the move
types defined by `move_strat` and the supplied `ensemble` and `pot`.

The function:

- selects an atom index,
- generates the proposed move,
- calculates the proposed energy using [`get_energy_neigh!`](@ref),
- performs the acceptance test using [`acc_test!`](@ref).

Returns the updated `mc_state`.
"""
function mc_move_neigh!(mc_state, move_strat, pot, ensemble)
    N = length(mc_state.config.positions)

    mc_state.ensemble_variables.index = rand(1:N)
    movetype = move_strat.movestrat[mc_state.ensemble_variables.index]

    mc_state = generate_move!(mc_state, movetype, ensemble)
    mc_state = get_energy_neigh!(mc_state, pot, movetype)
    acc_test!(mc_state, ensemble, movetype)

    return mc_state
end

"""
    mc_step!(mc_states::MCStateVector, move_strat::MoveStrategy{N, E}, pot, ensemble, n_steps::Int) where {N, E}

Distributes each state in `mc_state` to the [`mc_move!`](@ref) function in accordance with a `move_strat`, `ensemble` and `pot`.
"""
function mc_step!(
    mc_states::MCStateVector, move_strat::MoveStrategy{N,E}, pot, ensemble, n_steps::Int
) where {N,E}
    Threads.@threads for state in mc_states
        for i_step in 1:n_steps
            state = mc_move!(state, move_strat, pot, ensemble)
        end
    end
    return mc_states
end

"""
    mc_step_neigh!(
        mc_states::MCStateVector,
        move_strat::MoveStrategy{N,E},
        pot::Ptype,
        ensemble::Etype,
        n_steps::Int,
    ) where {N,E,Ptype,Etype}

Neighbour-restricted counterpart of [`mc_step!`](@ref).

Distributes the states in `mc_states` across Julia threads. For each state,
calls [`mc_step_neigh_single!`](@ref) to perform `n_steps` sequential
neighbour-restricted Monte Carlo moves according to `move_strat`, `pot`, and
`ensemble`.

Returns the updated `mc_states`.
"""
function mc_step_neigh!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    pot::Ptype,
    ensemble::Etype,
    n_steps::Int,
) where {N,E,Ptype,Etype}

    Threads.@threads for state in mc_states
        mc_step_neigh_single!(
            state,
            move_strat,
            pot,
            ensemble,
            n_steps,
        )
    end

    return mc_states
end

"""
    mc_step_neigh_single!(
        mc_state,
        move_strat,
        pot,
        ensemble,
        n_steps::Int,
    )

Performs `n_steps` sequential neighbour-restricted Monte Carlo moves for one
`mc_state`.

This helper function is called by [`mc_step_neigh!`](@ref), which distributes
different states across Julia threads. The Monte Carlo moves within each
individual state must be performed sequentially because every proposed move
depends on the accepted configuration produced by the preceding move.

Returns the updated `mc_state`.
"""
function mc_step_neigh_single!(
    mc_state,
    move_strat,
    pot,
    ensemble,
    n_steps::Int,
)
    for _ in 1:n_steps
        mc_state = mc_move_neigh!(
            mc_state,
            move_strat,
            pot,
            ensemble,
        )
    end

    return mc_state
end

"""
    mc_cycle!(mc_states::MCStateVector, move_strat::MoveStrategy{N, E}, mc_params::MCParams, pot, ensemble, n_steps::Int, index::Int) where {N, E}
    mc_cycle!(mc_states::MCStateVector, move_strat::MoveStrategy{N, E}, mc_params::MCParams, pot, ensemble, n_steps::Int, results::Output, idx::Int, rdfsave::Bool) where {N, E}

Basic function utilised by the simulation. For each of the `n_steps` run a single [`mc_step!`](@ref) on the `mc_states` according to `pot`, `move_strat` and `ensemble`, then complete the [`parallel_tempering_exchange!`](@ref) and `update_step_size!`.

Second method includes the [`sampling_step!`](@ref) which updates the `results` struct. The first method is used by the [`equilibration_cycle!`](@ref) and therefore does __not__ update the results struct.
"""
function mc_cycle!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot,
    ensemble,
    n_steps::Int,
    index::Int,
) where {N,E}
    mc_states = mc_step!(mc_states, move_strat, pot, ensemble, n_steps)

    if rand() < 0.1
        parallel_tempering_exchange!(mc_states, mc_params, ensemble)
    end
    if rem(index, mc_params.n_adjust) == 0
        for state in mc_states
            update_max_stepsize!(
                state, mc_params.n_adjust, ensemble, mc_params.min_acc, mc_params.max_acc
            )
        end
    end
    return mc_states
end
function mc_cycle!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot,
    ensemble,
    n_steps::Int,
    results::Output,
    idx::Int,
    rdfsave::Bool,
    potential,
) where {N,E}
    #TODO: Implement saving configurations after n steps

    mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, idx)

    if rem(idx, mc_params.mc_sample) == 0
        sampling_step!(mc_params, mc_states, ensemble, idx, results, rdfsave, idx)
    end

    return mc_states
end

"""
    mc_cycle_neigh!(
        mc_states::MCStateVector,
        move_strat::MoveStrategy{N,E},
        mc_params::MCParams,
        pot::Ptype,
        ensemble::Etype,
        n_steps::Int,
        index::Int,
    ) where {N,E,Ptype,Etype}

    mc_cycle_neigh!(
        mc_states::MCStateVector,
        move_strat::MoveStrategy{N,E},
        mc_params::MCParams,
        pot::Ptype,
        ensemble::Etype,
        n_steps::Int,
        results::Output,
        idx::Int,
        rdfsave::Bool,
    ) where {N,E,Ptype,Etype}

Neighbour-restricted counterpart of [`mc_cycle!`](@ref).

The first method advances every state by calling [`mc_step_neigh!`](@ref),
optionally attempts a [`parallel_tempering_exchange!`](@ref), and periodically
updates the maximum move step size according to the acceptance-rate bounds in
`mc_params`.

The second method additionally calls [`sampling_step!`](@ref) at the sampling
frequency defined by `mc_params.mc_sample`, updating the supplied `results`
structure. The first method is used during equilibration and therefore does not
sample results.

Returns the updated `mc_states`.
"""
function mc_cycle_neigh!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot::Ptype,
    ensemble::Etype,
    n_steps::Int,
    index::Int,
) where {N,E,Ptype,Etype}

    mc_states = mc_step_neigh!(
        mc_states,
        move_strat,
        pot,
        ensemble,
        n_steps,
    )

    if rand() < 0.1
        parallel_tempering_exchange!(
            mc_states,
            mc_params,
            ensemble,
        )
    end

    if rem(index, mc_params.n_adjust) == 0
        for state in mc_states
            update_max_stepsize!(
                state,
                mc_params.n_adjust,
                ensemble,
                mc_params.min_acc,
                mc_params.max_acc,
            )
        end
    end

    return mc_states
end


function mc_cycle_neigh!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot::Ptype,
    ensemble::Etype,
    n_steps::Int,
    results::Output,
    idx::Int,
    rdfsave::Bool,
) where {N,E,Ptype,Etype}

    mc_states = mc_cycle_neigh!(
        mc_states,
        move_strat,
        mc_params,
        pot,
        ensemble,
        n_steps,
        idx,
    )

    if rem(idx, mc_params.mc_sample) == 0
        sampling_step!(
            mc_params,
            mc_states,
            ensemble,
            idx,
            results,
            rdfsave,
            idx,
        )
    end

    return mc_states
end


"""
    check_e_bounds(energy::Number, ebounds::VorS)
Function to determine if an energy value is greater than or less than the min/max, used in equilibration cycle.
"""
function check_e_bounds(energy::Number, ebounds::VorS)
    if energy < ebounds[1]
        ebounds[1] = energy
    elseif energy > ebounds[2]
        ebounds[2] = energy
    end
    return ebounds
end
"""
    reset_counters(state::MCState)
After equilibration this resets the count stats to zero
"""
function reset_counters(state::MCState)
    state.count_atom = [0, 0]
    state.count_vol = [0, 0]
    state.count_vol_xy = [0, 0]
    state.count_vol_z = [0, 0]
    return state.count_exc = [0, 0]
end

"""
    equilibration_cycle!(mc_states::MCStateVector, move_strat::MoveStrategy{N, E}, mc_params::MCParams, pot, ensemble, n_steps::Int, results::Output) where {N, E}
Function to thermalise a set of `mc_states` ensuring that the number of equilibration cycles defined in `mc_params` are completed without updating the results before initialising the `results` struct according to the maximum and minimum energy determined throughout the equilibration cycle.
"""
function equilibration_cycle!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot,
    ensemble,
    n_steps::Int,
    results::Output,
) where {N,E}
    #set initial hamiltonian values and ebounds

    ebounds = [100.0, -100.0]
    # Don't touch ebound for the first half of the run in case energies
    # are very high at the beginning.
    for i in 1:(mc_params.eq_cycles ÷ 2)
        mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, i)
    end
    for i in (mc_params.eq_cycles ÷ 2 + 1):(mc_params.eq_cycles)
        for state in mc_states
            ebounds = check_e_bounds(state.en_tot, ebounds)
        end
    end
    #post equilibration reset
    for state in mc_states
        reset_counters(state)
    end
    results = initialise_histograms!(
        mc_params, results, ebounds, mc_states[1].config.boundary_condition
    )

    return mc_states, results
end

"""
    equilibration_cycle_neigh!(
        mc_states::MCStateVector,
        move_strat::MoveStrategy{N,E},
        mc_params::MCParams,
        pot::Ptype,
        ensemble::Etype,
        n_steps::Int,
        results::Output,
    ) where {N,E,Ptype,Etype}

Neighbour-restricted counterpart of [`equilibration_cycle!`](@ref).

Thermalises the states for `mc_params.eq_cycles` cycles without recording
production samples. During equilibration, the minimum and maximum encountered
energies are tracked and used to initialise the result histograms.

After equilibration, acceptance counters are reset and the result histograms
are initialised using the observed energy bounds.

Returns the updated `(mc_states, results)`.
"""
function equilibration_cycle_neigh!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot::Ptype,
    ensemble::Etype,
    n_steps::Int,
    results::Output,
) where {N,E,Ptype,Etype}

    ebounds = [100.0, -100.0]

    for i in 1:mc_params.eq_cycles
        mc_states = mc_cycle_neigh!(
            mc_states,
            move_strat,
            mc_params,
            pot,
            ensemble,
            n_steps,
            i,
        )

        for state in mc_states
            ebounds = check_e_bounds(
                state.en_tot,
                ebounds,
            )
        end
    end

    for state in mc_states
        reset_counters(state)
    end

    results = initialise_histograms!(
        mc_params,
        results,
        ebounds,
        mc_states[1].config.boundary_condition,
    )

    return mc_states, results
end

"""
    equilibration(mc_states::MCStateVector, move_strat::MoveStrategy{N, E}, mc_params::MCParams, pot, ensemble, n_steps::Int, results::Output, restart::Bool) where {N, E}
While initialisation sets `mc_states`, `params` etc. we require something to thermalise our simulation and set the histograms. This function is mostly a wrapper for the [`equilibration_cycle!`](@ref) function that optionally removes the thermalisation from restart.

N.B. Restart is currently non-functional, do not try use it
"""
function equilibration(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot,
    ensemble,
    n_steps::Int,
    results::Output,
    restart::Bool,
) where {N,E}
    for state in mc_states
        push!(state.ham, 0)
        push!(state.ham, 0)
    end

    if restart == true
        return mc_states, results
    else
        return equilibration_cycle!(
            mc_states, move_strat, mc_params, pot, ensemble, n_steps, results
        )
    end
end

"""
    equilibration_neigh!(
        mc_states::MCStateVector,
        move_strat::MoveStrategy{N,E},
        mc_params::MCParams,
        pot::Ptype,
        ensemble::Etype,
        n_steps::Int,
        results::Output,
        restart::Bool,
    ) where {N,E,Ptype,Etype}

Neighbour-restricted counterpart of [`equilibration`](@ref).

Initialises the Hamiltonian history required by the simulation. For a new run,
calls [`equilibration_cycle_neigh!`](@ref) to thermalise the states and
initialise the result histograms. If `restart` is true, equilibration is skipped
and the supplied states and results are returned unchanged.

Returns `(mc_states, results)`.
"""
function equilibration_neigh!(
    mc_states::MCStateVector,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    pot::Ptype,
    ensemble::Etype,
    n_steps::Int,
    results::Output,
    restart::Bool,
) where {N,E,Ptype,Etype}

    for state in mc_states
        push!(state.ham, 0)
        push!(state.ham, 0)
    end

    if restart
        return mc_states, results
    end

    return equilibration_cycle_neigh!(
        mc_states,
        move_strat,
        mc_params,
        pot,
        ensemble,
        n_steps,
        results,
    )
end

"""
    (ptmc_run!(mc_params::MCParams, temp::TempGrid, start_config::Config, potential, ensemble; rdfsave = false, restart = false, save = false, saveconfigs = false, configsname = "configuration", workingdirectory = pwd()))
    ptmc_run!(restart::Bool; rdfsave = false, save = 1000, eq_cycles = 0.2, saveconfigs = false, configsname = "configuration")

Main call for the ptmc program. Given `mc_params` dictating the number of cycles etc. the `temps` containing the temperature and beta values we aim to simulate, an initial `start_config` and the `potential` and `ensemble` we run a complete simulation, explicitly outputting the `mc_states` and `results` structs.
-   Second method:
The second method relies on a series of checkpoint files -see Checkpoint module [`ReadSave`](@ref)- to autoinitialise an MC cycle. Still accepts restart as an argument to indicate whether this is a clean start with configs or a restart from a checkpoint at a given index.


-   kwargs currently implemented are:
    -   `rdfsave::Bool` : tells the simulation whether or not to generate and save radial distribution functions (a resource intensive step) -- set to false
    -   `restart::Bool` : tells the simulation whether or not we are beginning from a partially complete simulation - set false for method one.
    -   `acc::Vector` : sets the min and max acceptance rates used to adjust stepsize for the simulation - set [0.4 0.6] for a target of 40-60% acceptance
    -   `save::Bool` or `Int` : tells the simulation whether to write checkpoints - set false for no save or integer expressing save frequency
    -   `saveconfigs::Bool` or `Int` : tells the simulation whether to save configurations - set false for no save or integer expressing save frequency
    -   `configsname::AbstractString` : tells the simulation what name to save configuration files under.

"""
function ptmc_run!(
    mc_params::MCParams,
    temp::TempGrid,
    start_config,
    potential,
    ensemble;
    rdfsave=false,
    restart=false,
    save=false,
    saveconfigs=false,
    configsname="configuration",
    workingdirectory=pwd(),
)
    # Initialisation
    cd(workingdirectory)
    if save ≢ false
        save_init(potential, ensemble, mc_params, temp)
    end

    mc_states, move_strategy, results, n_steps, start_counter = initialisation(
        mc_params, temp, start_config, potential, ensemble
    )

    # Equilibration
    mc_states, results = equilibration(
        mc_states, move_strategy, mc_params, potential, ensemble, n_steps, results, restart
    )
    if save ≢ false
        save_histparams(results)
    end

    @info "equilibration complete"

    # Main loop
    for i in start_counter:(mc_params.mc_cycles)
        mc_cycle!(
            mc_states,
            move_strategy,
            mc_params,
            potential,
            ensemble,
            n_steps,
            results,
            i,
            rdfsave,
            potential,
        )

        if save ≢ false && rem(i, save) == 0
            checkpoint(i, mc_states, results, ensemble, rdfsave)
        end
        if saveconfigs ≢ false && rem(i, saveconfigs) == 0
            save_configs(mc_states, string(configsname, i))
        end
        if rem(i, 100000) == 0 #TODO: this should be a progress bar
            @info "$i"
            #results = finalise_results_convergence(i,mc_states,mc_params,results)
            #println(results.heat_cap)
        end
    end
    @info "MC loop done."

    if save ≢ false && rem(mc_params.mc_cycles, save) ≠ 0
        # Save at the end if we didn't save in the last step.
        checkpoint(mc_params.mc_cycles, mc_states, results, ensemble, rdfsave)
    end

    #Finalisation of results
    results = finalise_results(mc_states, mc_params, results)
    return mc_states, results
end

# This method is used to resume a saved computation
function ptmc_run!(
    restart::Bool;
    rdfsave=false,
    save=1000,
    eq_cycles=0.2,
    saveconfigs=false,
    configsname="configuration",
)
    mc_params, ensemble, potential, mc_states, move_strategy, results, n_steps, start_counter = initialisation(
        restart, eq_cycles
    )

    mc_states, results = equilibration(
        mc_states, move_strategy, mc_params, potential, ensemble, n_steps, results, restart
    )
    @info "equilibration complete"

    if save ≢ false
        save_histparams(results)
    end

    for i in start_counter:(mc_params.mc_cycles)
        mc_cycle!(
            mc_states,
            move_strategy,
            mc_params,
            potential,
            ensemble,
            n_steps,
            results,
            i,
            rdfsave,
            potential,
        )
        if save ≢ false && rem(i, save) == 0
            checkpoint(i, mc_states, results, ensemble, rdfsave)
        end
        if saveconfigs ≢ false && rem(i, saveconfigs) == 0
            save_configs(mc_states, string(configsname, i))
        end
    end
    @info "MC loop done."

    results = finalise_results(mc_states, mc_params, results)

    return mc_states, results
end

"""
    ptmc_run_neigh!(
        mc_params::MCParams,
        temp::TempGrid,
        start_config::Config,
        potential::Ptype,
        ensemble::Etype;
        rdfsave::Bool=false,
        restart::Bool=false,
        save=false,
        workingdirectory=pwd(),
    ) where {Ptype,Etype}

Main entry point for a neighbour-restricted parallel-tempering Monte Carlo
simulation.

Initialises the Monte Carlo states, move strategy, result structures, and
temperature trajectories from `mc_params`, `temp`, `start_config`, `potential`,
and `ensemble`. The states are thermalised using
[`equilibration_neigh!`](@ref), then advanced through the production cycles
using [`mc_cycle_neigh!`](@ref).

The neighbour-restricted path updates only those neural-network symmetry
function values affected by each proposed atom move.

Keyword arguments:

- `rdfsave::Bool`: whether radial distribution functions are sampled and saved.
- `restart::Bool`: whether the simulation is being resumed from an existing run.
- `save`: `false` to disable checkpoints, or an integer checkpoint frequency.
- `workingdirectory`: directory in which simulation files are read and written.

Returns `(mc_states, results)` after finalising the simulation results.
"""
function ptmc_run_neigh!(
    mc_params::MCParams,
    temp::TempGrid,
    start_config::Config,
    potential::Ptype,
    ensemble::Etype;
    rdfsave::Bool=false,
    restart::Bool=false,
    save=false,
    workingdirectory=pwd(),
) where {Ptype,Etype}

    cd(workingdirectory)

    if save != false
        save_init(
            potential,
            ensemble,
            mc_params,
            temp,
        )
    end

    mc_states,
    move_strategy,
    results,
    n_steps,
    start_counter = initialisation(
        mc_params,
        temp,
        start_config,
        potential,
        ensemble,
    )

    println("Parameters set")
    println("Number of trajectories: ", length(mc_states))
    println("Moves per cycle per trajectory: ", n_steps)
    println("Equilibration cycles: ", mc_params.eq_cycles)
    println("Production cycles: ", mc_params.mc_cycles)

    mc_states, results = equilibration_neigh!(
        mc_states,
        move_strategy,
        mc_params,
        potential,
        ensemble,
        n_steps,
        results,
        restart,
    )

    if save != false
        save_histparams(results)
    end

    println("Equilibration complete")

    for i in start_counter:mc_params.mc_cycles
        mc_states = mc_cycle_neigh!(
            mc_states,
            move_strategy,
            mc_params,
            potential,
            ensemble,
            n_steps,
            results,
            i,
            rdfsave,
        )

        if save == false
            # No checkpoint.
        elseif rem(i, save) == 0
            checkpoint(
                i,
                mc_states,
                results,
                ensemble,
                rdfsave,
            )
        end
    end

    println("MC loop complete")

    results = finalise_results(
        mc_states,
        mc_params,
        results,
    )

    println("Results finalised")

    return mc_states, results
end


#---------------------------------------------------------#
#-------------Notes for Future Implementation-------------#
#---------------------------------------------------------#
"""
-- TO IMPLEMENT --

This version is not complete. While "under the hood" is working as it should, not a lot of effort has been put into:

    - Organising the keyword arguments to be more intuitive
    - Expanding the initialise functions to set the type of results we wish to collect (eg no RDF, save configs as well as checkpoints)
"""

end
