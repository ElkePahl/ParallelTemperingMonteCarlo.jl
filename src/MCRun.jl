module MCRun

export metropolis_condition, mc_cycle!, ptmc_run!, get_energy!
export exc_acceptance, exc_trajectories!
export acc_test!, check_e_bounds, reset_counters, equilibration_cycle!, equilibration
export mc_move!

using StaticArrays, DelimitedFiles, ProgressMeter, DataFrames, Arrow
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

include("Reporter.jl")

"""
    mc_cycle!(mc_states, move_strat::MoveStrategy, mc_params::MCParams, n_steps, index)
    mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, results, idx, rdfsave)

Basic function utilised by the simulation. For each of the `n_steps` run a single step on the `mc_states` according to `move_strat`, then complete the [`parallel_tempering_exchange!`](@ref) and `update_step_size!`.

Second method includes the [`sampling_step!`](@ref) which updates the `results` struct. The first method is used by the [`equilibration_cycle!`](@ref) and therefore does __not__ update the results struct.
"""
function mc_cycle!(
    mc_states,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    n_steps::Int,
    index::Int,
    reporter,
) where {N,E}
    Threads.@threads for trajectory_id in eachindex(mc_states)
        state = mc_states[trajectory_id]
        n_accepted = 0

        for i_step in 1:n_steps
            n_accepted += mc_move!(state, move_strat)
        end

        # stats get written to each state so we don't asynchronously push to the DataFrame
        state.step += 1
        state.acceptance = n_accepted / n_steps
        state.last_stats = (;
            cycle=state.step,
            trajectory_id=Int16(trajectory_id), # Int16 to conserve RAM
            temperature=state.temp,
            hamiltonian=hamiltonian_value(state, state.ensemble),
            total_energy=state.en_tot,
            acceptance=Float32(state.acceptance),
            report_stats(state, state.ensemble)...,
        )
    end
    ensemble = mc_states[1].ensemble

    if rand() < 0.1
        exchange_index = parallel_tempering_exchange!(mc_states, mc_params, ensemble)
    else
        exchange_index = 0
    end
    if rem(index, mc_params.n_adjust) == 0
        for state in mc_states
            update_max_stepsize!(
                state, mc_params.n_adjust, ensemble, mc_params.min_acc, mc_params.max_acc
            )
        end
    end
    for (i, state) in enumerate(mc_states)
        exchanged = i ∈ (exchange_index, exchange_index + 1)
        report!(reporter, (; state.last_stats..., exchanged))
    end

    flush!(reporter, index)
    return mc_states
end
function mc_cycle!(
    mc_states,
    move_strat,
    mc_params::MCParams,
    n_steps::Int,
    results::Output,
    idx::Int,
    rdfsave::Bool,
    potential,
    reporter,
)
    #TODO: Implement saving configurations after n steps

    mc_cycle!(mc_states, move_strat, mc_params, n_steps, idx, reporter)
    ensemble = mc_states[1].ensemble

    if rem(idx, mc_params.mc_sample) == 0
        sampling_step!(mc_params, mc_states, ensemble, idx, results, rdfsave, idx)
    end

    return mc_states
end

function hamiltonian_value(state::MCState, ::AbstractEnsemble)
    return state.en_tot
end
function hamiltonian_value(state::MCState, ensemble::NPT)
    return state.en_tot + ensemble.pressure * volume(state.config.boundary_condition)
end

"""
    check_e_bounds(energy, ebounds)
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
    equilibration_cycle!(mc_states, move_strat, mc_params::MCParams, n_steps, results::Output, stats)

Function to thermalise a set of `mc_states` ensuring that the number of equilibration cycles
defined in `mc_params` are completed without updating the results before initialising the
`results` struct according to the maximum and minimum energy determined throughout the
equilibration cycle.
"""
function equilibration_cycle!(
    mc_states,
    move_strat::MoveStrategy{N,E},
    mc_params::MCParams,
    n_steps::Int,
    results::Output,
    reporter,
) where {N,E}
    ebounds = [100.0, -100.0]
    # Don't touch ebound for the first half of the run in case energies
    # are very high at the beginning.
    progress = Progress(
        mc_params.eq_cycles;
        desc="Equilibration",
        enabled=(!reporter.verbose && !isinteractive()),
    )
    for i in 1:(mc_params.eq_cycles ÷ 2)
        mc_cycle!(mc_states, move_strat, mc_params, n_steps, i, reporter)
        next!(progress)
    end
    for i in (mc_params.eq_cycles ÷ 2 + 1):(mc_params.eq_cycles)
        mc_cycle!(mc_states, move_strat, mc_params, n_steps, i, reporter)
        for state in mc_states
            ebounds = check_e_bounds(state.en_tot, ebounds)
        end
        next!(progress)
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

#TODO: why is restart not functional?
"""
    equilibration(mc_states, move_strat, mc_params, pot, ensemble, n_steps::Int, results::Output, restart, stats)

While initialisation sets `mc_states`, `params` etc. we require something to thermalise our simulation and set the histograms. This function is mostly a wrapper for the [`equilibration_cycle!`](@ref) function that optionally removes the thermalisation from restart.

N.B. Restart is currently non-functional, do not try use it
"""
function equilibration(
    mc_states,
    move_strat,
    mc_params::MCParams,
    n_steps::Int,
    results::Output,
    restart::Bool,
    reporter,
) where {N,E}
    for state in mc_states
        push!(state.ham, 0)
        push!(state.ham, 0)
    end

    if restart == true
        return mc_states, results
    else
        return equilibration_cycle!(
            mc_states, move_strat, mc_params, n_steps, results, reporter
        )
    end
end
"""
    ptmc_run!(
        mc_params::MCParams,
        temp::TempGrid,
        start_config::Config,
        potential,
        ensemble;
        rdfsave = false,
        restart = false,
        save = false,
        saveconfigs = false,
        configsname = "configuration",
        workingdirectory = pwd(),
        stats_filename = nothing,
        flush_interval = 1_000_000,
    )
    ptmc_run!(
        restart::Bool;
        rdfsave = false,
        save = 1000,
        eq_cycles = 0.2,
        saveconfigs = false,
        configsname = "configuration",
    )

Main call for the ptmc program. Given `mc_params` dictating the number of cycles etc. the `temps` containing the temperature and beta values we aim to simulate, an initial `start_config` and the `potential` and `ensemble` we run a complete simulation, explicitly outputting the `mc_states` and `results` structs.
-   Second method:
The second method relies on a series of checkpoint files -see Checkpoint module [`ReadSave`](@ref)- to autoinitialise an MC cycle. Still accepts restart as an argument to indicate whether this is a clean start with configs or a restart from a checkpoint at a given index.


## Keyword arguments

- `rdfsave=false`: tells the simulation whether or not to generate and save radial
  distribution functions (a resource intensive step).
- `restart=false`: tells the simulation whether or not we are beginning from a partially
  complete simulation.
- `save::Bool` or `Int`: tells the simulation whether to write checkpoints - set false for
  no save or integer expressing save frequency.
- `saveconfigs::Bool` or `Int` : tells the simulation whether to save configurations - set
  false for no save or integer expressing save frequency
- `configsname::AbstractString` : tells the simulation what name to save configuration files
  under.
- `stats_filename=nothing`: if set to an arrow filename, the stats will be written to that
  file.
- `flush_interval=100_000`: if `stats_filename ≢ nothing`, the stats will be periodically
  flushed to disk.
- `verbose=!isinteractive()`: if set to `true` a message is printed on flush.
- `return_stats=true`: if set to `false`, the stats `DataFrame` is not returned.
- `extra_metadata`: additional metadata to be recorded in the `DataFrame`. Should be an
  iterable of key-value pairs. Both the keys and the values are converted to strings.
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
    stats_filename=nothing,
    flush_interval=100_000,
    verbose=(!isinteractive()),
    return_stats=true,
    extra_metadata=nothing,
)
    # Initialisation
    cd(workingdirectory)
    if save ≢ false
        save_init(potential, ensemble, mc_params, temp)
    end

    reporter = Reporter(
        start_config,
        potential,
        ensemble,
        mc_params;
        flush_interval,
        filename=stats_filename,
        verbose,
        return_stats,
        extra_metadata,
    )

    mc_states, move_strategy, results, n_steps, start_counter = initialisation(
        mc_params, temp, start_config, potential, ensemble
    )

    # Equilibration
    mc_states, results = equilibration(
        mc_states, move_strategy, mc_params, n_steps, results, restart, reporter
    )
    if save ≢ false
        save_histparams(results)
    end

    # Main loop
    progress = Progress(
        length(start_counter:(mc_params.mc_cycles));
        desc="Main loop",
        enabled=(!reporter.verbose && !isinteractive()),
    )
    for i in start_counter:(mc_params.mc_cycles)
        mc_cycle!(
            mc_states,
            move_strategy,
            mc_params,
            n_steps,
            results,
            i,
            rdfsave,
            potential,
            reporter,
        )
        if save ≢ false && rem(i, save) == 0
            checkpoint(i, mc_states, results, ensemble, rdfsave)
        end
        if saveconfigs ≢ false && rem(i, saveconfigs) == 0
            save_configs(mc_states, string(configsname, i))
        end
        next!(progress)
    end

    stats = finalise!(reporter)

    #Finalisation of results
    results = finalise_results(mc_states, mc_params, results)
    return mc_states, results, stats
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
    stats = NamedTuple[]

    mc_states, results = equilibration(
        mc_states, move_strategy, mc_params, n_steps, results, restart, stats
    )
    @info "equilibration complete"

    if save ≢ false
        save_histparams(results)
    end

    for i in start_counter:(mc_params.mc_cycles)
        mc_cycle!(mc_states, move_strategy, mc_params, n_steps, results, i, rdfsave, stats)
        if save ≢ false && rem(i, save) == 0
            checkpoint(i, mc_states, results, ensemble, rdfsave)
        end
        if saveconfigs ≢ false && rem(i, saveconfigs) == 0
            save_configs(mc_states, string(configsname, i))
        end
    end
    @info "MC loop done."

    results = finalise_results(mc_states, mc_params, results)

    return mc_states, results, stats
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
