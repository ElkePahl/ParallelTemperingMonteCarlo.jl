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
        )
    end

    return mc_states
end


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
        mc_states[1].config.bc,
    )

    return mc_states, results
end


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

function acc_test_with_rand!(mc_state::MCState, ensemble, movetype::String, r_accept::Float64)
    if metropolis_condition(movetype, mc_state, ensemble) >= r_accept
        swap_config!(mc_state, movetype)
    end

    return mc_state
end

function mc_move_neigh!(mc_state, move_strat, pot, ensemble)
    N = length(mc_state.config.pos)

    mc_state.ensemble_variables.index = rand(1:N)
    movetype = move_strat.movestrat[mc_state.ensemble_variables.index]

    mc_state = generate_move!(mc_state, movetype)
    mc_state = get_energy_neigh!(mc_state, pot, movetype)
    acc_test!(mc_state, ensemble, movetype)

    return mc_state
end

function neighbour_union_from_cutoff_vectors(f_old_row, f_new_vec, atomindex)
    neighbours = Int[]

    for j in eachindex(f_new_vec)
        if j != atomindex && (f_old_row[j] != 0.0 || f_new_vec[j] != 0.0)
            push!(neighbours, j)
        end
    end

    return neighbours
end