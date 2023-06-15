"""
    module Exchange

Here we include methods for calculating the metropolis condition and other exchange criteria required for Monte Carlo steps. This further declutters the MCRun module and allows us to split the cycle. Includes update_max_stepsize which controls the frequency of
"""

module Exchange


using ..MCStates
using ..Configurations
using ..EnergyEvaluation

export metropolis_condition, exc_acceptance,exc_trajectories!

export parallel_tempering_exchange!,update_max_stepsize!


"""
    metropolis_condition(ensemble, delta_en, beta)
Returns probability to accept a MC move at inverse temperature `beta` 
for energy difference `delta_en` between new and old configuration 
for given ensemble; implemented: 
    - `NVT`: canonical ensemble
    - `NPT`: NPT ensemble
Asymmetric Metropolis criterium, p = 1.0 if new configuration more stable, 
Boltzmann probability otherwise
"""
function metropolis_condition(::NVT, delta_energy, beta)
    prob_val = exp(-delta_energy*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function metropolis_condition(::NPT, delta_energy, beta)
    prob_val = exp(-delta_energy*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

#function metropolis_condition(::NPT, N, d_en, volume_changed, volume_unchanged, pressure, beta)
    #delta_h = d_en + pressure*(volume_changed-volume_unchanged)*JtoEh*Bohr3tom3
    #prob_val = exp(-delta_h*beta + NAtoms*log(volume_changed/volume_unchanged))
    #T = typeof(prob_val)
    #return ifelse(prob_val > 1, T(1), prob_val)
#end

function metropolis_condition(ensemble::NPT, N, d_en, volume_changed, volume_unchanged, beta)
    delta_h = d_en + ensemble.pressure*(volume_changed-volume_unchanged)
    prob_val = exp(-delta_h*beta + N*log(volume_changed/volume_unchanged))
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

"""
    exc_acceptance(beta_1, beta_2, en_1, en_2)
Returns probability to exchange configurations of two trajectories with energies `en_1` and `en_2` 
at inverse temperatures `beta_1` and `beta_2`. 
"""
function exc_acceptance(beta_1, beta_2, en_1, en_2)
    delta_energy_acc = en_1 - en_2
    delta_beta = beta_1 - beta_2
    exc_acc = min(1.0,exp(delta_beta * delta_energy_acc))
    return exc_acc
end

"""
    exc_trajectories!(state_1::MCState, state_2::MCState)
Exchanges configurations and distance and energy information between two trajectories;
information contained in `state_1` and `state_2`, see [`MCState`](@ref)   
"""
function exc_trajectories!(state_1::MCState, state_2::MCState)
    state_1.config, state_2.config = state_2.config, state_1.config
    state_1.dist2_mat, state_2.dist2_mat = state_2.dist2_mat, state_1.dist2_mat
    state_1.tan_mat, state_2.tan_mat = state_2.tan_mat, state_1.tan_mat
    state_1.en_atom_vec, state_2.en_atom_vec = state_2.en_atom_vec, state_1.en_atom_vec
    state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot
    return state_1, state_2
end 
function exc_trajectories!(state_1::NNPState, state_2::NNPState)
    state_1.config,state_2.config = state_2.config,state_1.config
    state_1.dist2_mat, state_2.dist2_mat = state_2.dist2_mat, state_1.dist2_mat
    state_1.en_atom_vec, state_2.en_atom_vec = state_2.en_atom_vec, state_1.en_atom_vec
    state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot

    #then the unique NNP variables
    state_1.g_matrix, state_2.g_matrix = state_2.g_matrix, state_1.g_matrix
    state_1.f_matrix, state_2.f_matrix = state_2.f_matrix, state_1.f_matrix


    return state_1,state_2
end

"""
    parallel_tempering_exchange!(mc_states,mc_params,ensemble:NVT)
This function takes a vector of mc_states as well as the parameters of the simulation and attempts to swap two trajectories according to the parallel tempering method. 
"""
function parallel_tempering_exchange!(mc_states,mc_params,ensemble::NVT)
    n_exc = rand(1:mc_params.n_traj-1)

    mc_states[n_exc].count_exc[1] += 1
    mc_states[n_exc+1].count_exc[1] += 1

    

    if exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot) > rand()
        mc_states[n_exc].count_exc[2] += 1
        mc_states[n_exc+1].count_exc[2] += 1

        mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
    end

    return mc_states
end

"""
    parallel_tempering_exchange!(mc_states,mc_params,ensemble:NPT)
This function takes a vector of mc_states as well as the parameters of the simulation and attempts to swap two trajectories according to the parallel tempering method.
Acceptance is determined by enthalpy instead of energy. 
"""
function parallel_tempering_exchange!(mc_states,mc_params,ensemble::NPT)
    n_exc = rand(1:mc_params.n_traj-1)

    mc_states[n_exc].count_exc[1] += 1
    mc_states[n_exc+1].count_exc[1] += 1

    

    if exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, (mc_states[n_exc].en_tot + ensemble.pressure * mc_states[n_exc].config.bc.box_length^3),  (mc_states[n_exc+1].en_tot + ensemble.pressure * mc_states[n_exc+1].config.bc.box_length^3)) > rand()
        mc_states[n_exc].count_exc[2] += 1
        mc_states[n_exc+1].count_exc[2] += 1

        mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
    end

    return mc_states
end

"""
    update_max_stepsize!(mc_state::MCState, n_update, a, v, r)
Increases/decreases the max. displacement of atom, volume, and rotation moves to 110%/90% of old values
if acceptance rate is >60%/<40%. Acceptance rate is calculated after `n_update` MC cycles; 
each cycle consists of `a` atom, `v` volume and `r` rotation moves.
Information on actual max. displacement and accepted moves between updates is contained in `mc_state`, see [`MCState`](@ref).  
"""
function update_max_stepsize!(mc_state::MCState, n_update, a, v, r; min_acc = 0.4, max_acc = 0.6)
    #atom moves
    acc_rate = mc_state.count_atom[2] / (n_update * a)
    if acc_rate < min_acc
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0
    #volume moves
    if v > 0
        acc_rate = mc_state.count_vol[2] / (n_update * v)
        #println("acc rate volume = ",acc_rate)
        if acc_rate < min_acc
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0
    end
    #rotation moves
    if r > 0
        acc_rate = mc_state.count_rot[2] / (n_update * r)
        if acc_rate < min_acc
            mc_state.max_displ[3] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[3] *= 1.1
        end
        mc_state.count_rot[2] = 0
    end
    return mc_state
end
function update_max_stepsize!(mc_state::NNPState, n_update, a, v, r; min_acc = 0.4, max_acc = 0.6)
    #atom moves
    acc_rate = mc_state.count_atom[2] / (n_update * a)
    if acc_rate < min_acc
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0
    #volume moves
    if v > 0
        acc_rate = mc_state.count_vol[2] / (n_update * v)
        #println("acc rate volume = ",acc_rate)
        if acc_rate < min_acc
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0
    end
    #rotation moves
    if r > 0
        acc_rate = mc_state.count_rot[2] / (n_update * r)
        if acc_rate < min_acc
            mc_state.max_displ[3] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[3] *= 1.1
        end
        mc_state.count_rot[2] = 0
    end
    return mc_state
end


end