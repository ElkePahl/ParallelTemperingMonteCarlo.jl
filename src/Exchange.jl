"""
    module Exchange

Here we include methods for calculating the metropolis condition and other exchange criteria required for Monte Carlo steps. This further declutters the MCRun module and allows us to split the cycle. Includes [`update_max_stepsize!`](@ref) which controls the frequency of
"""
module Exchange

using ..MCStates
using ..InputParams
using ..Configurations
using ..EnergyEvaluation
using ..Ensembles
export metropolis_condition, exc_acceptance,exc_trajectories!

export parallel_tempering_exchange!,update_max_stepsize!


# """
#     metropolis_condition(ensemble, delta_en, beta)
# Returns probability to accept a MC move at inverse temperature `beta` 
# for energy difference `delta_en` between new and old configuration 
# for given ensemble; implemented: 
#     - `NVT`: canonical ensemble
#     - `NPT`: NPT ensemble
# Asymmetric Metropolis criterium, p = 1.0 if new configuration more stable, 
# Boltzmann probability otherwise
# """
# function metropolis_condition(::NVT, delta_energy, beta)
#     prob_val = exp(-delta_energy*beta)
#     T = typeof(prob_val)
#     return ifelse(prob_val > 1, T(1), prob_val)
# end

# function metropolis_condition(::NPT, delta_energy, beta)
#     prob_val = exp(-delta_energy*beta)
#     T = typeof(prob_val)
#     return ifelse(prob_val > 1, T(1), prob_val)
# end


# function metropolis_condition(ensemble::NPT, N, d_en, volume_changed, volume_unchanged, beta)
#     delta_h = d_en + ensemble.pressure*(volume_changed-volume_unchanged)
#     prob_val = exp(-delta_h*beta + N*log(volume_changed/volume_unchanged))
#     T = typeof(prob_val)
#     return ifelse(prob_val > 1, T(1), prob_val)
# end
"""
    metropolis_condition(delta_energy::Number, beta::Number)
    metropolis_condition(ensemble::Etype, delta_energy::Float64, volume_changed::Float64, volume_unchanged::Float64, beta::Float64) where Etype <: NPT
    metropolis_condition(movetype::String, mc_state::MCState, ensemble::Etype) where Etype <: AbstractEnsemble
Function returning the probability value associated with a trial move. Four methods included. The last two methods are separatig functions taking a `movetype`, `mc_state` and `ensemble` and separating them into volume and atom moves defined in the first two functions, namely:
-   accepts `delta_energy` and `beta` and determines the thermodynamic probability of the single-atom move
-   accepts pressure by way of `ensemble`, `delta_energy`, `delta_volume` by way of `volume_changed` and `volume_unchanged` and `beta` and calculates the thermodynamic probability of the volume move.
"""
function metropolis_condition(delta_energy::Number, beta::Number)
    prob_val = exp(-delta_energy*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end
function metropolis_condition(ensemble::Etype, delta_energy::Float64,volume_changed::Float64,volume_unchanged::Float64,beta::Float64) where Etype <: NPT
    delta_h = delta_energy + ensemble.pressure*(volume_changed-volume_unchanged)
    prob_val = exp(-delta_h*beta + ensemble.n_atoms*log(volume_changed/volume_unchanged))
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end
function metropolis_condition(movetype::String,mc_state::MCState,ensemble::Etype) where Etype <: AbstractEnsemble
    if movetype == "atommove"
        return metropolis_condition((mc_state.new_en - mc_state.en_tot),mc_state.beta)
    elseif movetype == "volumemove"
        #return metropolis_condition(ensemble,(mc_state.new_en - mc_state.en_tot),mc_state.ensemble_variables.trial_config.bc.box_length^3,mc_state.config.bc.box_length^3,mc_state.beta )
        return metropolis_condition(ensemble,(mc_state.new_en - mc_state.en_tot),get_volume(mc_state.ensemble_variables.trial_config.bc),get_volume(mc_state.config.bc),mc_state.beta )
    elseif movetype == "atomswap"
        return metropolis_condition((mc_state.new_en - mc_state.en_tot),mc_state.beta)
    else   
        error("chosen move_type not implemented yet (see Exchange.jl)")
    end
end
# function metropolis_condition(::atommove,mc_state,ensemble)
#     return metropolis_condition((mc_state.new_en - mc_state.en_tot),mc_state.beta)
# end
# function metropolis_condition(::volumemove,mc_state,ensemble)
#     return metropolis_condition(ensemble,(mc_state.new_en - mc_state.en_tot),mc_state.ensemble_variables.trial_config.bc.box_length^3,mc_states.config.bc.box_length^3,mc_state.beta )
# end
"""
    exc_acceptance(beta_1::Number, beta_2::Number, en_1::Number, en_2::Number)
Returns probability to exchange configurations of two trajectories with energies `en_1` and `en_2` 
at inverse temperatures `beta_1` and `beta_2`. 
"""
function exc_acceptance(beta_1::Number, beta_2::Number, en_1::Number, en_2::Number)
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
    state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot
    state_1.ensemble_variables,state_2.ensemble_variables = state_2.ensemble_variables,state_1.ensemble_variables
    state_1.potential_variables,state_2.potential_variables = state_2.potential_variables,state_1.potential_variables
    return state_1, state_2
end 


"""
    parallel_tempering_exchange!(mc_states::Vector{T},mc_params::MCParams,ensemble::NVT) where T <: MCState
    parallel_tempering_exchange!(mc_states::Vector{T},mc_params::MCParams,ensemble::NPT) where T <: MCState
These functions take a vector `mc_states` as well as the parameters of the simulation and attempts to swap two trajectories according to the parallel tempering method. 
The second method uses enthalpy instead of energy to determine acceptance. 
"""
function parallel_tempering_exchange!(mc_states::MCStateVector,mc_params::MCParams,ensemble::AbstractEnsemble)
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
function parallel_tempering_exchange!(mc_states::MCStateVector,mc_params::MCParams,ensemble::NPT)
    n_exc = rand(1:mc_params.n_traj-1)

    mc_states[n_exc].count_exc[1] += 1
    mc_states[n_exc+1].count_exc[1] += 1

    

    #if exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, (mc_states[n_exc].en_tot + ensemble.pressure * mc_states[n_exc].config.bc.box_length^3),  (mc_states[n_exc+1].en_tot + ensemble.pressure * mc_states[n_exc+1].config.bc.box_length^3)) > rand()
    if exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, (mc_states[n_exc].en_tot + ensemble.pressure * get_volume(mc_states[n_exc].config.bc)),  (mc_states[n_exc+1].en_tot + ensemble.pressure * mc_states[n_exc+1].config.bc.box_length^3)) > rand()
    
        mc_states[n_exc].count_exc[2] += 1
        mc_states[n_exc+1].count_exc[2] += 1

        mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
    end

    return mc_states
end

"""
    update_max_stepsize!(mc_state::MCState, n_update::Int, ensemble::NPT, min_acc::Number, max_acc::Number)
    update_max_stepsize!(mc_state::MCState, n_update::Int, ensemble::Etype, min_acc::Number, max_acc::Number) where Etype <: AbstractEnsemble
Increases/decreases the max. displacement of atom, volume, and rotation moves to 110%/90% of old values
if acceptance rate is >60%/<40%. Acceptance rate is calculated after `n_update` MC cycles; 
each cycle consists of `a` atom, `v` volume moves.
Information on actual max. displacement and accepted moves between updates is contained in `mc_state`, see [`MCState`](@ref).  

Methods split for NVT/NPT ensemble to ensure we don't consider volume moves when dealing with the NVT ensemble.
"""
function update_max_stepsize!(mc_state::MCState, n_update::Int, ensemble::NPT,min_acc::Number,max_acc::Number)
    #atom moves
    acc_rate = mc_state.count_atom[2] / (n_update * ensemble.n_atom_moves) 
    if acc_rate < min_acc
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0
    #volume moves
    #if v > 0
    acc_rate = mc_state.count_vol[2] / (n_update * ensemble.n_volume_moves)
    #println("acc rate volume = ",acc_rate)
    if acc_rate < min_acc
        mc_state.max_displ[2] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[2] *= 1.1
    end
    mc_state.count_vol[2] = 0
    #end


    return mc_state
end
function update_max_stepsize!(mc_state::MCState, n_update::Int, ensemble::Etype ,min_acc::Number,max_acc::Number) where Etype <: AbstractEnsemble
    #atom moves
    acc_rate = mc_state.count_atom[2] / (n_update * ensemble.n_atom_moves)
    if acc_rate < min_acc
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0


    return mc_state
end


end