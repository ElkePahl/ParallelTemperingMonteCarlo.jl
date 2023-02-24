"""
    module Exchange

Here we include methods for calculating the metropolis condition and other exchange criteria required for Monte Carlo steps. This further declutters the MCRun module and allows us to split the cycle
"""

module Exchange

using ..MCStates
using ..Configurations
using ..EnergyEvaluation

export metropolis_condition, exc_acceptance,exc_trajectories!


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

function metropolis_condition(::NPT, N, d_en, volume_changed, volume_unchanged, pressure, beta)
    delta_h = d_en + pressure*(volume_changed-volume_unchanged)*JtoEh*Bohr3tom3
    prob_val = exp(-delta_h*beta + NAtoms*log(volume_changed/volume_unchanged))
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
    state_1.en_atom_vec, state_2.en_atom_vec = state_2.en_atom_vec, state_1.en_atom_vec
    state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot
    return state_1, state_2
end 



end