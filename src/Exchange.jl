"""
    module Exchange

Here we include methods for calculating the metropolis condition and other exchange 
criteria required for Monte Carlo steps. This further declutters the MCRun module and 
allows us to split the cycle. Includes [`update_max_stepsize!`](@ref) which controls the
frequency of.
"""
module Exchange

using ..MCStates
using ..InputParams
using ..BoundaryConditions
using ..Configurations
using ..EnergyEvaluation
using ..Ensembles
export get_metropolis_probability, metropolis_condition, exc_acceptance, exc_trajectories!

export parallel_tempering_exchange!, update_max_stepsize!

#=
TODO:
Elke:
I think that this is a bit chaotic. Would it make sense to separate last method from first
three as they do different things. First three could be renamed to get_metropolis_probability or
something like this.

Documentation has to be updated anyway as only three methods left.
Suggestions:

- metropolis_condition(movetype ...)
- Returns probability for given move_type (atom, volume or atom swap moves). Perhaps provide
  formulae here?
- get_metropolis_probability(...)
- get_metropolis_probability(...)
- Return probability for atom or atom swap moves (1st method) or volume move(2nd method)
=#
"""
    get_metropolis_probability(
    delta_energy::Number,
    beta::Number
    )
    get_metropolis_probability(
    ensemble::NPT,
    delta_energy::Float64,
    volume_changed::Float64,
    volume_unchanged::Float64,
    beta::Float64
    )
    get_metropolis_probability(
    ensemble::NPT,
    delta_energy::Float64,
    volume_changed::Float64,
    xy_changed::Float64,
    z_changed::Float64,
    volume_unchanged::Float64,
    xy_unchanged::Float64,
    z_unchanged::Float64,
    beta::Float64,
    reference_length::Float64=15.8 
    )
Function returning the probability value associated with a trial move. 
Three methods included, one for NVT, one for NPT, one for NσT. 
"""
function get_metropolis_probability(
    delta_energy::Number,
    beta::Number
)
    prob_val = exp(-delta_energy * beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function get_metropolis_probability(
    ensemble::NPT,
    delta_energy::Float64,
    volume_changed::Float64,
    volume_unchanged::Float64,
    beta::Float64
)
delta_h = delta_energy + ensemble.pressure * (volume_changed - volume_unchanged)
    prob_val = exp(
        -delta_h * beta + (ensemble.n_atoms + 1) * log(volume_changed / volume_unchanged)
    )
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function get_metropolis_probability(
    ensemble::NPT,
    delta_energy::Float64,
    volume_changed::Float64,
    xy_changed::Float64,
    z_changed::Float64,
    volume_unchanged::Float64,
    xy_unchanged::Float64,
    z_unchanged::Float64,
    beta::Float64,
    reference_length::Float64=23.0 #Variable encoding reference box size.
    #=The value 15.8 corresponds specifically to the reference length obtained for a 32 atom
    Argon box. (The value 23 correponds to the reference length for a 96 atom Argon box) This value will not impact the calculation unless the stress tensor for the
    NPT ensemble is nonzero. =#
)
    delta_h = delta_energy + ensemble.pressure * (volume_changed - volume_unchanged) +
    reference_length^3 * ensemble.stress_tensor[1] * (xy_unchanged + xy_changed)*
    (xy_changed - xy_unchanged)/(reference_length)^2 +
    reference_length^3 * ensemble.stress_tensor[2] * 0.5*(z_unchanged + z_changed)*
    (z_changed - z_unchanged)/(reference_length)^2
    prob_val = exp(
        -delta_h * beta + (ensemble.n_atoms + 1) * log(volume_changed / volume_unchanged)
    )
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

"""
    metropolis_condition(
    movetype::String,
    mc_state::MCState,
    ensemble
)
Separating functions taking a `movetype`, `mc_state` and `ensemble` and separating them
into volume and atom moves defined in [get_metropolis_probability](@ref) namely:
-   accepts `delta_energy` and `beta` and determines the thermodynamic probability of
the single-atom move
-   accepts pressure by way of `ensemble`, `delta_energy`, `delta_volume` by way of
`volume_changed` and `volume_unchanged` and `beta` and determines the thermodynamic
probability of the volume move.
-   accepts pressure and stress by way of `ensemble`, `delta_energy`, `delta_volume`, 
`delta_xy`, `delta_z`, and `reference length` and determines the thermodynamic property of
the deformation move.
"""
function metropolis_condition(movetype::String, mc_state::MCState, ensemble)
    if movetype == "atommove"
        return get_metropolis_probability(mc_state.new_en - mc_state.en_tot, mc_state.beta)
    elseif movetype == "volumemove"
        if ensemble.stress_tensor ≠ [0, 0] # If we are in NPT, just use simple version.
            # This doesn't waste time multiplying by zero.
            return get_metropolis_probability(
            ensemble,
            (mc_state.new_en - mc_state.en_tot),
            volume(mc_state.ensemble_variables.trial_config.boundary_condition),
            volume(mc_state.config.boundary_condition),
            mc_state.beta,
        )
        else # Otherwise, we need to include the stress.
            return get_metropolis_probability(
                ensemble,
                (mc_state.new_en - mc_state.en_tot),
                volume(mc_state.ensemble_variables.trial_config.boundary_condition),
                mc_state.ensemble_variables.trial_config.boundary_condition.box_length,
                mc_state.ensemble_variables.trial_config.boundary_condition.box_height,
                volume(mc_state.config.boundary_condition),
                mc_state.config.boundary_condition.box_length,
                mc_state.config.boundary_condition.box_height,
                mc_state.beta
                #Eventually, this line will also be an input, allowing the specification of a
                #reference length. This will ultimately be a part of the ensemble variables.
            )
        end
    elseif movetype == "atomswap"
        return get_metropolis_probability((mc_state.new_en - mc_state.en_tot), mc_state.beta)
    else
        error("chosen move_type not implemented yet (see Exchange.jl)")
    end
end
"""
    exc_acceptance(beta_1::Number, beta_2::Number, en_1::Number, en_2::Number)
Returns probability to exchange configurations of two trajectories with energies `en_1` and `en_2`
at inverse temperatures `beta_1` and `beta_2`.
"""
function exc_acceptance(beta_1::Number, beta_2::Number, en_1::Number, en_2::Number)
    delta_energy_acc = en_1 - en_2
    delta_beta = beta_1 - beta_2
    exc_acc = min(1.0, exp(delta_beta * delta_energy_acc))
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
    state_1.ensemble_variables, state_2.ensemble_variables = state_2.ensemble_variables,
    state_1.ensemble_variables
    state_1.potential_variables, state_2.potential_variables = state_2.potential_variables,
    state_1.potential_variables
    return state_1, state_2
end

"""
    parallel_tempering_exchange!(
    mc_states::Vector{T},
    mc_params::MCParams,
    ensemble::NVT
    ) where T <: MCState
    parallel_tempering_exchange!(
    mc_states::Vector{T},mc_params::MCParams,ensemble::NPT
    ) where T <: MCState
These functions take a vector `mc_states` as well as the parameters of the simulation
and attempt to swap two trajectories according to the parallel tempering method.
The second method uses enthalpy instead of energy to determine acceptance.
"""
function parallel_tempering_exchange!(
    mc_states::MCStateVector, mc_params::MCParams, ensemble::AbstractEnsemble
)
    n_exc = rand(1:(mc_params.n_traj - 1))

    mc_states[n_exc].count_exc[1] += 1
    mc_states[n_exc + 1].count_exc[1] += 1

    if exc_acceptance(
        mc_states[n_exc].beta,
        mc_states[n_exc + 1].beta,
        mc_states[n_exc].en_tot,
        mc_states[n_exc + 1].en_tot,
    ) > rand()
        mc_states[n_exc].count_exc[2] += 1
        mc_states[n_exc + 1].count_exc[2] += 1

        mc_states[n_exc], mc_states[n_exc + 1] = exc_trajectories!(
            mc_states[n_exc], mc_states[n_exc + 1]
        )
    end

    return mc_states
end
function parallel_tempering_exchange!(
    mc_states::MCStateVector, mc_params::MCParams, ensemble::NPT
)
    n_exc = rand(1:(mc_params.n_traj - 1))

    mc_states[n_exc].count_exc[1] += 1
    mc_states[n_exc + 1].count_exc[1] += 1

    if exc_acceptance(
        mc_states[n_exc].beta,
        mc_states[n_exc + 1].beta,
        (
            mc_states[n_exc].en_tot +
            ensemble.pressure * volume(mc_states[n_exc].config.boundary_condition)
        ),
        (
            mc_states[n_exc + 1].en_tot +
            ensemble.pressure * volume(mc_states[n_exc + 1].config.boundary_condition)
        ),
    ) > rand()
        mc_states[n_exc].count_exc[2] += 1
        mc_states[n_exc + 1].count_exc[2] += 1

        mc_states[n_exc], mc_states[n_exc + 1] = exc_trajectories!(
            mc_states[n_exc], mc_states[n_exc + 1]
        )
    end

    return mc_states
end

"""
    update_max_stepsize!(
    mc_state::MCState,
    n_update::Int,
    ensemble::NPT,
    min_acc::Number,
    max_acc::Number
    )
    update_max_stepsize!(
    mc_state::MCState,
    n_update::Int,
    ensemble,
    min_acc::Number,
    max_acc::Number
    )
Increases/decreases the max. displacement of atom, volume, and rotation moves to 110%/90%
of old values if acceptance rate is >60%/<40%. Acceptance rate is calculated after 
`n_update` MC cycles; each cycle consists of `a` atom and `v` volume moves.
Information on actual max. displacement and accepted moves between updates is contained in
`mc_state`, see [`MCState`](@ref).

Methods split for NVT/NPT ensemble to ensure we don't consider volume moves when dealing 
with the NVT ensemble.
"""
function update_max_stepsize!(
    mc_state::MCState, n_update::Int, ensemble::NPT, min_acc::Number, max_acc::Number
)
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
    if ensemble.separated_volume == false
        acc_rate = mc_state.count_vol[2] / (n_update * ensemble.n_volume_moves)
        if acc_rate < min_acc
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0
    else
        acc_rate = mc_state.count_vol[2] / (n_update * ensemble.n_volume_moves * 1 / 2)
        if acc_rate < min_acc
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0

        acc_rate = mc_state.count_vol_xy[2] / (n_update * ensemble.n_volume_moves * 1 / 3)
        if acc_rate < min_acc
            mc_state.max_displ[3] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[3] *= 1.1
        end
        mc_state.count_vol_xy[2] = 0

        acc_rate = mc_state.count_vol_z[2] / (n_update * ensemble.n_volume_moves / 6)
        if acc_rate < min_acc
            mc_state.max_displ[4] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[4] *= 1.1
        end
        mc_state.count_vol_z[2] = 0
    end
    #end

    return mc_state
end
function update_max_stepsize!(
    mc_state::MCState, n_update::Int, ensemble, min_acc::Number, max_acc::Number
)
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
