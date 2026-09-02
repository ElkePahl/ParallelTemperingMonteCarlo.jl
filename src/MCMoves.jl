module MCMoves

export atom_displacement, volume_change
export scale_xy, scale_z, volume_change_xy, volume_change_z, volume_change_xyz
export generate_move!

using StaticArrays

using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles
using ..EnergyEvaluation
using ..CustomTypes

"""
    abstract type AbstractMove end

Abstract type representing moves. Each move must implement
[`generate_move!(::AbstractMove, ::MCState)`](@ref).

Currently supported move types:
- [`AtomDisplacement`](@ref)
- [`AtomSwap`](@ref)
- [`VolumeChange`](@ref)
"""
abstract type AbstractMove end

"""
    AtomDisplacement() <: AbstractMove

An [`AbstractMove`](@ref) that moves a single atom. For a MC state `mc_state`, the move's
max move size is determined by `mc_state.max_displ[1]`.
"""
struct AtomDisplacement <: AbstractMove end

function generate_move!(::AtomDisplacement, mc_state::MCState)
    initial_position = mc_state.config[mc_state.ensemble_variables.index]
    max_displacement = mc_state.max_displ[1] # <- move variables?
    boundary_condition = mc_state.config.boundary_condition

    num_attempts = 0
    trial_position = nothing
    move_is_valid = false

    while !move_is_valid
        num_attempts += 1
        num_attempts ≥ 100 && error("Error: too many moves out of bounds")
        num_attempts == 50 && recentre!(mc_state.config)

        delta_move = SVector(
            (rand() - 0.5) * max_displacement,
            (rand() - 0.5) * max_displacement,
            (rand() - 0.5) * max_displacement,
        )
        trial_position = initial_position + delta_move
        trial_position = check_boundary(boundary_condition, trial_position)

        move_is_valid = !isnothing(trial_position)
    end
    mc_state.ensemble_variables.trial_move = trial_position

    # TODO: put me in potential variables?
    for (i, b) in enumerate(mc_state.config)
        mc_state.new_dist2_vec[i] = distance2(
            mc_state.ensemble_variables.trial_move, b, boundary_condition
        )
    end
    mc_state.new_dist2_vec[mc_state.ensemble_variables.index] = 0.0
    return mc_state
end

"""
    AtomSwap() <: AbstractMove

An [`AbstractMove`](@ref) that swaps two atoms in a configuration. Currently only works with
the [`NNVT`](@ref) ensemble.
"""
struct AtomSwap <: AbstractMove end

function generate_move!(::AtomSwap, mc_state::MCState)
    N1, N2 = mc_state.ensemble.n_atoms
    i1, i2 = rand(1:N1), N1 + rand(1:N2)
    mc_state.ensemble_variables.swap_indices = SVector{2}(i1, i2)
    return mc_state
end

"""
    VolumeChange(; separated_volume=false, max_asymmetry=0.1) <: AbstractMove

An [`AbstractMove`](@ref) that changes the volume of the configuration. Only works with the
[`NPT`](@ref) ensemble. For a MC state `mc_state`,the maximum size of the change is
controlled by `mc_state.max_displ[2]`. The maximum size of the configuration is controlled
by `mc_state.max_boxlength` and `mc_state.max_boxheight`.

## Keyword arguments
- `separated_volume`: if true, allow for separate moves in the `xy` and `z`-directions. The
  separated moves are controlled by `mc_state.max_displ[3]` (in `xy`) and
  `mc_state.max_displ[4]` (in `z`).
- `max_asymmetry`: controls how asymmetric the configuration is allowed to become. It limits
  the ratio between box length and box height to ``R/(1 + max_asymmetry)``and ``R (1 +
  max_asymmetry)``, where ``R`` is the lenth to height ratio of the initial configuration.
"""
struct VolumeChange{separated} <: AbstractMove
    max_asymmetry::Float64
end
VolumeChange(; separated=false, max_asymmetry=0.1) = VolumeChange{separated}(max_asymmetry)

function generate_move!(::VolumeChange{false}, mc_state)
    mc_state.ensemble_variables.trial_config, scale = volume_change_xyz(
        mc_state.config, mc_state.max_displ[2], mc_state.max_boxlength
    )
    if mc_state.potential_variables isa DimerPotentialBVariables
        mc_state.potential_variables.new_tan_mat .= mc_state.potential_variables.tan_mat
    end

    # Recalculating the distance matrix is necessary even on uniform moves. scaling it can
    # cause numerical issues.
    # mc_state.ensemble_variables.new_dist2_mat .= mc_state.dist2_mat .* scale^2
    get_distance2_mat!(
        mc_state.ensemble_variables.new_dist2_mat, mc_state.ensemble_variables.trial_config
    )
    return mc_state
end

function generate_move!(vol_move::VolumeChange{true}, mc_state)
    #change volume
    ra = rand(1:6)
    if ra == 1  # Choose z-direction volume change
        mc_state.ensemble_variables.xy_or_z = 2
        mc_state.ensemble_variables.trial_config, scale = volume_change_z(
            mc_state.config,
            mc_state.max_displ[4],
            mc_state.max_boxlength,
            mc_state.max_boxheight,
            vol_move.max_asymmetry,
        )
    elseif ra <= 3  # Choose xy-direction volume change
        mc_state.ensemble_variables.xy_or_z = 1
        mc_state.ensemble_variables.trial_config, scale = volume_change_xy(
            mc_state.config,
            mc_state.max_displ[3],
            mc_state.max_boxlength,
            mc_state.max_boxheight,
            vol_move.max_asymmetry,
        )
    else   # Choose all-direction volume change
        mc_state.ensemble_variables.xy_or_z = 0
        mc_state.ensemble_variables.trial_config, scale = volume_change_xyz(
            mc_state.config, mc_state.max_displ[2], mc_state.max_boxlength
        )
    end

    if mc_state.potential_variables isa DimerPotentialBVariables
        if ra <= 3
            get_tantheta_mat!(
                mc_state.potential_variables.new_tan_mat,
                mc_state.ensemble_variables.trial_config,
            )
        else
            mc_state.potential_variables.new_tan_mat .= mc_state.potential_variables.tan_mat
        end
    end

    # Recalculating the distance matrix is necessary even on uniform moves. scaling it can
    # cause numerical issues.
    get_distance2_mat!(
        mc_state.ensemble_variables.new_dist2_mat, mc_state.ensemble_variables.trial_config
    )
    return mc_state
end

"""
    volume_change_xyz(conf::Config, bc, max_vchange::Real, max_length::Real)

Scale the whole configuration, including positions and the box length by a random amount.
Returns the trial configuration.
"""
function volume_change_xyz(conf::Config, max_vchange, max_length)
    scale = exp((rand() - 0.5) * max_vchange)^(1 / 3)
    if conf.boundary_condition.box_length >= max_length && scale > 1.0
        scale = 1.0
    end

    trial_config = scale_xyz(conf, scale)
    return trial_config, scale
end

"""
    volume_change_xy(conf, max_vchange, max_length, max_height, max_asymmetry)

Scale the whole configuration, including positions and the box length by a random amount in
the ``x`` and ``y`` directions.

Returns the trial configuration and the amount it was scaled by.
"""
function volume_change_xy(conf::Config, max_vchange, max_length, max_height, max_asymmetry)
    scale = exp((rand() - 0.5) * max_vchange)^(1 / 2)
    lh_ratio = conf.boundary_condition.box_length / conf.boundary_condition.box_height
    init_ratio = max_length / max_height

    if lh_ratio >= initial_ratio * (1.0 + max_asymmetry) && scale > 1.0
        scale = 1 / scale
    elseif lh_ratio <= initial_ratio / (1 + max_asymmetry) && scale < 1.0
        scale = 1 / scale
    end
    if conf.boundary_condition.box_length >= max_length && scale > 1.0
        scale = 1 / scale
    end

    return scale_xy(conf, scale), scale
end

"""
    volume_change_z(conf::Config, max_vchange, max_length, max_height, max_asymmetry)

Scale the whole configuration, including positions and the box length by a random amount in
the ``z`` direction.

Returns the trial configuration and the amount it was scaled by.
"""
function volume_change_z(conf::Config, max_vchange, max_length, max_height, max_asymmetry)
    scale = exp((rand() - 0.5) * max_vchange)
    lh_ratio = conf.boundary_condition.box_length / conf.boundary_condition.box_height
    init_ratio = max_length / max_height

    if lh_ratio <= init_ratio * (1 + max_asymmetry) && scale > 1.0
        scale = 1 / scale
    elseif lh_ratio >= init_ratio / (1 + max_asymmetry) && scale < 1.0
        scale = 1 / scale
    end
    if conf.boundary_condition.box_height >= max_height && scale > 1.0
        scale = 1 / scale
    end

    return scale_z(conf, scale), 1 / scale
end

"""
    generate_move!(mc_state::MCState,movetype::String)
[`generate_move!`](@ref) is the currying function that takes `mc_state` and a `movetype`
and generates the variables required inside of the `ensemblevariables` struct within `mc_state`.
"""
function generate_move!(mc_state::MCState, movetype::String)
    if movetype == "atommove"
        return generate_move!(AtomDisplacement(), mc_state)
    elseif movetype == "atomswap"
        return generate_move!(AtomSwap(), mc_state)
    else
        move = VolumeChange(; separated=mc_state.ensemble.separated_volume)
        return generate_move!(move, mc_state)
    end
end

end
