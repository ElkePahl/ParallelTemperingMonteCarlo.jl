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
    atom_displacement(pos::PositionVector, max_displacement::Number, bc::SphericalBC)
    atom_displacement(pos::PositionVector, max_displacement::Number, bc::CubicBC)
    atom_displacement(pos::PositionVector, max_displacement::Number, bc::RhombicBC)
    atom_displacement(mc_state::MCState{T, N, BC}) where {T, N, BC <: PeriodicBC}
    atom_displacement(mc_state::MCState{T, N, BC}) where {T, N, BC <: SphericalBC}

Generates trial position for atom, moving it from `pos` by some random displacement.
Random displacement determined by `max_displacement`.
These variables are additionally contained in `mc_state` where the pos is determined by `index`.
Implemented for:
-   [`SphericalBC`](@ref): trial move is repeated until moved atom is within binding sphere
-   [`CubicBC`](@ref); [`RhombicBC`](@ref); [`RectangularBC`](@ref): periodic boundary condition enforced, an atom is moved into the box from the other side when it tries to get out.


The final method is a wrapper function which unpacks `mc_states`, which contains all the necessary arguments for the two methods above. When we have correctly implemented `move_strat` this wrapper will be expanded to include other methods.
"""
function atom_displacement(pos, max_displacement, bc::SphericalBC)
    delta_move = SVector(
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
    )
    trial_pos = pos + delta_move
    # count = 0
    # while check_boundary(bc, trial_pos)         #displace the atom until it's inside the binding sphere
    #     count += 1
    #     delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    #     trial_pos = pos + delta_move
    #     count == 100 && error("Error: too many moves out of binding sphere")
    # end
    return trial_pos
end

function atom_displacement(pos, max_displacement, bc::CubicBC)
    delta_move = SVector(
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
    )
    trial_pos = pos + delta_move
    trial_pos -=
        bc.box_length * SVector(
            round(trial_pos[1] / bc.box_length),
            round(trial_pos[2] / bc.box_length),
            round(trial_pos[3] / bc.box_length),
        )
    return trial_pos
end

function atom_displacement(pos, max_displacement, bc::RhombicBC)
    delta_move = SVector(
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
    )
    trial_pos = pos + delta_move
    trial_pos -= SVector(
        bc.box_length *
        round((trial_pos[1] - trial_pos[2] / 3^0.5 - bc.box_length / 2) / bc.box_length) +
        bc.box_length / 2 *
        round((trial_pos[2] - bc.box_length * 3^0.5 / 4) / (bc.box_length * 3^0.5 / 2)),
        bc.box_length * 3^0.5 / 2 *
        round((trial_pos[2] - bc.box_length * 3^0.5 / 4) / (bc.box_length * 3^0.5 / 2)),
        bc.box_height * round((trial_pos[3] - bc.box_height / 2) / bc.box_height),
    )
    return trial_pos
end

function atom_displacement(pos, max_displacement, bc::RectangularBC)
    delta_move = SVector(
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
        (rand() - 0.5) * max_displacement,
    )
    trial_pos = pos + delta_move
    trial_pos -= SVector(
        bc.box_length * round(trial_pos[1] / bc.box_length),
        bc.box_length * round(trial_pos[2] / bc.box_length),
        bc.box_height * round(trial_pos[3] / bc.box_height),
    )
    return trial_pos
end

function atom_displacement(mc_state::MCState{<:Any,<:PeriodicBC})
    mc_state.ensemble_variables.trial_move = atom_displacement(
        mc_state.config[mc_state.ensemble_variables.index],
        mc_state.max_displ[1],
        mc_state.config.boundary_condition,
    )
    for (i, b) in enumerate(mc_state.config)
        mc_state.new_dist2_vec[i] = distance2(
            mc_state.ensemble_variables.trial_move, b, mc_state.config.boundary_condition
        )
    end
    mc_state.new_dist2_vec[mc_state.ensemble_variables.index] = 0.0
    return mc_state
end

function atom_displacement(mc_state::MCState{<:Any,<:SphericalBC})
    count = 0.0
    trial_pos = atom_displacement(
        mc_state.config[mc_state.ensemble_variables.index],
        mc_state.max_displ[1],
        mc_state.config.boundary_condition,
    )
    while check_boundary(mc_state.config.boundary_condition, trial_pos)
        count += 1
        if count == 50
            recentre!(mc_state.config)
        else
            trial_pos = atom_displacement(
                mc_state.config[mc_state.ensemble_variables.index],
                mc_state.max_displ[1],
                mc_state.config.boundary_condition,
            )
            count == 100 && error("Error: too many moves out of binding sphere")
        end
    end
    mc_state.ensemble_variables.trial_move = trial_pos

    # mc_state.ensemble_variables.trial_move = atom_displacement(mc_state.config[mc_state.ensemble_variables.index],mc_state.max_displ[1],mc_state.config.boundary_condition)
    for (i, b) in enumerate(mc_state.config)
        mc_state.new_dist2_vec[i] = distance2(
            mc_state.ensemble_variables.trial_move, b, mc_state.config.boundary_condition
        )
    end
    mc_state.new_dist2_vec[mc_state.ensemble_variables.index] = 0.0
    return mc_state
end

"""
    scale_xyz(::RhombicBC, α)
    scale_xyz(::RectangularBC, α)
    scale_xyz(::Vector{<:SVector}, α)
    scale_xyz(::Config, α)

Scale boundary condition, vector, or configuration in all three dimensions by factor `α`.
"""
scale_xyz(bc::CubicBC, α) = CubicBC(α * bc.box_length)
scale_xyz(bc::RhombicBC, α) = RhombicBC(α * bc.box_length, α * bc.box_height)
scale_xyz(bc::RectangularBC, α) = RectangularBC(α * bc.box_length, α * bc.box_height)
scale_xyz(vector, α) = α * vector
function scale_xyz(config::Config, α)
    return Config(scale_xyz(config.positions, α), scale_xyz(config.boundary_condition, α))
end

"""
    volume_change_xyz(conf::Config, bc, max_vchange::Real, max_length::Real)

Scale the whole configuration, including positions and the box length by a random amount.
Returns the trial configuration.
"""
function volume_change_xyz(conf::Config, max_vchange::Real, max_length::Real)
    scale = exp((rand() - 0.5) * max_vchange)^(1 / 3)
    if conf.boundary_condition.box_length >= max_length && scale > 1.0
        scale = 1.0
    end

    trial_config = scale_xyz(conf, scale)
    return trial_config, scale
end

"""
    scale_xy(::RhombicBC, α)
    scale_xy(::RectangularBC, α)
    scale_xy(::Vector{<:SVector}, α)
    scale_xy(::Config, α)

Scale boundary condition, vector, or configuration in all ``x`` and ``y`` dimensions by
factor `α`.
"""
scale_xy(bc::RhombicBC, scale) = RhombicBC(bc.box_length * scale, bc.box_height)
scale_xy(bc::RectangularBC, scale) = RectangularBC(bc.box_length * scale, bc.box_height)
function scale_xy(pos, scale)
    new_pos = map(pos) do p
        SVector(p[1] * scale, p[2] * scale, p[3])
    end
    return new_pos
end
function scale_xy(config::Config, scale)
    return Config(
        scale_xy(config.positions, scale), scale_xy(config.boundary_condition, scale)
    )
end

"""
    scale_z(::RhombicBC, α)
    scale_z(::RectangularBC, α)
    scale_z(::Vector{<:SVector}, α)
    scale_z(::Config, α)

Scale boundary condition, vector, or configuration in the ``z`` dimension by factor `α`.
"""
scale_z(bc::RhombicBC, scale) = RhombicBC(bc.box_length, bc.box_height * scale)
scale_z(bc::RectangularBC, scale) = RectangularBC(bc.box_length, bc.box_height * scale)
function scale_z(pos, scale)
    new_pos = map(pos) do p
        SVector(p[1], p[2], p[3] * scale)
    end
    return new_pos
end
function scale_z(config::Config, scale)
    return Config(
        scale_z(config.positions, scale), scale_z(config.boundary_condition, scale)
    )
end

"""
    volume_change_xy(conf::Config, bc, max_vchange::Real, max_length::Real, lh_ratio)

Scale the whole configuration, including positions and the box length by a random amount in
the ``x`` and ``y`` directions.

Returns the trial configuration.
"""
function volume_change_xy(conf::Config, max_vchange, max_length, lh_ratio)
    scale = exp((rand() - 0.5) * max_vchange)^(1 / 2)
    if conf.boundary_condition.box_length / conf.boundary_condition.box_height >=
       lh_ratio * 1.1 && scale > 1.0
        scale = 1 / scale
    elseif conf.boundary_condition.box_length / conf.boundary_condition.box_height <=
           lh_ratio * 0.909 && scale < 1.0
        scale = 1 / scale
    end
    if conf.boundary_condition.box_length >= max_length && scale > 1.0
        scale = 1 / scale
    end

    return scale_xy(conf, scale), scale
end

"""
    volume_change_z(conf::Config, max_vchange::Real, max_length::Real, lh_ratio)

Scale the whole configuration, including positions and the box length by a random amount in
the ``z`` direction.

Returns the trial configuration.
"""
function volume_change_z(conf::Config, max_vchange, max_height, lh_ratio)
    scale = exp((rand() - 0.5) * max_vchange)
    if conf.boundary_condition.box_length / conf.boundary_condition.box_height <=
       lh_ratio * 1.1 && scale > 1.0
        scale = 1 / scale
    elseif conf.boundary_condition.box_length / conf.boundary_condition.box_height >=
           lh_ratio * 0.909 && scale < 1.0
        scale = 1 / scale
    end
    if conf.boundary_condition.box_height >= max_height && scale > 1.0
        scale = 1 / scale
    end

    return scale_z(conf, scale), 1 / scale
end

"""
    volume_change_uniform(mc_state::MCState)

Change the volume uniformly and update the `mc_state` accordingly.
"""
function volume_change_uniform(mc_state::MCState)
    mc_state.ensemble_variables.trial_config, scale = volume_change_xyz(
        mc_state.config, mc_state.max_displ[2], mc_state.max_boxlength
    )
    mc_state.ensemble_variables.new_r_cut = get_r_cut(
        mc_state.ensemble_variables.trial_config.boundary_condition
    )
    mc_state.ensemble_variables.new_dist2_mat .= mc_state.dist2_mat .* scale^2

    return mc_state
end

"""
    volume_change_separated(mc_state::MCState)

Change the volume
- in the ``z``-direction with probability ``1/3``,
- in the ``x``,``y``-directions with probability ``2/3`` or
- unioformly with probability ``1/2``.

Update the `mc_state` accordingly. If the potential used is [`ELJPotentialB`](@ref) or
[`LookupTablePotential`](@ref), update the tangent matrix as well.
"""
function volume_change_separated(mc_state::MCState)
    #change volume
    ra = rand(1:6)
    if ra == 1  # Choose z-direction volume change
        mc_state.ensemble_variables.xy_or_z = 2
        mc_state.ensemble_variables.trial_config, scale = volume_change_z(
            mc_state.config,
            mc_state.max_displ[4],
            mc_state.max_boxheight,
            mc_state.lh_ratio,
        )
    elseif ra <= 3  # Choose xy-direction volume change
        mc_state.ensemble_variables.xy_or_z = 1
        mc_state.ensemble_variables.trial_config, scale = volume_change_xy(
            mc_state.config,
            mc_state.max_displ[3],
            mc_state.max_boxlength,
            mc_state.lh_ratio,
        )
    else   # Choose all-direction volume change
        mc_state.ensemble_variables.xy_or_z = 0
        mc_state.ensemble_variables.trial_config, scale = volume_change_xyz(
            mc_state.config, mc_state.max_displ[2], mc_state.max_boxlength
        )
    end

    mc_state.ensemble_variables.new_r_cut = get_r_cut(
        mc_state.ensemble_variables.trial_config.boundary_condition
    )
    get_distance2_mat!(
        mc_state.ensemble_variables.new_dist2_mat, mc_state.ensemble_variables.trial_config
    )

    if ra <= 3 && (
        mc_state.potential_variables isa ELJPotentialBVariables{Float64} ||
        mc_state.potential_variables isa LookupTableVariables{Float64}
    )
        get_tantheta_mat!(
            mc_state.potential_variables.new_tan_mat,
            mc_state.ensemble_variables.trial_config,
        )
    end
    return mc_state
end

"""
    volume_change(mc_state::MCState, separated_volume=false)

MC move that changes volume. If `separated_volume == true`, the volume is changed in the ``x``,``y`` directions or in the ``z`` direction separately.
"""
function volume_change(mc_state::MCState, separated_volume=false)
    if separated_volume
        mc_state = volume_change_separated(mc_state)
    else
        mc_state = volume_change_uniform(mc_state)
    end
    return mc_state
end

"""
    (swap_atoms(mc_state::MCState{T, N, BC, PV, EV}) where {T, N, BC, PV, EV <: NNVTVariables{tee, n, N1, N2}}) where {tee, n, N1, N2}
Swaps two atoms in the configuration.
"""
function swap_atoms(
    mc_state::MCState{T,BC,PV,EV}
) where {T,BC,PV,EV<:NNVTVariables{tee,n,N1,N2}} where {tee,n,N1,N2}
    # TODO: make extracting Ns nicer than this.
    i1, i2 = rand(1:N1), rand((N1 + 1):length(mc_state.config))
    mc_state.ensemble_variables.swap_indices = SVector{2}(i1, i2)
    return mc_state
end

"""
    generate_move!(mc_state::MCState,movetype::String)
[`generate_move!`](@ref) is the currying function that takes `mc_state` and a `movetype`
and generates the variables required inside of the `ensemblevariables` struct within `mc_state`.
"""
function generate_move!(mc_state::MCState, movetype::String, ensemble)
    if movetype == "atommove"
        return atom_displacement(mc_state)
    elseif movetype == "atomswap"
        return swap_atoms(mc_state)
    else
        return volume_change(mc_state, ensemble.separated_volume)
    end
end

end
