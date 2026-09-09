module MCMoves

export atom_displacement, volume_change
export scale_xy, scale_z, volume_change_xy, volume_change_z, volume_change_xyz, get_energy!
export generate_move!, swap_config!, mc_move!

export generate_move!, AtomDisplacement, AtomSwap, VolumeChange, metropolis_condition
export MoveStrategy

using StaticArrays

using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles
using ..EnergyEvaluation
using ..CustomTypes

#TODO: better doc
"""
    MoveStrategy(moves, weights)

Used to (randomly) select moves on each MC cycle.
"""
struct MoveStrategy{N,T<:Tuple}
    moves::T
    weights::NTuple{N,Int}
end
function MoveStrategy(pairs...)
    moves = Tuple(first.(pairs))
    weights = Tuple(Int.(last.(pairs)))

    return MoveStrategy(moves, weights)
end

function MoveStrategy(ensemble::NVT)
    return MoveStrategy(
        AtomDisplacement() => ensemble.n_atom_moves,
    )
end
function MoveStrategy(ensemble::NPT)
    n_atoms = ensemble.n_atoms
    return MoveStrategy(
        AtomDisplacement() => ensemble.n_atom_moves,
        VolumeChange(; separated=ensemble.separated_volume) => ensemble.n_volume_moves,
    )
end
function MoveStrategy(ensemble::NNVT)
    return MoveStrategy(
        AtomDisplacement() => ensemble.n_atom_moves,
        AtomSwap() => ensemble.n_atom_swaps,
    )
end
Base.length(ms::MoveStrategy) = sum(ms.weights)

function mc_move!(
    mc_state::MCState, move_strat::MoveStrategy, selected=rand(1:length(move_strat))
)
    mc_state.ensemble_variables.index = selected
    return _perform_move!(
        mc_state, move_strat.moves, move_strat.weights, selected, false
    )
end
@inline _perform_move!(_, ::Tuple{}, ::Tuple{}, _, _) = false
@inline function _perform_move!(mc_state, (m, ms...), (w, ws...), selected, done)
    selected -= w
    if !done && selected ≤ 0
        return mc_move!(m, mc_state) | _perform_move!(mc_state, ms, ws, selected, true)
    else
        return _perform_move!(mc_state, ms, ws, selected, false)
    end
end

@inline function mc_move!(move, mc_state)
    generate_move!(move, mc_state)
    get_energy!(move, mc_state)
    prob = metropolis_probability(move, mc_state)
    if isnan(prob)
        error("metropolis probability NaN!")
    elseif rand() ≤ metropolis_probability(move, mc_state)
        swap_config!(move, mc_state)
        return true
    else
        return false
    end
end

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
    return
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
    return
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
    metropolis_probability(::AbstractMove, mc_state)

Get the probability of accepting a given move.
"""
function metropolis_probability(::Union{AtomDisplacement,AtomSwap}, mc_state)
    delta_energy = mc_state.new_en - mc_state.en_tot
    return min(exp(-delta_energy * mc_state.beta), 1.0)
end
function metropolis_probability(::VolumeChange, mc_state)
    ensemble = mc_state.ensemble::NPT

    reference_length = ensemble.reference_length
    old_bc = mc_state.config.boundary_condition
    new_bc = mc_state.ensemble_variables.trial_config.boundary_condition

    delta_energy = mc_state.new_en - mc_state.en_tot
    new_volume = volume(new_bc)
    old_volume = volume(old_bc)
    new_xy = new_bc.box_length
    new_z = new_bc.box_height
    old_xy = old_bc.box_length
    old_z = old_bc.box_height
    σ = ensemble.stress_tensor

    if reference_length ≠ 0
        delta_h =
            delta_energy + ensemble.pressure * (new_volume - old_volume) +
            reference_length * (
                σ[1] * (old_xy + new_xy) * (new_xy - old_xy) +
                σ[2] * (old_z + new_z) * (new_z - old_z) / 2
            )
    else
        delta_h = delta_energy + ensemble.pressure * (new_volume - old_volume)
    end

    probability = exp(
        -delta_h * mc_state.beta + (ensemble.n_atoms + 1) * log(new_volume / old_volume)
    )
    return min(probability, 1.0)
end

"""
    get_energy!(::AbstractMove, mc_state)

Update the energy `mc_state.en_new` according to the move.
"""
function get_energy!(::AtomDisplacement, mc_state)
    mc_state.potential_variables, mc_state.new_en = energy_update!(
        mc_state.ensemble_variables,
        mc_state.config,
        mc_state.potential_variables,
        mc_state.dist2_mat,
        mc_state.new_dist2_vec,
        mc_state.en_tot,
        mc_state.potential,
    )
end
function get_energy!(::AtomSwap, mc_state)
    mc_state.potential_variables, mc_state.new_en = swap_energy_update(
        mc_state.ensemble_variables,
        mc_state.config,
        mc_state.potential_variables,
        mc_state.dist2_mat,
        mc_state.en_tot,
        mc_state.potential,
    )
end
function get_energy!(::VolumeChange, mc_state)
    mc_state.new_en = dimer_energy_config(
        mc_state.ensemble_variables.trial_config,
        mc_state.ensemble_variables.new_dist2_mat,
        mc_state.potential_variables,
        mc_state.potential;
        new=true,
    )
end

function swap_config!(::AtomDisplacement, mc_state)
    atom_index = mc_state.ensemble_variables.index
    mc_state.config[atom_index] = mc_state.ensemble_variables.trial_move
    mc_state.dist2_mat[atom_index, :] = mc_state.new_dist2_vec
    mc_state.dist2_mat[:, atom_index] = mc_state.new_dist2_vec
    mc_state.en_tot, mc_state.new_en = mc_state.new_en, mc_state.en_tot
    mc_state.count_atom[1] += 1
    mc_state.count_atom[2] += 1

    return swap_vars!(atom_index, mc_state.potential_variables)
end
function swap_config!(::AtomSwap, mc_state)
    i, j = mc_state.ensemble_variables.swap_indices

    #swap energy and positions
    mc_state.en_tot, mc_state.new_en = mc_state.new_en, mc_state.en_tot
    mc_state.config[i], mc_state.config[j] = mc_state.config[j], mc_state.config[i]

    #swap dist2mat
    dist2_mat = mc_state.dist2_mat
    dist2_mat[i, :], dist2_mat[j, :] = dist2_mat[j, :], dist2_mat[i, :]
    dist2_mat[:, i], dist2_mat[:, j] = dist2_mat[:, j], dist2_mat[:, i]
    dist2_mat[i, i], dist2_mat[j, j] = 0.0, 0.0

    #swap fmat
    f_matrix = mc_state.potential_variables.f_matrix
    f_matrix[i, :], f_matrix[j, :] = f_matrix[j, :], f_matrix[i, :]
    f_matrix[:, i], f_matrix[:, j] = f_matrix[:, j], f_matrix[:, i]
    f_matrix[i, i], f_matrix[j, j] = 1.0, 1.0

    #swap en_atom_vec and gmat
    pot_vars = mc_state.potential_variables
    pot_vars.en_atom_vec, pot_vars.new_en_atom = pot_vars.new_en_atom, pot_vars.en_atom_vec
end
function swap_config!(::VolumeChange, mc_state)
    trial_config = mc_state.ensemble_variables.trial_config

    # TODO: swap instead of copying
    mc_state.config = deepcopy(trial_config)
    mc_state.dist2_mat .= mc_state.ensemble_variables.new_dist2_mat

    # if xy_or_z == 0, tangents don't change.
    if mc_state.potential isa AbstractDimerPotentialB &&
        mc_state.ensemble_variables.xy_or_z ≥ 1
        mc_state.potential_variables.tan_mat .= mc_state.potential_variables.new_tan_mat
    end

    mc_state.en_tot = mc_state.new_en
    if mc_state.ensemble_variables.xy_or_z == 0
        mc_state.count_vol[1] += 1
        mc_state.count_vol[2] += 1
    elseif mc_state.ensemble_variables.xy_or_z == 1
        mc_state.count_vol_xy[1] += 1
        mc_state.count_vol_xy[2] += 1
    else
        mc_state.count_vol_z[1] += 1
        mc_state.count_vol_z[2] += 1
    end
    return nothing
end

# THIS IS A BIT MESSY, FIX IT
function swap_vars!(i_atom::Int, potential_variables::DimerPotentialVariables)
    return nothing
end
function swap_vars!(i_atom::Int, potential_variables::DimerPotentialBVariables)
    potential_variables.tan_mat[i_atom, :] .= potential_variables.new_tan_vec
    potential_variables.tan_mat[:, i_atom] .= potential_variables.new_tan_vec
    return nothing
end
function swap_vars!(i_atom::Int, potential_variables::EmbeddedAtomVariables)
    potential_variables.component_vector, potential_variables.new_component_vector = potential_variables.new_component_vector,
    potential_variables.component_vector
    return nothing
end
function swap_vars!(i_atom::Int, potential_variables::NNPVariables)
    potential_variables.g_matrix, potential_variables.new_g_matrix = potential_variables.new_g_matrix,
    potential_variables.g_matrix

    potential_variables.f_matrix[i_atom, :] = potential_variables.new_f_vec
    potential_variables.f_matrix[:, i_atom] = potential_variables.new_f_vec
    return nothing
end
function swap_vars!(i_atom::Int, potential_variables::NNPVariables2a)
    potential_variables.g_matrix, potential_variables.new_g_matrix = potential_variables.new_g_matrix,
    potential_variables.g_matrix

    potential_variables.f_matrix[i_atom, :] = potential_variables.new_f_vec
    potential_variables.f_matrix[:, i_atom] = potential_variables.new_f_vec
    return nothing
end

end
