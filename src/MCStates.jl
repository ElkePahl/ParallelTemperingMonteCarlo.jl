module MCStates

using ..BoundaryConditions
using ..Configurations
using ..MachineLearningPotential
using ..EnergyEvaluation
using ..Ensembles
using ..CustomTypes
#using ..InputParams

export MCState, max_length#, NNPState

"""
    MCState(
        temp,
        beta,
        config,
        dist2_mat,
        new_dist2_vec,
        new_en,
        en_tot,
        potential,
        potential_variables,
        ensemble,
        ensemble_variables;
        max_displ = [0.1, 0.1, 1.0],
        max_boxlength = max_length(config.boundary_condition),
        count_atom = [0, 0],
        count_vol = [0, 0],
        count_exc = [0, 0]
    )
    MCState(temp, config, ensemble, potential; kwargs...)

Monte Carlo state for a given temperature `temp` containing all information required to
perform a Monte Carlo step.

## Fields
    -   `temp`: temperature
    -   `beta`: inverse temperature
    -   `config`: actual configuration in Markov chain [`Config`](@ref)
    -   `dist_2mat`: matrix of squared distances d_ij between atoms i and j; generated automatically when potential `potential` given
    -   `new_dist2_vec`: calculates the new r2 between atoms based on a trial move
    -   `new_en` : new energy value for trial configuraiton
    -   `en_tot`: total energy of `config`; generated automatically when `potential` given
    -   `potential_variables` : mutable struct containing energy-related variables for the current configuration
    -   `ensemble_variables` : mutable struct containing ensemble-related variables for the current configuraiton
    -   `ham`: vector containing sampled energies - generated in MC run
    -   `count_atom`: number of accepted atom moves - total and between adjustment of step sizes; key-word argument
    -   `count_vol`: number of accepted volume moves - total and between adjustment of step sizes; key-word argument
    -   `count_exc`: number of attempted (10%) and accepted exchanges with neighbouring trajectories; key-word argument
"""
mutable struct MCState{BC,P,PV,E,EV}
    temp::Float64
    beta::Float64
    config::Config{Float64,BC}
    dist2_mat::Matrix{Float64}
    new_dist2_vec::Vector{Float64}
    new_en::Float64
    en_tot::Float64
    potential::P
    potential_variables::PV
    ensemble::E
    ensemble_variables::EV
    ham::Vector{Float64}
    max_displ::Vector{Float64}
    max_boxlength::Float64
    max_boxheight::Float64
    lh_ratio::Float64
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_vol_xy::Vector{Int}
    count_vol_z::Vector{Int}
    count_exc::Vector{Int}
    acceptance::Float64
    step::Int
    last_stats::NamedTuple
end

"""
    max_length(bc::SphericalBC)
    max_length(bc::CubicBC)
    max_length(bc::RhombicBC)
Returns the max box_length allowed when a volume change step is performed. For spherical boundary, it is not used during the MC steps.
"""
function max_length(bc::SphericalBC)
    return 30.0
end
function max_length(bc::CubicBC)
    return bc.box_length * 1.8
end
function max_length(bc::RhombicBC)
    return bc.box_length * 1.8
end
function max_length(bc::RectangularBC)
    return bc.box_length * 1.8
end

function max_height(bc::SphericalBC)
    return 30.0
end
function max_height(bc::CubicBC)
    return bc.box_length * 1.8
end
function max_height(bc::RhombicBC)
    return bc.box_height * 1.8
end
function max_height(bc::RectangularBC)
    return bc.box_height * 1.8
end

function MCState(
    temp::Float64,
    beta::Float64,
    config::Config{Float64,BC},
    dist2_mat,
    new_dist2_vec,
    new_en,
    en_tot,
    potential::P,
    potential_variables::PV,
    ensemble::E,
    ensemble_variables::EV;
    max_displ=[0.1, 0.1, 0.1, 0.1],
    max_boxlength=max_length(config.boundary_condition),
    max_boxheight=max_height(config.boundary_condition),
    lh_ratio=max_boxlength / max_boxheight,
    count_atom=[0, 0],
    count_vol=[0, 0],
    count_vol_xy=[0, 0],
    count_vol_z=[0, 0],
    count_exc=[0, 0],
) where {BC,P,PV,E,EV}
    return MCState{BC,P,PV,E,EV}(
        temp,
        beta,
        deepcopy(config),
        copy(dist2_mat),
        copy(new_dist2_vec),
        new_en,
        en_tot,
        potential,
        deepcopy(potential_variables),
        ensemble,
        deepcopy(ensemble_variables),
        Float64[],
        copy(max_displ),
        copy(max_boxlength),
        copy(max_boxheight),
        copy(lh_ratio),
        copy(count_atom),
        copy(count_vol),
        copy(count_vol_xy),
        copy(count_vol_z),
        copy(count_exc),
        0.0,
        0,
        NamedTuple(),
    )
end
function MCState(temp, config, ensemble, potential; kwargs...)
    beta = inv(temp * 3.16681196E-6)

    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config)

    potential_variables = set_variables(config, dist2_mat, potential)
    ensemble_variables = set_ensemble_variables(config, ensemble)

    en_tot, potential_variables = initialise_energy(
        config, dist2_mat, potential_variables, ensemble_variables, potential
    )

    return MCState(
        temp,
        beta,
        config,
        dist2_mat,
        zeros(n_atoms),
        0.0,
        en_tot,
        potential,
        potential_variables,
        ensemble,
        ensemble_variables;
        kwargs...,
    )
end

end
