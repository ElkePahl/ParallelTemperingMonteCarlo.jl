"""
    EmbeddedAtomPotential
Struct containing the important quantities for calculating EAM (specifically Sutton-Chen type) potentials.
-   Fields:
    -   `n::Float64` the exponent for the two-body repulsive ϕ component
    -   `m::Float64` the exponent for the embedded electron density ρ
    -   `ean::Float64` multiplicative factor `ϵa^n /2` for ϕ
    -   `eCam::Float64` multiplicative factor `ϵCa^(m/2)` for ρ

"""
struct EmbeddedAtomPotential <: AbstractPotential
    n::Float64
    m::Float64
    ean::Float64
    eCam::Float64
end
"""
    EmbeddedAtomPotential(n::Real,m::Real,ϵ::Real,C::Real,a::Real)
Function to initalise the EAM struct given the actual constants cited in papers. The exponents `n`,`m`, the energy constant `ϵ` the distance constant `a` standard in all EAM models, and a dimensionless parameter `C` scaling ρ with respect to ϕ.
"""
function EmbeddedAtomPotential(n::Real, m::Real, ϵ::Real, C::Real, a::Real)
    epsan = ϵ * a^n / 2
    epsCam = ϵ * C * a^(m / 2)
    return EmbeddedAtomPotential(n, m, epsan, epsCam)
end

"""
    EmbeddedAtomVariables{T}
Contains the `component_vector::Matrix{T}` and `new_component_vector::Matrix{T}` for the EAM potential.
"""
mutable struct EmbeddedAtomVariables{T} <: AbstractPotentialVariables
    component_vector::Matrix{T}
    new_component_vector::Matrix{T}
end
#-------------------Component Calculation------------------#
"""
    invrexp(r2::Real,n::Real,m::Real)
Function to calculate the `ϕ,ρ` components given a square distance `r2` and the exponents `n,m`
"""
function invrexp(r2::Real, n::Real, m::Real)
    if r2 != 0.0
        r_term = 1 / sqrt(r2)
        return r_term^n, r_term^m
    else
        return 0.0, 0.0
    end
end
"""
    calc_components(componentvec,distancevec,n::Real,m::Real)
    calc_components(new_component_vec::Matrix{Float64},atomindex::Int,old_r2_vec,new_r2_vec,n::Real,m::Real)
    calc_components(component_vec::Matrix{Float64},new_component_vec::Matrix{Float64},atomindex::Int,old_r2_vec,new_r2_vec,n::Real,m::Real)

Primary calculation of ϕ,ρ for atom i, given each other atom's distance to i in `distancevec`. `eatomvec` stores the ϕ and ρ components.

Second method also includes an existing `new_component_vec` `atomindex` and old and new interatomic distances from an atom at `atomindex` stored in vectors `new_r2_vec,old_r2_vec`. This calculates the `new_component_vec` based on the updated distances and returns this.
"""
function calc_components(componentvec, distancevec, n::Real, m::Real)
    for dist in distancevec
        componentvec .+= invrexp(dist, n, m)
    end
    return componentvec
end
function calc_components(
    new_component_vec::Matrix{Float64},
    atomindex::Int,
    old_r2_vec,
    new_r2_vec,
    n::Real,
    m::Real,
)
    for j_index in eachindex(new_r2_vec)
        j_term = invrexp(new_r2_vec[j_index], n, m) .- invrexp(old_r2_vec[j_index], n, m)

        @views new_component_vec[j_index, :] .+= j_term
        @views new_component_vec[atomindex, :] .+= j_term
    end

    return new_component_vec
end

function calc_components(
    component_vec::Matrix{Float64},
    new_component_vec::Matrix{Float64},
    atomindex::Int,
    old_r2_vec,
    new_r2_vec,
    n::Real,
    m::Real,
)
    for j_index in eachindex(new_r2_vec)
        j_term = invrexp(new_r2_vec[j_index], n, m) .- invrexp(old_r2_vec[j_index], n, m)

        new_component_vec[j_index, 1] = component_vec[j_index, 1] + j_term[1]
        new_component_vec[atomindex, 1] = component_vec[atomindex, 1] + j_term[1]
        new_component_vec[j_index, 2] = component_vec[j_index, 2] + j_term[2]
        new_component_vec[atomindex, 2] = component_vec[atomindex, 2] + j_term[2]
    end

    return new_component_vec
end
"""
    calc_energies_from_components(component_vector,ean::Float64,ecam::Float64)
Takes a `component_vector` containing ϕ,ρ for each atom. Using the multiplicative factors `ean,ecam` we sum the atomic contributions and return the energy. Commented version used more allocations due to broadcasting defaulting to copying arrays. New version uses minimal allocations.
"""
# function calc_energies_from_components(component_vector,ean,ecam)
# @views    return sum(ean.*component_vector[:,1] - ecam*sqrt.(component_vector[:,2]))
# end
function calc_energies_from_components(component_vector, ean::Float64, ecam::Float64)
    en_val = 0.0
    for componentrow in eachrow(component_vector)
        en_val += ean * componentrow[1] - ecam * sqrt(componentrow[2])
    end
    return en_val
end

function set_variables(
    config::Config{T}, dist2_matrix::Matrix{Float64}, pot::EmbeddedAtomPotential
) where {T}
    N = length(config)
    componentvec = zeros(N, 2)
    for row_index in 1:N
        componentvec[row_index, :] = calc_components(
            componentvec[row_index, :], dist2_matrix[row_index, :], pot.n, pot.m
        )
    end
    return EmbeddedAtomVariables{T}(componentvec, zeros(N, 2))
end

function initialise_energy(
    config::Config,
    dist2_mat::Matrix{Float64},
    potential_variables::AbstractPotentialVariables,
    ensemble_variables::AbstractEnsembleVariables,
    pot::EmbeddedAtomPotential,
)
    en_tot = calc_energies_from_components(
        potential_variables.component_vector, pot.ean, pot.eCam
    )

    return en_tot, potential_variables
end

function energy_update!(
    ensemblevariables::AbstractEnsembleVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::EmbeddedAtomPotential,
)
    potential_variables.new_component_vector .= potential_variables.component_vector

    potential_variables.new_component_vector = calc_components(
        potential_variables.new_component_vector,
        ensemblevariables.index,
        dist2_mat[ensemblevariables.index, :],
        new_dist2_vec,
        pot.n,
        pot.m,
    )

    new_energy = calc_energies_from_components(
        potential_variables.new_component_vector, pot.ean, pot.eCam
    )
    return potential_variables, new_energy
end
