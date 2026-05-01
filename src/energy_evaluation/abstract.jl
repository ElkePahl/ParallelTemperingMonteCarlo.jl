"""
    AbstractPotential

Abstract type for potentials.

# Subtypes
- [`AbstractDimerPotential`](@ref):
  - [`ELJPotentialEven`](@ref)
  - [`ELJPotential`](@ref)
  - [`AbstractDimerPotentialB`](@ref):
    - [`ELJPotentialB`](@ref)
    - [`LookupPotential`](@ref)
- [`EmbeddedAtomPotential`](@ref)
- [`AbstractMachineLearningPotential`](@ref):
  - [`RuNNerPotential`](@ref)
  - [`RuNNerPotential2Atom`](@ref)

# Inteface

When defining a new type, the functions relating a potential to the rest of the Monte Carlo code are explicated at the end of this file. Each potential also requires a PotentialVariable [`AbstractPotentialVariables`](@ref) struct to hold all non-static information relating a potential to the current configuration.

- [`energy_update!`](@ref)
- [`initialise_energy`](@ref)
- [`set_variables`](@ref)
- [`long_range_correction`](@ref) (optional, necessary for the potential to work with
  periodic boundary conditions)

"""
abstract type AbstractPotential end
const Ptype = T where {T<:AbstractPotential}
export Ptype

"""
    AbstractPotentialVariables

An abstract type defining a class of mutable struct containing all the relevant vectors and arrays each potential will need throughout the course of a simulation to prevent over-definitions inside the MCState struct.
Implemented subtypes:
- [`DimerPotentialVariables`](@ref)
- [`ELJPotentialBVariables`](@ref)
- [`EmbeddedAtomVariables`](@ref)
- [`NNPVariables`](@ref)

"""
abstract type AbstractPotentialVariables end

"""
    AbstractDimerPotential <: AbstractPotential

# Subtypes

- [`ELJPotential`](@ref)
- [`ELJPotentialEven`](@ref)
- [`AbstractDimerPotentialB`](@ref):
  - [`ELJPotentialB`](@ref)
  - [`LookupPotential`](@ref)

# Interface

- [`dimer_energy_atom`](@ref)
- [`dimer_energy_config`](@ref)
"""
abstract type AbstractDimerPotential <: AbstractPotential end

"""
    dimer_energy_atom(i, d2vec, potential)
    dimer_energy_atom(i, d2vec, r_cut, potential)
    dimer_energy_atom(i, d2vec, tanvec, potential)
    dimer_energy_atom(i, d2vec, tanvec, r_cut, potential)

Sums the dimer energies for atom `i` with all other atoms Needs vector of squared distances
`d2vec` between atom `i` and all other atoms in configuration See
[`get_distance2_mat`](@ref) and potential information `pot` [`AbstractPotential`](@ref)

Second method includes additional variable `r_cut` to exclude distances outside the cutoff
radius of the potential.

Final two methods relate to the use of magnetic field potentials such as
[`ELJPotentialB`](@ref).
"""
function dimer_energy_atom(i::Int, d2vec, pot::AbstractDimerPotential)
    sum1 = 0.0
    for j in 1:(i - 1)
        sum1 += dimer_energy(pot, d2vec[j])
    end
    for j in (i + 1):size(d2vec, 1)
        sum1 += dimer_energy(pot, d2vec[j])
    end
    return sum1
end
function dimer_energy_atom(i::Int, d2vec, r_cut::Real, pot::AbstractDimerPotential)
    sum1 = 0.0
    for j in 1:(i - 1)
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j])
        end
    end
    for j in (i + 1):size(d2vec, 1)
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j])
        end
    end
    return sum1
end

"""
    dimer_energy_config(distmat, num_atoms, potential_variables, pot)
    dimer_energy_config(distmat, num_atoms, potential_variables, r_cut, pot)
    dimer_energy_config(distmat, num_atoms, potential_variables, r_cut, boundary_condition, potential)

Stores the total of dimer energies of one atom with all other atoms in vector and calculates
total energy of configuration.

First two methods are for standard dimer potentials, one with a cutoff radius, one without a
cutoff radius. The final two methods are for the same calculation using a magnetic potential
such as the ELJB potential.

Needs squared distances matrix, see [`get_distance2_mat`](@ref) and potential information
`potential` [`AbstractPotential`](@ref)
"""
function dimer_energy_config(
    distmat, NAtoms, potential_variables, pot::AbstractDimerPotential
)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.0

    for i in 1:NAtoms
        for j in (i + 1):NAtoms
            e_ij = dimer_energy(pot, distmat[i, j])
            dimer_energy_vec[i] += e_ij
            dimer_energy_vec[j] += e_ij
            energy_tot += e_ij
        end
    end
    return dimer_energy_vec, energy_tot
end
function dimer_energy_config(
    distmat, NAtoms, potential_variables, r_cut, bc, pot::AbstractDimerPotential
)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.0

    for i in 1:NAtoms
        for j in (i + 1):NAtoms
            if distmat[i, j] <= r_cut
                e_ij = dimer_energy(pot, distmat[i, j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end
    return dimer_energy_vec, energy_tot + long_range_correction(bc, pot, NAtoms, r_cut)
end

"""
    set_variables(config, dist_2_mat, potential)

Initialises the PotentialVariable struct for the various potentials. Defined in this way to
generalise the [`MCState`](@ref Main.ParallelTemperingMonteCarlo.MCStates.MCState) function
as this must be type-invariant with respect to the potential.
"""
function set_variables(
    config::Config{T}, dist_2_mat::Matrix{Float64}, pot::AbstractDimerPotential
) where {T}
    N = length(config)
    return DimerPotentialVariables{T}(zeros(N))
end

"""
    initialise_energy(config, dist2_mat, potential_variables, ensemble_variables, potential)

Initialise energy is used during the MCState call to set the starting energy of a `config`
according to the potential as `pot` and the configurational variables
`potential_variables`. Written with general input means the top-level is type-invariant.
"""
function initialise_energy(
    config::Config,
    dist2_mat::Matrix{Float64},
    potential_variables::AbstractPotentialVariables,
    ensemble_variables::NPTVariables,
    pot::AbstractDimerPotential,
)
    potential_variables.en_atom_vec, en_tot = dimer_energy_config(
        dist2_mat,
        length(config),
        potential_variables,
        ensemble_variables.r_cut,
        config.boundary_condition,
        pot,
    )

    return en_tot, potential_variables
end
function initialise_energy(
    config::Config,
    dist2_mat::Matrix{Float64},
    potential_variables::AbstractPotentialVariables,
    ensemble_variables::NVTVariables,
    pot::AbstractDimerPotential,
)
    potential_variables.en_atom_vec, en_tot = dimer_energy_config(
        dist2_mat, length(config), potential_variables, pot
    )

    return en_tot, potential_variables
end

"""
    energy_update!(ensemblevariables, config, potential_variables, dist2_mat, new_dist2_vec, en_tot, pot)

Energy update function for use within a cycle. at the top level this is called with the new
position `trial_pos` which is the `index`-th atom in the `config` it operates on the
`potential_variables` along with the `dist2_mat`. Using `pot` the potential to find the
`new_en`.

Has additional methods including `r_cut` where appropriate for use with periodic boundary
conditions.

This function is designed as a curry function. The generic [`get_energy!`](@ref
Main.ParallelTemperingMonteCarlo.MCRun.get_energy!) function operates on a __vector__ of
states, this function takes each state and the set potential and calls the potential
specific [`energy_update!`](@ref) function.

-   Methods defined for:
    -   [`AbstractDimerPotential`](@ref)
    -   [`AbstractDimerPotentialB`](@ref)
    -   [`EmbeddedAtomPotential`](@ref)
    -   [`RuNNerPotential`](@ref)
    -   [`RuNNerPotential2Atom`](@ref)
"""
function energy_update!(
    ensemblevariables::NVTVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::AbstractDimerPotential,
)
    new_energy = dimer_energy_update!(
        ensemblevariables.index, dist2_mat, new_dist2_vec, en_tot, pot
    )
    return potential_variables, new_energy
end
function energy_update!(
    ensemblevariables::NPTVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::AbstractDimerPotential,
)
    new_energy = dimer_energy_update!(
        ensemblevariables.index,
        dist2_mat,
        new_dist2_vec,
        en_tot,
        ensemblevariables.r_cut,
        pot,
    )
    return potential_variables, new_energy
end

# TODO: once interface materialises, explain here
"""
    AbstractDimerPotentialB <: AbstractDimerPotential

# Subtypes

- [`ELJPotentialB`](@ref)
- [`LookupTablePotential`](@ref)
"""
abstract type AbstractDimerPotentialB <: AbstractDimerPotential end

function dimer_energy_atom(i::Int, d2vec, tanvec, r_cut::Real, pot::AbstractDimerPotentialB)
    sum1 = 0.0
    for j in 1:(i - 1)
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
        end
    end
    for j in (i + 1):size(d2vec, 1)
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
        end
    end
    return sum1
end
function dimer_energy_atom(i::Int, d2vec, tanvec, pot::AbstractDimerPotentialB)
    sum1 = 0.0
    for j in 1:(i - 1)
        sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
    end
    for j in (i + 1):size(d2vec, 1)
        sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
    end
    return sum1
end
function dimer_energy_config(
    distmat, NAtoms, potential_variables, pot::AbstractDimerPotentialB
)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.0

    for i in 1:NAtoms
        for j in (i + 1):NAtoms
            e_ij = dimer_energy(pot, distmat[i, j], potential_variables.tan_mat[i, j])
            dimer_energy_vec[i] += e_ij
            dimer_energy_vec[j] += e_ij
            energy_tot += e_ij
        end
    end
    return dimer_energy_vec, energy_tot
end
function dimer_energy_config(
    distmat, NAtoms, potential_variables, r_cut, bc, pot::AbstractDimerPotentialB
)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.0

    for i in 1:NAtoms
        for j in (i + 1):NAtoms
            if distmat[i, j] <= r_cut
                e_ij = dimer_energy(pot, distmat[i, j], potential_variables.tan_mat[i, j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end
    return dimer_energy_vec, energy_tot + long_range_correction(bc, pot, NAtoms, r_cut)
end

function initialise_energy(
    config::Config,
    dist2_mat::Matrix{Float64},
    potential_variables::AbstractPotentialVariables,
    ensemble_variables::NPTVariables,
    pot::AbstractDimerPotentialB,
)
    potential_variables.en_atom_vec, en_tot = dimer_energy_config(
        dist2_mat,
        length(config),
        potential_variables,
        ensemble_variables.r_cut,
        config.boundary_condition,
        pot,
    )
    return en_tot, potential_variables
end
function initialise_energy(
    config::Config,
    dist2_mat::Matrix{Float64},
    potential_variables::AbstractPotentialVariables,
    ensemble_variables::NVTVariables,
    pot::AbstractDimerPotentialB,
)
    potential_variables.en_atom_vec, en_tot = dimer_energy_config(
        dist2_mat, length(config), potential_variables, pot
    )
    return en_tot, potential_variables
end

function energy_update!(
    ensemble_variables::NPTVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::AbstractDimerPotentialB,
)
    trial_pos = ensemble_variables.trial_move
    index = ensemble_variables.index

    potential_variables.new_tan_vec .= (
        get_tan(trial_pos, b, config.boundary_condition) for b in config
    )
    potential_variables.new_tan_vec[index] = 0

    new_energy = dimer_energy_update!(
        index,
        dist2_mat,
        potential_variables.tan_mat,
        new_dist2_vec,
        potential_variables.new_tan_vec,
        en_tot,
        ensemble_variables.r_cut,
        pot,
    )
    return potential_variables, new_energy
end
function energy_update!(
    ensemble_variables::NVTVariables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    pot::AbstractDimerPotentialB,
)
    trial_pos = ensemble_variables.trial_move
    index = ensemble_variables.index

    potential_variables.new_tan_vec .= (
        get_tan(trial_pos, b, config.boundary_condition) for b in config
    )
    potential_variables.new_tan_vec[index] = 0

    new_energy = dimer_energy_update!(
        index,
        dist2_mat,
        potential_variables.tan_mat,
        new_dist2_vec,
        potential_variables.new_tan_vec,
        en_tot,
        pot,
    )
    return potential_variables, new_energy
end

"""
    DimerPotentialVariables

Potential variables for simple dimer potentials. Contains the energy per atom in the system.
"""
mutable struct DimerPotentialVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Vector{T}
end #TODO: make immutable

"""
    DimerPotentialBVariables

Potential variables for dimer potentials in magnetic field. Contains the energy per atom in
the system and the tangent matrix.
"""
mutable struct DimerPotentialBVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Vector{T}
    tan_mat::Matrix{T}
    new_tan_mat::Matrix{T}
    new_tan_vec::Vector{T}
end #TODO: make immutable

"""
    dimer_energy_update!(index::Int,dist2_mat::Matrix{Float64},new_dist2_vec,en_tot::Float64,pot::AbstractDimerPotential)
    dimer_energy_update!(index::Int,dist2_mat::Matrix{Float64},new_dist2_vec,en_tot::Float64,r_cut::Real,pot::AbstractDimerPotential)
    dimer_energy_update!(index::Int,dist2_mat::Matrix{Float64},tanmat::Matrix{Float64},new_dist2_vec,new_tan_vec,en_tot::Float64,pot::AbstractDimerPotentialB)
    dimer_energy_update!(index::Int,dist2_mat::Matrix{Float64},tanmat::Matrix{Float64},new_dist2_vec,new_tan_vec,en_tot::Float64,r_cut::Real,pot::AbstractDimerPotentialB)

`dimer_energy_update` is the potential-level-call where for a single `mc_state` we take the new position `pos`, for atom at `index` , inside the current `config` , where the interatomic distances `dist2_mat` and the new vector based on the new position `new_dist2_vec`; these use the `potential` to calculate a delta_energy and modify the current `en_tot`. These quantities are modified in place and returned.

Final two methods are for use with a dimer potential in a magnetic field, where there is anisotropy in the coefficients.
"""
function dimer_energy_update!(
    index, dist2_mat, new_dist2_vec, en_tot, pot::AbstractDimerPotential
)
    @views delta_en =
        dimer_energy_atom(index, new_dist2_vec, pot) -
        dimer_energy_atom(index, dist2_mat[index, :], pot)

    return delta_en + en_tot
end
function dimer_energy_update!(
    index, dist2_mat, new_dist2_vec, en_tot, r_cut, pot::AbstractDimerPotential
)
    @views delta_en =
        dimer_energy_atom(index, new_dist2_vec, r_cut, pot) -
        dimer_energy_atom(index, dist2_mat[index, :], r_cut, pot)

    return delta_en + en_tot
end
function dimer_energy_update!(
    index,
    dist2_mat,
    tanmat,
    new_dist2_vec,
    new_tan_vec,
    en_tot,
    pot::AbstractDimerPotentialB,
)
    @views delta_en =
        dimer_energy_atom(index, new_dist2_vec, new_tan_vec, pot) -
        dimer_energy_atom(index, dist2_mat[index, :], tanmat[index, :], pot)

    return delta_en + en_tot
end
function dimer_energy_update!(
    index,
    dist2_mat,
    tanmat,
    new_dist2_vec,
    new_tan_vec,
    en_tot,
    r_cut,
    pot::AbstractDimerPotentialB,
)
    @views delta_en =
        dimer_energy_atom(index, new_dist2_vec, new_tan_vec, r_cut, pot) -
        dimer_energy_atom(index, dist2_mat[index, :], tanmat[index, :], r_cut, pot)

    return delta_en + en_tot
end

"""
    swap_energy_update(ensemble_variables,config,potential_variables,dist2_matrix,en_tot,pot)
This is used as a replacement for the energy_update! function when swapping atoms. It does not function in quite the same way, but stands as a replacement. First calculates `get_new_state_vars!` and then `calc_new_runner_energy!` returning the new_energy.
"""
function swap_energy_update(
    ensemble_variables, config, potential_variables, dist2_matrix, en_tot, pot
)
    potential_variables = get_new_state_vars!(
        ensemble_variables.swap_indices, config, potential_variables, dist2_matrix, pot
    )

    potential_variables, new_en = calc_new_runner_energy!(potential_variables, pot)

    return potential_variables, new_en
end
