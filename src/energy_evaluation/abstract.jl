"""
    AbstractPotential

Abstract type for potentials.

# Subtypes
- [`AbstractDimerPotential`](@ref):
  - [`ELJPotentialEven`](@ref)
  - [`ELJPotential`](@ref)
  - [`AbstractDimerPotentialB`](@ref):
    - [`ELJPotentialB`](@ref)
    - [`LookupTablePotential`](@ref)
- [`EmbeddedAtomPotential`](@ref)
- [`AbstractMachineLearningPotential`](@ref):
  - [`RuNNerPotential`](@ref)
  - [`RuNNerPotential2Atom`](@ref)

# Interface

- [`energy_update!`](@ref)
- [`initialise_energy`](@ref)
- [`set_variables`](@ref)
- [`long_range_correction`](@ref) (optional, necessary for the potential to work with
  periodic boundary conditions)

Each potential also requires a potential variable struct
([`AbstractPotentialVariables`](@ref)) to hold all non-static information relating a
potential to the current configuration.
"""
abstract type AbstractPotential end

"""
    AbstractPotentialVariables

Defines abstract type for mutable structs containing relevant potential information updated
throughout the Monte Carlo simulation.

# Subtypes

- [`DimerPotentialVariables`](@ref)
- [`ELJPotentialBVariables`](@ref)
- [`EmbeddedAtomVariables`](@ref)
- [`LookupTableVariables`](@ref)
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
  - [`LookupTablePotential`](@ref)

# Interface

- [`dimer_energy`](@ref)
"""
abstract type AbstractDimerPotential <: AbstractPotential end

"""
    dimer_energy(potential::AbstractDimerPotential, r2::Real)
    dimer_energy(potential::AbstractDimerPotentialB, r2::Real, z_angle::Real)

Calculate the energy of dimer for given potential `potential` and squared distance `r2`
between atoms. For an [`AbstractDimerPotentialB`](@ref), the angle between the line
connecting them and z-direction (`z_angle`) is also required.
"""
dimer_energy

"""
    AbstractDimerPotentialB <: AbstractDimerPotential

# Subtypes

- [`ELJPotentialB`](@ref)
- [`LookupTablePotential`](@ref)
"""
abstract type AbstractDimerPotentialB <: AbstractDimerPotential end

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
function dimer_energy_atom(i::Int, d2vec, r_cut, pot::AbstractDimerPotential)
    result = 0.0
    @inbounds for j in eachindex(d2vec)
        if i ≠ j && d2vec[j] <= r_cut
            result += dimer_energy(pot, d2vec[j])
        end
    end
    return result
end
function dimer_energy_atom(i::Int, d2vec, tanvec, r_cut, pot::AbstractDimerPotentialB)
    result = 0.0
    @inbounds for j in eachindex(d2vec)
        if i ≠ j && d2vec[j] <= r_cut
            result += dimer_energy(pot, d2vec[j], tanvec[j])
        end
    end
    return result
end

"""
    dimer_energy_config!(dimer_energies, config, dist2_mat, potential_variables, potential)

Store the dimer energies of one atom with all other atoms in `dimer_energies` and return
the total energy of configuration.

`config` is the configuration, `dist2_mat` a squared distance matrix, (see
[`get_distance2_mat`](@ref)), and the potential information is in `potential_variables` and
`potential` (see [`AbstractPotential`](@ref)).  `potential` [`AbstractPotential`](@ref).

The energy is calculated through a call of [`dimer_energy`](@ref).
"""
function dimer_energy_config!(
    dimer_energy_vec, config, dist2_mat, tan_mat, pot::AbstractDimerPotentialB
)
    num_atoms = length(config)
    energy_tot = 0.0

    for i in 1:num_atoms
        for j in (i + 1):num_atoms
            if dist2_mat[i, j] <= r_cut(config.boundary_condition)
                e_ij = dimer_energy(pot, dist2_mat[i, j], tan_mat[i, j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end
    return energy_tot + long_range_correction(config.boundary_condition, pot, num_atoms)
end
function dimer_energy_config!(
    dimer_energy_vec, config, dist2_mat, pot::AbstractDimerPotential
)
    num_atoms = length(config)
    energy_tot = 0.0

    for i in 1:num_atoms
        for j in (i + 1):num_atoms
            if dist2_mat[i, j] <= r_cut(config.boundary_condition)
                e_ij = dimer_energy(pot, dist2_mat[i, j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end
    return energy_tot + long_range_correction(config.boundary_condition, pot, num_atoms)
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
    dist2_mat,
    potential_variables,
    ensemble_variables,
    potential::AbstractDimerPotential,
)
    if potential isa AbstractDimerPotentialB
        en_tot = dimer_energy_config!(
            potential_variables.en_atom_vec,
            config,
            dist2_mat,
            potential_variables.tan_mat,
            potential,
        )
    else
        en_tot = dimer_energy_config!(
            potential_variables.en_atom_vec, config, dist2_mat, potential
        )
    end
    return en_tot, potential_variables # TODO: why return potential variables? They aren't modified.
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
    ensemble_variables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    en_tot,
    potential::AbstractDimerPotential,
)
    trial_pos = ensemble_variables.trial_move
    index = ensemble_variables.index
    cutoff = r_cut(config.boundary_condition)

    # TODO: move tan calculation into an update_potential_variables function
    # TODO: make dimer_energy_atom take potential variables as argument
    if potential isa AbstractDimerPotentialB
        potential_variables.new_tan_vec .= (
            get_tan(trial_pos, b, config.boundary_condition) for b in config
        )
        potential_variables.new_tan_vec[index] = 0

        new_tan_vec = potential_variables.new_tan_vec
        tan_mat = potential_variables.tan_mat

        @views delta_en =
            dimer_energy_atom(index, new_dist2_vec, new_tan_vec, cutoff, potential) -
            dimer_energy_atom(
                index, dist2_mat[index, :], tan_mat[index, :], cutoff, potential
            )
    else
        @views delta_en =
            dimer_energy_atom(index, new_dist2_vec, cutoff, potential) -
            dimer_energy_atom(index, dist2_mat[index, :], cutoff, potential)
    end

    return potential_variables, delta_en + en_tot
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

# TODO: this is hardcoded to be used with ruNNer?
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
