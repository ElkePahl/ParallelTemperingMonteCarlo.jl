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
- [`DimerPotentialBVariables`](@ref)
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
  - [`LookupTablePotential`](@ref)

# Interface

- [`dimer_energy`](@ref)
"""
abstract type AbstractDimerPotential <: AbstractPotential end

"""
    dimer_energy(potential::AbstractDimerPotential, r2::Real)
    dimer_energy(potential::AbstractDimerPotentialB, r2::Real, z_angle::Real)

Calculate the energy of dimer for given potential `potential` and squared distance `r2`
between atoms. For an [`AbstractDimerPotentialB`](@ref), the angle between the molecular
axis and the z-direction (`z_angle`) is also required.
"""
dimer_energy

"""
    AbstractDimerPotentialB <: AbstractDimerPotential

# Subtypes

- [`ELJPotentialB`](@ref)
- [`LookupTablePotential`](@ref)
"""
abstract type AbstractDimerPotentialB <: AbstractDimerPotential end

# TODO: this function needs more work
"""
    dimer_energy_config!(config, dist2_mat, potential_variables, potential; new=false)

Return the total energy of configuration. If `new=false`, calculate the energy of the
current configuration, while if `new=true`, calculate the energy for the trial
configuration.

`config` is the configuration, `dist2_mat` a squared distance matrix, (see
[`get_distance2_mat`](@ref)), and the potential information is in `potential_variables` and
`potential` (see [`AbstractPotential`](@ref)).

The energy is calculated through a call of [`dimer_energy`](@ref).
"""
function dimer_energy_config(
    config, dist2_mat, potential_variables, pot::AbstractDimerPotentialB; new=false
)
    tan_mat = new ? potential_variables.new_tan_mat : potential_variables.tan_mat

    num_atoms = length(config)
    total_energy = 0.0

    for i in 1:num_atoms
        for j in (i + 1):num_atoms
            if dist2_mat[j, i] <= r_cut(config.boundary_condition)
                e_ij = dimer_energy(pot, dist2_mat[j, i], tan_mat[j, i])
                total_energy += e_ij
            end
        end
    end
    return total_energy + long_range_correction(config.boundary_condition, pot, num_atoms)
end

function dimer_energy_config(
    config, dist2_mat, potential_variables, pot::AbstractDimerPotential; new=false
)
    num_atoms = length(config)
    total_energy = 0.0

    for i in 1:num_atoms
        for j in (i + 1):num_atoms
            if dist2_mat[j, i] <= r_cut(config.boundary_condition)
                e_ij = dimer_energy(pot, dist2_mat[j, i])
                total_energy += e_ij
            end
        end
    end
    return total_energy + long_range_correction(config.boundary_condition, pot, num_atoms)
end

"""
    set_variables(config, dist_2_mat, potential)

Initialises the PotentialVariable struct for the various potentials. Defined in this way to
generalise the [`MCState`](@ref Main.ParallelTemperingMonteCarlo.MCStates.MCState) function
as this must be type-invariant with respect to the potential.
"""
function set_variables(_, _, pot::AbstractDimerPotential)
    return DimerPotentialVariables()
end
function set_variables(config, _, pot::AbstractDimerPotentialB)
    n = length(config)
    tan_matrix = get_tantheta_mat(config)

    return DimerPotentialBVariables(tan_matrix, copy(tan_matrix), zeros(n))
end

"""
    initialise_energy(config, dist2_mat, potential_variables, ensemble_variables, potential)

Initialise energy is used during the MCState call to set the starting energy of a `config`
according to the potential as `pot` and the configurational variables
`potential_variables`. Written with general input means the top-level is type-invariant.
"""
function initialise_energy(
    config::Config, dist2_mat, potential_variables, _, potential::AbstractDimerPotential
)
    return dimer_energy_config(
        config, dist2_mat, potential_variables, potential; new=false
    ),
    potential_variables
end

"""
    dimer_energy_atom(potential::AbstractDimerPotential, index, cutoff, dist2)
    dimer_energy_atom(potential::AbstractDimerPotentialB, index, cutoff, dist2, tan)

Sum the dimer energies for atom `index` with all other atoms.
"""
function dimer_energy_atom(potential, index, cutoff, dist2)
    energy = 0.0
    @inbounds for j in eachindex(dist2)
        if j ≠ index && dist2[j] ≤ cutoff
            energy += dimer_energy(potential, dist2[j])
        end
    end
    return energy
end
function dimer_energy_atom(potential, index, cutoff, dist2, tan)
    energy = 0.0
    @boundscheck if length(dist2) ≠ length(tan)
        throw(DimensionMismatch("lengths of `dist2` and `tan2` don't match"))
    end
    @inbounds for j in eachindex(dist2)
        if j ≠ index && dist2[j] ≤ cutoff
            energy += dimer_energy(potential, dist2[j], tan[j])
        end
    end
    return energy
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
    total_energy,
    potential::AbstractDimerPotential,
)
    index = ensemble_variables.index
    cutoff = r_cut(config.boundary_condition)

    old_dist2_vec = view(dist2_mat, :, index)

    delta_energy =
        dimer_energy_atom(potential, index, cutoff, new_dist2_vec) -
        dimer_energy_atom(potential, index, cutoff, old_dist2_vec)

    return potential_variables, total_energy + delta_energy
end
function energy_update!(
    ensemble_variables,
    config,
    potential_variables,
    dist2_mat,
    new_dist2_vec,
    total_energy,
    potential::AbstractDimerPotentialB,
)
    index = ensemble_variables.index
    cutoff = r_cut(config.boundary_condition)

    trial_pos = ensemble_variables.trial_move
    potential_variables.new_tan_vec .= (
        get_tan(trial_pos, b, config.boundary_condition) for b in config
    )
    potential_variables.new_tan_vec[index] = 0

    old_dist2_vec = view(dist2_mat, :, index)
    old_tan_vec = view(potential_variables.tan_mat, :, index)
    new_tan_vec = potential_variables.new_tan_vec

    delta_energy =
        dimer_energy_atom(potential, index, cutoff, new_dist2_vec, new_tan_vec) -
        dimer_energy_atom(potential, index, cutoff, old_dist2_vec, old_tan_vec)

    return potential_variables, total_energy + delta_energy
end

"""
    DimerPotentialVariables

Potential variables for simple dimer potentials. Contains the energy per atom in the system.
"""
struct DimerPotentialVariables <: AbstractPotentialVariables end

"""
    DimerPotentialBVariables

Potential variables for dimer potentials in magnetic field. Contains the energy per atom in
the system and the tangent matrix.
"""
struct DimerPotentialBVariables <: AbstractPotentialVariables
    tan_mat::Matrix{Float64}
    new_tan_mat::Matrix{Float64}
    new_tan_vec::Vector{Float64}
end

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
