module Ensembles

using ..Configurations
using ..BoundaryConditions
import ..BoundaryConditions: report_stats
using StaticArrays, Random

export AbstractEnsemble, NVT, NPT, NNVT

export AbstractEnsembleVariables,
    NVTVariables, NPTVariables, NNVTVariables, set_ensemble_variables, hamiltonian

export report_stats
export MoveStrategy

"""
    AbstractEnsemble
Abstract type for ensemble:
-   [`NVT`](@ref): canonical ensemble
-   [`NPT`](@ref): isothermal,isobaric ensemble, with option to include isostress.

Each subtype requires a corresponding [`AbstractEnsembleVariables`](@ref) struct.
"""
abstract type AbstractEnsemble end

"""
    report_stats(mc_state, ensemble)

Return a `NamedTuple` of statistics to report in the result table. The default
implementation used by [`NVT`](@ref), [`NNVT`](@ref) returns an empty `NamedTuple`.
"""
function report_stats(_, ::AbstractEnsemble)
    return NamedTuple()
end

"""
    AbstractEnsembleVariables
Abstract struct for variables specific to ensemble that change during MC run (moves).
"""
abstract type AbstractEnsembleVariables end

"""
    NVT
Canonical ensemble.
-   Fieldnames:
    -   `n_atoms::Int64`: number of atoms
    -   `n_atom_moves::Int64`: number of atom moves; defaults to `n_atoms`
    -   `n_atom_swaps::Int64`: number of atom exchanges made; defaults to 0
"""
struct NVT <: AbstractEnsemble
    n_atoms::Int64
    n_atom_moves::Int64
    n_atom_swaps::Int64
end

function NVT(n_atoms)
    return NVT(n_atoms, n_atoms, 0)
end

"""
    NVTVariables <: AbstractEnsembleVariables
NVT ensemble specific variables that change during MC run:
-   Fields:
    -   `index::Int64`
    -   `trial_move::SVector{3,T}`
When trialing a new configuration we select an atom at `index` to move to position given by `trial_move`.
"""
mutable struct NVTVariables{T} <: AbstractEnsembleVariables
    index::Int64
    trial_move::SVector{3,T}
end

"""
    NPT
Isothermal, isobaric ensemble.
-   Fieldnames:
    -   `n_atoms::Int64`: number of atoms
    -   `n_atom_moves::Int64`: number of atom moves; defaults to `n_atoms`
    -   `n_volume_moves::Int64`: number of volume moves; defaults to 1
    -   `n_atom_swaps::Int64`: number of atom exchanges made; defaults to 0
    -   `pressure::Float64`: the fixed pressure of the system
    -   `separated_volume::Bool`: allows independent volume changes in different directions.
    -   `stress_tensor::SVector{2, Float64}`: the fixed internal stress of the system. First entry
    corresponds to the stress in the x and y directions, assumed the same, second entry is z.
    This is set to zero by default.
    -   `reference_length::Float64`: This is a specialised variable specifically related
    to NσT ensemble. In order to discuss strain, one needs to make reference to an
    unstrained length, which is what this parameter encodes. This is set to zero by default.
"""
struct NPT <: AbstractEnsemble
    n_atoms::Int64
    n_atom_moves::Int64
    n_volume_moves::Int64
    n_atom_swaps::Int64
    pressure::Float64
    separated_volume::Bool
    stress_tensor::SVector{2,Float64}
    reference_length::Float64
end
#= This first method is to ensure that any prior code which constructed an NPT ensemble
using 6 variables continues to construct the correct ensemble now that there are 8 options.=#
function NPT(
    n_atoms::Int64,
    n_atom_moves::Int64,
    n_volume_moves::Int64,
    n_atom_swaps::Int64,
    pressure::Float64,
    separated_volume::Bool,
)
    return NPT(
        n_atoms,
        n_atom_moves,
        n_volume_moves,
        n_atom_swaps,
        pressure,
        separated_volume,
        [0, 0],
        0,
    )
end
# This generates the appropriate ensemble by assuming omitted parameters take on default values.
function NPT(
    n_atoms::Int64,
    pressure::Float64;
    separated_volume::Bool=false,
    stress_tensor=[0, 0],
    reference_length::Float64=0.0,
)
    return NPT(
        n_atoms, n_atoms, 1, 0, pressure, separated_volume, stress_tensor, reference_length
    )
end
#=For users who are using regular NPT, this function allows a shortcut for constructing
a separated volume NPT ensemble.=#
function NPT(n_atoms, pressure, separated_volume)
    return NPT(n_atoms, n_atoms, 1, 0, pressure, separated_volume, [0, 0], 0)
end
function report_stats(mc_state, ::NPT)
    bc = mc_state.config.boundary_condition
    return (; volume=volume(bc), report_stats(bc)...)
end

"""
    NPTVariables <: AbstractEnsembleVariables

Contains [`NPT`](@ref) ensemble variables that change during MC run.
-   Field names:
    -   `index::Int64`
    -   `trial_move::SVector{3,T}`
    -   `trial_config::Config`
    -   `new_dist2_mat::Matrix{T}`

Using an NPT ensemble, the type of move is selected according to `index`. For indices
smaller or equal to the number of atoms in the system, a `trial_move` for the `index`-th
atom is generated. If `index` is larger that the number of atoms, a volume move is trialled
involving the generation of a scaled `trial_config` and corresponding `new_dist2_mat`.
"""
mutable struct NPTVariables{T} <: AbstractEnsembleVariables
    index::Int64
    trial_move::SVector{3,T}
    trial_config::Config
    new_dist2_mat::Matrix{T}
    xy_or_z::Int
end

#---------------------------------------------------------------------#
#--------------------------------NNVT---------------------------------#
#---------------------------------------------------------------------#
"""
    NNVT <: AbstractEnsemble

Ensemble designed for systems with two types of atoms.

## Field names:
- `n_atoms`: vector specifying how much of each species we have in the system
- `n_atom_moves`: defaults to n_total
- `n_atom_swaps`: defaults to 1 per cycle
"""
struct NNVT <: AbstractEnsemble
    n_atoms::SVector{2,Int}
    n_atom_moves::Int
    n_atom_swaps::Int
end
function NNVT(n_atoms_vec; n_atom_swaps=1, n_atom_moves=sum(n_atoms_vec))
    if isa(n_atoms_vec, Vector)
        n_atoms = SVector{2}(n_atoms_vec)
    elseif isa(n_atoms_vec, SVector)
        n_atoms = n_atoms_vec
    end
    return NNVT(n_atoms, n_atom_moves, n_atom_swaps)
end

"""
    NNVTVariables <: AbstractEnsembleVariables

NNVT - specific ensembles for moves made during an NNVT run.
Fields include:
    - index: Used for standard atom moves
    - trial_move: Used for standard atom moves
    - atom_list1: index of atoms of type one
    - atom_list2: index of atoms of type two
"""
mutable struct NNVTVariables{T,N,N1,N2} <: AbstractEnsembleVariables
    index::Int64
    trial_move::SVector{3,T}
    swap_indices::SVector{2,Int}
end

#---------------------------------------------------------------------#
#------------------------global functions-----------------------------#
#---------------------------------------------------------------------#
function hamiltonian(state, ::AbstractEnsemble)
    return state.en_tot
end

"""
    set_ensemble_variables(config::Config, ensemble::NVT)
    set_ensemble_variables(config::Config, ensemble::NPT)
    set_ensemble_variables(config::Config, ensemble::NNVT)

Initialises the instance of EnsembleVariables (with ensemble being `NVT` or `NPT`);
required to allow for neutral initialisation in defining the MCState [`Main.ParallelTemperingMonteCarlo.MCStates.MCState`](@ref) struct.
"""
function set_ensemble_variables(config::Config{T}, ensemble::NVT) where {T}
    return NVTVariables{T}(1, SVector{3}(zeros(3)))
end

function set_ensemble_variables(config::Config{T}, ensemble::NPT) where {T}
    if config.boundary_condition isa SphericalBC
        error("SphericalBC cannot be used in an NPT ensemble.")
    end
    return NPTVariables{T}(
        1,
        SVector{3}(zeros(3)),
        deepcopy(config),
        zeros(ensemble.n_atoms, ensemble.n_atoms),
        0,
    )
end
function set_ensemble_variables(config::Config{T}, ensemble::NNVT) where {T}
    N1, N2 = ensemble.n_atoms[1], ensemble.n_atoms[2]
    return NNVTVariables{T,length(config),N1,N2}(
        1, SVector{3}(zeros(3)), SVector{2}(1, N1 + 1)
    )
end

function hamiltonian(state, ensemble::NPT)
    V = volume(state.config.boundary_condition)
    p = ensemble.pressure
    E = state.en_tot
    if ensemble.separated_volume
        σ = ensemble.stress_tensor
        xy = state.config.boundary_condition.box_length
        z = state.config.boundary_condition.box_height
        L0 = ensemble.reference_length

        return E + p*V + L0 * (σ[1] * xy + σ[2] * z / 2)
    else
        return E + p*V
    end
end

end
