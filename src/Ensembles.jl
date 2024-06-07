module Ensembles 

using ..Configurations
using ..BoundaryConditions
using StaticArrays

export AbstractEnsemble,NVT,NPT 

export AbstractEnsembleVariables,NVTVariables,NPTVariables,set_ensemble_variables

export MoveType,atommove,volumemove,atomswap 
export MoveStrategy
export get_r_cut

"""
    AbstractEnsemble
abstract type for ensemble:
    - NVT: canonical ensemble [`NVT`](@ref)
    - NPT: isothermal,isobaric [`NPT`](@ref)

    Each subtype requires a corresponding AbstractEnsembleVariable struct
"""
abstract type AbstractEnsemble end

"""
    AbstractEnsembleVariables
Abstract struct for variables specific to ensemble that change during MC run (moves)
"""
abstract type AbstractEnsembleVariables end

"""
    NVT
canonical ensemble
fieldnames: 
    -n_atoms: number of atoms
    -n_atom_moves: number of atom moves; defaults to n_atoms
    -n_swap_moves: number of atom exchanges made; defaults to 0
"""
struct NVT <: AbstractEnsemble
    n_atoms::Int64
    n_atom_moves::Int64
    n_atom_swaps::Int64
end

function NVT(n_atoms)
    return NVT(n_atoms,n_atoms,0)
end

"""
    NVTVariables <: AbstractEnsembleVariables 
NVT ensemble specific variables that change during MC run
Fields for the NVT ensemble include:
        - index 
        - trial_move 
    When trialing a new configuration we select an atom at `index` to move to position given by `trial_move`
"""
mutable struct  NVTVariables{T} <: AbstractEnsembleVariables
    index::Int64
    trial_move:: SVector{3,T}
end

"""
    NPT
isothermal, isobaric ensemble
fieldnames: 
    -n_atoms: number of atoms
    -n_atom_moves: number of atom moves; defaults to n_atoms
    -n_volume_moves: number of volume moves; defaults to 1
    -n_swap_moves: number of atom exchanges made; defaults to 0
    -pressure: the fixed pressure of the system
"""
struct NPT <: AbstractEnsemble
    n_atoms::Int64
    n_atom_moves::Int64
    n_volume_moves::Int64
    n_atom_swaps::Int64
    pressure::Float64
    separated_volume::Bool
end

function NPT(n_atoms,pressure,separated_volume)
    return NPT(n_atoms,n_atoms,1,0,pressure,separated_volume)
end

"""
    NPTVariables <: AbstractEnsembleVariables 
NPT ensemble specific variable that change during MC run
Field names:
        - index 
        - trial_move 
        - trial_config
        - new_dist2_mat
        - r_cut 
        - new_r_cut
    When trialing a new configuration we select an atom at `index` to move to new position `trial_move`, 
    the index can be greater than n_atoms in which case we trial a volume move, 
    involving a scaled `trial_config` with a `new_r_cut` having a `new_dist2_mat` this being a volume move
"""
mutable struct NPTVariables{T} <: AbstractEnsembleVariables
    index::Int64
    trial_move::SVector{3,T}
    trial_config::Config
    new_dist2_mat::Matrix{T}
    r_cut::T
    new_r_cut::T
    xy_or_z::Int
end

"""
get_r_cut(bc<:PeriodicBC)
finds the square of the cut-off radius `r_cut` that is implied by periodic boundary conditions (to avoid double-counting).
implemented for `CubicBC` and `RhombicBC`.
"""
function get_r_cut(bc::CubicBC)
    return bc.box_length^2/4
end

function get_r_cut(bc::RhombicBC)
    return min(bc.box_length^2*3/16,bc.box_height^2/4)
    #return bc.box_length^2*3/16
end

"""
set_ensemble_variables(config::Config{N,BC,T}, ensemble)
initialises the instance of EnsembleVariables (with ensemble being `NVT` or `NPT`);
required to allow for neutral initialisation in defining the MCState [`MCStates.MCState`](@ref) struct. 
"""
function set_ensemble_variables(config::Config{N,BC,T}, ensemble::NVT) where {N,BC,T}
    return NVTVariables{T}(1,SVector{3}(zeros(3)))
end

function set_ensemble_variables(config::Config{N,BC,T},ensemble::NPT) where {N,BC,T}
    return NPTVariables{T}(1,SVector{3}(zeros(3)),deepcopy(config),zeros(ensemble.n_atoms,ensemble.n_atoms),get_r_cut(config.bc),0.,0)
end

"""
    MoveType
defines the abstract type for moves to establish the movestrat struct. Basic types are:
    - `atommove`: basic move of a single atom 
    - `volumemove`: NPT ensemble requires volume changes to maintain pressure as constant 
    - `atomswap`: for systems with different atom types we need to exchange atoms (not yet implemented)
"""
abstract type MoveType end

struct atommove <: MoveType end

struct volumemove <: MoveType end

struct atomswap <: MoveType end

"""
    MoveStrategy{N,Etype}
A struct to define the types of moves performed per MC cycle 
Field names:
    - ensemble: type of ensemble (NVT, NPT)
    - movestrat: vector of strings that describes moves made per MC cycle (see `MoveType`)
"""
# for the time being we substitute 0,1,2 as the basic input for atom,volume and swaps. 
struct MoveStrategy{N,Etype}
    ensemble::Etype
    movestrat::Vector{String}
end

function MoveStrategy(ensemble::NPT)
    movestrat = []
    for m_index in 1:ensemble.n_atom_moves
        push!(movestrat,"atommove")
    end
    for m_index in 1:ensemble.n_volume_moves
        push!(movestrat,"volumemove")
    end
    for m_index in 1:ensemble.n_atom_swaps
        push!(movestrat,"atomswap")
    end

    return MoveStrategy{ensemble.n_atom_moves+ensemble.n_atom_swaps+ensemble.n_volume_moves,typeof(ensemble)}(ensemble,movestrat)
end

function MoveStrategy(ensemble::NVT)
    movestrat = []
    for m_index in 1:ensemble.n_atom_moves
        push!(movestrat,"atommove")
    end
    for m_index in 1:ensemble.n_atom_swaps
        push!(movestrat,"atomswap")
    end

    return MoveStrategy{ensemble.n_atom_moves+ensemble.n_atom_swaps,typeof(ensemble)}(ensemble,movestrat)
end

Base.length(::MoveStrategy{N}) where N = N


end