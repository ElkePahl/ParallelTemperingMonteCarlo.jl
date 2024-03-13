module Ensembles 

using ..Configurations
using ..BoundaryConditions
using StaticArrays

export AbstractEnsemble,NVT,NPT 

export EnsembleVariables,NVTVariables,NPTVariables,set_ensemble_variables

export MoveType,atommove,volumemove,atomswap 
export MoveStrategy
export get_r_cut
"""
    abstract type AbstractEnsemble
abstract type for ensemble:
    - NVT: canonical ensemble
    - NPT: isotherman,isobaric
"""
abstract type AbstractEnsemble end
"""
    abstract type EnsembleVariables

Basic type for the struct containing mutable variables, this will sit in MCStates and be changed in the displacement steps.
"""
abstract type EnsembleVariables end
"""
    NVT
canonical ensemble
fieldname: 
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
    NVTVariables <: EnsembleVariables 
Fields for the NVT ensemble include
        - Index 
        - trial_move 
    When trialing a new configuration we select an atom at `index` to move to `trial_move`
"""
mutable struct  NVTVariables{T} <: EnsembleVariables
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
end
function NPT(n_atoms,pressure)
    return NPT(n_atoms,n_atoms,1,0,pressure)
end
"""
    NPTVariables <: EnsembleVariables 
Fields for the NPT ensemble include
        - Index 
        - trial_move 
        - trial_config
        - new_dist2_mat
        - r_cut 
        - new_r_cut
    When trialing a new configuration we select an atom at `index` to move to `trial_move`, the index can be over n_atoms in which case we trial a scaled `new_config` with a `new_r_cut` having a `new_dist2_mat`
"""
mutable struct NPTVariables{T} <: EnsembleVariables
    index::Int64
    trial_move::SVector{3,T}
    trial_config::Config
    new_dist2_mat::Matrix{T}
    r_cut::T
    new_r_cut::T
end

function get_r_cut(bc::CubicBC)
    return bc.box_length^2/4
end

function get_r_cut(bc::RhombicBC)
    #return min(bc.box_length^2*3/16,bc.box_height^2/4)
    return bc.box_length^2*3/16
end

function set_ensemble_variables(config::Config{N,BC,T}, ensemble::NVT) where {N,BC,T}
    return NVTVariables{T}(1,SVector{3}(zeros(3)))
end
function set_ensemble_variables(config::Config{N,BC,T},ensemble::NPT) where {N,BC,T}
    return NPTVariables{T}(1,SVector{3}(zeros(3)),deepcopy(config),zeros(ensemble.n_atoms,ensemble.n_atoms),get_r_cut(config.bc),0.)
end

"""
    abstract type MoveType
defines the type of move to establish the movestrat struct. Basic types are:
    - atommove: basic move of a single atom 
    - volumemove: NPT ensemble requires volume changes to maintain pressure as constant 
    - atomswap: for diatomic species we need to exchange atoms of differing types
"""
abstract type MoveType end

struct atommove <: MoveType end

struct volumemove <: MoveType end
struct atomswap <: MoveType end
"""
    struct MoveStrategy
        - MoveStrategy(ensemble::NPT)
        - MoveStrategy(ensemble::NVT)
A struct containing an ensemble and a movestrategy vector. This vector has movetypes in the appropriate ratio so that when we generate a trial index, we select the appropriate move type. 
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