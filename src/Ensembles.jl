module Ensembles 

using ..Configurations
using StaticArrays

export AbstractEnsemble,NVT,NPT 

export EnsembleVariables,NVTVariables,NPTVariables,set_ensemble_variables

export MoveType,atommove,volumemove,atomswap 
export MoveStrategy
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
    n_atoms::Int
    n_atom_moves::Int
    n_atom_swaps::Int
end
function NVT(n_atoms)
    return NVT(n_atoms,n_atoms,0)
end

mutable struct  NVTVariables <: EnsembleVariables
    index::Int
    trial_move:: SVector
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
    n_atoms::Int
    n_atom_moves::Int
    n_volume_moves::Int
    n_atom_swaps::Int
    pressure::Real
end
function NPT(n_atoms,pressure)
    return NPT(n_atoms,n_atoms,1,0,pressure)
end
mutable struct NPTVariables <: EnsembleVariables
    index::Int
    trial_move::SVector
    trial_config::Config
    new_dist2_mat::Matrix
    r_cut::Real
    new_r_cut::Real
end

function set_ensemble_variables(config, ensemble::NVT)
    return NVTVariables(1,SVector{3}(zeros(3)))
end
function set_ensemble_variables(config,ensemble::NPT)
    return NPTVariables(1,SVector{3}(zeros(3)),deepcopy(config),zeros(ensemble.n_atoms,ensemble.n_atoms),config.bc.box_length^2/4,0.)
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

"""

struct MoveStrategy{N} 
    ensemble::AbstractEnsemble
    movestrat::Vector
end
function MoveStrategy(ensemble::NPT)
    movestrat = []
    for m_index in 1:ensemble.n_atom_moves
        push!(movestrat,atommove())
    end
    for m_index in 1:ensemble.n_volume_moves
        push!(movestrat,volumemove())
    end
    for m_index in 1:ensemble.n_atom_swaps
        push!(movestrat,atomswap())
    end

    return MoveStrategy{ensemble.n_atom_moves+ensemble.n_atom_swaps+ensemble.n_volume_moves}(ensemble,movestrat)
end
function MoveStrategy(ensemble::NVT)
    movestrat = []
    for m_index in 1:ensemble.n_atom_moves
        push!(movestrat,atommove())
    end
    for m_index in 1:ensemble.n_atom_swaps
        push!(movestrat,atomswap())
    end

    return MoveStrategy{ensemble.n_atom_moves+ensemble.n_atom_swaps}(ensemble,movestrat)
end
Base.length(::MoveStrategy{N}) where N = N


end