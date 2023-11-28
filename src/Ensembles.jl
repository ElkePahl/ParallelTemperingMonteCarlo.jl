module Ensembles 

using ..Configurations


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
fieldname: natoms: number of atoms   
"""
struct NVT <: AbstractEnsemble
    n_atoms::Int
end
mutable struct  NVTVariables <: EnsembleVariables
    index::Int
    trial_move:: Vector
end

"""
    NPT
isothermal, isobaric ensemble
fieldnames: 
- natoms: number of atoms
- pressure
"""
struct NPT <: AbstractEnsemble
    n_atoms::Int
    pressure::Real
end
mutable struct NPTVariables <: EnsembleVariables
    index::Int
    trial_move::Vector
    trial_config::Config
    r_cut::Real
end

function set_ensemble_variables(config, ensemble::NVT)
    return NVTVariables(1,[0.,0.,0.])
end
function set_ensemble_variables(config,ensemble::NPT)
    return NPTVariables(1,[0.,0.,0.],deepcopy(config),config.bc.box_length^2/4)
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
function MoveStrategy(ensemble,a,v,s)
    movestrat = []
    for m_index in 1:a
        push!(movestrat,atommove())
    end
    for m_index in 1:v
        push!(movestrat,volumemove())
    end
    for m_index in 1:s
        push!(movestrat,atomswap())
    end

    return MoveStrategy{a+v+s}(ensemble,movestrat)
end

Base.length(::MoveStrategy{N}) where N = N


end