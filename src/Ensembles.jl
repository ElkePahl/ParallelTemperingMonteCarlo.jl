module Ensembles 

export AbstractEnsemble,NVT,NPT 
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
    NVT
canonical ensemble
fieldname: natoms: number of atoms   
"""
struct NVT <: AbstractEnsemble
    n_atoms::Int
end
"""
    NPT
isothermal, isobaric ensemble
fieldnames: 
- natoms: number of atoms
- pressure
"""
struct NPT <: AbstractEnsemble
    pressure::Real
    n_atoms::Int
end

"""
    abstract type MoveType
defines the type of move to establish the movestrat struct. Basic types are:
    - atommove: basic move of a single atom 
    - volumemove: NPT ensemble requires volume changes to maintain pressure as constant 
    - atomswap: for diatomic species we need to exchange atoms of differing types
"""
abstract type MoveType end

abstract type atommove <: MoveType end

abstract type volumemove <: MoveType end
abstract type atomswap <: MoveType end
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
        push!(movestrat,atommove)
    end
    for m_index in 1:v
        push!(movestrat,volumemove)
    end
    for m_index in 1:s
        push!(movestrat,atomswap)
    end

    return MoveStrategy{a+v+s}(ensemble,movestrat)
end

Base.length(::MoveStrategy{N}) where N = N


end