"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    
module EnergyEvaluation

using StaticArrays
using ..Configurations
using ..RuNNer

export AbstractPotential, AbstractDimerPotential, AbstractMLPotential 
export ELJPotential, ELJPotentialEven
export dimer_energy, dimer_energy_atom, dimer_energy_config
export energy_update
export AbstractEnsemble, NVT, NPT
export EnHist

"""   
    AbstractPotential
Abstract type for possible potentials
implemented subtype: 
- AbstractDimerPotential

Needs method for dimer_energy [`dimer_energy`](@ref)
"""
abstract type AbstractPotential end

"""
    AbstractDimerPotential <: AbstractPotential
 implemented dimer potentials:   
    - ELJPotential [`ELJPotential`](@ref)
    - ELJPotentialEven [`ELJPotentialEven`](@ref)

Needs methods for 
    - dimer_energy_atom [`dimer_energy_atom`](@ref)
    - dimer_energy_config [`dimer_energy_config`](@ref)
"""   
abstract type AbstractDimerPotential <: AbstractPotential end

struct AbstractMLPotential <: AbstractPotential 
    dir::String
    atomtype::String
end

#function AbstractMLPotential(dir::String,atomtype::String)
#    return AbstractMLPotential(dir,atomtype)
#end

"""
    dimer_energy_atom(i, pos, d2vec, pot<:AbstractPotential)
Sums the dimer energies for atom `i` with all other atoms
Needs vector of squared distances `d2vec` between atom `i` and all other atoms in configuration
see  `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_atom(i, d2vec, pot::AbstractDimerPotential)
    sum1 = 0.
    for j in 1:i-1
        sum1 += dimer_energy(pot, d2vec[j])
    end
    for j in i+1:size(d2vec,1)
        sum1 += dimer_energy(pot, d2vec[j])
    end 
    return sum1
end

"""
    dimer_energy_config(distmat, NAtoms, pot::AbstractPotential)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, NAtoms, pot::AbstractDimerPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    for i in 1:NAtoms #eachindex(),enumerate()..?
        dimer_energy_vec[i] = dimer_energy_atom(i, distmat[i, :], pot) #@view distmat[i, :]
        energy_tot += dimer_energy_vec[i]
    end 
    return dimer_energy_vec, 0.5*energy_tot
end    

function energy_update(i_atom, dist2_new, pot::AbstractDimerPotential)
    return dimer_energy_atom(i_atom, dist2_new, pot)
end

function energy_update(pos, i_atom, config, dist2_mat, pot::AbstractDimerPotential)
    dist2_new = [distance2(pos,b) for b in config.pos]
    dist2_new[i_atom] = 0.
    delta_en = dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, dist2_mat[i_atom,:], pot)
    return delta_en, dist2_new
end

function energy_update(pos,i_atom,config,dist2_mat,pot::AbstractMLPotential)

    dist2_new = [distance2(pos,b) for b in config.pos]
    dist2_new[i_atom] = 0.

    Evec = RuNNer.getenergy(pot.dir,config,pot.atomtype,i_atom,pos)
    delta_en = Evec[2] - Evec[1]

    return delta_en, dist2_new

end

"""
    ELJPotential{N,T} 
Implements type for extended Lennard Jones potential; subtype of [`AbstractDimerPotential`](@ref)<:[`AbstractPotential`](@ref);
as sum over c_i r^(-i), starting with i=6 up to i=N+6
field name: coeff : contains ELJ coefficients c_ifrom i=6 to i=N+6, coefficient for every power needed.
"""
struct ELJPotential{N,T} <: AbstractDimerPotential
    coeff::SVector{N,T}
end

function ELJPotential{N}(c) where N
    @boundscheck length(c) == N || error("number of ELJ coefficients does not match given length")
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotential{N,T}(coeff)
end

function ELJPotential(c) 
    N = length(c)
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotential{N,T}(coeff)
end

"""
    dimer_energy(pot::ELJPotential{N}, r2)
Calculates energy of dimer for given potential `pot` and squared distance `r2` between atoms
methods implemented for:

    - ELJPotential [`ELJPotential`](@ref)

    - ELJPotentialEven [`ELJPotentialEven`](@ref)
"""
function dimer_energy(pot::ELJPotential{N}, r2) where N
    r = sqrt(r2)
    r6inv = 1/(r2*r2*r2)
    sum1 = 0.
    for i = 1:N
        sum1 += pot.coeff[i] * r6inv
        r6inv /= r 
    end
    return sum1
end

"""
    ELJPotentialEven{N,T} 
Implements type for extended Lennard Jones potential with only even powers; subtype of [`AbstractDimerPotential`](@ref)<:[`AbstractPotential`](@ref);
as sum over c_i r^(-i), starting with i=6 up to i=N+6 with only even integers i
field name: coeff : contains ELJ coefficients c_i from i=6 to i=N+6 in steps of 2, coefficient for every even power needed.
"""
struct ELJPotentialEven{N,T} <: AbstractDimerPotential
    coeff::SVector{N,T}
end

function ELJPotentialEven{N}(c) where N
    @boundscheck length(c) == N || error("number of ELJ coefficients does not match given length")
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialEven{N,T}(coeff)
end

function ELJPotentialEven(c) 
    N = length(c)
    coeff = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialEven{N,T}(coeff)
end

function dimer_energy(pot::ELJPotentialEven{N}, r2) where N
    r6inv = 1/(r2*r2*r2)
    sum1 = 0.
    for i = 1:N
        sum1 += pot.coeff[i] * r6inv
        r6inv /= r2 
    end
    return sum1
end

"""
    LJPotential{T} 
Implements Lennard Jones potential; subtype of [`AbstractDimerPotential`](@ref)<:[`AbstractPotential`](@ref);
as c_6 r^(-6) + c_12 r^(-12)
field name: coeff : contains LJ coefficients c_6 and c_12
"""
struct LJPotential{T} <: AbstractDimerPotential
    coeff::SVector{2,T}
end

function LJPotential(c)
    @boundscheck length(c) == 2 || error("exactly 2 LJ coefficients (c_6 and c_12) needed")
    coeff = SVector{2}(c)
    T = eltype(c)
    return ELJPotential{N,T}(coeff)
end

function dimer_energy(pot::LJPotential, r2)
    r6inv = 1/(r2*r2*r2)
    sum1 = pot.coeff[1] * r6inv + pot.coeff[2] * r6inv * r6inv
    return sum1
end

"""
    AbstractEnsemble
abstract type for ensemble:
    - NVT: canonical ensemble
    - NPT: isothermal, isobaric
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
    n_atoms::Int
    pressure::Real
end

"""
    EnHist(n_bin, en_min::T, en_max::T)
    EnHist(n_bin; en_min=-0.006, en_max=-0.001)
Collects data for energy histograms per temperature
Field names:    
- `n_bins`: number of energy bins
- `en_min`,`en_max`: minimum and maximum energy between which data is collected
- `delta_en_bin`: energy spacing of bins
- `en_hist`: stores number of sampled configurations per energy bins
"""
struct EnHist{T}
    n_bin::Int
    en_min::Ref{T}
    en_max::Ref{T}
    delta_en_bin::Ref{T}
    en_hist::Vector{Int}
end

function EnHist(n_bin, en_min::T, en_max::T) where T
    delta_en_bin = (en_max-en_min)/n_bin
    en_hist = zeros(Int, n_bin)
    return EnHist{T}(n_bin,en_min,en_max,delta_en_bin,en_hist)
end

function EnHist(n_bin; en_min=-0.006, en_max=-0.001)
    T = eltype(en_min)
    delta_en_bin = (en_max-en_min)/n_bin
    en_hist = zeros(Int, n_bin)
    return EnHist{T}(n_bin,en_min,en_max,delta_en_bin,en_hist)
end

end
