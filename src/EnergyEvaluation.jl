"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    

module EnergyEvaluation

#using Configurations
using StaticArrays

export AbstractPotential, AbstractDimerPotential 
export ELJPotential
export dimer_energy, dimer_energy_atom

#Dimer energies - no angle dependence
#extended Lennard Jones, ELJ, general definition n=6 to N+6

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

Needs method for dimer_energy [`dimer_energy`](@ref)
"""   
abstract type AbstractDimerPotential <: AbstractPotential

"""
    ELJPotential{N,T} 
Implements type for extended Lennard Jones potential; subtype of [`AbstractPotential`](@ref);
as sum over c_i r^(-i), starting with i=6 up to i=N+6
field name: coeff : contains ELJ coefficients c_ifrom i=6 to i=N+6, coefficient for every power needed.
"""
struct ELJPotential{N,T} <: AbstractDimerPotential
    coeff::SVector{N,T}
end

function ELJPotential{N}(c) where {N}
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
Calculates energy of dimer for given potential `pot` of type ELJPotential [`ELJPotential`](@ref), 
and squared distance `r2` between atoms
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
    dimer_energy_atom(i, pos, d2vec, pot<:AbstractPotential)
Sums the dimer energies for atom `i` with all other atoms
Needs vector of squared distances `d2vec` between atom `i` and all other atoms in configuration
see  `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_atom(i, d2vec, pot::AbstractPotential)
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

function dimer_energy_config(distmat, NAtoms, pot::AbstractPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    for i in 1:NAtoms
        dimer_energy_vec[i] = dimer_energy_atom(i, distmat[i, :], pot)
        energy_tot += dimer_energy_vec[i]
    end 
    return dimer_energy_vec, 0.5*energy_tot
end

end
