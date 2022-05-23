"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    

module EnergyEvaluation

#using Configurations
using StaticArrays

export ELJPotential, AbstractPotential 
export dimer_energy, dimer_energy_atom, dimer_energy_config, elj_ne

#Dimer energies - no angle dependence
#extended Lennard Jones, ELJ, general definition n=6 to N+6

"""   
    AbstractPotential
Encompasses possible potentials; implemented: 
- ELJPotential [`ELJPotential`](@ref)


Needs method for dimer_energy [`dimer_energy`](@ref)
"""
abstract type AbstractPotential end

"""   
    ELJPotential{N,T} 
Implements type for extended Lennard Jones potential; subtype of [`AbstractPotential`](@ref);
as sum over c_i r^(-i), starting with i=6 up to i=N+6
field name: coeff : contains ELJ coefficients c_ifrom i=6 to i=N+6, coefficient for every power needed.
"""
struct ELJPotential{N,T} <: AbstractPotential
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
    dimer_energy_mat = zeros(NAtoms,NAtoms)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    for i in 1:NAtoms
        for j in 1:NAtoms
            if i!=j
                dimer_energy_mat[i,j]=dimer_energy(pot, distmat[i,j])
            end
        end
        dimer_energy_vec[i] = dimer_energy_atom(i, distmat[i, :], pot)
        energy_tot += dimer_energy_vec[i]
    end 
    return dimer_energy_mat, dimer_energy_vec, 0.5*energy_tot
end



c=[-10.5097942564988, 0., 989.725135614556, 0., -101383.865938807, 0., 3918846.12841668, 0., -56234083.4334278, 0., 288738837.441765]

println("extended LJ potential:")
E=ELJPotential(c)
println(E)
println()


elj_ne = ELJPotential{11}(c)


end
