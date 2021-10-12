"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    

module EnergyEvaluation

#using Configurations
using StaticArrays

export ELJPotential
export dimer_energy, dimer_energy_atom

#Dimer energies - no angle dependence
#extended Lennard Jones, ELJ, general definition n=6 to N+6

struct ELJPotential{N,T}
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


# function dimer_energy1(pot::ELJPotential{N},r2) where N
#     r = sqrt(r2)
#     sum1 = 0.
#     for i = 1:N
#         sum1 += pot.coeff[i] * r^(-i-5)
#     end
#     return sum1
# end

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

function dimer_energy_atom(i, d2mat, pot)
    sum1 = 0.
    for j in 1:i-1
        sum1 += dimer_energy(pot, d2mat[j,i])
    end
    for j in i+1:size(d2mat,1)
        sum1 += dimer_energy(pot, d2mat[i,j])
    end 
    return sum1
end

end
