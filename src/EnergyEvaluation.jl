"""
    module EnergyEvaluation
Structs and functions relating to the calculation of energy. Includes both low and high level functions from individual PES calculations to state-specific functions. The structure is as follows:
    Define abstract potential and potential_variables
    -Define PES functions
        -  DimerPotential 
        -  ELJB 
        -  Embedded Atom Model 
        -  Machine Learning Potentials 
    -EnergyUpdate function
        Calculates a new energy based on a trialpos for each PES type 
    - InitialiseEnergy function 
            Calculates potentialvariables and total energy from a new config to be used when initialising MCStates 
    -SetVariables function 
            Initialises the potential variables, aka creates a blank version of the struct for each type of PES
"""

module EnergyEvaluation 

using StaticArrays,LinearAlgebra,StructArrays

using ..MachineLearningPotential
using ..Configurations
using ..Ensembles 
using ..BoundaryConditions


export AbstractPotential,AbstractDimerPotential,ELJPotential,ELJPotentialEven,AbstractMachineLearningPotential
export AbstractDimerPotentialB,ELJPotentialB,EmbeddedAtomPotential,RuNNerPotential
export AbstractPotentialVariables,DimerPotentialVariables,ELJPotentialBVariables
export EmbeddedAtomVariables,NNPVariables


export dimer_energy,dimer_energy_atom, dimer_energy_update!, calc_energies_from_components, invrexp, lrc


export energy_update!,set_variables,initialise_energy,dimer_energy_config, calc_components
#-------------------------------------------------------------#
#----------------------Universal Structs----------------------#
#-------------------------------------------------------------#


#-------------------------------------------------------------#
"""   
    AbstractPotential
Abstract type for possible potentials.
implemented subtype: 
- AbstractDimerPotential
- AbstractDimerPotentialB
- EmbeddedAtomPotential
- AbstractMachineLearningPotential


When defining a new type, the functions relating a potential to the rest of the Monte Carlo code are explicated at the end of this file. Each potential also requires a PotentialVariable [`AbstractPotentialVariables`](@ref) struct to hold all non-static information relating a potential to the current configuration. 

 Needs method for:
    energy_update! [`energy_update!`](@ref)
    initialise_energy [`initialise_energy`](@ref) 
    set_variables [`set_variables`](@ref)

"""
abstract type AbstractPotential end

"""
    AbstractPotentialVariables
An abstract type defining a class of mutable struct containing all the relevant vectors and arrays each potential will need throughout the course of a simulation to prevent over-definitions inside the MCState struct.
    implemented subtype:
    -DimerPotentialVariables
    -ELJPotentialBVariables
    -EmbeddedAtomVariables
    -NNPVariables

"""
abstract type AbstractPotentialVariables end
#-----------------------------------------------------------------------#
#-----------------------Explicit Dimer Potentials-----------------------#
#-----------------------------------------------------------------------#

#--------------------------Dimer Structs--------------------------#
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
abstract type AbstractDimerPotentialB <: AbstractPotential end
"""
    DimerPotentialVariables
The struct contains only the `en_atom_vec``, particular special features for this potential type. 
This vector is the energy per atom in the system.
"""
mutable struct DimerPotentialVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Vector{T}
end
##
#Need to include ELJB here to prevent recursive definitions erroring 
##
"""
ELJPotentialB{N,T}
   Extended Lennard-Jones Potential in a magnetic field where there is anisotropy in the coefficient vectors `coeff_a`, `coeff_b`, `coeff_c`
"""
struct ELJPotentialB{N,T} <: AbstractDimerPotentialB
    coeff_a::SVector{N,T}
    coeff_b::SVector{N,T}
    coeff_c::SVector{N,T}
end
function ELJPotentialB{N}(a,b,c) where N
    @boundscheck length(c) == N || error("number of ELJ coefficients does not match given length")
    coeff_a = SVector{N}(a)
    coeff_b = SVector{N}(b)
    coeff_c = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialB{N,T}(coeff_a,coeff_b,coeff_c)
end

function ELJPotentialB(a,b,c) 
    N = length(c)
    coeff_a = SVector{N}(a)
    coeff_b = SVector{N}(b)
    coeff_c = SVector{N}(c)
    T = eltype(c)
    return ELJPotentialB{N,T}(coeff_a,coeff_b,coeff_c)
end

mutable struct ELJPotentialBVariables{T} <: AbstractPotentialVariables
    en_atom_vec::Array{T}
    tan_mat::Matrix{T}
    new_tan_vec::Vector{T}
end
#---------------------------------------------------------------------------#
#--------------------------Universal functions------------------------------#
#---------------------------------------------------------------------------#
"""
    dimer_energy_atom(i, d2vec, pot::AbstractDimerPotential)
    dimer_energy_atom(i, d2vec, r_cut, pot::AbstractDimerPotential)
    dimer_energy_atom(i, d2vec, tanvec,pot::AbstractDimerPotentialB)
    dimer_energy_atom(i, d2vec, tanvec, r_cut, pot::AbstractDimerPotentialB)

Sums the dimer energies for atom `i` with all other atoms 
Needs vector of squared distances `d2vec` between atom `i` and all other atoms in configuration
see  `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 

second method includes additional variable `r_cut` to exclude distances outside the cutoff radius of the potential.

Final two methods relate to the use of magnetic field potentials such as the ELJB potential.
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
function dimer_energy_atom(i, d2vec, r_cut, pot::AbstractDimerPotential)
    sum1 = 0.
    for j in 1:i-1
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j])
        end
    end
    for j in i+1:size(d2vec,1)
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j])
        end
    end 
    return sum1
end
function dimer_energy_atom(i, d2vec, tanvec,pot::AbstractDimerPotentialB)
    sum1 = 0.
    for j in 1:i-1
        sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
    end
    for j in i+1:size(d2vec,1)
        sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
    end 
    return sum1
end
function dimer_energy_atom(i, d2vec, tanvec, r_cut, pot::AbstractDimerPotentialB)
    sum1 = 0.
    for j in 1:i-1
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
        end
    end
    for j in i+1:size(d2vec,1)
        if d2vec[j] <= r_cut
            sum1 += dimer_energy(pot, d2vec[j], tanvec[j])
        end
    end 
    return sum1
end
"""
    dimer_energy_config(distmat, NAtoms, pot::AbstractDimerPotential)
    dimer_energy_config(distmat, NAtoms,potential_variables::DimerPotentialVariables, r_cut, pot::AbstractDimerPotential)
    dimer_energy_config(distmat, NAtoms,potential_variables::ELJPotentialBVariables, pot::AbstractDimerPotentialB)
    dimer_energy_config(distmat, NAtoms,potential_variables::ELJPotentialBVariables, r_cut, pot::AbstractDimerPotentialB)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration.

First two methods are for standard dimer potentials, one with a cutoff radius, one without a cutoff radius. The final two methods are for the same calculation using a magnetic potential such as the ELJB potential. 

Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, NAtoms, potential_variables::DimerPotentialVariables,pot::AbstractDimerPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.

    for i in 1:NAtoms
        for j=i+1:NAtoms
            e_ij=dimer_energy(pot,distmat[i,j])
            dimer_energy_vec[i] += e_ij
            dimer_energy_vec[j] += e_ij
            energy_tot += e_ij
        end
    end 
    #energy_tot=sum(dimer_energy_vec)
    return dimer_energy_vec, energy_tot
end
function dimer_energy_config(distmat, NAtoms,potential_variables::DimerPotentialVariables, r_cut, bc::CubicBC, pot::AbstractDimerPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.

    for i in 1:NAtoms
        for j=i+1:NAtoms
            if distmat[i,j] <= r_cut
                e_ij=dimer_energy(pot,distmat[i,j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end 

    return dimer_energy_vec, energy_tot + lrc(NAtoms,r_cut,pot)   #no 0.5*energy_tot
end 
function dimer_energy_config(distmat, NAtoms,potential_variables::DimerPotentialVariables, r_cut, bc::RhombicBC, pot::AbstractDimerPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.

    for i in 1:NAtoms
        for j=i+1:NAtoms
            if distmat[i,j] <= r_cut
                e_ij=dimer_energy(pot,distmat[i,j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end 

    return dimer_energy_vec, energy_tot + lrc(NAtoms,r_cut,pot) * 3/4 * bc.box_length/bc.box_height  
end 
function dimer_energy_config(distmat, NAtoms,potential_variables::ELJPotentialBVariables, pot::AbstractDimerPotentialB)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.

    for i in 1:NAtoms
        for j=i+1:NAtoms
            e_ij=dimer_energy(pot,distmat[i,j],potential_variables.tan_mat[i,j])
            dimer_energy_vec[i] += e_ij
            dimer_energy_vec[j] += e_ij
            energy_tot += e_ij
        end
    end 
    #energy_tot=sum(dimer_energy_vec)
    return dimer_energy_vec, energy_tot
end 
function dimer_energy_config(distmat, NAtoms,potential_variables::ELJPotentialBVariables, r_cut, bc::CubicBC, pot::AbstractDimerPotentialB)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.

    for i in 1:NAtoms
        for j=i+1:NAtoms
            if distmat[i,j] <= r_cut
                e_ij=dimer_energy(pot,distmat[i,j],potential_variables.tan_mat[i,j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end 

    return dimer_energy_vec, energy_tot + lrc(NAtoms,r_cut,pot)   #no 0.5*energy_tot
end 
"""
    dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
    dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,r_cut,pot::AbstractDimerPotential)
    dimer_energy_update!(index,dist2_mat,tanmat,new_dist2_vec,new_tan_vec,en_tot,pot::AbstractDimerPotentialB)
    dimer_energy_update!(index,dist2_mat,tanmat,new_dist2_vec,new_tan_vec,en_tot,r_cut,pot::AbstractDimerPotentialB)

dimer_energy_update is the potential-level-call where for a single mc_state we take the new position `pos`, for atom at `index` , inside the current `config` , where the interatomic distances `dist2_mat` and the new vector based on the new position `new_dist2_vec`; these use the `potential` to calculate a delta_energy and modify the current `en_tot`. These quantities are modified in place and returned 

Final two methods are for use with a dimer potential in a magnetic field, where there is anisotropy in the coefficients.
""" 
function dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
    @views  delta_en = dimer_energy_atom(index,new_dist2_vec,pot) - dimer_energy_atom(index,dist2_mat[index,:], pot)

    return  delta_en + en_tot
end
function dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,r_cut,pot::AbstractDimerPotential)
    @views delta_en = dimer_energy_atom(index,new_dist2_vec,r_cut,pot) - dimer_energy_atom(index,dist2_mat[index,:],r_cut, pot)

    return delta_en + en_tot 
end

function dimer_energy_update!(index,dist2_mat,tanmat,new_dist2_vec,new_tan_vec,en_tot,pot::AbstractDimerPotentialB)

    delta_en = dimer_energy_atom(index,new_dist2_vec,new_tan_vec,pot) - dimer_energy_atom(index,dist2_mat[index,:],tanmat[index,:], pot)

    return  delta_en + en_tot
end
function dimer_energy_update!(index,dist2_mat,tanmat,new_dist2_vec,new_tan_vec,en_tot,r_cut,pot::AbstractDimerPotentialB)

    delta_en = dimer_energy_atom(index,new_dist2_vec,new_tan_vec,r_cut,pot) - dimer_energy_atom(index,dist2_mat[index,:],tanmat[index,:],r_cut, pot)

    return  delta_en + en_tot
end
#----------------------------------------------------------#
#-----------------Specific Dimer Potentials----------------#
#----------------------------------------------------------#

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
    lrc(NAtoms,r_cut,pot::ELJPotentialEven{N})
    lrc(NAtoms,r_cut,pot::ELJPotentialB{N}) where N
The long range correction for the extended Lennard-Jones potential (even). r_cut is the cutoff distance.
lrc is an integral of all interactions outside the cutoff distance, using the uniform density approximation.
Second method applies to the ELJ potential in extreme magnetic fields ELJB
"""
function lrc(NAtoms,r_cut,pot::ELJPotentialEven{N}) where N
    
    r_cut_sqrt=r_cut^0.5
    rc3 = r_cut*r_cut_sqrt
    e_lrc = 0.
    for i = 1:N
        e_lrc += pot.coeff[i] / rc3 / (2i+1)
        rc3 *= r_cut
    end
    e_lrc *= pi*NAtoms^2/4/r_cut_sqrt^3
    return e_lrc
end
function lrc(NAtoms,r_cut,pot::ELJPotentialB{N}) where N
    coeff=[-0.1279111890228638, -1.328138539967966, 12.260941135261255,41.12212408251662]
    r_cut_sqrt=r_cut^0.5
    rc3 = r_cut*r_cut_sqrt
    e_lrc = 0.
    for i = 1:4
        e_lrc += coeff[i] / rc3 / (2i+1)
        rc3 *= r_cut
    end
    e_lrc *= pi*NAtoms^2/4/r_cut_sqrt^3
    return e_lrc
end
#----------------------------------------------------------#

#-----------------Magnetic Dimer Potential-----------------#
# """
# ELJPotentialB{N,T}
#    Extended Lennard-Jones Potential in a magnetic field where there is anisotropy in the coefficient vectors `coeff_a`, `coeff_b`, `coeff_c`
# """
# struct ELJPotentialB{N,T} <: AbstractDimerPotentialB
#     coeff_a::SVector{N,T}
#     coeff_b::SVector{N,T}
#     coeff_c::SVector{N,T}
# end
# function ELJPotentialB{N}(a,b,c) where N
#     @boundscheck length(c) == N || error("number of ELJ coefficients does not match given length")
#     coeff_a = SVector{N}(a)
#     coeff_b = SVector{N}(b)
#     coeff_c = SVector{N}(c)
#     T = eltype(c)
#     return ELJPotentialB{N,T}(coeff_a,coeff_b,coeff_c)
# end

# function ELJPotentialB(a,b,c) 
#     N = length(c)
#     coeff_a = SVector{N}(a)
#     coeff_b = SVector{N}(b)
#     coeff_c = SVector{N}(c)
#     T = eltype(c)
#     return ELJPotentialB{N,T}(coeff_a,coeff_b,coeff_c)
# end

# mutable struct ELJPotentialBVariables{T} <: AbstractPotentialVariables
#     en_atom_vec::Array{T}
#     tan_mat::Matrix{T}
#     new_tan_vec::Vector{T}
# end
"""
    dimer_energy(pot::ELJPotentialB{N}, r2, tan) where N
Dimer energy when the distance square between two atom is r2 and the angle between the line connecting them and z-direction is tan.
When r2 < 5.30, returns 1.
"""
function dimer_energy(pot::ELJPotentialB{N}, r2, tan) where N
   
    if r2>=5.30
        r6inv = 1/(r2*r2*r2)
        t2=2/(tan^2+1)-1     #cos(2*theta)
        t4=2*t2^2-1
        sum1 = pot.coeff_c[1] * r6inv * (1 + pot.coeff_a[1]*t2 + pot.coeff_b[1]*t4)
        r6inv/=r2
        for i = 2:N
            sum1 += pot.coeff_c[i] * r6inv * (1 + pot.coeff_a[i]*t2 + pot.coeff_b[i]*t4)
            r6inv /= r2^0.5 
        end
    else
        sum1=0.1
    end
    return sum1
end 


#----------------------------------------------------------#
#-------------------Embedded Atom Model--------------------#
#----------------------------------------------------------#
"""
    EmbeddedAtomPotential
Struct containing the important quantities for calculating EAM (specifically Sutton-Chen type) potentials.
    Fields:
    `n` the exponent for the two-body repulsive ϕ component
    `m` the exponent for the embedded electron density ρ
    `ean` multiplicative factor ϵa^n /2 for ϕ
    `eCam` multiplicative factor ϵCa^(m/2) for ρ 

"""
struct EmbeddedAtomPotential <: AbstractPotential
    n::Float64
    m::Float64
    ean::Float64
    eCam::Float64
end
"""
    EmbeddedAtomPotential(n,m,ϵ,C,a)
Function to initalise the EAM struct given the actual constants cited in papers. The exponents `n`,`m`, the energy constant `ϵ` the distance constant `a` standard in all EAM models, and a dimensionless parameter `C` scaling ρ with respect to ϕ.
"""
function EmbeddedAtomPotential(n,m,ϵ,C,a)
    epsan = ϵ * a^n / 2
    epsCam = ϵ * C * a^(m/2)
    return EmbeddedAtomPotential(n,m,epsan,epsCam)
end

mutable struct EmbeddedAtomVariables{T} <: AbstractPotentialVariables
    component_vector::Matrix{T}
    new_component_vector::Matrix{T}
end
#-------------------Component Calculation------------------#
"""
    invrexp(r2,n,m)
Function to calculate the `ϕ,ρ` components given a square distance `r2` and the exponents `n,m`
"""
function invrexp(r2,n,m)
    if r2 != 0.
        r_term = 1/sqrt(r2)
        return r_term^n , r_term^m
    else
        return 0. , 0.
    end    
end
"""
    calc_components(eatomvec,distancevec,n,m)
    calc_components(new_component_vec,atomindex,old_r2_vec,new_r2_vec,n,m)

Primary calculation of ϕ,ρ for atom i, given each other atom's distance to i in `distancevec`. `eatomvec` stores the ϕ and ρ components.

Second method also includes an existing `new_component_vec` `atomindex` and old and new interatomic distances from an atom at `atomindex` stored in vectors `new_r2_vec,old_r2_vec`. This calculates the `new_component_vec` based on the updated distances and returns this.
"""
function calc_components(componentvec,distancevec,n,m)
    for dist in distancevec
        componentvec .+= invrexp(dist,n,m)
    end
    return componentvec
end
function calc_components(new_component_vec,atomindex,old_r2_vec,new_r2_vec,n,m)

    for j_index in eachindex(new_r2_vec)

        j_term = invrexp(new_r2_vec[j_index],n,m) .- invrexp(old_r2_vec[j_index],n,m)



@views        new_component_vec[j_index,:] .+= j_term 
@views        new_component_vec[atomindex,:] .+= j_term 

    end

    return new_component_vec
end

function calc_components(component_vec,new_component_vec,atomindex,old_r2_vec,new_r2_vec,n,m)
    for j_index in eachindex(new_r2_vec)

        j_term = invrexp(new_r2_vec[j_index],n,m) .- invrexp(old_r2_vec[j_index],n,m)

        new_component_vec[j_index,1] = component_vec[j_index,1] + j_term[1]
        new_component_vec[atomindex,1]=component_vec[atomindex,1] + j_term[1] 
        new_component_vec[j_index,2] = component_vec[j_index,2] + j_term[2]
        new_component_vec[atomindex,2]=component_vec[atomindex,2] + j_term[2] 
    end

    return new_component_vec
end
"""
    calc_energies_from_components(component_vector,ean,ecam)
Takes a `component_vector` containing ϕ,ρ for each atom. Using the multiplicative factors `ean,ecam` we sum the atomic contributions and return the energy. Commented version used more allocations due to broadcasting defaulting to copying arrays. New version uses minimal allocations. 
"""
# function calc_energies_from_components(component_vector,ean,ecam)
# @views    return sum(ean.*component_vector[:,1] - ecam*sqrt.(component_vector[:,2]))
# end
function calc_energies_from_components(component_vector,ean,ecam)
    en_val = 0.
    for componentrow in eachrow(component_vector)
        en_val += ean*componentrow[1] - ecam*sqrt(componentrow[2])
    end
return en_val
end

#-----------------------------------------------------------#
#----------------Machine Learning Potentials----------------#
#-----------------------------------------------------------#

abstract type AbstractMachineLearningPotential <: AbstractPotential end

"""
    RuNNerPotential <: AbstractMachineLearningPotential
Contains the important structs required for a neural network potential defined in the MachineLearningPotential package:
    Fields are:
    nnp -- a struct containing the weights, biases and neural network parameters.
    symmetryfunctions -- a vector containing the hyperparameters used to calculate symmetry function values
    r_cut -- every symmetry function has an r_cut, but saving it here saves annoying memory unpacking 
"""
struct  RuNNerPotential{Nrad,Nang} <: AbstractMachineLearningPotential
    nnp:: NeuralNetworkPotential
    radsymfunctions::StructVector{RadialType2{Float64}} #SVector{Nrad,RadialType2}
    angsymfunctions::StructVector{AngularType3{Float64}}
    r_cut::Float64
end
function RuNNerPotential(nnp,radsymvec,angsymvec)
    r_cut = radsymvec[1].r_cut
    nrad = length(radsymvec)
    nang = length(angsymvec)
    radvec=StructVector([rsymm for rsymm in radsymvec])
    angvec = StructVector([asymm for asymm in angsymvec])
    return RuNNerPotential{nrad,nang}(nnp,radvec,angvec,r_cut)
end
mutable struct NNPVariables{T} <: AbstractPotentialVariables

    en_atom_vec::Vector{T}

    new_en_atom::Vector{T}
    g_matrix::Matrix{T}
    f_matrix::Matrix{T}
    new_g_matrix::Matrix{T}
    new_f_vec::Vector{T}
end
"""
    get_new_state_vars!(trial_pos,atomindex,config::Config,potential_variables::NNPVariables,dist2_mat,new_dist2_vec,pot)
Function for finding the new state variables for calculating an NNP. Redefines new_f and new_g matrices based on the `trial_pos` of atom at `atomindex` and adjusts the parameters in the `potential_variables` according to the variables in `pot`.
"""
function get_new_state_vars!(trial_pos,atomindex,config::Config,potential_variables::NNPVariables,dist2_mat,new_dist2_vec,pot::RuNNerPotential{Nrad,Nang}) where {Nrad,Nang}
    # new_dist2_vec = [ distance2(trial_pos,b,config.bc) for b in config.pos]
    # new_dist2_vec[atomindex] = 0.
    potential_variables.new_f_vec = cutoff_function.(sqrt.(new_dist2_vec),Ref(pot.r_cut))
    potential_variables.new_g_matrix = copy(potential_variables.g_matrix)
    potential_variables.new_g_matrix = total_thr_symm!(potential_variables.new_g_matrix,config.pos,trial_pos,dist2_mat,new_dist2_vec,potential_variables.f_matrix,potential_variables.new_f_vec,atomindex,pot.radsymfunctions,pot.angsymfunctions,Nrad,Nang)
    return potential_variables
end
"""
    calc_new_runner_energy!(potential_variables::NNPVariables,new_en,pot)
function designed to calculate the new per-atom energy according to the RuNNer forward pass with parameters defined in `pot`. utilises the `new_g_matrix` to redefine the `new_en` and `new_en_atom` variables within the `potential_variables` struct.
"""
function calc_new_runner_energy!(potential_variables::NNPVariables,pot::RuNNerPotential)
    potential_variables.new_en_atom = forward_pass(potential_variables.new_g_matrix,length(potential_variables.en_atom_vec),pot.nnp)
    new_en = sum(potential_variables.new_en_atom)
    return potential_variables,new_en
end


#----------------------------------------------------------#
#----------------------Top Level Call----------------------#
#----------------------------------------------------------#
"""
    energy_update!(trial_pos,index,config::Config,potential_variables,dist2_mat,en_tot,pot::AbstractDimerPotential)
    energy_update!(trial_pos,index,config::Config,potential_variables,dist2_mat,en_tot,r_cut,pot::AbstractDimerPotential)
    energy_update!(trial_pos,index,config::Config,potential_variables::ELJPotentialBVariables,dist2_mat,en_tot,pot::AbstractDimerPotentialB)
    energy_update!(trial_pos,index,config::Config,potential_variables::ELJPotentialBVariables,dist2_mat,en_tot,r_cut,pot::AbstractDimerPotentialB)
    energy_update!(trial_pos,index,config::Config,potential_variables::EmbeddedAtomVariables,dist2_mat,en_tot,pot::EmbeddedAtomPotential)
    energy_update!(trial_pos,index,config::Config,potential_variables::NNPVariables,dist2_mat,en_tot,pot::RuNNerPotential)

Energy update function for use within a cycle. at the top level this is called with the new position `trial_pos` which is the `index`-th atom in the `config` it operates on the `potential_variables` along with the `dist2_mat`. Using `pot` the potential to find the `new_en`. 

    Has additional methods including `r_cut` where appropriate for use with periodic boundary conditions
    
    This function is designed as a curry function. The generic get_energy function operates on a __vector__ of states, this function takes each state and the set potential and calls the potential specific energy_update function.

        Methods defined for :
            - AbstractDimerPotential
            - AbstractDimerPotentialB
            - EmbeddedAtomPotential
            - RuNNerPotential
"""
function energy_update!(trial_pos,index,config::Config,potential_variables,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)

    # new_dist2_vec = [distance2(trial_pos,b,config.bc) for b in config.pos]
    # new_dist2_vec[index] = 0.

    new_en = dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,pot)

    return potential_variables,new_en
end
function energy_update!(trial_pos,index,config::Config,potential_variables,dist2_mat,new_dist2_vec,en_tot,r_cut,pot::AbstractDimerPotential)

    # new_dist2_vec = [distance2(trial_pos,b,config.bc) for b in config.pos]
    # new_dist2_vec[index] = 0.

    new_en = dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,r_cut,pot)

    return potential_variables,new_en
end

function energy_update!(trial_pos,index,config::Config,potential_variables::ELJPotentialBVariables,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotentialB)

    # new_dist2_vec = [distance2(trial_pos,b,config.bc) for b in config.pos]
    # new_dist2_vec[index] = 0.

    potential_variables.new_tan_vec = [get_tan(trial_pos,b,config.bc) for b in config.pos]
    potential_variables.new_tan_vec[index] = 0

    new_en = dimer_energy_update!(index,dist2_mat,potential_variables.tan_mat,new_dist2_vec,potential_variables.new_tan_vec,en_tot,pot)

    return potential_variables,new_en
end
function energy_update!(trial_pos,index,config::Config,potential_variables::ELJPotentialBVariables,dist2_mat,new_dist2_vec,en_tot,r_cut,pot::AbstractDimerPotentialB)

    # new_dist2_vec = [distance2(trial_pos,b,config.bc) for b in config.pos]
    # new_dist2_vec[index] = 0.

    potential_variables.new_tan_vec = [get_tan(trial_pos,b,config.bc) for b in config.pos]
    potential_variables.new_tan_vec[index] = 0

    new_en = dimer_energy_update!(index,dist2_mat,potential_variables.tan_mat,new_dist2_vec,potential_variables.new_tan_vec,en_tot,r_cut,pot)

    return potential_variables,new_en
end
function energy_update!(trial_pos,atomindex,config::Config,potential_variables::EmbeddedAtomVariables,dist2_mat,new_dist2_vec,en_tot,pot::EmbeddedAtomPotential)
    # new_dist2_vec = [distance2(trial_pos,b) for b in config.pos]

    # new_dist2_vec[atomindex] = 0.

    potential_variables.new_component_vector = copy(potential_variables.component_vector)
    
    potential_variables.new_component_vector = calc_components(potential_variables.new_component_vector,atomindex,dist2_mat[atomindex,:],new_dist2_vec,pot.n,pot.m)

    new_en = calc_energies_from_components(potential_variables.new_component_vector,pot.ean,pot.eCam)

    return potential_variables,new_en
end
function energy_update!(trial_pos,index,config::Config,potential_variables::NNPVariables,dist2_mat,new_dist2_vec,en_tot,pot::RuNNerPotential)

    potential_variables = get_new_state_vars!(trial_pos,index,config,potential_variables,dist2_mat,new_dist2_vec,pot)

    potential_variables,new_en = calc_new_runner_energy!(potential_variables,pot)

    return potential_variables,new_en
end
#------------------------------------------------------------#
#----------------Initialising State Functions----------------#
#------------------------------------------------------------#
"""
    initialise_energy(config,dist2_mat,potential_variables,pot::AbstractDimerPotential)
    

Initialise energy is used during the MCState call to set the starting energy of a `config` according to the potential as `pot` and the configurational variables `potential_variables`. Written with general input means the top-level is type-invariant. 
Methods included for:
    - Dimer Potential with and without magnetic field and with and without PBC 
    - EmbeddedAtomModel 
    - Machine Learning Potentials 
"""
function initialise_energy(config,dist2_mat,potential_variables,ensemble_variables::NVTVariables,pot::AbstractDimerPotential)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),potential_variables,pot)

    return en_tot,potential_variables
end
function initialise_energy(config,dist2_mat,potential_variables,ensemble_variables::NPTVariables,pot::AbstractDimerPotential)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),potential_variables,ensemble_variables.r_cut, config.bc, pot)

    return en_tot,potential_variables
end
function initialise_energy(config,dist2_mat,potential_variables,ensemble_variables::NVTVariables,pot::AbstractDimerPotentialB)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),potential_variables,pot)
    return en_tot,potential_variables 
end
function initialise_energy(config,dist2_mat,potential_variables,ensemble_variables::NPTVariables,pot::AbstractDimerPotentialB)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),potential_variables,ensemble_variables.r_cut, config.bc, pot)
    return en_tot,potential_variables 
end
function initialise_energy(config,dist2_mat,potential_variables,ensemble_variables,pot::EmbeddedAtomPotential)
    en_tot = calc_energies_from_components(potential_variables.component_vector,pot.ean,pot.eCam)

    return en_tot,potential_variables
end
function initialise_energy(config,dist2_mat,potential_variables,ensemble_variables,pot::RuNNerPotential)
    potential_variables.en_atom_vec = forward_pass(potential_variables.g_matrix,length(config),pot.nnp)
    en_tot = sum(potential_variables.en_atom_vec)
    return en_tot,potential_variables
end

"""
    set_variables(config,dist_2_mat,pot::AbstractDimerPotential)
    set_variables(config::Config,dist2_matrix::Matrix,pot::AbstractDimerPotentialB)
    set_variables(config,dist2_matrix,pot::EmbeddedAtomPotential)
    set_variables(config,dist_2_mat,pot::AbstractDimerPotential)

initialises the PotentialVariable struct for the various potentials. Defined in this way to generalise the MCState function as this must be type-invariant with respect to the potential. 
    
"""
function set_variables(config::Config{N,BC,T},dist_2_mat,pot::AbstractDimerPotential) where {N,BC,T}
    return DimerPotentialVariables{T}(zeros(N))
end
function set_variables(config::Config{N,BC,T},dist2_matrix::Matrix,pot::AbstractDimerPotentialB) where {N,BC,T}
    tan_matrix = get_tantheta_mat(config,config.bc)

    return ELJPotentialBVariables{T}(zeros(N),tan_matrix,zeros(N))
end
function set_variables(config::Config{N,BC,T},dist2_matrix,pot::EmbeddedAtomPotential) where {N,BC,T}
    
    componentvec = zeros(N,2)
    for row_index in 1:N
        componentvec[row_index,:] = calc_components(componentvec[row_index,:],dist2_matrix[row_index,:],pot.n,pot.m)
    end
    return EmbeddedAtomVariables{T}(componentvec,zeros(N,2))
end
function set_variables(config::Config{N,BC,T},dist2_mat,pot::RuNNerPotential{nrad,nang}) where {N,BC,T} where {nrad,nang}
    
    f_matrix = cutoff_function.(sqrt.(dist2_mat),Ref(pot.r_cut))
    g_matrix = total_symm_calc(config.pos,dist2_mat,f_matrix,pot.radsymfunctions,pot.angsymfunctions,nrad,nang)
    
    return NNPVariables{T}(zeros(N) ,zeros(N),g_matrix,f_matrix,copy(g_matrix), zeros(N))
end

end