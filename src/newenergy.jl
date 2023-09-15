"""
    module EnergyEvaluation
Rewrite of the original energyevaluation module to standardise and clean the functions up. 
"""

module EnergyEvaluation 
using StaticArrays,LinearAlgebra,DFTK
using ..MachineLearningPotential
using ..Configurations


#-------------------------------------------------------------#
#----------------------Universal Structs----------------------#
#-------------------------------------------------------------#

#--------------------------Ensembles--------------------------#
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

#-------------------------------------------------------------#
"""   
    AbstractPotential
Abstract type for possible potentials
implemented subtype: 
- AbstractDimerPotential

Needs method for dimer_energy [`dimer_energy`](@ref)
"""
abstract type AbstractPotential end

"""
    PotentialVariables
An abstract type defining a class of mutable struct containing all the relevant vectors and arrays each potential will need throughout the course of a simulation to prevent over-definitions inside the MCState struct.
"""
abstract type PotentialVariables end
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
"""
    DimerPotenitalVariables
The struct contains only the new_dist2_vec as this doesn't explicitly require any particular special features.
"""
mutable struct DimerPotentialVariables <: PotentialVariables
    new_dist2_vec::Vector
    en_atom_vec::Vector
    new_en::Float64
end

"""
    dimer_energy_atom(i, pos, d2vec, pot<:AbstractPotential)
    dimer_energy_atom(i, d2vec, r_cut, pot::AbstractDimerPotential)

Sums the dimer energies for atom `i` with all other atomsdimer_energy_update(pos,index,config,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
Needs vector of squared distances `d2vec` between atom `i` and all other atoms in configuration
see  `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 

second method includes r_cut for use with the NPT ensemble
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
"""
    dimer_energy_config(distmat, NAtoms, pot::AbstractPotential)
    dimer_energy_config(distmat, NAtoms, r_cut, pot::AbstractPotential)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, NAtoms, pot::AbstractDimerPotential)
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
function dimer_energy_config(distmat, NAtoms, r_cut, pot::AbstractDimerPotential)
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
    return dimer_energy_vec, energy_tot + lrc(NAtoms,r_cut,pot)
end
"""
    dimer_energy_update(pos,index,config,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
dimer_energy_update is the potential-level-call where for a single mc_state we take the new position `pos`, for atom at `index` , inside the current `config` , where the interatomic distances `dist2_mat` and the new vector based on the new position `new_dist2_vec`; these use the `potential` to calculate a delta_energy and modify the current `en_tot`. These quantities are modified in place and returned 
""" 
function dimer_energy_update!(pos,index,config,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
    new_dist2_vec = [distance2(pos,b,config.bc) for b in config.pos]
    new_dist2_vec[index] = 0.
    delta_en = dimer_energy_atom(index,new_dist2_vec,pot) - dimer_energy_atom(index,dist2_mat, pot)

    return new_dist2_vec, delta_en + en_tot
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
#----------------------------------------------------------#

#----------------------------------------------------------#
#----------------------Top Level Call----------------------#
#----------------------------------------------------------#
"""
    energy_update(trial_pos,index,mc_state,pot::AbstractDimerPotential)

Energy update function for use within a cycle. at the top level this is called with the new position `trial_pos` which is the `index`-th atom in the `config` contained in `mc_state`. Using `pot` the potential. 
    
    This function is designed as a curry function. The generic get_energy function operates on a __vector__ of states, this function takes each state and the set potential and calls the potential specific energy_update function.
"""
function energy_update!(trial_pos,index,mc_state,pot::AbstractDimerPotential)

    mc_state.potential_variables.new_dist2_vec,mc_state.potential_variables.new_en = dimer_energy_update!(trial_pos,index,mc_state.config,mc_state.dist2_mat,mc_state.potential_variables.new_dist2_vec,mc_state.en_tot,pot)

    return mc_state
end


"""
    get_energy(trial_positions,indices,mc_states,pot)
curry function used as the top call within each mc_step. Passes a vector of `mc_states` with updated `trial_positions` for atom at `index` where we use `pot` to calculate the new energy in the non-vectorised energy_update function. 
"""

function get_energy(trial_positions,indices,mc_states,pot)
    return energy_update!.(trial_positions,indices,mc_states,Ref(pot))
end

#------------------------------------------------------------#
#----------------Initialising State Functions----------------#
#------------------------------------------------------------#
"""
    initialise_energy(config,dist2_mat,potential_variables,pot::AbstractDimerPotential)
    initialise_energy(config,dist2_mat,potential_variables,r_cut,pot::AbstractDimerPotential)

Initialise energy is a function to set the starting parameters of the 
"""
function initialise_energy(config,dist2_mat,potential_variables,pot::AbstractDimerPotential)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),pot)

    return en_tot,potential_variables
end
function initialise_energy(config,dist2_mat,potential_variables,r_cut,pot::AbstractDimerPotential)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),r_cut,pot)

    return en_tot,potential_variables
end
"""
    set_variables(config,dist_2_mat,pot::AbstractDimerPotential)
initialises the PotentialVariable struct for the various potentials. 
    -   Method one functions for abstract dimer potentials such as the ELJ
    -   
    
"""
function set_variables(config,dist_2_mat,pot::AbstractDimerPotential)
    return DimerPotentialVariable(zeros(length(config)),0. )
end


end