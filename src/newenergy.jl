"""
    module EnergyEvaluation
Rewrite of the original energyevaluation module to standardise and clean the functions up. 
"""

module EnergyEvaluation 
using StaticArrays,LinearAlgebra,DFTK
using ..MachineLearningPotential
using ..Configurations

export AbstractEnsemble,NVT,NPT
export AbstractPotential,AbstractDimerPotential,ELJPotential,ELJPotentialEven
export AbstractDimerPotentialB,ELJPotentialB,EmbeddedAtomPotential,RuNNerPotential
export PotentialVariables,DimerPotentialVariables,ELJPotentialBVariables
export EmbeddedAtomVariables,NNPVariables

export energy_update!,set_variables,initialise_energy
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
    #new_dist2_vec::Vector
    en_atom_vec::Vector
    #new_en::Float64
end

"""
    dimer_energy_atom(i, pos, d2vec, pot<:AbstractPotential)
    dimer_energy_atom(i, d2vec, r_cut, pot:abstract type AbstractDimerPotential <: AbstractPotential end:AbstractDimerPotential)

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
"""
    dimer_energy_update(pos,index,config,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
dimer_energy_update is the potential-level-call where for a single mc_state we take the new position `pos`, for atom at `index` , inside the current `config` , where the interatomic distances `dist2_mat` and the new vector based on the new position `new_dist2_vec`; these use the `potential` to calculate a delta_energy and modify the current `en_tot`. These quantities are modified in place and returned 
""" 
function dimer_energy_update!(index,dist2_mat,new_dist2_vec,en_tot,pot::AbstractDimerPotential)
    delta_en = dimer_energy_atom(index,new_dist2_vec,pot) - dimer_energy_atom(index,dist2_mat[index,:], pot)

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
#----------------------------------------------------------#

#-----------------Magnetic Dimer Potential-----------------#
abstract type AbstractDimerPotentialB <: AbstractPotential end
"""
   potential in B
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

mutable struct ELJPotentialBVariables <: PotentialVariables
    en_atom_vec::Array
    tan_mat::Array
    new_tan_vec::Vector
end
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

"""
    dimer_energy_atom(i, pos, d2vec, pot<:AbstractDimerPotentialB)
Sums the dimer energies for atom `i` with all other atoms
Needs vector of squared distances `d2vec` between atom `i` and all other atoms in configuration
see  `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
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

"""
    dimer_energy_config(distmat, NAtoms, pot::AbstractDimerPotentialB)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, tanmat, NAtoms, pot::AbstractDimerPotentialB)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.

    for i in 1:NAtoms
        for j=i+1:NAtoms
            e_ij=dimer_energy(pot,distmat[i,j],tanmat[i,j])
            dimer_energy_vec[i] += e_ij
            dimer_energy_vec[j] += e_ij
            energy_tot += e_ij
        end
    end 
    #energy_tot=sum(dimer_energy_vec)
    return dimer_energy_vec, energy_tot
end 
function dimer_energy_update!(index,dist2_mat,tanmat,new_dist2_vec,new_tan_vec,en_tot,pot::AbstractDimerPotentialB)
    # new_dist2_vec = [distance2(pos,b,config.bc) for b in config.pos]
    # new_dist2_vec[index] = 0.
    delta_en = dimer_energy_atom(index,new_dist2_vec,new_tan_vec,pot) - dimer_energy_atom(index,dist2_mat[index,:],tanmat[index,:], pot)

    return  delta_en + en_tot
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
struct EmbeddedAtomPotential <: AbstractDimerPotential
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

mutable struct EmbeddedAtomVariables <: PotentialVariables
    component_vector::Matrix
    new_component_vector::Matrix
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

        new_component_vec[j_index,:] .+= j_term 
        new_component_vec[atomindex,:] .+= j_term 
    end

    return new_component_vec
end
"""
    calc_energies_from_components(component_vector,ean,ecam)
Takes a `component_vector` containing ϕ,ρ for each atom. Using the multiplicative factors `ean,ecam` we sum the atomic contributions and return the energy.
"""
function calc_energies_from_components(component_vector,ean,ecam)
    return sum(ean.*component_vector[:,1] - ecam*sqrt.(component_vector[:,2]))
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
struct  RuNNerPotential <: AbstractMachineLearningPotential
    nnp::NeuralNetworkPotential
    symmetryfunctions::Vector{AbstractSymmFunction}
    r_cut::Float64
end
function RuNNerPotential(nnp,symmetryfunction)
    r_cut = symmetryfunction[1].r_cut
    return RuNNerPotential(nnp,symmetryfunction,r_cut)
end


mutable struct NNPVariables <: PotentialVariables
    en_atom_vec::Vector
    new_en::Float64
    new_en_atom::Vector
    g_matrix::Array
    f_matrix::Array
    new_g_matrix::Array
    new_f_vec::Vector
end
"""
    get_new_state_vars!(trial_pos,atomindex,nnp_state,pot)
Function for finding the new state variables for calculating an NNP. Redefines new_f and new_g matrices based on the `trial_pos` of atom at `atomindex` and adjusts the parameters in the `nnp_state` according to the variables in `pot`.
"""
function get_new_state_vars!(trial_pos,atomindex,nnp_state,pot)

    nnp_state.potential_variables.new_dist2_vec = [ distance2(trial_pos,b,nnp_state.config.bc) for b in nnp_state.config.pos]
    nnp_state.potential_variables.new_dist2_vec[atomindex] = 0.
    
    nnp_state.potential_variables.new_f_vec = cutoff_function.(sqrt.(nnp_state.potential_variables.new_dist2_vec),Ref(pot.r_cut))


    nnp_state.potential_variables.new_g_matrix = copy(nnp_state.potential_variables.g_matrix)

    nnp_state.potential_variables.new_g_matrix = total_thr_symm!(nnp_state.potential_variables.new_g_matrix,nnp_state.config.pos,trial_pos,nnp_state.dist2_mat,nnp_state.potential_variables.new_dist2_vec,nnp_state.potential_variables.f_matrix,nnp_state.potential_variables.new_f_vec,atomindex,pot.symmetryfunctions)

    return nnp_state
end
"""
    calc_new_runner_energy!(potential_variables,pot)
function designed to calculate the new per-atom energy according to the RuNNer forward pass with parameters defined in `pot`. utilises the `new_g_matrix` to redefine the `new_en` and `new_en_atom` variables within the `potential_variables` struct.
"""
function calc_new_runner_energy!(mc_state,pot)
    mc_state.new_en_atom = forward_pass(mc_state.potential_variables.new_g_matrix,length(mc_state.en_atom_vec),pot.nnp)
    mc_state.new_en = sum(mc_state.potential_variables.new_en_atom)
    return mc_state
end


#----------------------------------------------------------#
#----------------------Top Level Call----------------------#
#----------------------------------------------------------#
"""
    energy_update(trial_pos,index,mc_state,pot::AbstractDimerPotential)
    energy_update!(trial_pos,index,mc_state,pot::AbstractDimerPotentialB)
    energy_update!(trial_pos,atomindex,mc_state,pot::EmbeddedAtomPotential)
    energy_update!(trial_pos,index,state,pot::RuNNerPotential)
Energy update function for use within a cycle. at the top level this is called with the new position `trial_pos` which is the `index`-th atom in the `config` contained in `mc_state`. Using `pot` the potential. 
    
    This function is designed as a curry function. The generic get_energy function operates on a __vector__ of states, this function takes each state and the set potential and calls the potential specific energy_update function.

        Methods defined for :
            - AbstractDimerPotential
            - AbstractDimerPotentialB
            - EmbeddedAtomPotential
            - RuNNerPotential
"""
function energy_update!(trial_pos,index,mc_state,pot::AbstractDimerPotential)

    mc_state.new_dist2_vec = [distance2(trial_pos,b,mc_state.config.bc) for b in mc_state.config.pos]
    mc_state.new_dist2_vec[index] = 0.

    mc_state.new_en = dimer_energy_update!(index,mc_state.dist2_mat,mc_state.new_dist2_vec,mc_state.en_tot,pot)

    return mc_state
end
function energy_update!(trial_pos,index,mc_state,pot::AbstractDimerPotentialB)

    mc_state.new_dist2_vec = [distance2(trial_pos,b,mc_state.config.bc) for b in mc_state.config.pos]
    mc_state.new_dist2_vec[index] = 0.

    mc_state.potential_variables.new_tan_vec = [get_tan(trial_pos,b,mc_state.config.bc) for b in mc_state.config.pos]
    mc_state.potential_variables.new_tan_vec[index] = 0

    mc_state.new_en = dimer_energy_update!(index,mc_state.dist2_mat,mc_state.potential_variables.tan_mat,mc_state.new_dist2_vec,mc_state.potential_variables.new_tan_vec,mc_state.en_tot,pot)

    return mc_state
end
function energy_update!(trial_pos,atomindex,mc_state,pot::EmbeddedAtomPotential)
    mc_state.new_dist2_vec = [distance2(trial_pos,b) for b in mc_state.config.pos]

    mc_state.new_dist2_vec[atomindex] = 0.

    mc_state.potential_variables.new_component_vector = deepcopy(mc_state.potential_variables.component_vector)
    
    mc_state.potential_variables.new_component_vector = calc_components(mc_state.potential_variables.new_component_vector,atomindex,mc_state.dist2_mat[atomindex,:],mc_state.new_dist2_vec,pot.n,pot.m)

    mc_state.new_en = calc_energies_from_components(mc_state.potential_variables.new_component_vector,pot.ean,pot.eCam)

    return  mc_state
end
function energy_update!(trial_pos,index,mc_state,pot::RuNNerPotential)

    mc_state = get_new_state_vars(trial_pos,index,mc_state,pot)

    mc_state = calc_new_runner_energy!(mc_state,pot)
    return mc_state
end

"""
    get_energy(trial_positions,indices,mc_states,pot)
    get_energy(trial_positions,indices,mc_states,pot::RuNNerPotential)
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
    initialise_energy(config,dist2_mat,potential_variables,r_cut,pot::RuNNerPotential)

Initialise energy is used during the MCState call to set the starting energy of a `config` according to the potential as `pot` and the configurational variables `potential_variables`. Written with general input means the top-level is type-invariant. 
Methods included for:
    - Dimer Potential 
    - Machine Learning Potentials 
"""
function initialise_energy(config,dist2_mat,potential_variables,pot::AbstractDimerPotential)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,length(config),pot)

    return en_tot,potential_variables
end
function initialise_energy(config,dist2_mat,potential_variables,pot::AbstractDimerPotentialB)
    potential_variables.en_atom_vec,en_tot = dimer_energy_config(dist2_mat,potential_variables.tan_mat,length(config),pot)
    return en_tot,potential_variables 
end
function initialise_energy(config,dist2_mat,potential_variables,pot::EmbeddedAtomPotential)
    en_tot = calc_energies_from_components(potential_variables.component_vector,pot.ean,pot.eCam)

    return en_tot,potential_variables
end
function initialise_energy(config,dist2_mat,potential_variables,pot::RuNNerPotential)
    potential_variables.en_atom_vec = forward_pass(potential_variables.g_matrix,length(config),pot.nnp)
    en_tot = sum(potential_variables.en_atom_vec)
    return en_tot,potential_variables
end

"""
    set_variables(config,dist_2_mat,pot::AbstractDimerPotential)
    set_variables(config,dist2_matrix,pot::EmbeddedAtomPotential)
    set_variables(config,dist_2_mat,pot::AbstractDimerPotential)

initialises the PotentialVariable struct for the various potentials. 
    -   Method one functions for abstract dimer potentials such as the ELJ
    -   Embeded Atom Potential involving the dimer and density components
    -   The RuNNer Potential involving the cutoff and symmetry matrix.
    
"""
function set_variables(config,dist_2_mat,pot::AbstractDimerPotential)
    return DimerPotentialVariables()
end
function set_variables(config::Config,dist2_matrix::Matrix,pot::AbstractDimerPotentialB)
    n_atoms = length(config)
    tan_matrix = get_tantheta_mat(config,config.pos)

    return ELJPotentialBVariables(tan_matrix,zeros(n_atoms))
end
function set_variables(config,dist2_matrix,pot::EmbeddedAtomPotential)
    n_atoms = length(config)
    componentvec = zeros(n_atoms,2)
    for row_index in 1:n_atoms
        componentvec[row_index,:] = calc_components(componentvec[row_index,:],dist2_matrix[row_index,:],pot.n,pot.m)
    end
    return EmbeddedAtomVariables(componentvec,zeros(n_atoms,2))
end
function set_variables(config,dist2_mat,pot::RuNNerPotential)
    
    n_atoms = length(config)
    f_matrix = cutoff_function.(sqrt.(dist2_mat),Ref(pot.r_cut))
    g_matrix = total_symm_calc(config.pos,dist2_mat,f_matrix,pot.symmetryfunctions)
    
    return NNPVariables(0. ,zeros(n_atoms),g_matrix,f_matrix,zeros(length(pot.symmetryfunctions)), zeros(n_atoms))
end

end