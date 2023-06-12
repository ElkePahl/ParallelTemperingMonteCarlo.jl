"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    
module EnergyEvaluation

using StaticArrays 
using DFTK 
using LinearAlgebra
using SplitApplyCombine
using ..MachineLearningPotential
using ..Configurations

#using ..RuNNer


export AbstractPotential, AbstractDimerPotential, AbstractDimerPotentialB, AbstractMachineLearningPotential,EmbeddedAtomPotential


export RuNNerPotential

export DFTPotential

export ELJPotential, ELJPotentialEven, ELJPotentialB
export dimer_energy, dimer_energy_atom, dimer_energy_config 
export getenergy_DFT, get_energy_dimer#,get_energy_
export energy_update
export get_energy
export AbstractEnsemble, NVT, NPT
export EnHist


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

#The potential struct for magnetic field (tan(theta) as variable)
abstract type AbstractDimerPotentialB <: AbstractPotential end


abstract type AbstractMachineLearningPotential <: AbstractPotential end




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
    dimer_energy_atom(i, d2vec, r_cut, pot<:AbstractPotential)
    dimer energy of an atom with other atoms, with a cutoff distance r_cut.
"""
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
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, NAtoms, pot::AbstractDimerPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    #This uses dimer_energy_atom for all atoms, which means one dimer energy will be calculated twice.
    #for i in 1:NAtoms #eachindex(),enumerate()..?
        #dimer_energy_vec[i] = dimer_energy_atom(i, distmat[:, i], pot) #@view distmat[i, :]
        #energy_tot += dimer_energy_vec[i]
    #end 
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
    dimer_energy_config(distmat, NAtoms, r_cut, pot::AbstractPotential)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration, with a cutoff distance r_cut
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, NAtoms, r_cut, pot::AbstractDimerPotential)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    #for i in 1:NAtoms #eachindex(),enumerate()..?
        #dimer_energy_vec[i] = dimer_energy_atom(i, distmat[:, i], r_cut, pot) #@view distmat[i, :]
        #energy_tot += dimer_energy_vec[i]
    #end 

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

"""
    energy_update(i_atom, dist2_new, pot::AbstractDimerPotential)
Gives new energy when an atom is moved. Without cutoff ditance.
"""
function energy_update(i_atom, dist2_new, pot::AbstractDimerPotential)
    return dimer_energy_atom(i_atom, dist2_new, pot)
end

"""
    energy_update(i_atom, dist2_new, r_cut, pot::AbstractDimerPotential)
Gives new energy when an atom is moved. With cutoff distance r_cut
"""
function energy_update(i_atom, dist2_new, r_cut, pot::AbstractDimerPotential)
    return dimer_energy_atom(i_atom, dist2_new, r_cut, pot)
end

"""
    energy_update(pos, i_atom, config, dist2_mat, pot::AbstractDimerPotential)
Gives energy difference and distances when an atom is moved. Without cutoff distance.
"""
function energy_update(pos, i_atom, config, dist2_mat, pot::AbstractDimerPotential)
    dist2_new = [distance2(pos,b,config.bc) for b in config.pos]
    dist2_new[i_atom] = 0.
    d_en = dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, dist2_mat[:,i_atom], pot)
    return d_en, dist2_new
end

"""
    energy_update(pos, i_atom, config, dist2_mat, r_cut, pot::AbstractDimerPotential)
Gives energy difference and distances when an atom is moved. With cutoff distance r_cut.
"""
function energy_update(pos, i_atom, config, dist2_mat, r_cut, pot::AbstractDimerPotential)
    dist2_new = [distance2(pos,b,config.bc) for b in config.pos]
    dist2_new[i_atom] = 0.
    d_en = dimer_energy_atom(i_atom, dist2_new, r_cut, pot) - dimer_energy_atom(i_atom, dist2_mat[:,i_atom], r_cut, pot)
    return d_en, dist2_new
end
"""
    get_energy_dimer(pos,i_atom,mc_state,pot)
A get_energy function similar to the energy_update function. This simply returns the current energy rather than delta_en
"""
function get_energy_dimer(pos,i_atom,mc_state,pot::AbstractDimerPotential)
    # dist2_new = [distance2(pos,b) for b in mc_state.config.pos]
    # dist2_new[i_atom] = 0.
    # delta_energy= dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, mc_state.dist2_mat[:,i_atom], pot)

    delta_energy,dist2_new = energy_update(pos,i_atom,mc_state.config,mc_state.dist2_mat,pot)
    energy = mc_state.en_tot + delta_energy
    return energy,dist2_new
end
"""
    get_energy_dimer(pos,i_atom,r_cut,mc_state,pot)
A get_energy function similar to the energy_update function. This simply returns the current energy rather than delta_en
with a cutoff distance r_cut
"""
function get_energy_dimer(pos,i_atom,r_cut,mc_state,pot::AbstractDimerPotential)
    # dist2_new = [distance2(pos,b) for b in mc_state.config.pos]
    # dist2_new[i_atom] = 0.
    # delta_energy= dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, mc_state.dist2_mat[:,i_atom], pot)

    delta_energy,dist2_new = energy_update(pos,i_atom,mc_state.config,mc_state.dist2_mat,r_cut,pot)
    energy = mc_state.en_tot + delta_energy
    return energy,dist2_new
end

"""
    get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotential)
Top scope get energy function for dimer potentials returning the energy vector and new distance squared vector as this must be calculated in order to calculate the potential.

"""
function get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotential,ensemble::NVT)
    energyvector, dist2_new = invert(get_energy_dimer.(trial_positions,indices,mc_states,Ref(pot)))

    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector,dist2_new
end
#this will be the format for this part of the get_energy function.

"""
    get_energy(trial_positions,indices,r_cut,mc_states,pot::AbstractDimerPotential,ensemble::NPT)
Top scope get energy function for dimer potentials returning the energy vector and new distance squared vector as this must be calculated in order to calculate the potential.
with the cutoff distance r_cut
"""
function get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotential,ensemble::NPT)
    
    n=length(mc_states)
    r_cut_all=Array{Float64}(undef,n)
    for i=1:n
        r_cut_all[i]=mc_states[i].config.bc.box_length^2/4
    end
    energyvector, dist2_new = invert(get_energy_dimer.(trial_positions,indices,r_cut_all,mc_states,Ref(pot)))
    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector,dist2_new
end

"""
    get_energy function when the whole configuration is scaled
    find the new distance matrix first, and use dimer_energy_config to find the new total energy and energy matrix
"""
function get_energy(trial_configs_all,mc_states,pot::AbstractDimerPotential)
    dist2_mat_new = get_distance2_mat.(trial_configs_all)
    #en_atom_vec, en_tot_new = invert(dimer_energy_config.(dist2_mat_new, length(trial_configs_all[1].pos), Ref(pot)))
    n=length(trial_configs_all)
    r_cut_all=Array{Float64}(undef,n)
    for i=1:n
        r_cut_all[i]=trial_configs_all[i].bc.box_length^2/4
    end
    en_atom_vec, en_tot_new = invert(dimer_energy_config.(dist2_mat_new, length(trial_configs_all[1].pos), r_cut_all, Ref(pot)))
    #println(invert(dimer_energy_config.(dist2_mat_new, length(trial_configs_all[1].pos), Ref(pot))))
    #println()
    return dist2_mat_new,en_atom_vec,en_tot_new
end

# energyvector, dist2new = invert(get_energy_dimer.(trial_positions,indices,mc_states,Ref(pot)))
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
#-----------------------------------------------------------------------------#
#----------------------------EmbeddedAtomPotential----------------------------#
#-----------------------------------------------------------------------------#
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
Primary calculation of ϕ,ρ for atom i, given each other atom's distance to i in `distancevec`. `eatomvec` stores the ϕ and ρ components.
"""
function calc_components(eatomvec,distancevec,n,m)
    for dist in distancevec
        eatomvec .+= invrexp(dist,n,m)
    end
    return eatomvec
end
"""
    calc_components(dist2_mat,n,m) 
    calc_components(componentvec,atomindex,old_r2_vec,new_r2_vec,n,m)
Given a matrix `dist2_mat` of each square-radial distance and the exponents `n,m` calculates the Nx2 vector of ϕ and ρ components for each atom and returning them as a `component_vector`.

Second method also includes an existing `componentvec` `atomindex` and old and new interatomic distances from an atom at `atomindex` stored in vectors `new_r2_vec,old_r2_vec`. This calculates the `new_component_vec` based on the updated distances and returns this.
"""
function calc_components(dist2_mat,n,m)
    n_atoms = size(dist2_mat)[1]
    component_vector = zeros(n_atoms,2)
    for row_index in 1:n_atoms
        component_vector[row_index,:] = calc_components(component_vector[row_index,:],dist2_mat[row_index,:],n,m)
    end
    return component_vector
end
function calc_components(new_component_vec,atomindex,old_r2_vec,new_r2_vec,n,m)

    #new_component_vec = copy(componentvec)

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
"""
    calc_new_energy(componentvec,atomindex,old_r2,new_r2,pot::EmbeddedAtomPotential)
Function to calculate updated energy after a move of an atom at `atomindex` resulting in a change in `componentvec` to `new_component_vector`. The `dist2_mat` and `newpos` are used with the potential `pot` to calculate the updated components and `newenergy` 
"""
function calc_new_energy(mc_state,atomindex,newpos,pot::EmbeddedAtomPotential)

    dist2_new = [distance2(newpos,b) for b in mc_state.config.pos]
    dist2_new[atomindex] = 0.
    new_component_vector = deepcopy(mc_state.en_atom_vec)
    new_component_vector = calc_components(new_component_vector,atomindex,mc_state.dist2_mat[atomindex,:],dist2_new,pot.n,pot.m)

    newenergy = calc_energies_from_components(new_component_vector,pot.ean,pot.eCam)

    return  newenergy,dist2_new,new_component_vector 
end

"""
    get_energy(dist2_mat,pot::EmbeddedAtomPotential)
Initialises the energy of an EAM potential. Using the interatomic distances stored in `dist2_mat` and the parameters in `pot` we return the atomic two body components in `componentvec` as well as the total energy. 
"""
function get_energy(dist2_mat,pot::EmbeddedAtomPotential)
    componentvec = calc_components(dist2_mat,pot.n,pot.m)
    return componentvec, calc_energies_from_components(componentvec,pot.ean,pot.eCam) 
end

function get_energy(trial_positions,indices,mc_states,pot::EmbeddedAtomPotential)

    energyvector,dist2_new,new_components = invert(calc_new_energy.(mc_states,indices,trial_positions,Ref(pot)))



    return energyvector,dist2_new,new_components
end
"""
    dimer_energy_config(distmat,NAtoms,pot::EmbeddedAtomPotential)
Currying function to match the initialisation of MCStates with the embedded atom model. This now matches the form of the general dimer potential for use.
"""
function dimer_energy_config(distmat,NAtoms,pot::EmbeddedAtomPotential)
    
    return get_energy(distmat,pot)
end
#-----------------------------------------------------------------------------#
#-----------------------------Non-Dimer Potentials----------------------------#
#-----------------------------------------------------------------------------#

"""
    lrc(NAtoms,r_cut,pot::ELJPotentialEven{N})
The long range correction for the extended Lennard-Jones potential (even). r_cut is the cutoff distance.
lrc is an integral of all interactions outside the cutoff distance, using the uniform density approximation.
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

"""
    dimer_energy(pot::ELJPotentialB{N}, r2, tan) where N
Dimer energy when the distance square between two atom is r2 and the angle between the line connecting them and z-direction is tan.
When r2 < 5.30, returns 1.
"""
function dimer_energy(pot::ELJPotentialB{N}, r2, tan) where N
    #d^(-6)*c6*(1+a6*t2+b6*t4)+d^(-8)*(c8*(1+a8*t2+b8*t4)+d_inv*(c9*(1+a9*t2+b9*t4)+d_inv*(c10*(1+a10*t2+b10*t4)+d_inv*(c11*(1+a11*t2+b11*t4)+d_inv*c12*(1+a12*t2+b12*t4)))))
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
    lrc(NAtoms,r_cut,pot::ELJPotentialEven{N})
The long range correction for the energy in magnetic field. r_cut is the cutoff distance.
lrc is an integral of all interactions outside the cutoff distance, using the uniform density approximation.
At long range, dimer energy changes very little with tan(theta), the coefficients are fitted to energies from theta = 0 to 90 degrees.
"""
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

"""
    all associated energy calculation function with ELJ potential (with B)
"""

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
    dimer_energy_atom(i, d2vec, r_cut, pot<:AbstractDimerPotentialB)
    dimer energy of an atom with other atoms, with a cutoff distance r_cut.
"""
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
    dimer_energy_config(distmat, NAtoms, pot::AbstractDimerPotentialB)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, tanmat, NAtoms, pot::AbstractDimerPotentialB)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    #This uses dimer_energy_atom for all atoms, which means one dimer energy will be calculated twice.
    #for i in 1:NAtoms #eachindex(),enumerate()..?
        #dimer_energy_vec[i] = dimer_energy_atom(i, distmat[:, i], pot) #@view distmat[i, :]
        #energy_tot += dimer_energy_vec[i]
    #end 
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


"""
    dimer_energy_config(distmat, NAtoms, r_cut, pot::AbstractDimerPotentialB)
Stores the total of dimer energies of one atom with all other atoms in vector and
calculates total energy of configuration, with a cutoff distance r_cut
Needs squared distances matrix, see `get_distance2_mat` [`get_distance2_mat`](@ref) 
and potential information `pot` [`Abstract_Potential`](@ref) 
"""
function dimer_energy_config(distmat, tanmat, NAtoms, r_cut, pot::AbstractDimerPotentialB)
    dimer_energy_vec = zeros(NAtoms)
    energy_tot = 0.
    #for i in 1:NAtoms #eachindex(),enumerate()..?
        #dimer_energy_vec[i] = dimer_energy_atom(i, distmat[:, i], r_cut, pot) #@view distmat[i, :]
        #energy_tot += dimer_energy_vec[i]
    #end 

    for i in 1:NAtoms
        for j=i+1:NAtoms
            if distmat[i,j] <= r_cut
                e_ij=dimer_energy(pot,distmat[i,j],tanmat[i,j])
                dimer_energy_vec[i] += e_ij
                dimer_energy_vec[j] += e_ij
                energy_tot += e_ij
            end
        end
    end 

    return dimer_energy_vec, energy_tot + lrc(NAtoms,r_cut,pot)   #no 0.5*energy_tot
end    

"""
    energy_update(i_atom, dist2_new, pot::AbstractDimerPotentialB)
returns new energy when an atom is moved. In magnetuic field, without cutoff distance.
"""
function energy_update(i_atom, dist2_new, pot::AbstractDimerPotentialB)
    return dimer_energy_atom(i_atom, dist2_new, pot)
end

"""
    energy_update(i_atom, dist2_new, r_cut, pot::AbstractDimerPotentialB)
returns new energy when an atom is moved. In magnetuic field, with cutoff distance r_cut.
"""
function energy_update(i_atom, dist2_new, r_cut, pot::AbstractDimerPotentialB)
    return dimer_energy_atom(i_atom, dist2_new, r_cut, pot)
end

"""
    energy_update(pos, i_atom, config, dist2_mat, tan_mat, pot::AbstractDimerPotentialB)
Gives difference in energy, new distances, new tan when an atom is moved. In magnetic field, without cutoff distance.
"""
function energy_update(pos, i_atom, config, dist2_mat, tan_mat, pot::AbstractDimerPotentialB)
    dist2_new = [distance2(pos,b,config.bc) for b in config.pos]
    dist2_new[i_atom] = 0.
    tan_new = [get_tan(pos,b,config.bc) for b in config.pos]
    tan_new[i_atom] = 0
    d_en = dimer_energy_atom(i_atom, dist2_new, tan_new, pot) - dimer_energy_atom(i_atom, dist2_mat[:,i_atom], tan_mat[:,i_atom],pot)
    return d_en, dist2_new, tan_new
end

"""
    energy_update(pos, i_atom, config, dist2_mat, tan_mat, r_cut, pot::AbstractDimerPotentialB)
Gives difference in energy, new distances, new tan when an atom is moved. In magnetic field, with cutoff distance r_cut.
"""
function energy_update(pos, i_atom, config, dist2_mat, tan_mat, r_cut, pot::AbstractDimerPotentialB)
    dist2_new = [distance2(pos,b,config.bc) for b in config.pos]
    dist2_new[i_atom] = 0.
    tan_new = [get_tan(pos,b,config.bc) for b in config.pos]
    tan_new[i_atom] = 0
    d_en = dimer_energy_atom(i_atom, dist2_new, tan_new, r_cut, pot) - dimer_energy_atom(i_atom, dist2_mat[:,i_atom], tan_mat[:,i_atom],r_cut, pot)
    return d_en, dist2_new, tan_new
end
"""
    get_energy_dimer(pos,i_atom,mc_state,pot::AbstractDimerPotentialB)
A get_energy function similar to the energy_update function. This simply returns the current energy rather than delta_en
"""
function get_energy_dimer(pos,i_atom,mc_state,pot::AbstractDimerPotentialB)
    # dist2_new = [distance2(pos,b) for b in mc_state.config.pos]
    # dist2_new[i_atom] = 0.
    # delta_energy= dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, mc_state.dist2_mat[:,i_atom], pot)

    delta_energy,dist2_new,tan_new = energy_update(pos,i_atom,mc_state.config,mc_state.dist2_mat,mc_state.tan_mat,pot)
    energy = mc_state.en_tot + delta_energy
    return energy,[dist2_new,tan_new]
end
"""
    get_energy_dimer(pos,i_atom,r_cut,mc_state,pot::AbstractDimerPotentialB)
A get_energy function similar to the energy_update function. This simply returns the current energy rather than delta_en
with a cutoff distance r_cut
"""
function get_energy_dimer(pos,i_atom,r_cut,mc_state,pot::AbstractDimerPotentialB)
    # dist2_new = [distance2(pos,b) for b in mc_state.config.pos]
    # dist2_new[i_atom] = 0.
    # delta_energy= dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, mc_state.dist2_mat[:,i_atom], pot)

    delta_energy,dist2_new,tan_new = energy_update(pos,i_atom,mc_state.config,mc_state.dist2_mat,mc_state.tan_mat,r_cut,pot)
    energy = mc_state.en_tot + delta_energy
    return energy,[dist2_new,tan_new]
end

"""
    get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotentialB)
Top scope get energy function for dimer potentials returning the energy vector and new distance squared vector as this must be calculated in order to calculate the potential.

"""
function get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotentialB,ensemble::NVT)
    energyvector, mats_new = invert(get_energy_dimer.(trial_positions,indices,mc_states,Ref(pot)))

    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector, mats_new
end
#this will be the format for this part of the get_energy function.

"""
    get_energy(trial_positions,indices,r_cut,mc_states,pot::AbstractDimerPotentialB,ensemble::NPT)
Top scope get energy function for dimer potentials returning the energy vector and new distance squared vector as this must be calculated in order to calculate the potential.
with the cutoff distance r_cut
"""
function get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotentialB,ensemble::NPT)
    
    n=length(mc_states)
    r_cut_all=Array{Float64}(undef,n)
    for i=1:n
        r_cut_all[i]=mc_states[i].config.bc.box_length^2/4
    end
    energyvector, mats_new = invert(get_energy_dimer.(trial_positions,indices,r_cut_all,mc_states,Ref(pot)))
    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector,mats_new
end

"""
    get_energy function when the whole configuration is scaled
        find the new distance matrix first, and use dimer_energy_config to find the new total energy and energy matrix
"""
function get_energy(trial_configs_all,mc_states,pot::AbstractDimerPotentialB)
    dist2_mat_new = get_distance2_mat.(trial_configs_all)
    #en_atom_vec, en_tot_new = invert(dimer_energy_config.(dist2_mat_new, length(trial_configs_all[1].pos), Ref(pot)))
    n=length(trial_configs_all)
    r_cut_all=Array{Float64}(undef,n)
    for i=1:n
        r_cut_all[i]=trial_configs_all[i].bc.box_length^2/4
    end
    tan_mat=Array{Matrix}(undef,n)
    for i=1:n
        tan_mat[i]=mc_states[i].tan_mat
    end
    en_atom_vec, en_tot_new = invert(dimer_energy_config.(dist2_mat_new, tan_mat, length(trial_configs_all[1].pos), r_cut_all, Ref(pot)))
    #println(invert(dimer_energy_config.(dist2_mat_new, length(trial_configs_all[1].pos), Ref(pot))))
    #println()

    return dist2_mat_new,en_atom_vec,en_tot_new
end
"""
    end here
"""


""" 
    DFTPotential 
Implements type for a "density functional theory" potential (calcuate energies in DFT); subtype of AbstractPotential 
field names: a: specifies the box length, lattice: specifies the 3x3 cube/box from a, El: specifies the atom type,
pseudopotential and functional, atoms: a vector containing the atom type from El, functional: specifies the functional, 
n_atoms:: specifies the number of atoms, kgrid: is the k-point sampling grid, Ecut: is energy cutoff. 
""" 
struct DFTPotential <:AbstractPotential
    a::Float64                     
    lattice::Mat3                  
    El::ElementPsp                
    atoms::Vector                 
    functional::Vector{Symbol}    
    n_atoms::Int                  
    kgrid::Vector                 
    Ecut::Int                      
end  

function DFTPotential(a, n_atoms) 
    kgrid = [1, 1, 1] 
    Ecut = 6
    lattice = a * I(3) 
    El = ElementPsp(:Ga, psp=load_psp("hgh/pbe/ga-q3")) 
    atoms = Vector{ElementPsp}(undef,n_atoms)
    for i in 1:n_atoms 
        atoms[i] = El 
    end  
    functional = [:gga_x_pbe, :gga_c_pbe] 
    return DFTPotential(a, lattice, El, atoms, functional, n_atoms, kgrid, Ecut)
end  
""" 
    getenergy_DFT(pos1, pot) 
Calculates total energy of a given configuration for an arbitrary number of gallium atoms; 
note that this function depends only on the positions of the atoms within the configuration, 
so no bc's are to be included. 
"""
function getenergy_DFT(pos1, pot::DFTPotential) 
    pos1 = pos1 / pot.a 
    model = model_DFT(pot.lattice, pot.atoms, pos1, pot.functional)
    basis = PlaneWaveBasis(model; pot.Ecut, pot.kgrid) 
    scfres = self_consistent_field(basis; tol = 1e-7, callback=info->nothing) 
    return scfres.energies.total 
end  

function energy_update(pos, i_atom, config::Config, dist2_mat, en_old, pot::DFTPotential) #pos is SVector, i_atom is integer 
    dist2_new = [distance2(pos,b) for b in config.pos]
    dist2_new[i_atom] = 0.  
    config.pos[i_atom] = copy(pos)
    pos_new = copy(config.pos) 
    delta_en = getenergy_DFT(pos_new, pot) - en_old
    return delta_en, dist2_new
end   

#--------------------------------------------#
#--------------RuNNer methods----------------#
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
"""
    get_new_state_vars!(trial_pos,atomindex,nnp_state,pot)
calculates the changes to the g_matrix, dist2_matrix etc based on the new `trial_pos` of atom `atomindex` and adjusts these values inside the `nnp_state` struct. Note the use of the threaded function for the calculation of the symmetry functions
"""
function get_new_state_vars!(trial_pos,atomindex,nnp_state,pot)

    nnp_state.new_dist2_vec = [ distance2(trial_pos,b,nnp_state.config.bc) for b in nnp_state.config.pos]
    nnp_state.new_dist2_vec[atomindex] = 0.
    
    nnp_state.new_f_vec = cutoff_function.(sqrt.(nnp_state.new_dist2_vec),Ref(pot.r_cut))


    nnp_state.new_g_matrix = copy(nnp_state.g_matrix)

    nnp_state.new_g_matrix = total_thr_symm!(nnp_state.new_g_matrix,nnp_state.config.pos,trial_pos,nnp_state.dist2_mat,nnp_state.new_dist2_vec,nnp_state.f_matrix,nnp_state.new_f_vec,atomindex,pot.symmetryfunctions)

    return nnp_state
end

"""
    calc_new_energy!(nnp_state,pot)
Function to curry the PTMC struct `nnp_state` and `pot` into the format required by the MachineLearningPotential package, output is the updated `nnp_state`
"""
function calc_new_energy!(nnp_state,pot)
    nnp_state.new_en_atom = forward_pass(nnp_state.new_g_matrix,length(nnp_state.en_atom_vec),pot.nnp)
    return nnp_state
end
"""
    get_runner_energy!(trial_pos,atomindex,nnp_state,pot)
base-level single-state calculation of the `energy` and adjusted `nnp_state` given that we have displaced atom `atomindex` to position `trial_pos`.
"""
function get_runner_energy!(trial_pos,atomindex,nnp_state,pot)
    nnp_state = get_new_state_vars!(trial_pos,atomindex,nnp_state,pot)

    nnp_state = calc_new_energy!(nnp_state,pot)
    
    return sum(nnp_state.new_en_atom) , nnp_state
end
"""
    get_energy!(trial_positions,indices,nnp_states,pot::RuNNerPotential)
Total, vectorised function for the calculation of the energy change in each state within `nnp_states` based on a vector of moved `indices` to `trial_positions` based on the potential `pot`.
"""
function get_energy(trial_positions,indices,nnp_states,pot::RuNNerPotential)
    energyvector = Vector{Float64}(undef,length(indices))

    energyvector,nnp_states = invert(get_runner_energy!.(trial_positions,indices,nnp_states,Ref(pot)))
    
    return energyvector, nnp_states
end

#---------------------------------------------#

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
