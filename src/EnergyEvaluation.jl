"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    
module EnergyEvaluation

using StaticArrays 
#using DFTK 
using LinearAlgebra
using SplitApplyCombine
using ..Configurations

using ..RuNNer

export AbstractPotential, AbstractDimerPotential, AbstractMachineLearningPotential,SerialMLPotential,ParallelMLPotential

#export DFTPotential

export ELJPotential, ELJPotentialEven, ELJPotentialB
export dimer_energy, dimer_energy_atom, dimer_energy_config 
# export getenergy_DFT 
export get_energy_dimer,get_energy_RuNNer
export energy_update
export get_energy
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


abstract type AbstractMachineLearningPotential <: AbstractPotential end

struct SerialMLPotential <: AbstractMachineLearningPotential #remove the Abstract from the name
    dir::String
    atomtype::String
end
struct ParallelMLPotential <: AbstractMachineLearningPotential
    dir::String
    atomtype::String
    index::Int64
    total::Int64
end



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
        dimer_energy_vec[i] = dimer_energy_atom(i, distmat[:, i], pot) #@view distmat[i, :]
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
    d_en = dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, dist2_mat[:,i_atom], pot)
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
    get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotential)
Top scope get energy function for dimer potentials returning the energy vector and new distance squared vector as this must be calculated in order to calculate the potential.

"""
function get_energy(trial_positions,indices,mc_states,pot::AbstractDimerPotential)
    energyvector, dist2_new = invert(get_energy_dimer.(trial_positions,indices,mc_states,Ref(pot)))

    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector,dist2_new
end
#this will be the format for this part of the get_energy function.

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

function dimer_energy(pot::ELJPotentialB{N}, r2, tan) where N
    if r2>=5.30
        r6inv = 1/(r2*r2*r2)
        t2=2/(tan^2+1)-1     #cos(2*theta)
        #t2=0
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

function energy_update(i_atom, dist2_new, pot::AbstractDimerPotentialB)
    return dimer_energy_atom(i_atom, dist2_new, pot)
end

function energy_update(i_atom, dist2_new, r_cut, pot::AbstractDimerPotentialB)
    return dimer_energy_atom(i_atom, dist2_new, r_cut, pot)
end

function energy_update(pos, i_atom, config, dist2_mat, tan_mat, pot::AbstractDimerPotentialB)
    dist2_new = [distance2(pos,b,config.bc) for b in config.pos]
    dist2_new[i_atom] = 0.
    tan_new = [get_tan(pos,b,config.bc) for b in config.pos]
    tan_new[i_atom] = 0
    #println("i_atom ",i_atom)
    #println("dimer energy atom new ",dimer_energy_atom(i_atom, dist2_new, tan_new, pot))
    d_en = dimer_energy_atom(i_atom, dist2_new, tan_new, pot) - dimer_energy_atom(i_atom, dist2_mat[:,i_atom], tan_mat[:,i_atom],pot)
    #println("d_en= ",d_en)
    return d_en, dist2_new, tan_new
end

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
    energyvector, vecs_new = invert(get_energy_dimer.(trial_positions,indices,mc_states,Ref(pot)))

    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector, vecs_new
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
        #r_cut_all[i]=mc_states[i].config.bc.box_length^2/4
        r_cut_all[i]=get_r_cut(mc_states[i].config.bc)
    end
    energyvector, vecs_new = invert(get_energy_dimer.(trial_positions,indices,r_cut_all,mc_states,Ref(pot)))
    # energyvector = mc_state.en_tot .+ delta_energyvector
    return energyvector,vecs_new
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
        #r_cut_all[i]=trial_configs_all[i].bc.box_length^2/4
        r_cut_all[i]=get_r_cut(trial_configs_all[i].bc)
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
# struct DFTPotential <:AbstractPotential
#     a::Float64                     
#     lattice::Mat3                  
#     El::ElementPsp                
#     atoms::Vector                 
#     functional::Vector{Symbol}    
#     n_atoms::Int                  
#     kgrid::Vector                 
#     Ecut::Int                      
# end  

# function DFTPotential(a, n_atoms) 
#     kgrid = [1, 1, 1] 
#     Ecut = 6
#     lattice = a * I(3) 
#     El = ElementPsp(:Ga, psp=load_psp("hgh/pbe/ga-q3")) 
#     atoms = Vector{ElementPsp}(undef,n_atoms)
#     for i in 1:n_atoms 
#         atoms[i] = El 
#     end  
#     functional = [:gga_x_pbe, :gga_c_pbe] 
#     return DFTPotential(a, lattice, El, atoms, functional, n_atoms, kgrid, Ecut)
# end  
""" 
    getenergy_DFT(pos1, pot) 
Calculates total energy of a given configuration for an arbitrary number of gallium atoms; 
note that this function depends only on the positions of the atoms within the configuration, 
so no bc's are to be included. 
 """
# function getenergy_DFT(pos1, pot::DFTPotential) 
#     pos1 = pos1 / pot.a 
#     model = model_DFT(pot.lattice, pot.atoms, pos1, pot.functional)
#     basis = PlaneWaveBasis(model; pot.Ecut, pot.kgrid) 
#     scfres = self_consistent_field(basis; tol = 1e-7, callback=info->nothing) 
#     return scfres.energies.total 
# end  

# function energy_update(pos, i_atom, config::Config, dist2_mat, en_old, pot::DFTPotential) #pos is SVector, i_atom is integer 
#     dist2_new = [distance2(pos,b) for b in config.pos]
#     dist2_new[i_atom] = 0.  
#     config.pos[i_atom] = copy(pos)
#     pos_new = copy(config.pos) 
#     delta_en = getenergy_DFT(pos_new, pot) - en_old
#     return delta_en, dist2_new
# end   

#--------------------------------------------#
#--------------RuNNer methods----------------#
"""
    get_energy_RuNNer(pos_vec,i_atom_vec,mc_states,pot::AbstractMLPotential)
get energy function for the RuNNer potential. Accepts the updated positions, indices and mc_states as well as the potential and returns the energy vector. 
"""
function get_energy_RuNNer(pos_vec,i_atom_vec,mc_states,pot::AbstractMachineLearningPotential)
    file = RuNNer.writeinit(pot.dir)

    writeconfig.(Ref(file),mc_states,i_atom_vec,pos_vec,Ref(pot.atomtype))
    close(file)

    energyvec = getRuNNerenergy(pot.dir,length(pos_vec))

    return energyvec
end
"""
    get_energy(trial_positions,indices,mc_states,pot::AbstractMachineLearningPotential)
top-scope function for RuNNer returning the energy vector. Blank vector is presently a placeholder recognising that the dist2_new vector is returned for dimer potentials. This will be calculated later for RuNNer and DFT.
"""
function get_energy(trial_positions,indices,mc_states,pot::AbstractMachineLearningPotential)
    energyvector = get_energy_RuNNer(trial_positions,indices,mc_states,pot)

    return energyvector,zeros(length(energyvector))
end

#---------------------------------------------#
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
