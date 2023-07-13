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

export AbstractPotential, AbstractDimerPotential, AbstractMachineLearningPotential
export RuNNerPotential

export DFTPotential

export ELJPotential, ELJPotentialEven
export dimer_energy, dimer_energy_atom, dimer_energy_config 
export getenergy_DFT, get_energy_dimer#,get_energy_RuNNer
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
    dist2_new = [distance2(pos,b,config.bc) for b in config.pos]
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

"""
    get_energy function when the whole configuration is scaled
    find the new distance matrix first, and use dimer_energy_config to find the new total energy and energy matrix
"""
function get_energy(trial_configs_all,pot::AbstractDimerPotential)
    dist2_mat_new = get_distance2_mat.(trial_configs_all)
    en_atom_vec, en_tot_new = invert(dimer_energy_config.(dist2_mat_new, length(trial_configs_all), Ref(pot)))

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
