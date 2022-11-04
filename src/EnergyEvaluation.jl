"""
    module EnergyEvaluation

this module provides data, structs and methods for dimer energy and total energy evaluation
"""    
module EnergyEvaluation

using StaticArrays 
using DFTK 
using LinearAlgebra
using ..Configurations

using ..RuNNer

export AbstractPotential, AbstractDimerPotential, AbstractMLPotential , ParallelMLPotential

export DFTPotential

export ELJPotential, ELJPotentialEven
export dimer_energy, dimer_energy_atom, dimer_energy_config 
export getenergy_DFT
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


abstract type AbstractMachineLearningPotential <: AbstractPotential end

struct AbstractMLPotential <: AbstractMachineLearningPotential #remove the Abstract from the name
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
        dimer_energy_vec[i] = dimer_energy_atom(i, distmat[i, :], pot) #@view distmat[i, :]
        energy_tot += dimer_energy_vec[i]
    end 
    return dimer_energy_vec, 0.5*energy_tot
end    

function energy_update(i_atom, dist2_new, en_old, pot::AbstractDimerPotential)
    return dimer_energy_atom(i_atom, dist2_new, pot)
end

function energy_update(pos, i_atom, config, dist2_mat, en_old, pot::AbstractDimerPotential)
    dist2_new = [distance2(pos,b) for b in config.pos]
    dist2_new[i_atom] = 0.
    d_en = dimer_energy_atom(i_atom, dist2_new, pot) - dimer_energy_atom(i_atom, dist2_mat[i_atom,:], pot)
    return d_en, dist2_new
end

function energy_update(pos,i_atom,config,dist2_mat,pot::AbstractMLPotential)


    dist2_new = [distance2(pos,b) for b in config.pos]
    dist2_new[i_atom] = 0.

    Evec = RuNNer.getenergy(pot.dir,config,pot.atomtype,i_atom,pos)

    d_en = Evec[2] - Evec[1]

    return d_en, dist2_new

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
