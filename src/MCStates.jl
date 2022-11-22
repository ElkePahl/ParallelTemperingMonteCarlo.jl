module MCStates 

using ..BoundaryConditions
using ..Configurations
using ..RuNNer
using ..EnergyEvaluation
#using ..InputParams

export MCState
"""
    MCState(temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_vec, en_tot; 
        max_displ = [0.1,0.1,1.], count_atom = [0,0], count_vol = [0,0], count_rot = [0,0], count_exc = [0,0])
    MCState(temp, beta, config::Config, pot; kwargs...) 
Creates an MC state vector at a given temperature `temp` containing temperature-dependent information

Fieldnames:
- `temp`: temperature
- `beta`: inverse temperature
- `config`: actual configuration in Markov chain [`Config`](@ref)  
- `dist_2mat`: matrix of squared distances d_ij between atoms i and j; generated automatically when potential `pot` given
- `en_atom_vec`: vector of energy contributions per atom i; generated automatically when `pot` given
- `en_tot`: total energy of `config`; generated automatically when `pot` given
- `ham`: vector containing sampled energies - generated in MC run
- `max_displ`: max_diplacements for atom, volume and rotational moves; key-word argument
- `count_atom`: number of accepted atom moves - total and between adjustment of step sizes; key-word argument
- `count_vol`: number of accepted volume moves - total and between adjustment of step sizes; key-word argument
- `count_rot`: number of accepted rotational moves - total and between adjustment of step sizes; key-word argument
- `count_exc`: number of attempted (10%) and accepted exchanges with neighbouring trajectories; key-word argument
"""
mutable struct MCState{T,N,BC}
    temp::T
    beta::T
    config::Config{N,BC,T}
    dist2_mat::Matrix{T}
    en_atom_vec::Vector{T}
    en_tot::T
    ham::Vector{T}
    max_displ::Vector{T}
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_rot::Vector{Int}
    count_exc::Vector{Int}
end    

function MCState(
    temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_vec, en_tot; 
    max_displ = [0.1,0.1,1.], count_atom = [0,0], count_vol = [0,0], count_rot = [0,0], count_exc = [0,0]
) where {T,N,BC}
    ham = T[]
    MCState{T,N,BC}(
        temp, beta, deepcopy(config), copy(dist2_mat), copy(en_atom_vec), en_tot, 
        ham, copy(max_displ), copy(count_atom), copy(count_vol), copy(count_rot), copy(count_exc)
        )
end

function MCState(temp, beta, config::Config, pot::AbstractDimerPotential; kwargs...) 
   dist2_mat = get_distance2_mat(config)
   n_atoms = length(config.pos)
   en_atom_vec, en_tot = dimer_energy_config(dist2_mat, n_atoms, pot)
   MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end

function MCState(temp,beta, config::Config, pot::AbstractMLPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    en_tot = RuNNer.getenergy(pot.dir, config,pot.atomtype)

    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)

end
function MCState(temp,beta, config::Config, pot::DFTPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    en_tot = getenergy_DFT(config.pos, pot)

    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end
function MCState(temp,beta, config::Config, pot::ParallelMLPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    
    en_tot = RuNNer.getenergy(pot.dir, config,pot.atomtype,pot.index)


    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end


end