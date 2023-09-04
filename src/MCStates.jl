module MCStates 

using ..BoundaryConditions
using ..Configurations
using ..MachineLearningPotential
using ..EnergyEvaluation
#using ..InputParams

export MCState, NNPState
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
    tan_mat::Matrix{T}
    en_atom_vec::Vector{T}
    en_tot::T
    ham::Vector{T}
    max_displ::Vector{T}
    max_boxlength::T
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_rot::Vector{Int}
    count_exc::Vector{Int}
end    

function MCState(
    temp, beta, config::Config{N,BC,T}, dist2_mat, tan_mat, en_atom_vec, en_tot; 
    max_displ = [0.1,0.1,1.], max_boxlength = 10., count_atom = [0,0], count_vol = [0,0], count_rot = [0,0], count_exc = [0,0]
) where {T,N,BC}
    ham = T[]
    MCState{T,N,BC}(
        temp, beta, deepcopy(config), copy(dist2_mat), copy(tan_mat),copy(en_atom_vec), en_tot, 
        ham, copy(max_displ), copy(max_boxlength), copy(count_atom), copy(count_vol), copy(count_rot), copy(count_exc)
        )
end

function MCState(temp, beta, config::Config, pot::AbstractDimerPotential; kwargs...) 
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    tan_mat = zeros(n_atoms,n_atoms)
    if typeof(config.bc) == PeriodicBC{Float64}
        en_atom_vec, en_tot = dimer_energy_config(dist2_mat, n_atoms, config.bc.box_length^2/4, pot)
    else
        en_atom_vec, en_tot = dimer_energy_config(dist2_mat, n_atoms, pot)
    end
    MCState(temp, beta, config, dist2_mat, tan_mat, en_atom_vec, en_tot; kwargs...)
end

function MCState(temp, beta, config::Config, pot::AbstractDimerPotentialB; kwargs...) 
    dist2_mat = get_distance2_mat(config)
    tan_mat = get_tantheta_mat(config,config.bc)
    n_atoms = length(config.pos)
    if typeof(config.bc) == PeriodicBC{Float64}
        en_atom_vec, en_tot = dimer_energy_config(dist2_mat, tan_mat, n_atoms, config.bc.box_length^2/4, pot)
    else
        en_atom_vec, en_tot = dimer_energy_config(dist2_mat, tan_mat, n_atoms, pot)
    end
    MCState(temp, beta, config, dist2_mat, tan_mat, en_atom_vec, en_tot; kwargs...)
end

function MCState(temp,beta, config::Config, pot::AbstractMachineLearningPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    en_tot = 0.

    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)

end
function MCState(temp,beta, config::Config, pot::DFTPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    en_tot = getenergy_DFT(config.pos, pot)

    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end
#-------------------------------------------------------------#
#------------Neural Networks need more information------------#
#-------------------------------------------------------------#
"""
    NNPState(
        temp::T, beta::T, config::Config{N,BC,T}, dist2_mat::Matrix{T}, en_atom_vec::Vector{T}, en_tot::T, ham::Vector{T}, max_displ::Vector{T}, count_atom::Vector{Int}, count_vol::Vector{Int}, count_rot::Vector{Int}, count_exc::Vector{Int}

    new_en_atom::Vector{T}, g_matrix::Array{T}, f_matrix::Array{T}, new_g_matrix::Array{T}, new_dist2_vec::Vector{T}, new_f_vec::Vector{T}

Separate struct defining mc states in the context of a neural network potential. most fields are identical to the MCStates struct with the following additions:
    new_en_atoms -- saves on memory allocation when calculating new energies
    g_matrix -- current symmety function values per atom
    f_matrix -- an extention of the dist2_mat containing the cutoff function

    new_g_matrix,new_dist2_vec,new_f_vec -- like new_en_atom these are designed to save on memory allocation when updating the position during the mc_cycle.
"""
mutable struct NNPState{T,N,BC}
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

    new_en_atom::Vector{T}

    g_matrix::Array{T}
    f_matrix::Array{T}

    new_g_matrix::Array{T}
    new_dist2_vec::Vector{T}
    new_f_vec::Vector{T}
end
function NNPState(
    temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_vec, en_tot,n_symm_func,g_matrix,f_matrix; 
    max_displ = [0.1,0.1,1.], count_atom = [0,0], count_vol = [0,0], count_rot = [0,0], count_exc = [0,0]
) where {T,N,BC}
    ham = T[]
    NNPState{T,N,BC}(
        temp, beta, deepcopy(config), copy(dist2_mat), copy(en_atom_vec), en_tot, 
        ham, copy(max_displ), copy(count_atom), copy(count_vol), copy(count_rot), copy(count_exc),zeros(length(config.pos)),g_matrix,f_matrix,zeros(n_symm_func,length(config.pos)),zeros(length(config.pos)),zeros(length(config.pos))
        )
end
function NNPState(temp,beta, config::Config, pot::RuNNerPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    f_matrix = cutoff_function.(sqrt.(dist2_mat),Ref(pot.r_cut))
    n_atoms = length(config.pos)
    g_matrix = total_symm_calc(config.pos,dist2_mat,f_matrix,pot.symmetryfunctions)


    en_atom_vec = zeros(n_atoms)
    en_tot = 0.

    NNPState(temp, beta, config, dist2_mat, en_atom_vec, en_tot,length(pot.symmetryfunctions),g_matrix,f_matrix ; kwargs...)

end
#     MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
# end


end
