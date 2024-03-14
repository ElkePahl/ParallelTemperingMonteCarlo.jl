module MCStates 

using ..BoundaryConditions
using ..Configurations
using ..MachineLearningPotential
using ..EnergyEvaluation
using ..Ensembles
#using ..InputParams

export MCState#, NNPState
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
-`new_dist2_vec`: calculates the new r2 between atoms based on a trial move
- `new_en` : new energy value for trial configuraiton
- `en_tot`: total energy of `config`; generated automatically when `pot` given

- `potential_variables` : mutable struct containing energy-related variables for the current configuration
- `ensemble_variables` : mutable struct containing ensemble-related variables for the current configuraiton
- `ham`: vector containing sampled energies - generated in MC run
- `count_atom`: number of accepted atom moves - total and between adjustment of step sizes; key-word argument
- `count_vol`: number of accepted volume moves - total and between adjustment of step sizes; key-word argument

- `count_exc`: number of attempted (10%) and accepted exchanges with neighbouring trajectories; key-word argument
"""
mutable struct MCState{T,N,BC,PVType,EVType}
    temp::T
    beta::T
    config::Config{N,BC,T}
    dist2_mat::Matrix{T}
    new_dist2_vec::Vector{T}

    new_en::T
    en_tot::T

    potential_variables::PVType
    ensemble_variables::EVType

    ham::Vector{T}
    max_displ::Vector{T}
    max_boxlength::T
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_exc::Vector{Int}
end    

function max_length(bc::SphericalBC)
    return 30.
end

function max_length(bc::CubicBC)
    return bc.box_length*1.8
end

function max_length(bc::RhombicBC)
    return bc.box_length*1.8
end


function MCState(
    temp, beta, config::Config{N,BC,T}, dist2_mat, new_dist2_vec,new_en, en_tot,potentialvariables,ensemble_variables; 
    max_displ = [0.1,0.1,1.], max_boxlength = max_length(config.bc), count_atom = [0,0], count_vol = [0,0], count_exc = [0,0]
) where {T,N,BC}
    ham = T[]
    MCState{T,N,BC,typeof(potentialvariables),typeof(ensemble_variables)}(
        temp, beta, deepcopy(config), copy(dist2_mat), copy(new_dist2_vec),new_en, en_tot,deepcopy(potentialvariables),deepcopy(ensemble_variables),ham, copy(max_displ), copy(max_boxlength), copy(count_atom), copy(count_vol), copy(count_exc)
        )
end
function MCState(temp,beta,config::Config,ensemble::Etype,pot::Ptype;
    kwargs...) where Ptype <: AbstractPotential where Etype <: AbstractEnsemble
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config)
    
    potential_variables = set_variables(config,dist2_mat,pot)
    ensemble_variables = set_ensemble_variables(config,ensemble)

    en_tot, potential_variables=initialise_energy(config,dist2_mat,potential_variables,ensemble_variables,pot)

    MCState(temp,beta,config,dist2_mat,zeros(n_atoms),0.,en_tot,potential_variables,ensemble_variables;kwargs...
    )

end

# function MCState(temp, beta, config::Config, pot::AbstractDimerPotential; kwargs...) 
#     dist2_mat = get_distance2_mat(config)
#     n_atoms = length(config.pos)
#     # tan_mat = zeros(n_atoms,n_atoms)
#     if typeof(config.bc) == PeriodicBC{Float64}
#         en_atom_vec, en_tot = dimer_energy_config(dist2_mat, n_atoms, config.bc.box_length^2/4, pot)
#     else
#         en_atom_vec, en_tot = dimer_energy_config(dist2_mat, n_atoms, pot)
#     end
#     MCState(temp, beta, config, dist2_mat, tan_mat, en_atom_vec, en_tot; kwargs...)
# end

# function MCState(temp, beta, config::Config, pot::AbstractDimerPotentialB; kwargs...) 
#     dist2_mat = get_distance2_mat(config)
#     tan_mat = get_tantheta_mat(config,config.bc)
#     n_atoms = length(config.pos)
#     if typeof(config.bc) == PeriodicBC{Float64}
#         en_atom_vec, en_tot = dimer_energy_config(dist2_mat, tan_mat, n_atoms, config.bc.box_length^2/4, pot)
#     else
#         en_atom_vec, en_tot = dimer_energy_config(dist2_mat, tan_mat, n_atoms, pot)
#     end
#     MCState(temp, beta, config, dist2_mat, tan_mat, en_atom_vec, en_tot; kwargs...)
# end

# function MCState(temp,beta, config::Config, pot::AbstractMachineLearningPotential;kwargs...)
#     dist2_mat = get_distance2_mat(config)
#     n_atoms = length(config.pos)
#     en_atom_vec = zeros(n_atoms)
#     en_tot = 0.

#     MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)

# end
# function MCState(temp,beta, config::Config, pot::DFTPotential;kwargs...)
#     dist2_mat = get_distance2_mat(config)
#     n_atoms = length(config.pos)
#     en_atom_vec = zeros(n_atoms)
#     en_tot = getenergy_DFT(config.pos, pot)

#     MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
# end



end
