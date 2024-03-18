module Initialization


using StaticArrays,DelimitedFiles,Random

export initialisation


using ..MachineLearningPotential
using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..Exchange
using ..ReadSave
using ..Ensembles

"""
    initialisation(mc_params::MCParams,temp::TempGrid,start_config::Config,potential::Ptype,ensemble::NVT)

Basic function for establishing the structs and parameters required for the simulation. Inputs are:
    - `mc_params`: parameters defining MC specifics [`mc_params`](@ref)  
    - `temp`: temperature/inverse temperature grid [`temp`](@ref) 
    - `start_config`: initial configuration (same for each trajectory) [`start_config`](@ref) 
    - `potential`: defines potential energy of N atomic system [`potential`](@ref) 
    - `ensemble`: ensemble used (defines move strategy presently) [`ensemble`](@ref) 
 
Returns the following structs:
    - `mc_states`: collects `MCState` structs for all trajectories [`MCState`](@ref) 
    - `move_strategy`: defines move types used (displacemet, volume, swap, ...) [`move_strategy`](@ref) 
    - `results`: collects MC output (eg. heat capacity and histogram information) [`Output`](@ref) 
    - `start_counter`: defines starting point of simulation (new or restart) 
    - `n_steps`: total number of moves per MC cycle 

***NOTES FOR FUTURE IMPLEMENTATION***
    - re-introduce a restart function once save exists
    - consider shuffling mc_params to include the tempgrid and cut down the number of inputs.
"""
function initialisation(mc_params::MCParams,temp::TempGrid,start_config::Config,potential::Ptype,ensemble::Etype) where Ptype <: AbstractPotential where Etype <:AbstractEnsemble



    move_strategy = MoveStrategy(ensemble)
    n_steps = length(move_strategy)
    
    mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i],start_config,ensemble,potential) for i in 1:mc_params.n_traj]

    results = Output{Float64}(mc_params.n_bin;en_min = mc_states[1].en_tot)

    return mc_states,move_strategy,results,n_steps
end


end