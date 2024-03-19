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
    -mc_params: the basic values and parameters concerning how long our simulation runs.
    -temp: a grid of temp and beta values passed to the mc_states struct.
    -start_config: the initial configuration used to populate each starting state with
    -potential: the potential energy used for the simulation
    -ensemble: the ensemble used for the simulation, contains the move strat inherently
 
returns the following structs:
    -mc_states: a vector of MCState structs each at a different temperature
    -move_strategy: struct containing a vector of movetypes
    -results: struct countaining the output such as Cv and histograms
    -start_counter: where to begin the sims
    -n_steps: total moves per mc_cycle 

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