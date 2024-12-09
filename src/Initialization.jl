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
    initialisation(restart::Bool;eq_cycles)
Basic function for establishing the structs and parameters required for the simulation. Inputs for method one are:
-   `mc_params`: the basic values and parameters concerning how long our simulation runs.
-   `temp`: a grid of temp and beta values passed to the mc_states struct.
-   `start_config`: the initial configuration used to populate each starting state with.
-   `potential`: the potential energy used for the simulation.
-   `ensemble`: the ensemble used for the simulation, contains the move strat inherently.
Inputs for method two are:
-   `restart`: A boolean determining whether the simulation is restarting or being read-in from a file. 
-   `eq_cycles`: the proportion of n_cycles to be run in equilibration.

Method one and two return the following structs:
-   `mc_states`: a vector of [`MCState`](@ref) structs each at a different temperature
-   `move_strategy`: struct containing a vector of [`MoveType`](@ref)
-   `results`: struct countaining the output such as Cv and histograms
-   `start_counter`: where to begin the sims
-   `n_steps`: total moves per mc_cycle 
Method two also returns:
-   `mc_params`: the static parameters determining the scope of the simulation
-   `ensemble`: determines the ensemble and move_strategy followed by the simulation
-   `potential`: the potential energy surface explored by trajectories

***NOTES FOR FUTURE IMPLEMENTATION***

-   consider shuffling `mc_params` to include the `tempgrid` and cut down the number of inputs.

"""
function initialisation(mc_params::MCParams,temp::TempGrid,start_config::Config,potential::Ptype,ensemble::Etype) where Ptype <: AbstractPotential where Etype <:AbstractEnsemble

    move_strategy = MoveStrategy(ensemble)
    n_steps = length(move_strategy)
    
    mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i],start_config,ensemble,potential) for i in 1:mc_params.n_traj]

    results = Output{Float64}(mc_params.n_bin;en_min = mc_states[1].en_tot)
    start_counter=1
    return mc_states,move_strategy,results,n_steps,start_counter
end
function initialisation(restart::Bool,eq_cycles)

    mc_params,temp,ensemble,potential = read_init(restart,eq_cycles)
    
    if restart == true
        start_counter = Int(readdlm("./checkpoint/index.txt")[1])
        mc_states,results = rebuild_states(mc_params.n_traj,ensemble,temp,potential)
    else
        mc_states,results = build_states(mc_params,ensemble,temp,potential)
        start_counter=1
    end

    for state in mc_states
        recentre!(state.config)
    end

    move_strategy = MoveStrategy(ensemble)
    n_steps = length(move_strategy)

    return mc_params,ensemble,potential,mc_states,move_strategy,results,n_steps,start_counter
end

end