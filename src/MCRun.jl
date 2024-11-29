module MCRun


export metropolis_condition, mc_step!, mc_cycle!,ptmc_cycle!, ptmc_run!,save_states,save_params,save_results,get_energy!
export atom_move!
export exc_acceptance, exc_trajectories!

using StaticArrays,DelimitedFiles
using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..Exchange

using ..ReadSave
using ..MCSampling

using ..Initialization


include("swap_config.jl")


"""
    get_energy!(mc_state::MCState{T,N,BC,P,E},pot::PType,movetype::String) where PType <: AbstractPotential where {T,N,BC,P<:PotentialVariables,E<:NVTVariables}
    get_energy!(mc_state::MCState{T,N,BC,P,E},pot::PType,movetype::String) where PType <: AbstractPotential where {T,N,BC,P<:PotentialVariables,E<:NPTVariables}
Curry function designed to separate energy calculations into their respective ensembles and move types. Currently implemented for: 

        - NVT ensemble without r_cut
        - NPT ensemble with r_cut
"""
function get_energy!(mc_state::MCState{T,N,BC,P,E},pot::PType,movetype::String) where PType <: AbstractPotential where {T,N,BC,P<:AbstractPotentialVariables,E<:NVTVariables}
    if movetype == "atommove"
        mc_state.potential_variables,mc_state.new_en = energy_update!(mc_state.new_dist2_vec,mc_state.ensemble_variables.trial_move,mc_state.ensemble_variables.index,mc_state.config,mc_state.potential_variables,mc_state.dist2_mat,mc_state.en_tot,pot)
    end
    return mc_state

end
# function get_energy!(mc_state::MCState{T,N,BC,P,E},pot::PType,movetype::Int64) where PType <: AbstractPotential where {T,N,BC,P<:PotentialVariables,E<:NPTVariables}
#     mc_state.potential_variables,mc_state.new_dist2_vec,mc_state.new_en = energy_update!(mc_state.ensemble_variables.trial_move,mc_state.ensemble_variables.index,mc_state.config,mc_state.potential_variables,mc_state.dist2_mat,mc_state.en_tot,mc_state.ensemble_variables.r_cut,pot)

#     return mc_state

# end
function get_energy!(mc_state::MCState{T,N,BC,P,E},pot::PType,movetype::String) where PType <: AbstractPotential where {T,N,BC,P<:AbstractPotentialVariables,E<:NPTVariables}
    if movetype == "atommove"
        mc_state.potential_variables,mc_state.new_en = energy_update!(mc_state.new_dist2_vec,mc_state.ensemble_variables.trial_move,mc_state.ensemble_variables.index,mc_state.config,mc_state.potential_variables,mc_state.dist2_mat,mc_state.en_tot,mc_state.ensemble_variables.r_cut,pot)
    else
        mc_state.potential_variables.en_atom_vec,mc_state.new_en = dimer_energy_config(mc_state.ensemble_variables.new_dist2_mat,N,mc_state.potential_variables,mc_state.ensemble_variables.new_r_cut,mc_state.config.bc,pot)
    end
    return mc_state
end
"""
    acc_test!(mc_state::MCState,ensemble::Etype,movetype::String) where Etype <: AbstractEnsemble
acc_test! function now significantly contracted as a method of calculating the metropolis condition, comparing it to a random variable and if the condition is met using the swap_config! function to exchange the current `mc_state` with the internally defined new variables. `ensemble` and `movetype` dictate the exact calculation of the metropolis condition, and the internal `potential_variables` within the mc_states dictate how swap_config! operates. 
"""
function acc_test!(mc_state::MCState,ensemble::Etype,movetype::String) where Etype <: AbstractEnsemble #where Mtype <: MoveType
    if metropolis_condition(movetype,mc_state,ensemble) >=rand()
        swap_config!(mc_state,movetype)
    end
end
"""
    mc_move!(mc_state::MCState,move_strat::MoveStrategy{N,E},pot::Ptype,ensemble::Etype) where Ptype <: AbstractPotential where Etype <: AbstractEnsemble where {N,E}
basic move for one `mc_state` according to a `move_strat` dictating the types of moves allowed within the `ensemble` when moving across a `pot` defining the PES.
     calculates an index for the move
     generates either a volume or atom move depending on movestrat[index]
     calculates energy based on the pot and new move 
     tests acc and swaps if relevant 
"""
function mc_move!(mc_state::MCState,move_strat::MoveStrategy{N,E},pot::Ptype,ensemble::Etype) where Ptype <: AbstractPotential where Etype <: AbstractEnsemble where {N,E}
    mc_state.ensemble_variables.index = rand(1:N)

    mc_state = generate_move!(mc_state,move_strat.movestrat[mc_state.ensemble_variables.index],ensemble)

    mc_state = get_energy!(mc_state,pot,move_strat.movestrat[mc_state.ensemble_variables.index])

    acc_test!(mc_state,move_strat.ensemble,move_strat.movestrat[mc_state.ensemble_variables.index])
    return mc_state
end
"""
    mc_step!((mc_states::Vector{stype},move_strat,pot,ensemble) where stype <: MCState
Distributes each state in `mc_state` to the mc_move function in accordance with a `move_strat`, `ensemble` and `pot`
"""
function mc_step!(mc_states::Vector{stype},move_strat,pot,ensemble,n_steps) where stype <: MCState
    Threads.@threads for state in mc_states
        for i_step in 1:n_steps
            state = mc_move!(state,move_strat,pot,ensemble)
        end
    end
    return mc_states
end
"""
    mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,index)
    mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,results,idx)
Basic function utilised by the simulation. For each of the `n_steps` run a single mc_step on the `mc_states` according to `pot`, `move_strat` and `ensemble`. then complete the parallel_tempering_exchange and update_step_size.

    Second method includes the sampling_step! which updates the `results` struct. The first method is used by the equilibration_cycle and therefore does __not__ update the results funciton. 
"""
function mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,index, save_directory::String)
    
    mc_states=  mc_step!(mc_states,move_strat,pot,ensemble,n_steps)
    
    if rand() < 0.1
        parallel_tempering_exchange!(mc_states,mc_params,ensemble)
    end
    if rem(index,mc_params.n_adjust) == 0
        for state in mc_states 
            update_max_stepsize!(state,mc_params.n_adjust,ensemble)
        end
    end
    if rem(index, 50000) == 0 
        for (i, state) in enumerate(mc_states)
            filename = "configurations_cycle_state_$(i).dat"
            save_config(filename, state, index, save_directory)
            filename_xyz = "configurations_cycle_state_$(i).xyz"
            save_config_xyz(filename_xyz, state, index, save_directory) # Addition of index to track cycle number
        end
    end


    return mc_states
end
function mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,results,idx, save_directory::String,rdfsave)

    mc_states = mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,idx, save_directory)
    sampling_step!(mc_params,mc_states,ensemble,idx,results,rdfsave)
    #     if save == true
#         if rem(i,1000) == 0
#             save_states(mc_params,mc_states,i,save_dir)
#             save_results(results,save_dir)
#         end
#     end
    return mc_states 
end
"""
    check_e_bounds(energy,ebounds)
Function to determine if an energy value is greater than or less than the min/max, used in equilibration cycle.
"""
function check_e_bounds(energy,ebounds)
    if energy < ebounds[1]
        ebounds[1]=energy
    elseif energy > ebounds[2]
        ebounds[2] = energy
    end
    return ebounds
end
"""
    reset_counters(state)
        after equilibration this resets the count stats to zero
"""
function reset_counters(state)
    state.count_atom = [0,0]
    state.count_vol = [0,0]
    state.count_exc = [0,0]
end

"""
    equilibration_cycle(mc_states,move_strat,mc_params,pot,ensemble,n_steps,results)
Function to thermalise a set of `mc_states` ensuring that the number of equilibration cycles defined in `mc_params` are completed without updating the results before initialising the `results` struct according to the maximum and minimum energy determined throughout the equilibration cycle. 
"""
function equilibration_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,results, save_directory::String)
    #set initial hamiltonian values and ebounds
    for state in mc_states
        push!(state.ham, 0)
        push!(state.ham, 0)
    end
    ebounds = [100. , -100.]
    #begin equilibration
    for i = 1:mc_params.eq_cycles
        mc_states = mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,i, save_directory::String)
        for state in mc_states
            ebounds = check_e_bounds(state.en_tot,ebounds)
        end
    end
    #post equilibration reset
    for state in mc_states
        reset_counters(state)
    end
    results = initialise_histograms!(mc_params,results,ebounds,mc_states[1].config.bc)
    return mc_states,results
end
"""

    equilibration(mc_states::Vector{stype},move_strat,mc_params,pot,ensemble,n_steps,results,restart) where stype <: MCState
while initialisation sets mc_states,params etc we require something to thermalise our simulation and set the histograms. This function is mostly a wrapper for the equilibration_cycle! function that optionally removes the thermalisation from restart.

    N.B. Restart is currently non-functional, do not try use it
"""
function equilibration(mc_states::Vector{stype},move_strat,mc_params,pot,ensemble,n_steps,results,restart, save_directory::String) where stype <: MCState
    if restart == true
        println("Restart not implemented yet")
    else
        return equilibration_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,results, save_directory::String)

    end
end
"""
    ptmc_run!(mc_params::MCParams,temps::TempGrid,start_config::Config,potential::Ptype,ensemble::Etype;restart=false, min_acc=0.4,max_acc=0.6,save=false,save_dir=pwd()) where Ptype <: AbstractPotential where Etype <: AbstractEnsemble

Main call for the ptmc program. Given `mc_params` dictating the number of cycles etc. the `temps` containing the temperature and beta values we aim to simulate, an initial `start_config` and the `potential` and `ensemble` we run a complete simulation, explicitly outputting the `mc_states` and `results` structs. 

    the kwargs are the __unimplemented portion__ of the code that needs to be reinserted through reimplementing save/restart and dealing with the update_max_stepsize function in case the user wants to vary the acceptance ratios. 
"""
function ptmc_run!(save_directory::String, mc_params::MCParams,temp::TempGrid,start_config::Vector,potential::Ptype,ensemble::Etype;rdfsave=true,restart=false,start_counter=1, min_acc=0.4,max_acc=0.6,save=false,save_dir=pwd()) where Ptype <: AbstractPotential where Etype <: AbstractEnsemble

    #initialise the states and results etc
    mc_states,move_strategy,results,n_steps = initialisation(mc_params,temp,start_config,potential,ensemble)
    println("params set")
    #Equilibration 
    mc_states,results = equilibration(mc_states,move_strategy,mc_params,potential,ensemble,n_steps,results,restart, save_directory)
    println("equilibration complete")

    #main loop 
    for i = start_counter:mc_params.mc_cycles 
        @inbounds mc_cycle!(mc_states,move_strategy,mc_params,potential,ensemble,n_steps,results,i, save_directory, rdfsave)
    end
    println("MC loop done.")
    println("testing revise pt2")
    #Finalisation of results
    results = finalise_results(mc_states,mc_params,results)
    println("done")
    return mc_states,results
end

#---------------------------------------------------------#
#-------------Notes for Future Implementation-------------#
#---------------------------------------------------------#
"""
-- TO IMPLEMENT --

This version is not complete. While "under the hood" is working as it should, not a lot of effort has been put into:

    - Organising the keyword arguments to be more intuitive
    - Expanding the initialise functions to set the type of results we wish to collect (eg no RDF, save configs as well as checkpoints)
"""

end
