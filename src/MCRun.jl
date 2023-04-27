module MCRun


export metropolis_condition, mc_step!, mc_cycle!,ptmc_cycle!, ptmc_run!,save_states,save_params,save_results
export atom_move!
export exc_acceptance, exc_trajectories!

using StaticArrays,DelimitedFiles
using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..Exchange
using ..RuNNer
using ..ReadSave
using ..MCSampling

using ..Initialization




"""  

    swap_config!(mc_state, i_atom, trial_pos, dist2_new, new_energy)
        Designed to input one mc_state, the atom to be changed, the trial position, the new distance squared vector and the new energy. 
        If the Metropolis condition is satisfied, these are used to update mc_state. 
"""
function swap_config!(mc_state, i_atom, trial_pos, dist2_new, energy)

    mc_state.config.pos[i_atom] = trial_pos #copy(trial_pos)
    mc_state.dist2_mat[i_atom,:] = dist2_new #copy(dist2_new)
    mc_state.dist2_mat[:,i_atom] = dist2_new    
    mc_state.en_tot = energy
    mc_state.count_atom[1] += 1
    mc_state.count_atom[2] += 1

end
"""
    acc_test!(ensemble, mc_state, new_energy, i_atom, trial_pos, dist2_new::Vector)  

            (ensemble, mc_state, energy, i_atom, trial_pos, dist2_new::Float64)

        The acc_test function works in tandem with the swap_config function, only adding the metropolis condition. Separate functions was benchmarked as very marginally faster. The method for a float64 only calculates the dist2 vector if it's required, as for RuNNer, where the distance matrix is not given during energy calculation.

"""
function acc_test!(ensemble, mc_state, energy, i_atom, trial_pos, dist2_new::Vector)
    
    
        if metropolis_condition(ensemble,(energy -mc_state.en_tot), mc_state.beta) >= rand()

            swap_config!(mc_state,i_atom,trial_pos,dist2_new, energy)
        end   
end
function acc_test!(ensemble, mc_state, energy, i_atom, trial_pos, dist2_new::Float64)
    
    
    if metropolis_condition(ensemble,(energy -mc_state.en_tot), mc_state.beta) >= rand()


        dist2new = [distance2(trial_pos,b) for b in mc_state.config.pos]

        swap_config!(mc_state,i_atom,trial_pos,dist2new, energy)
    end   
end

"""
    function mc_step!(mc_states,mc_params,pot,ensemble)
        New mc_step function, vectorised displacements and energies are batch-passed to the acceptance test function, which determines whether or not to accept the moves.
"""
function mc_step!(mc_states,mc_params,pot,ensemble)

    indices,trial_positions = generate_displacements(mc_states,mc_params)


    energy_vector, dist2_new = get_energy(trial_positions,indices,mc_states,pot)


    for idx in eachindex(mc_states)
        @inbounds acc_test!(ensemble,mc_states[idx],energy_vector[idx],indices[idx],trial_positions[idx],dist2_new[idx])
    end

    return mc_states
    

end

"""
    function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
             mc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r,results,save,i,save_dir,delta_en_hist,delta_r2)
        Current iteration of mc_cycle! using the vectorised mc_step! followed by an attempted trajectory exchange. Ultimately we will add more move types requiring the move strat to be implemented, but this is presently redundant. 

        second method used to be called ptmc_cycle! this is used in the main run as it includes sampling results and save functions. 
"""
function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)

    for i_steps = 1:n_steps
        mc_states = mc_step!(mc_states,mc_params,pot,ensemble)
    end

    if rand() < 0.1 #attempt to exchange trajectories
        parallel_tempering_exchange!(mc_states,mc_params)
    end


    return mc_states
end
function mc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r,results,save,i,save_dir,delta_en_hist,delta_r2)


    mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot,  ensemble, n_steps, a, v, r) 

    sampling_step!(mc_params,mc_states,i,results,delta_en_hist,delta_r2)
    
    #step adjustment
    if rem(i, mc_params.n_adjust) == 0
        for i_traj = 1:mc_params.n_traj
            update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
        end 
    end

    if save == true
        if rem(i,1000) == 0
            save_states(mc_params,mc_states,i,save_dir)
            save_results(results,save_dir)
        end
    end



end
"""
    check_e_bounds(energy,ebounds)
Function to determine if an energy value is greater than or less than the min/max, used in equilibration cycle.
"""
function check_e_bounds(energy,ebounds)
    if energy<ebounds[1]
        ebounds[1]=energy
    elseif energy>ebounds[2]
        ebounds[2] = energy
    else
    end
    return ebounds
end

function reset_counters(state)
    state.count_atom = [0,0]
    state.count_vol = [0,0]
    state.count_rot = [0,0]
    state.count_exc = [0,0]
end

"""
    equilibration_cycle(mc_states,move_strat,mc_params,pot,ensemble)
    
Determines the parameters of a fully thermalised set of mc_states. The method involving complete parameters assumes we begin our simulation from the same set of mc_states. In theory we could pass it one single mc_state which it would then duplicate, passing much more responsibility on to this function. An idea to discuss in future. 

outputs are: thermalised states(mc_states),initialised results(results),the histogram stepsize(delta_en_hist),rdf histsize(delta_r2),starting step for restarts(start_counter),n_steps,a,v,r

"""
function equilibration_cycle!(mc_states,move_strat,mc_params,results,pot,ensemble,n_steps,a,v,r)

    for mc_state in mc_states
        push!(mc_state.ham, 0)
        push!(mc_state.ham, 0)
    end

    ebounds = [100. , -100.]

    for i = 1:mc_params.eq_cycles
        mc_states = mc_cycle!(mc_states,move_strat,mc_params,pot,ensemble,n_steps,a,v,r)       
        for state in mc_states
            ebounds = check_e_bounds(state.en_tot,ebounds)
        end

        if rem(i, mc_params.n_adjust) == 0
            for state in mc_states
                update_max_stepsize!(state,mc_params.n_adjust,a,v,r)
            end
        end        
    end
    #reset counter-variables
    for state in mc_states
        reset_counters(state)
    end


    delta_en_hist,delta_r2 = initialise_histograms!(mc_params,results,ebounds,mc_states[1].config.bc)
    return mc_states,results,delta_en_hist,delta_r2

end

"""

    equilibration(mc_states,move_strat,mc_params,results,pot,ensemble,n_steps,a,v,r,restart)
while initialisation sets mc_states,params etc we require something to thermalise our simulation and set the histograms. This function is mostly a wrapper for the equilibration_cycle! function that optionally removes the thermalisation from restart.


"""
function equilibration(mc_states,move_strat,mc_params,results,pot,ensemble,n_steps,a,v,r,restart)
    if restart == true

        delta_en_hist = (results.en_max - results.en_min) / (results.n_bin - 1)
        delta_r2 = 4*mc_states[1].config.bc.radius2/results.n_bin/5

    else
        mc_states,results,delta_en_hist,delta_r2 = equilibration_cycle!(mc_states,move_strat,mc_params,results,pot,ensemble,n_steps,a,v,r)

    end

    return mc_states,results,delta_en_hist,delta_r2
end


"""
    ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results)

Main function, controlling the parallel tempering MC run.
Calculates number of MC steps per cycle.
Performs equilibration and main MC loop.  
Energy is sampled after `mc_sample` MC cycles. 
Step size adjustment is done after `n_adjust` MC cycles.    
Evaluation: including calculation of inner energy, heat capacity, energy histograms;
saved in `results`.

The booleans control:
save_ham: whether or not to save every energy in a vector, or calculate averages on the fly.
save: whether or not to save the parameters and configurations every 1000 steps
restart: this controls whether to run an equilibration cycle, it additionally requires an integer restartindex which says from which cycle we have restarted the process.
"""

#function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results; save::Bool=true, restart::Bool=false,save_dir = pwd())
function ptmc_run!(input ; restart=false,startfile="input.data",save::Bool=true,save_dir = pwd(), n_config::Int64)

    #first we initialise the simulation with arguments matching the initialise function's various methods
    mc_states,mc_params,move_strat,pot,ensemble,results,start_counter,n_steps,a,v,r = initialisation(restart,input...; startfile=startfile)
    #equilibration thermalises new simulaitons and sets the histograms and results
    mc_states,results,delta_en_hist,delta_r2= equilibration(mc_states,move_strat,mc_params,results,pot,ensemble,n_steps,a,v,r,restart)

   
    println("equilibration done")



    if save == true
        save_states(mc_params,mc_states,0,save_dir,move_strat,ensemble)
    end
    
    save_configs = []
    
    configsfile = open("$save_dir/configs.data", "w+")
    #main MC loop
    for i = start_counter:mc_params.mc_cycles

        @inbounds mc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps,a ,v ,r,results,save,i,save_dir,delta_en_hist,delta_r2)

        if rem(i,n_config) == 0
            for j in length(mc_states)
                position = mc_states[j].config.pos
                push!(save_configs, position)
                #write(configsfile, "$position \n")

            end
        end
    end
    write(configsfile, "$save_configs")
    close(configsfile)
    println("MC loop done.")



    results = finalise_results(mc_states,mc_params,results)
    #TO DO
    # volume (NPT ensemble),rot moves ...
    # move boundary condition from config to mc_params?
    # rdfs

    println("done")
    return 
end
#---------------------------------------------------------#
#-------------Notes for Future Implementation-------------#
#---------------------------------------------------------#
"""

-- TO IMPLEMENT -- 

This version is not complete. While "under the hood" is working as it should, not a lot of effort has been put into:
    - organising the dependencies, properly categorising these is a job for the future.
    - Making the input script order-invariant by making the I/O smarter
    - Organising the keyword arguments to be more intuitive
    - Expanding the initialise functions to set the type of results we wish to collect (eg no RDF, save configs as well as checkpoints)



"""

#---------------------------------------------------------#
end
