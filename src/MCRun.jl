module MCRun


export metropolis_condition, mc_step!, mc_cycle!,ptmc_cycle!, ptmc_run!,save_states,save_params,save_results
export atom_move!,update_max_stepsize!
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


"""
    update_max_stepsize!(mc_state::MCState, n_update, a, v, r)
Increases/decreases the max. displacement of atom, volume, and rotation moves to 110%/90% of old values
if acceptance rate is >60%/<40%. Acceptance rate is calculated after `n_update` MC cycles; 
each cycle consists of `a` atom, `v` volume and `r` rotation moves.
Information on actual max. displacement and accepted moves between updates is contained in `mc_state`, see [`MCState`](@ref).  
"""
function update_max_stepsize!(mc_state::MCState, n_update, a, v, r; min_acc = 0.4, max_acc = 0.6)
    #atom moves
    acc_rate = mc_state.count_atom[2] / (n_update * a)
    if acc_rate < min_acc
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0
    #volume moves
    if v > 0
        acc_rate = mc_state.count_vol[2] / (n_update * v)
        if acc_rate < min_acc
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0
    end
    #rotation moves
    if r > 0
        acc_rate = mc_state.count_rot[2] / (n_update * r)
        if acc_rate < min_acc
            mc_state.max_displ[3] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[3] *= 1.1
        end
        mc_state.count_rot[2] = 0
    end
    return mc_state
end



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
        Current iteration of mc_cycle! using the vectorised mc_step! followed by an attempted trajectory exchange. Ultimately we will add more move types requiring the move strat to be implemented, but this is presently redundant. 
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

"""
    function ptmc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i ;delta_en=0. ) 
functionalised the main body of the ptmc_run! code. Runs a single mc_state, samples the results, updates the histogram and writes the savefile if necessary.
"""
function ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save, i,save_dir,delta_en_hist,delta_r2)


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

function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results; save_ham::Bool = false, save::Bool=true, restart::Bool=false,restartindex::Int64=0,save_dir = pwd())


    
    
    
            #If we do not save the hamiltonian we still need to store the E, E**2 terms at each cycle
    for i_traj = 1:mc_params.n_traj
        push!(mc_states[i_traj].ham, 0)
        push!(mc_states[i_traj].ham, 0)
    end

   

    a = atom_move_frequency(move_strat)
    v = vol_move_frequency(move_strat)
    r = rot_move_frequency(move_strat)
    #number of MC steps per MC cycle
    n_steps = a + v + r

    println("Total number of moves per MC cycle: ", n_steps)
    println()

    
    #equilibration cycle
    #if restart == false
        #this initialises the max and min energies for histograms
        
        ebounds = [100. , -100.] #emin,emax
       
        
        for i = 1:mc_params.eq_cycles
            @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
            #verbose way to save the highest and lowest energies
                for i_traj = 1:mc_params.n_traj
                    if mc_states[i_traj].en_tot < ebounds[1]
                        ebounds[1] = mc_states[i_traj].en_tot
                    end

                    if mc_states[i_traj].en_tot > ebounds[2]
                        ebounds[2] = mc_states[i_traj].en_tot
                    end
                end
            #update stepsizes

            if rem(i, mc_params.n_adjust) == 0
                for i_traj = 1:mc_params.n_traj
                    update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
                end 
            end
        end
    #re-set counter variables to zero
        for i_traj = 1:mc_params.n_traj
            mc_states[i_traj].count_atom = [0, 0]
            mc_states[i_traj].count_vol = [0, 0]
            mc_states[i_traj].count_rot = [0, 0]
            mc_states[i_traj].count_exc = [0, 0]
        end
        #initialise histogram for non-saving hamiltonian 
        #if save_ham == false

            delta_en_hist,delta_r2 = initialise_histograms!(mc_params,results,ebounds,mc_states[1].config.bc)

        #end

        println("equilibration done")


        if save == true
            save_states(mc_params,mc_states,0,save_dir)
        end
    #end



    #main MC loop

    #if restart == false

    for i = 1:mc_params.mc_cycles
        @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r,save,i,save_dir,delta_en_hist,delta_r2)
    end

    # else #if restarting

    #     for i = restartindex:mc_params.mc_cycles
    #         @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r,save,i,save_dir,delta_en_hist,delta_r2)
    #     end 
    # end
    println("MC loop done.")
    #Evaluation
    #average energy
    n_sample = mc_params.mc_cycles / mc_params.mc_sample

    if save_ham == true
        en_avg = [sum(mc_states[i_traj].ham) / n_sample for i_traj in 1:mc_params.n_traj] #floor(mc_cycles/mc_sample)
        en2_avg = [sum(mc_states[i_traj].ham .* mc_states[i_traj].ham) / n_sample for i_traj in 1:mc_params.n_traj]
    else
        en_avg = [mc_states[i_traj].ham[1] / n_sample  for i_traj in 1:mc_params.n_traj]
        en2_avg = [mc_states[i_traj].ham[2] / n_sample  for i_traj in 1:mc_params.n_traj]
    end


    results.en_avg = en_avg

    #heat capacity
    results.heat_cap = [(en2_avg[i]-en_avg[i]^2) * mc_states[i].beta 
    
    for i in 1:mc_params.n_traj]

    #acceptance statistics
    results.count_stat_atom = [mc_states[i_traj].count_atom[1] / (mc_params.n_atoms * mc_params.mc_cycles) for i_traj in 1:mc_params.n_traj]
    results.count_stat_exc = [mc_states[i_traj].count_exc[2] / mc_states[i_traj].count_exc[1] for i_traj in 1:mc_params.n_traj]

    println(results.heat_cap)

    #energy histograms

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
- possible struct containing params relating to the simulation such as min/max acceptance rate, save/checkpoint frequency and whether to store RDF info. 
- equilibration should have its own function with variable methods for restart etc.


"""

#---------------------------------------------------------#
end
