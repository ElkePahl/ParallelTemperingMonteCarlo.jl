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
    swap_config!(mc_state, trial_configs_all, dist2_mat_new, en_vect_new, en_tot)
        Update the configurations, but this time the whole config, including all coordinates, box length, distance matrix and energy matrix
        If the Metropolis condition is satisfied, these are used to update mc_state. 
"""
#function swap_config!(mc_state, trial_configs_all::mc_states.config, dist2_mat_new::Matrix, en_vec_new::Vector, en_tot)
    #println("swap_config")

    #mc_state.config.pos = trial_configs_all.pos
    #mc_state.config.bc = trial_configs_all.bc
    #mc_state.dist2_mat = dist2_mat_new  
    #mc_state.en_atom_vec = en_vec_new
    #mc_state.en_tot = en_tot

#end

function swap_config_v!(mc_state,trial_config,dist2_mat_new,en_vec_new,en_tot)
    #println("swap_v_config")
    #for i=1:length(mc_state.config.pos)
        #mc_state.config.pos[i] = trial_config.pos[i]
    #end
    #mc_state.config.bc.box_length = copy(trial_config.bc.box_length)
    mc_state.config = Config(trial_config.pos,PeriodicBC(trial_config.bc.box_length))
    mc_state.dist2_mat = dist2_mat_new
    mc_state.en_atom_vec = en_vec_new
    mc_state.en_tot = en_tot
    mc_state.count_vol[1] += 1
    mc_state.count_vol[2] += 1
end


"""
    swap_config!(mc_state, i_atom, trial_pos, dist2_new, new_energy)
        Designed to input one mc_state, the atom to be changed, the trial position, the new distance squared vector and the new energy. 
        If the Metropolis condition is satisfied, these are used to update mc_state. 
"""
function swap_config!(mc_state, i_atom::Int64, trial_pos, dist2_new, energy)
    #println("swap_config_displacement")

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
function acc_test!(ensemble, mc_state, energy::Float64, i_atom::Int64, trial_pos, dist2_new::Vector)
    
    
    if metropolis_condition(ensemble,(energy -mc_state.en_tot), mc_state.beta) >= rand()

        swap_config!(mc_state,i_atom,trial_pos,dist2_new, energy)
    end   
end

"""
acc_test! function for volume change
the acceptance depends on both energy and volume differences
"""

function acc_test!(ensemble::NPT, mc_state, trial_config::Config, dist2_mat_new::Matrix, en_vec_new::Vector, en_tot_new::Float64)


    if metropolis_condition(ensemble, ensemble.n_atoms, (en_tot_new-mc_state.en_tot), trial_config.bc.box_length^3, mc_state.config.bc.box_length^3, mc_state.beta) >= rand()
        #println("accepted")
        #println("swap")

        #swap_config!(mc_state, trial_config, dist2_mat_new, en_vec_new, en_tot_new)
        swap_config_v!(mc_state,trial_config,dist2_mat_new,en_vec_new,en_tot_new)
    end   
end

#function acc_test!(ensemble, mc_state, energy, i_atom, trial_pos, dist2_new::Float64)
    #if metropolis_condition(ensemble,(energy -mc_state.en_tot), mc_state.beta) >= rand()
        #dist2new = [distance2(trial_pos,b,mc_state.config.bc) for b in mc_state.config.pos]

        #swap_config!(mc_state,i_atom,trial_pos,dist2new, energy)
    #end   
#end



"""
    function mc_step!(mc_states,mc_params,pot,ensemble)
        New mc_step function, vectorised displacements and energies are batch-passed to the acceptance test function, which determines whether or not to accept the moves.
"""
function mc_step!(mc_states,move_strat,mc_params,pot,ensemble)

    a,v,r = atom_move_frequency(move_strat),vol_move_frequency(move_strat),rot_move_frequency(move_strat)
    if rand(1:a+v+r)<=a
        #println("d")
        indices,trial_positions = generate_displacements(mc_states,mc_params)
        energy_vector, dist2_new = get_energy(trial_positions,indices,mc_states,pot)
        for idx in eachindex(mc_states)
            @inbounds acc_test!(ensemble,mc_states[idx],energy_vector[idx],indices[idx],trial_positions[idx],dist2_new[idx])
        end
    else
        #println("v")
        trial_configs = generate_vchange(mc_states)   #generate_vchange gives an array of configs
        #get the new distance matrix, energy matrix and total energy for each trajectory
        dist2_mat_new,en_mat_new,en_tot_new = get_energy(trial_configs,pot::AbstractDimerPotential)
        #println(en_tot_new)
        for idx in eachindex(mc_states)
            @inbounds acc_test!(ensemble, mc_states[idx], trial_configs[idx], dist2_mat_new[idx], en_mat_new[idx], en_tot_new[idx])
        end
    end

    return mc_states
    

end

"""
    function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
        Current iteration of mc_cycle! using the vectorised mc_step! followed by an attempted trajectory exchange. Ultimately we will add more move types requiring the move strat to be implemented, but this is presently redundant. 
"""
function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)

    for i_steps = 1:n_steps
        mc_states = mc_step!(mc_states,move_strat,mc_params,pot,ensemble)
    end

    if rand() < 0.1 #attempt to exchange trajectories
        parallel_tempering_exchange!(mc_states,mc_params)
    end


    return mc_states
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
    ( pot; save_dir=pwd() )
Determines the parameters of a fully thermalised set of mc_states. The method involving complete parameters assumes we begin our simulation from the same set of mc_states. In theory we could pass it one single mc_state which it would then duplicate, passing much more responsibility on to this function. An idea to discuss in future. 
Second method "equilibrates" the simulation by reading a complete checkpoint, and returning all required parameters for a ptmc run

outputs are: thermalised states(mc_states),initialised results(results),the histogram stepsize(delta_en_hist),rdf histsize(delta_r2),starting step for restarts(start_counter),n_steps,a,v,r

"""
function equilibration_cycle!(mc_states,move_strat,mc_params,results,pot,ensemble)


    a,v,r = atom_move_frequency(move_strat),vol_move_frequency(move_strat),rot_move_frequency(move_strat)
    n_steps = a + v + r

    println("Total number of moves per MC cycle: ", n_steps)
    println()

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

    start_counter = 1

    println("equilibration done")

    return mc_states,move_strat,ensemble,results,delta_en_hist,delta_r2,start_counter,n_steps,a,v,r

end
function equilibration( pot; save_dir=pwd() )

    results,ensemble,move_strat,mc_params,mc_states,step = restart_ptmc(pot,directory=save_dir)

    a,v,r = atom_move_frequency(move_strat),vol_move_frequency(move_strat),rot_move_frequency(move_strat)
    n_steps = a + v + r

    delta_en_hist = (results.en_max - results.en_min) / (results.n_bin - 1)
    delta_r2 = 4*mc_states[1].config.bc.radius2/results.n_bin/5

    return mc_states,move_strat,ensemble,results,delta_en_hist,delta_r2,step,n_steps,a,v,r

end

"""
    function ptmc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i ;delta_en=0. ) 
functionalised the main body of the ptmc_run! code. Runs a single mc_state, samples the results, updates the histogram and writes the savefile if necessary.
"""
function ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save, i,save_dir,delta_en_hist,delta_r2)


    mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot,  ensemble, n_steps, a, v, r) 

    sampling_step!(mc_params,mc_states,ensemble,i,results,delta_en_hist,delta_r2)
    
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
function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results; save_ham::Bool = false, save::Bool=true, restart::Bool=false,save_dir = pwd())



  
    mc_states,move_strat,ensemble,results,delta_en_hist,delta_r2,start_counter,n_steps,a,v,r = equilibration_cycle!(mc_states,move_strat,mc_params,results,pot,ensemble)

   
    println("equilibration done")




    if save == true
        save_states(mc_params,mc_states,0,save_dir,move_strat,ensemble)
    end
    
    #main MC loop

    for i = start_counter:mc_params.mc_cycles
        @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r,save,i,save_dir,delta_en_hist,delta_r2)
    end

    

 
    println("MC loop done.")
    #Evaluation
    #average energy
    n_sample = mc_params.mc_cycles / mc_params.mc_sample

    # if save_ham == true
    #     en_avg = [sum(mc_states[i_traj].ham) / n_sample for i_traj in 1:mc_params.n_traj] #floor(mc_cycles/mc_sample)
    #     en2_avg = [sum(mc_states[i_traj].ham .* mc_states[i_traj].ham) / n_sample for i_traj in 1:mc_params.n_traj]
    # else
    en_avg = [mc_states[i_traj].ham[1] / n_sample  for i_traj in 1:mc_params.n_traj]
    en2_avg = [mc_states[i_traj].ham[2] / n_sample  for i_traj in 1:mc_params.n_traj]
    #end


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
