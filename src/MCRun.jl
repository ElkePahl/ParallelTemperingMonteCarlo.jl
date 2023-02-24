module MCRun

#export MCState
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
    swap_var_function(mc_state, i_atom, trial_pos, dist2_new, new_energy)
        Designed to input one mc_state, the atom to be changed, the trial position, the new distance squared vector and the new energy. 
        If the Metropolis condition is satisfied, these are used to update mc_state. 
"""
function swap_var_function!(mc_state, i_atom, trial_pos, dist2_new, energy)
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

        The acc_test function works in tandem with the swap_var_function, only adding the metropolis condition. Separate functions was benchmarked as very marginally faster. The method for a float64 only calculates the dist2 vector if it's required, as for RuNNer
"""
function acc_test!(ensemble, mc_state, energy, i_atom, trial_pos, dist2_new::Vector)
    
    
        if metropolis_condition(ensemble,(energy -mc_state.en_tot), mc_state.beta) >= rand()
            swap_var_function!(mc_state,i_atom,trial_pos,dist2_new, energy)
        end   
end
function acc_test!(ensemble, mc_state, energy, i_atom, trial_pos, dist2_new::Float64)
    
    
    if metropolis_condition(ensemble,(energy -mc_state.en_tot), mc_state.beta) >= rand()

        dist2_new = [distance2(trial_pos,b) for b in mc_state.config.pos]

        swap_var_function!(mc_state,i_atom,trial_pos,dist2_new, energy)
    end   
end
"""
    parallel_tempering_exchange!(mc_states,mc_params)
This function takes a vector of mc_states as well as the parameters of the simulation and attempts to swap two trajectories according to the parallel tempering method. 
"""
function parallel_tempering_exchange!(mc_states,mc_params)
    n_exc = rand(1:mc_params.n_traj-1)

    mc_states[n_exc].count_exc[1] += 1
    mc_states[n_exc+1].count_exc[1] += 1

    

    if exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot) > rand()
        mc_states[n_exc].count_exc[2] += 1
        mc_states[n_exc+1].count_exc[2] += 1

        mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
    end

    return mc_states
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

    sampling_step(mc_params, mc_states, save_index, saveham::Bool)
A function to store the information at the end of an MC_Cycle, replacing the manual if statements previously in PTMC_run. 
"""
function sampling_step!(mc_params,mc_states,save_index, saveham::Bool)  
        if rem(save_index, mc_params.mc_sample) == 0
            for indx_traj=1:mc_params.n_traj
                if saveham == true
                    push!(mc_states[indx_traj].ham, mc_states[indx_traj].en_tot) #to build up ham vector of sampled energies
                else
                    mc_states[indx_traj].ham[1] += mc_states[indx_traj].en_tot
                    #add E,E**2 to the correct positions in the hamiltonian
                    mc_states[indx_traj].ham[2] += (mc_states[indx_traj].en_tot*mc_states[indx_traj].en_tot)
                end
            end
        end 
end
"""

    initialise_histograms!(mc_params,results,T)
functionalised the step in which we build the energy histograms  
"""
function initialise_histograms!(mc_params,mc_states,results; full_ham = true,e_bounds = [0,0])    
    T = typeof(mc_states[1].en_tot)
    en_min = T[]
    en_max = T[]

    r_max = 4*mc_states[1].config.bc.radius2 #we will work in d^2
    delta_r = r_max/results.n_bin/5 #we want more bins for the RDFs

    if full_ham == true
        for i_traj in 1:mc_params.n_traj
            push!(en_min,minimum(mc_states[i_traj].ham))
            push!(en_max,maximum(mc_states[i_traj].ham))
        end
    
        global_en_min = minimum(en_min)
        global_en_max = maximum(en_max)
    else

        #we'll give ourselves a 6% leeway here
        global_en_min = e_bounds[1] - abs(0.03*e_bounds[1])
        global_en_max = e_bounds[2] + abs(0.03*e_bounds[2])
    end

    for i_traj = 1:mc_params.n_traj
        histogram = zeros(results.n_bin + 2)
        push!(results.en_histogram, histogram)
        RDF = zeros(results.n_bin*5)
        push!(results.rdf,RDF)
    end
    

    delta_en_hist = (global_en_max - global_en_min) / (results.n_bin - 1)


    results.en_min = global_en_min
    results.en_max = global_en_max

  
        return  delta_en_hist
end
"""
    updaterdf!(mc_states,results,delta_r2)
For each state in a vector of mc_states, we use the distance squared matrix to determine which bin (between zero and 2*r_bound) the distance falls into, we then update results.rdf[bin] to build the radial distribution function
"""
function updaterdf!(mc_states,results,delta_r2)
    for j_traj in eachindex(mc_states)
        for element in mc_states[j_traj].dist2_mat 
            rdf_index=floor(Int,(element/delta_r2))
            if rdf_index != 0
                results.rdf[j_traj][rdf_index] +=1
            end
        end
    end
end
"""
    updatehistogram!(mc_params,mc_states,results,delta_en_hist ; fullham=true)
Performed either at the end or during the mc run according to fullham=true/false (saved all datapoints or calculated on the fly). Uses the energy bounds and the previously defined delta_en_hist to calculate the bin in which te current energy value falls for each trajectory. This is used to build up the energy histograms for post-analysis.
"""
function updatehistogram!(mc_params,mc_states,results,delta_en_hist ; fullham=true)

    for update_traj_index in 1:mc_params.n_traj
        
        if fullham == true #this is done at the end of the cycle

            hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
            for en in mc_states[update_traj_index].ham
                hist_index = floor(Int,(en - results.en_min) / delta_en_hist) + 1
                hist[hist_index] += 1

            end
        push!(results.en_histogram, hist)

        else #this is done throughout the simulation

            en = mc_states[update_traj_index].en_tot

            hist_index = floor(Int,(en - results.en_min) / delta_en_hist) + 1 

            if hist_index < 1 #if energy too low
                results.en_histogram[update_traj_index][1] += 1 #add to place 1
            elseif hist_index > results.n_bin #if energy too high
                results.en_histogram[update_traj_index][(results.n_bin +2)] += 1 #add to place n_bin +2
            else
                results.en_histogram[update_traj_index][(hist_index+1)] += 1
            end

        end
    end

end
"""
    function ptmc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i ;delta_en=0. ) 
functionalised the main body of the ptmc_run! code. Runs a single mc_state, samples the results, updates the histogram and writes the savefile if necessary.
"""
function ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir ;delta_en_hist=0.)


    mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot,  ensemble, n_steps, a, v, r) 
    #sampling step
    sampling_step!(mc_params,mc_states,i,save_ham)

    if save_ham == false
        updatehistogram!(mc_params,mc_states,results,delta_en_hist,fullham=save_ham)
        updaterdf!(mc_states,results,(4*mc_states[1].config.bc.radius2/(results.n_bin*5)))

    end

    #step adjustment
    if rem(i, mc_params.n_adjust) == 0
        for i_traj = 1:mc_params.n_traj
            update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
        end 
    end

    if save == true
        if rem(i,1000) == 0

            save_states(mc_params,mc_states,i,save_dir)
            if save_ham == false
                save_results(results)
            end
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

function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results; save_ham::Bool = true, save::Bool=true, restart::Bool=false,restartindex::Int64=0,save_dir = pwd())

    #restart isn't compatible with saving the hamiltonian at the moment

    if restart == true
        save_ham = false
    end
    
    if save_ham == false
        if restart == false
            #If we do not save the hamiltonian we still need to store the E, E**2 terms at each cycle
            for i_traj = 1:mc_params.n_traj
                push!(mc_states[i_traj].ham, 0)
                push!(mc_states[i_traj].ham, 0)
            end

        end
    end

    a = atom_move_frequency(move_strat)
    v = vol_move_frequency(move_strat)
    r = rot_move_frequency(move_strat)
    #number of MC steps per MC cycle
    n_steps = a + v + r

    println("Total number of moves per MC cycle: ", n_steps)
    println()

    
    #equilibration cycle
    if restart == false
        #this initialises the max and min energies for histograms
        if save_ham == false
            ebounds = [100. , -100.] #emin,emax
        end
        
        for i = 1:mc_params.eq_cycles
            @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
            #verbose way to save the highest and lowest energies

            if save_ham == false
                for i_traj = 1:mc_params.n_traj
                    if mc_states[i_traj].en_tot < ebounds[1]
                        ebounds[1] = mc_states[i_traj].en_tot
                    end

                    if mc_states[i_traj].en_tot > ebounds[2]
                        ebounds[2] = mc_states[i_traj].en_tot
                    end
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
        if save_ham == false

            delta_en_hist = initialise_histograms!(mc_params,mc_states,results, full_ham=false,e_bounds=ebounds)

        end

        println("equilibration done")


        if save == true
            save_states(mc_params,mc_states,0,save_dir)
        end
    end



    #main MC loop

    if restart == false

        for i = 1:mc_params.mc_cycles
            if save_ham == false

                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir;delta_en_hist=delta_en_hist)
            else
                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir)
            end

        end

    else #if restarting

        for i = restartindex:mc_params.mc_cycles
            if save_ham == false

                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir;delta_en_hist=delta_en_hist)
            else
                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir)
            end
            

        end 
    end
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
    results.heat_cap = [(en2_avg[i]-en_avg[i]^2) * mc_states[i].beta for i in 1:mc_params.n_traj]

    #acceptance statistics
    results.count_stat_atom = [mc_states[i_traj].count_atom[1] / (mc_params.n_atoms * mc_params.mc_cycles) for i_traj in 1:mc_params.n_traj]
    results.count_stat_exc = [mc_states[i_traj].count_exc[2] / mc_states[i_traj].count_exc[1] for i_traj in 1:mc_params.n_traj]

    println(results.heat_cap)

    #energy histograms
    if save_ham == true
        # T = typeof(mc_states[1].ham[1])

        delta_en_hist= initialise_histograms!(mc_params,mc_states,results)
        updatehistogram!(mc_params,mc_states,results,delta_en_hist)

    
    end
    #     for i_traj in 1:mc_params.n_traj
    #         hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
    #         for en in mc_states[i_traj].ham
    #             index = floor(Int,(en - global_en_min) / delta_en) + 1
    #             hist[index] += 1
    #         end
    #         push!(results.en_histogram, hist)
    #     end
    # end

    #TO DO
    # volume (NPT ensemble),rot moves ...
    # move boundary condition from config to mc_params?
    # rdfs

    println("done")
    return 
end

end
