module ParallelRun

using Distributed
@everywhere begin
    using StaticArrays,DelimitedFiles

    using ..BoundaryConditions
    using ..Configurations
    using ..InputParams
    using ..MCMoves
    using ..EnergyEvaluation
    using ..RuNNer
    using ..MCRun

    import ..MCRun: initialise_histograms!,updatehistogram!,update_max_stepsize!,sampling_step!,save_results,save_states
    
end

function pptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results; save_ham::Bool = true, save::Bool=true, restart::Bool=false,restartindex::Int64=0)
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
        
        Threads.@threads for i = 1:mc_params.eq_cycles
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
            delta_en = initialise_histograms!(mc_params,mc_states,results, full_ham=false,e_bounds=ebounds)
        end

        println("equilibration done")
        

        if save == true
            save_states(mc_params,mc_states,0)
        end
    end


    #main MC loop

    if restart == false

        Threads.@threads for i = 1:mc_params.mc_cycles
            if save_ham == false
                ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i;delta_en=delta_en)
            else
                ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i)
            end
            
        end

    else #if restarting

        Threads.@threads for i = restartindex:mc_params.mc_cycles
            if save_ham == false
                ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i;delta_en=delta_en)
            else
                ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i)
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
        delta_en= initialise_histograms!(mc_params,mc_states,results)
        updatehistogram!(mc_params,mc_states,results,delta_en)
    
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