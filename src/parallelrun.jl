module ParallelRun

using Distributed

    using StaticArrays,DelimitedFiles

    using ..BoundaryConditions
    using ..Configurations
    using ..InputParams
    using ..MCMoves
    using ..EnergyEvaluation
    using ..RuNNer
    using ..MCRun

    import ..MCRun: initialise_histograms!,updatehistogram!,update_max_stepsize!,sampling_step!,save_results,save_states
    import .MCRun.mc_cycle!

    export ParallelRun,mc_cycle!,pptmc_cycle,parallel_equilibration


"""
    parallel_equilibration(mc_states,move_strat,mc_params,pot,ensemble,results)
function takes as input a vector of mc_states and the standard MC parameters. It runs n_eq times saving one configuration as an initial state for one entire thread's vector of states. It returns parallel_states, a vector of vectors of states all initialised with the same energies and parameters, but sharing a common histogram and results vector. 
"""
function parallel_equilibration(mc_states,move_strat,mc_params,pot,ensemble,results)
    parallel_states = []
    pot_vector = []

     #initialise state and potentials

    n_threads = Threads.nthreads()
    sample_index = Int64(floor(mc_params.eq_cycles / n_threads)) #number of eq cycles per thread

    a = atom_move_frequency(move_strat)
    v = vol_move_frequency(move_strat)
    r = rot_move_frequency(move_strat)# function init_parallel_RuNNer(pot::AbstractMLPotential; n_threads = Threads.nthreads())
    n_steps = a + v + r
    println()

    ebounds = [100. , -100.]

    
    for i_thread = 1:n_threads
        for i_eq = 1:sample_index
            
            i = i_thread*sample_index + i_eq

            mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)#mc cycle
            
            for i_traj = 1:mc_params.n_traj#check energy bounds
                if mc_states[i_traj].en_tot < ebounds[1]
                    ebounds[1] = mc_states[i_traj].en_tot
                end

                if mc_states[i_traj].en_tot > ebounds[2]
                    ebounds[2] = mc_states[i_traj].en_tot
                end

            end

            if rem(i, mc_params.n_adjust) == 0 #adjust stepsize
                for i_traj = 1:mc_params.n_traj
                    update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
                end 
            end

        end
        temp_pot = ParallelMLPotential(pot.dir,pot.atomtype,i_thread)
        push!(pot_vector,temp_pot)

        states_vec = [MCState(mc_states[i_traj].temp,mc_states[i_traj].beta,mc_states[i_traj].config,pot_vector[i_thread];max_displ = mc_states[i_traj].max_displ ) for i_traj in 1:mc_params.n_traj] #initialise a new mc_states vector based on current state

        for i_traj = 1:mc_params.n_traj
            push!(states_vec[i_traj].ham, 0)
            push!(states_vec[i_traj].ham, 0)
        end

        push!(parallel_states,states_vec) #add to vector of parallel states


    end
    delta_en = initialise_histograms!(mc_params,mc_states,results, full_ham=false,e_bounds=ebounds) #start histogram

    println("equilibration done")
    
    return parallel_states,pot_vector,a,v,r,delta_en,n_threads

end

 function threadexchange!(parallel_states,n_threads,idx)
    if rand() < 0.2  #20% change per trajectory of an attempted exchange
        thrid = rand(1:n_threads,2)
        if thrid[1] == thrid[2] && thrid[2] == n_threads
            thrid[2] = rand(1:n_threads-1)
        elseif thrid[1] == thrid[2] && thrid[2] != n_threads
            thrid[2] +=1
        end #which threads are talking

        exc_acc = exc_acceptance(parallel_states[thrid[1]][idx].beta,parallel_states[thrid[2]][idx].beta,parallel_states[thrid[1]][idx].en_tot,parallel_states[thrid[2]][idx].en_tot) #calc acceptance
        
        if exc_acc > rand() #if exchange is likely
            exc_trajectories!(parallel_states[thrid[1]][idx] ,parallel_states[thrid[2]][idx] )#swap
        end
     end
    
end
"""
    function mc_cycle!(mc_states, move_strat, mc_params, pot::ParallelMLPotential, ensemble, n_steps, a, v, r)
updated version of the mc_cycle! function for parallelised RuNNer versions
"""
function mc_cycle!(mc_states, move_strat, mc_params, pot::ParallelMLPotential, ensemble, n_steps, a, v, r)
    file = RuNNer.writeinit(pot.dir;input_idx=pot.index)
    #this for loop creates n_traj perturbed atoms
    indices = []
    trials = []
    #we require parallelisation here, but will need to avoid a race condition
    for mc_state in mc_states
        #for i_step = 1:n_steps
            ran = rand(1:(a+v+r))
            trial_pos = atom_displacement(mc_state.config.pos[ran], mc_state.max_displ[1], mc_state.config.bc)
            writeconfig(file,mc_state.config,ran,trial_pos, pot.atomtype)
            push!(indices,ran)
            push!(trials,trial_pos)
        #end
    end
    #after which we require energy evaluations of the n_traj new configurations
    close(file)    
    energyvec = getRuNNerenergy(pot.dir,mc_params.n_traj; input_idx=pot.index)    
    #this replaces the atom_move! function
    #parallelisation here is fine
    for i in 1:mc_params.n_traj
        if metropolis_condition(ensemble, (mc_states[i].en_tot - energyvec[i]), mc_states[i].beta ) >=rand()
            mc_states[i].config.pos[indices[i]] = trials[i]
            mc_states[i].en_tot = energyvec[i]
            mc_states[i].count_atom[1] +=1
            mc_states[i].count_atom[2] += 1
        end
    end


    if rand() < 0.1 #attempt to exchange trajectories
        n_exc = rand(1:mc_params.n_traj-1)
        mc_states[n_exc].count_exc[1] += 1
        mc_states[n_exc+1].count_exc[1] += 1
        exc_acc = exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot)
        if exc_acc > rand()
            mc_states[n_exc].count_exc[2] += 1
            mc_states[n_exc+1].count_exc[2] += 1
            mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
        end
    end


    return mc_states
end

function pptmc_cycle(parallel_states,mc_params,results,move_strat,pot_vector,ensemble,n_threads,delta_en,n_steps,a,v,r)
    # for i = 1:500

        Threads.@threads for threadindex = 1:n_threads

            for i = 1:100
                ptmc_cycle!(parallel_states[threadindex],results,move_strat,mc_params,pot_vector[threadindex],ensemble,n_steps,a,v,r,false,false,i,pwd();delta_en=delta_en) #force save = false
            end

        end
        #we run 500 mc cycles per thread
    #end

    for idx = 1:mc_params.n_traj
        threadexchange!(parallel_states,n_threads,idx)
    end
    #then run n_traj exchanges

    return parallel_states
end

function pptmc_run!(mc_states,move_strat,mc_params,pot,ensemble,results)

    parallel_states,pot_vector,a,v,r,delta_en,n_threads=parallel_equilibration(mc_states,move_strat,mc_params,pot,ensemble,results)
    n_steps = a+v+r
    println("equilibration complete")
    n_run_per_thread = Int64(floor(mc_params.mc_cycles / n_threads / 500)) #this gives us the number of pptmc threads to run, 500 per thread per cycle
    n_sample = 500*n_run_per_thread

    for run_index = 1:n_run_per_thread
        parallel_states = pptmc_cycle(pparallel_states,mc_params,results,move_strat,pot_vector,ensemble,n_threads,delta_en,n_steps,a,v,r)

        
    end
    println("main loop complete")

    println("beginning statistics")

    total_en_avg = zeros(mc_params.n_traj)
    total_en2_avg = zeros(mc_params.n_traj)
    for mc_states in parallel_states
        en_avg = [mc_states[i_traj].ham[1] / n_sample  for i_traj in 1:mc_params.n_traj]
        en2_avg = [mc_states[i_traj].ham[2] / n_sample  for i_traj in 1:mc_params.n_traj]

        total_en_avg += en_avg
        total_en2_avg += en2_avg
    end

    total_en_avg = total_en_avg ./ n_threads
    total_en2_avg = total_en2_avg ./ n_threads
    results.en_avg = en_avg

    results.heat_cap = [(en2_avg[i]-en_avg[i]^2) * mc_states[i].beta for i in 1:mc_params.n_traj]

    println("done")
    return 

end

end