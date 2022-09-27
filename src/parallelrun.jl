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

"""
    parallel_equilibration(mc_states,move_strat,mc_params,pot,ensemble,results)
function takes as input a vector of mc_states and the standard MC parameters. It runs n_eq times saving one configuration as an initial state for one entire thread's vector of states. It returns parallel_states, a vector of vectors of states all initialised with the same energies and parameters, but sharing a common histogram and results vector.
"""
function parallel_equilibration(mc_states,move_strat,mc_params,pot,ensemble,results)
    parallel_states = []

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
        
        states_vec = [MCState(mc_states[i_traj].temp,mc_states[i_traj].beta,mc_states[i_traj].config,pot_vector[i_thread];max_displ = mc_states[i_traj].max_displ ) for i_traj in 1:mc_params.n_traj] #initialise a new mc_states vector based on current state

        for i_traj = 1:mc_params.n_traj
            push!(states_vec[i_traj].ham, 0)
            push!(states_vec[i_traj].ham, 0)
        end

        push!(parallel_states,states_vec) #add to vector of parallel states


    end
    delta_en = initialise_histograms!(mc_params,mc_states,results, full_ham=false,e_bounds=ebounds) #start histogram

    println("equilibration done")
    
    return parallel_states,a,v,r,delta_en,n_threads

end
"""
    threadexchange!(parallel_states,n_threads,idx)
function takes a series of parallel states along with a number of threads and exchanges two of them randomly. 
"""
@everywhere function threadexchange!(parallel_states,n_threads,idx)
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





end