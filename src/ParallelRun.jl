module ParallelRun
using Base.Threads
using Distributed
addprocs(nthreads())

@everywhere begin
    using ..BoundaryConditions
    using ..Configurations
    using ..InputParams
    using ..MCMoves
    using ..EnergyEvaluation
    using ..MCRun
    using SharedArrays
    using StaticArray
#using ..RuNNer #this is essential for after the pull request
end

s
# """
#     store_config(mc_states,mc_params)
# A function that finds the lowest energy configuration in a series of MC_State configs and returns it. 
# """
# function store_state(mc_states,mc_params)
#     en_vec = []
#     for idx = 1:mc_params.n_traj
#         push!(en_vec, mc_states[idx].en_tot)
#     end
#     idx = whichmin(en_vec)
#     copy_state = deepcopy(mc_states[idx].config) 

#     return copy_state
# end


# function ptmc_initialise(mc_states, move_strat, mc_params, pot, ensemble,n_bin; num_threads = Threads.nthreads() )
#     #initialise the move frequency
#     a = atom_move_frequency(move_strat)
#     v = vol_move_frequency(move_strat)
#     r = rot_move_frequency(move_strat)

#     #steps per mc_cycle

#     n_steps = a + v + r

#     println("Total number of moves per MC cycle: ", n_steps)
#     println()


#     #set steps to equilibrate and determine the starting configs
#     steps = Int(round( mc_params.eq_cycles/num_threads ))

#     state_vec = []
#     results_vec = []
#     config_vec = []

#     for j = 1:num_threads
#         for i = 1:steps
#             #begin cycles
#             @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
#             #adjust stepsize
#             if rem(i, mc_params.n_adjust) == 0
#                 for i_traj = 1:mc_params.n_traj
#                 update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
#                 end 
#             end
#         end
#         #store the lowest energy config
#         push!(config_vec,store_config(mc_states,mc_params))
#     end
#     # we now want to initialise n_thread mc_state vectors
#     for idx = 1:num_threads

#         states = [MCState(mc_states[i].temp, mc_states[i].beta, config_vec[idx], pot; max_displ=mc_states[i].max_displ ) for i in 1:n_traj]
#         push!(state_vec,states)

#         result = Output{Float64}(n_bin)
#         push!(results_vec,result)

#     end
    
#     return state_vec, results_vec

# end
@everywhere function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r, parallel_run::Bool)
    #perform one MC cycle
    @threads for i_traj = 1:mc_params.n_traj
        for i_step = 1:n_steps
            #mc_states[i_traj] = mc_step!(type_moves[ran][2], type_moves[ran][1], mc_states[i_traj], ran, pot, ensemble)
            @inbounds mc_states[i_traj] = mc_step!(mc_states[i_traj], pot, ensemble, a, v, r)
        end
        #push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
    end
    #parallel tempering
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


@everywhere function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results,parallel_run::Bool; save_ham::Bool = true, )
    if save_ham == false
        #If we do not save the hamiltonian we still need to store the E, E**2 terms at each cycle
        for i_traj = 1:mc_params.n_traj
            push!(mc_states[i_traj].ham, 0)
            push!(mc_states[i_traj].ham, 0)
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
    @sync begin
        for i = 1:mc_params.eq_cycles
            @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r, parallel_run)
            if rem(i, mc_params.n_adjust) == 0
                for i_traj = 1:mc_params.n_traj
                    @async update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
                end 
            end
        end

    end

    for i_traj = 1:mc_params.n_traj
        mc_states[i_traj].count_atom = [0, 0]
        mc_states[i_traj].count_vol = [0, 0]
        mc_states[i_traj].count_rot = [0, 0]
        mc_states[i_traj].count_exc = [0, 0]
    end 
    println("equilibration done")


    for i = 1:mc_params.mc_cycles
        @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r, parallel_run) 
        #sampling step
        MCRun.sampling_step!(mc_params,mc_states,i,save_ham)
        # if rem(i, mc_params.mc_sample) == 0
        #     for i_traj=1:mc_params.n_traj
        #         push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
        #     end
        # end 
        #step adjustment
        if rem(i, mc_params.n_adjust) == 0
            for i_traj = 1:mc_params.n_traj
                update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
            end 
        end
    end

    println("MC loop done.")
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
    T = typeof(mc_states[1].ham[1])
    en_min = T[]
    en_max = T[]
    
    for i_traj in 1:mc_params.n_traj
        push!(en_min,minimum(mc_states[i_traj].ham))
        push!(en_max,maximum(mc_states[i_traj].ham))
    end 
    global_en_min = minimum(en_min)
    global_en_max = maximum(en_max)
    delta_en = (global_en_max - global_en_min) / (results.n_bin - 1)

    results.en_min = global_en_min
    results.en_max = global_en_max


    for i_traj in 1:mc_params.n_traj
        hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
        for en in mc_states[i_traj].ham
            index = floor(Int,(en - global_en_min) / delta_en) + 1
            hist[index] += 1
        end
        push!(results.en_histogram, hist)
    end

    #TO DO
    # volume (NPT ensemble),rot moves ...
    # move boundary condition from config to mc_params?
    # rdfs

    println("done")
    return 
end




end