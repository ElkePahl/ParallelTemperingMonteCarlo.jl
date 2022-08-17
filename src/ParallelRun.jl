module ParallelRun
using Threads
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
#using ..RuNNer #this is essential for after the pull request
end

using StaticArrays
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
        @threads for i = 1:mc_params.eq_cycles
            @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
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


    @threads for i = 1:mc_params.mc_cycles
        @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r) 
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




end