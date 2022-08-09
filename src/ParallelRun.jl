module ParallelRun

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..MCRun
#using ..RuNNer #this is essential for after the pull request

using StaticArrays

function store_config(mc_states,mc_params)
    en_vec = []
    for idx = 1:mc_params.n_traj
        push!(en_vec, mc_states[idx].en_tot)
    end
    idx = whichmin(en_vec)
    copy_config = deepcopy(mc_states[idx].config) 

    return copy_config
end


function ptmc_initialise(mc_states, move_strat, mc_params, pot, ensemble, results; num_threads = Threads.nthreads() )
    #initialise the move frequency
    a = atom_move_frequency(move_strat)
    v = vol_move_frequency(move_strat)
    r = rot_move_frequency(move_strat)

    #steps per mc_cycle

    n_steps = a + v + r

    println("Total number of moves per MC cycle: ", n_steps)
    println()

    #set steps to equilibrate and determine the starting configs
    steps = Int(round( mc_params.eq_cycles/num_threads ))

    initial_configuration = []

    for j = 1:num_threads
        for i = 1:steps
            #begin cycles
            @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
            #adjust stepsize
            if rem(i, mc_params.n_adjust) == 0
                for i_traj = 1:mc_params.n_traj
                update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
                end 
            end
        end
        #store the lowest energy config
        push!(initial_configuration,store_config(mc_states,mc_params))
    end
    
    for i_traj = 1:mc_params.n_traj
        mc_states[i_traj].count_atom = [0, 0]
        mc_states[i_traj].count_vol = [0, 0]
        mc_states[i_traj].count_rot = [0, 0]
        mc_states[i_traj].count_exc = [0, 0]
    end 

    thread_states = []
    for idx = 1:num_threads
        
    end

end



end