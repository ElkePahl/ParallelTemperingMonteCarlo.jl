module MCRun

export MCState
export metropolis_condition, mc_step!, mc_cycle!, ptmc_run!
export atom_move!
export exc_acceptance, exc_trajectories!

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation

mutable struct MCState{T,N,BC}
    temp::T
    beta::T
    config::Config{N,BC,T}
    dist2_mat::Matrix{T}
    en_atom_mat::Vector{T}
    en_tot::T
    ham::Vector{T}
    max_displ::Vector{T}
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_rot::Vector{Int}
    count_exc::Vector{Int}
end    

function MCState(temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_mat, en_tot, ham; max_displ=[0.1,0.1,1.] ,count_atom=[0,0], count_vol=[0,0], count_rot=[0,0], count_exc=[0,0]) where {T,N,BC}
    MCState{T,N,BC}(temp,beta,config,dist2_mat,en_atom_mat,en_tot,ham,max_displ,count_atom,count_vol,count_rot,count_exc)
end

function MCState(temp, beta, config::Config{N,BC,T}, ham, max_displ; count_atom=[0,0], count_vol=[0,0], count_rot=[0,0], count_exc=[0,0]) where {T,N,BC}
    dist2_mat = get_distance2_mat(config)
    en_atom_mat, en_tot = dimer_energy_config(dist2_mat_0, n_atoms, pot)
    MCState{T,N,BC}(temp,beta,config,dist2_mat,en_atom_mat,en_tot,ham,max_displ,count_atom,count_vol,count_rot,count_exc)
end

"""
    metropolis_condition(energy_unmoved, energy_moved, beta)

Determines probability to accept a MC move at inverse temperature beta, takes energies of new and old configurations
"""
function metropolis_condition(energy_unmoved, energy_moved, beta)
    prob_val = exp(-(energy_moved-energy_unmoved)*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function metropolis_condition(::NVT, delta_en, beta)
    prob_val = exp(-delta_en*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function exc_acceptance(beta_1,beta_2,en_1,en_2)
    delta_en = en_1 - en_2
    delta_beta = beta_1 - beta_2
    exc_acc=min(1.0,exp(delta_beta*delta_en))
    return exc_acc
end

function exc_trajectories!(state_1, state_2)
    state_1.config, state_2.config = state_2.config, state_1.config
    state_1.dist2_mat, state_2.dist2_mat = state_2.dist2_mat, state_1.dist2_mat
    state_1.en_atom_mat, state_2.en_atom_mat = state_2.en_atom_mat, state_1.en_atom_mat
    state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot
    return state_1, state_2
end 

function update_max_stepsize!(mc_state, n_update, a, v, r)
    #atom moves
    acc_rate = mc_state.count_atom[2]/ (n_update * a)
    if acc_rate < 0.4
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > 0.6
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0
    #volume moves
    if v>0
        acc_rate = mc_state.count_vol[2]/ (n_update * v)
        if acc_rate < 0.4
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > 0.6
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0
    end
    #rotation moves
    if r>0
        acc_rate = mc_state.count_rot[2]/ (n_update * r)
        if acc_rate < 0.4
            mc_state.max_displ[3] *= 0.9
        elseif acc_rate > 0.6
            mc_state.max_displ[3] *= 1.1
        end
        mc_state.count_rot[2] = 0
    end
    return mc_state
end
    
#    for i in 1:length(count_acc)
#       acc_rate =  count_accept[i] / (displ.update_step * n_atom)
#        if acc_rate < 0.4
#            displ.max_displacement[i] *= 0.9
#        elseif acc_rate > 0.6
#            displ.max_displacement[i] *= 1.1
#        end
#        count_accept[i] = 0
#    end
#    return displ, count_accept
#end

#function mc_step_atom!(config, beta, dist2_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adjust, pot)
    #move randomly selected atom (obeying the boundary conditions)
    #trial_pos = atom_displacement(config.pos[i_atom], max_displacement, config.bc)
    #find new distances of moved atom - might not be always needed?
    #dist2_new = [distance2(trial_pos,b) for b in config.pos]
    #en_moved = energy_update(i_atom, dist2_new, pot)
    #recalculate old 
    #en_unmoved = energy_update(i_atom, dist2_mat[i_atom,:], pot)
    #one might want to store dimer energies per atom in vector?
    #decide acceptance
    #if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        #new config accepted
    #    config.pos[i_atom] = copy(trial_pos)
    #    dist2_mat[i_atom,:] = copy(dist2_new)
    #    dist2_mat[:,i_atom] = copy(dist2_new)
    #    en_tot = en_tot - en_unmoved + en_moved
    #    count_acc += 1
    #    count_acc_adjust += 1
    #end 
    #return config, entot, dist2mat, count_acc, count_acc_adjust
#end

function atom_move!(mc_state::MCState, i_atom, pot, ensemble)
    #move randomly selected atom (obeying the boundary conditions)
    trial_pos = atom_displacement(mc_state.config.pos[i_atom], mc_state.max_displ[1], mc_state.config.bc)
    #find new distances of moved atom 
    delta_en, dist2_new = energy_update(trial_pos, i_atom, mc_state.config, mc_state.dist2_mat, pot)
    #dist2_new = [distance2(trial_pos,b) for b in config.pos]
    #en_moved = energy_update(i_atom, dist2_new, pot)
    #recalculate old 
    #en_unmoved = energy_update(i_atom, dist2_mat[i_atom,:], pot)
    #one might want to store dimer energies per atom in vector?
    #decide acceptance
    if metropolis_condition(ensemble, delta_en, mc_state.beta) >= rand()
        #new config accepted
        mc_state.config.pos[i_atom] = copy(trial_pos)
        mc_state.dist2_mat[i_atom,:] = copy(dist2_new)
        mc_state.dist2_mat[:,i_atom] = copy(dist2_new)
        mc_state.en_tot += delta_en
        mc_state.count_atom[1] += 1
        mc_state.count_atom[2] += 1
    end 
    return mc_state #config, entot, dist2mat, count_acc, count_acc_adjust
end

function mc_step!(mc_state::MCState, move_strat, pot, ensemble, a, v, r)
    #a = atom_move_frequency(move_strat)
    #v = vol_move_frequency(move_strat)
    #r = rot_move_frequency(move_strat)

    ran = rand(1:(a+v+r)) #choose move randomly
    if ran <= a
        mc_state = atom_move!(mc_state, ran, pot, ensemble)
    #else if ran <= v
    #    vol_move!(mc_state, pot, ensemble)
    #else if ran <= r
    #    rot_move!(mc_state, pot, ensemble)
    end
    return mc_state
end 


function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
    #a = atom_move_frequency(move_strat)
    #v = vol_move_frequency(move_strat)
    #r = rot_move_frequency(move_strat)
    #n_steps = a + v + r
    for i_traj = 1:mc_params.n_traj
        for i_step = 1:n_steps
            #mc_states[i_traj] = mc_step!(type_moves[ran][2], type_moves[ran][1], mc_states[i_traj], ran, pot, ensemble)
            @inbounds mc_states[i_traj] = mc_step!(mc_states[i_traj], move_strat, pot, ensemble, a, v, r)
        end
        #push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
    end
    if rand()<0.1 #attempt to exchange trajectories
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

function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, n_bin)
    
    #number of moves per MC cycle
    #moves = mc_states[1].moves
    #n_moves = 0
    #type_moves = []
    #for i in eachindex(moves)
    #    n_moves += moves[i].frequency
    #    for j=1:moves[i].frequency
    #        push!(type_moves,(i,moves[i]))
    #    end 
    #end
    a = atom_move_frequency(move_strat)
    v = vol_move_frequency(move_strat)
    r = rot_move_frequency(move_strat)
    n_steps = a + v + r

    for i = 1:mc_params.eq_cycles
        @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
        if rem(i, mc_params.n_adjust) == 0
            for i_traj = 1:mc_params.n_traj
                update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
            end 
        end
    end
    #re-set counter variables to zero
    for i_traj = 1:mc_params.n_traj
        mc_states[i_traj].count_atom = mc_states[i_traj].count_vol = mc_states[i_traj].count_rot = mc_states[i_traj].count_exc = [0,0]
    end 

    for i = 1:mc_params.mc_cycles
        @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
        for i_traj=1:mc_params.n_traj
            push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
        end 
        if rem(i, mc_params.n_adjust) == 0
            for i_traj = 1:mc_params.n_traj
                update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
            end 
        end
    end

    results = Output{Float64}(n_bin=n_bin)




    #TO DO
    # volume,rot moves ...
    # adjustment of step size
    # do the averages, energy; cv
    # Histograms
    # make report struct

    return 
end

end
