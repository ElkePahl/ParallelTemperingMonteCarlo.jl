module MCRun

export MCState
export metropolis_condition, mc_step_atom!, mc_step!, mc_cycle!, ptmc_run!

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation

struct MCState{T,N,BC,M}
    temp::T
    beta::T
    config::Config{N,BC,T}
    dist2_mat::Matrix{T}
    en_atom_mat::Vector{T}
    en_tot::Ref{T}
    en_hist::EnHist{T}
    ham::Vector{T}
    moves::M # Tuple
    count_exc::SVector{2,Int}
end    

function MCState(temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_mat, en_tot, en_hist, ham, moves::M; count_exc=SVector(0,0)) where {T,N,BC,M}
    MCState{T,N,BC,M}(temp,beta,config,dist2_mat,en_atom_mat,en_tot,en_hist,ham,moves,count_exc)
end

function MCState(temp, beta, config::Config{N,BC,T}, en_hist, ham, moves::M; count_exc=SVector(0,0)) where {T,N,BC,M}
    dist2_mat = get_distance2_mat(config)
    en_atom_mat, en_tot = dimer_energy_config(dist2_mat_0, n_atoms, pot)
    MCState{T,N,BC,M}(temp,beta,config,dist2_mat,en_atom_mat,en_tot,en_hist,ham,moves,count_exc)
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

function mc_step_atom!(config, beta, dist2_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adjust, pot)
    #move randomly selected atom (obeying the boundary conditions)
    trial_pos = atom_displacement(config.pos[i_atom], max_displacement, config.bc)
    #find new distances of moved atom - might not be always needed?
    dist2_new = [distance2(trial_pos,b) for b in config.pos]
    en_moved = energy_update(i_atom, dist2_new, pot)
    #recalculate old 
    en_unmoved = energy_update(i_atom, dist2_mat[i_atom,:], pot)
    #one might want to store dimer energies per atom in vector?
    #decide acceptance
    if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        #new config accepted
        config.pos[i_atom] = copy(trial_pos)
        dist2_mat[i_atom,:] = copy(dist2_new)
        dist2_mat[:,i_atom] = copy(dist2_new)
        en_tot = en_tot - en_unmoved + en_moved
        count_acc += 1
        count_acc_adjust += 1
    end 
    return config, entot, dist2mat, count_acc, count_acc_adjust
end

function mc_step!(move::AtomMove, mc_state::MCState, i_atom, pot, ensemble)
    #println(move, mc_state, i_atom, pot, ensemble)
    #move randomly selected atom (obeying the boundary conditions)
    trial_pos = atom_displacement(mc_state.config.pos[i_atom], move.max_displacement, mc_state.config.bc)
    #find new distances of moved atom - might not be always needed?
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
        mc_state.en_tot[] = mc_state.en_tot[] + delta_en
        move.count_acc += 1
        move.count_acc_adj += 1
    end 
    return move, mc_state #config, entot, dist2mat, count_acc, count_acc_adjust
end

function mc_cycle!(n_moves, type_moves, mc_states, mc_params, pot, ensemble)
    for i_traj=1:mc_params.n_traj
        for i_move=1:n_moves
            ran = rand(1:n_moves) #choose move randomly
            type_moves[ran][2], mc_states[i_traj] = mc_step!(type_moves[ran][2], mc_states[i_traj], type_moves[ran][1], pot, ensemble)
        end
        push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot[]) #to build up ham vector of sampled energies
    end
    return type_moves, mc_states
end

function ptmc_run!(mc_states, mc_params, pot, ensemble)
    #number of moves per MC cycle
    moves = mc_states[1].moves
    n_moves = 0
    type_moves = []
    for i in eachindex(moves)
        n_moves += moves[i].frequency
        for j=1:moves[i].frequency
            push!(type_moves,[j,moves[i]])
        end 
    end
    
    for i=1:mc_params.mc_cycles
        type_moves,mc_states = mc_cycle!(n_moves, type_moves, mc_states, mc_params, pot, ensemble)
    end 
    for j=1:13
        println(mc_states[j].ham)
    end
    #push!(ham[i_traj],mc_states[i_traj].en_tot[]) to build up ham vector of sampled energies
    
    #cv=Array{Float64}(undef,n_traj)

    #MC loop ... 

    return 
end

end
