module MCRun

export metropolis_condition, mc_step_atom!, mc_step!, ptmc_run

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation

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

function mc_step!(::AtomMove, config, beta, dist2_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adjust, pot, ensemble)
    #move randomly selected atom (obeying the boundary conditions)
    trial_pos = atom_displacement(config.pos[i_atom], max_displacement, config.bc)
    #find new distances of moved atom - might not be always needed?
    delta_en, dist2_new = energy_update(trial_pos, i_atom, config, dist2_mat, pot)
    #dist2_new = [distance2(trial_pos,b) for b in config.pos]
    #en_moved = energy_update(i_atom, dist2_new, pot)
    #recalculate old 
    #en_unmoved = energy_update(i_atom, dist2_mat[i_atom,:], pot)
    #one might want to store dimer energies per atom in vector?
    #decide acceptance
    if metropolis_condition(delta_en, beta, ensemble) >= rand()
        #new config accepted
        config.pos[i_atom] = copy(trial_pos)
        dist2_mat[i_atom,:] = copy(dist2_new)
        dist2_mat[:,i_atom] = copy(dist2_new)
        en_tot = en_tot + delta_en
        count_acc += 1
        count_acc_adjust += 1
    end 
    return config, entot, dist2mat, count_acc, count_acc_adjust
end

function ptmc_run!(temp, mc_params, conf_ne13, bc_ne13, elj_ne, moves, ensemble, displ_param, stat_param)
    #number of moves per MC cycle
    n_moves = 0
    for i in eachindex(moves)
        n_moves += moves[i].frequency
    end
    println(n_moves)
    #to select a type of move for one of n_moves MC step per cycle
    i_move = rand(1:n_moves)
    
    #to be checked/improved ...
    dist2_mat_0=get_distance2_mat(conf_ne13)

    en_atom_mat_0=dimer_energy_config(dist2_mat_0, NAtoms, elj_ne)[1]
    en_tot_0=dimer_energy_config(dist2_mat_0, NAtoms, elj_ne)[3]

    config=Array{Config}(undef,n_traj)
    dist2_mat = Array{Matrix}(undef,n_traj) 
    en_atom_mat = Array{Array}(undef,n_traj) 
    en_tot = zeros(n_traj)
    max_displacement=displ_param.max_displacement
    for i=1:n_traj
        config[i]=Config(copy(conf_ne13.pos),bc_ne13)
        dist2_mat[i]=copy(dist2_mat_0)
        en_atom_mat[i]=copy(en_atom_mat_0)
        en_tot[i]=en_tot_0
    end
    
    energies=Array{Array}(undef,n_traj)
    for i=1:n_traj
        energies[i]=zeros(mc_cycles)
    end
    
    cv=Array{Float64}(undef,n_traj)

    #histograms
    Ebins = 100
    Emin = -0.006
    Emax = -0.001

    dE = (Emax-Emin)/Ebins
    Ehistogram = Array{Array}(undef,n_traj)      #initialization
    for i=1:n_traj
        Ehistogram[i]=zeros(Ebins)
    end

    return 
end

end
