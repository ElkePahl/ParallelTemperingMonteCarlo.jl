module MCRun

export metropolis_condition, mc_step_atom!

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves

"""
    metropolis_condition(energy_unmoved, energy_moved, beta)

Determines probability to accept a MC move at inverse temperature beta, takes energies of new and old configurations
"""
function metropolis_condition(energy_unmoved, energy_moved, beta)
    prob_val = exp(-(energy_moved-energy_unmoved)*beta)
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
        config.pos[i_atom] = copy(trial_pos)
        dist2_mat[i_atom,:] = copy(dist2_new)
        dist2_mat[:,i_atom] = copy(dist2_new)
        en_tot = en_tot - en_unmoved + en_moved
        count_acc += 1
        count_acc_adjust += 1
    end 
    #restore or accept
    return config, entot, dist2mat, count_acc, count_acc_adjust
end
