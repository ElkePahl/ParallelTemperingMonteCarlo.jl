# Not a module, just supplementary code for the swap_config function, this is essential given the inclusion of the new energy types.
export swap_config!, swap_atom_config!, swap_config_v!, swap_vars!
"""
    swap_config!(mc_state,movetype)

function for replacing the MC state and potential variables values with the updated values when metropolis condition is met. 
Implemented for the following `move_type`:
    - atommove 
    - volumemove
    - swapmove for atom swaps
All methods also call the swap_vars! function which distributes the appropriate `mc_states.potential_variables` values into the current mc_state struct.
"""
function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::String) where {T,N,BC,P<:AbstractPotentialVariables,E<:AbstractEnsembleVariables}
    if movetype == "atommove"
        swap_atom_config!(mc_state, mc_state.ensemble_variables.index, mc_state.ensemble_variables.trial_move)
    elseif movetype == "atomswap"
        swap_move_config!(mc_state,mc_state.ensemble_variables.swap_indices)
    else
        swap_config_v!(mc_state, mc_state.config.bc, mc_state.ensemble_variables.trial_config, mc_state.ensemble_variables.new_dist2_mat, mc_state.potential_variables.en_atom_vec, mc_state.new_en)
    end

end

"""
    swap_atom_config(mc_state::MCState,i_atom,trial_pos)
"""
function swap_atom_config!(mc_state::MCState{T,N,BC,P,E},i_atom,trial_pos) where {T,N,BC,P<:AbstractPotentialVariables,E<:AbstractEnsembleVariables}
    mc_state.config.pos[i_atom] = trial_pos
    mc_state.dist2_mat[i_atom,:] = mc_state.new_dist2_vec
    mc_state.dist2_mat[:,i_atom] = mc_state.new_dist2_vec
    mc_state.en_tot,mc_state.new_en = mc_state.new_en, mc_state.en_tot
    mc_state.count_atom[1] += 1
    mc_state.count_atom[2] += 1
    
    swap_vars!(i_atom,mc_state.potential_variables)
    
end
"""
    swap_config_v!(mc_state,trial_config,dist2_mat_new,en_vec_new,new_en_tot)
swaps mc states and ensemble variables in case of accepted volume move for NPT ensemble
implemented for `CubicBC` and  `RhombicBC`
"""
function swap_config_v!(mc_state::MCState,bc::CubicBC,trial_config::Config,new_dist2_mat,en_vec_new,new_en_tot)
    mc_state.config = Config(trial_config.pos,CubicBC(trial_config.bc.box_length))
    mc_state.dist2_mat = new_dist2_mat
    mc_state.potential_variables.en_atom_vec = en_vec_new
    mc_state.en_tot = new_en_tot
    mc_state.count_vol[1] += 1
    mc_state.count_vol[2] += 1

    mc_state.ensemble_variables.r_cut = mc_state.ensemble_variables.new_r_cut
end

function swap_config_v!(mc_state::MCState,bc::RhombicBC,trial_config::Config,new_dist2_mat,en_vec_new,new_en_tot)
    mc_state.config = Config(trial_config.pos,RhombicBC(trial_config.bc.box_length, trial_config.bc.box_height))
    mc_state.dist2_mat = new_dist2_mat
    mc_state.potential_variables.en_atom_vec = en_vec_new
    mc_state.en_tot = new_en_tot
    mc_state.count_vol[1] += 1
    mc_state.count_vol[2] += 1

    mc_state.ensemble_variables.r_cut = mc_state.ensemble_variables.new_r_cut
end

"""
    swap_vars!(i_atom::Int, potential_variables::V) where V <: DimerPotentialVariables
    swap_vars!(i_atom::Int, potential_variables::ELJPotentialBVariables)
    swap_vars!(i_atom::Int, potential_variables::EmbeddedAtomVariables)
    swap_vars!(i_atom::Int, potential_variables::NNPVariables)
Called by `swap_atom_config!` function; 
takes the appropriate `potential_variables` that are specific to the potential energy surface under consideration 
and replaces the current values with the new values such as:
    - Under magnetic fields, the new tan matrix replaces the old
    - In the EAM, we replace the rho and phi vectors with the new updated versions
    - Using an NNP we require the new G matrix and F matrix to replace the old versions. 
implemented for potential variables = `DimerPotentialVariables`,`ELJPotentialBVariables`,`EmbeddedAtomVariables`,`NNPVariables`
"""
function swap_vars!(i_atom,potential_variables::V) where V <: DimerPotentialVariables
end
function swap_vars!(i_atom,potential_variables::ELJPotentialBVariables)
    potential_variables.tan_mat[i_atom,:] = potential_variables.new_tan_vec
    potential_variables.tan_mat[:,i_atom] = potential_variables.new_tan_vec

end
function swap_vars!(i_atom,potential_variables::EmbeddedAtomVariables)
    potential_variables.component_vector,potential_variables.new_component_vector = potential_variables.new_component_vector,potential_variables.component_vector
end
function swap_vars!(i_atom,potential_variables::NNPVariables)
    potential_variables.en_atom_vec,potential_variables.new_en_atom = potential_variables.new_en_atom ,potential_variables.en_atom_vec
    potential_variables.g_matrix, potential_variables.new_g_matrix = potential_variables.new_g_matrix , potential_variables.g_matrix

    potential_variables.f_matrix[i_atom,:] = potential_variables.new_f_vec
    potential_variables.f_matrix[:,i_atom] = potential_variables.new_f_vec
end

function swap_vars!(i_atom,potential_variables::NNPVariables2a)

    potential_variables.en_atom_vec,potential_variables.new_en_atom = potential_variables.new_en_atom,potential_variables.en_atom_vec

    potential_variables.g_matrix,potential_variables.new_g_matrix = potential_variables.new_g_matrix,potential_variables.g_matrix

    potential_variables.f_matrix[i_atom,:] = potential_variables.new_f_vec
    potential_variables.f_matrix[:,i_atom] = potential_variables.new_f_vec
    
end
"""
    swap_move_config!(mc_state,indices)
Function designed to exchange relevant variables when swapping an atom. Accepts the `mc_state` and the atom `indices` and exchanges atom `indices[1]` with atom `indices[2]`
"""
function swap_move_config!(mc_state,indices)
    #swap energy
    mc_state.en_tot, mc_state.new_en = mc_state.new_en,mc_state.en_tot
    #swap positions 
    mc_state.config.pos[indices[1]],mc_state.config.pos[indices[2]] = mc_state.config.pos[indices[2]],mc_state.config.pos[indices[1]]
    #swap dist2mat
    mc_state.dist2_mat[indices[1],:],mc_state.dist2_mat[indices[2],:] = mc_state.dist2_mat[indices[2],:],mc_state.dist2_mat[indices[1],:]

    mc_state.dist2_mat[:,indices[1]],mc_state.dist2_mat[:,indices[2]] = mc_state.dist2_mat[:,indices[2]],mc_state.dist2_mat[:,indices[1]]

    mc_state.dist2_mat[indices[1],indices[1]],mc_state.dist2_mat[indices[2],indices[2]] = 0.,0.

    #swap fmat
    mc_state.potential_variables.f_matrix[indices[1],:],mc_state.potential_variables.f_matrix[indices[2],:] = mc_state.potential_variables.f_matrix[indices[2],:],mc_state.potential_variables.f_matrix[indices[1],:]

    mc_state.potential_variables.f_matrix[:,indices[1]],mc_state.potential_variables.f_matrix[:,indices[2]] = mc_state.potential_variables.f_matrix[:,indices[2]],mc_state.potential_variables.f_matrix[:,indices[1]]

    mc_state.potential_variables.f_matrix[indices[1],indices[1]],mc_state.potential_variables.f_matrix[indices[2],indices[2]] = 1.,1.

    #swap en_atom_vec and gmat
    mc_state.potential_variables.en_atom_vec,mc_state.potential_variables.new_en_atom = mc_state.potential_variables.new_en_atom,mc_state.potential_variables.en_atom_vec
    mc_state.potential_variables.g_matrix,mc_state.potential_variables.new_g_matrix = mc_state.potential_variables.new_g_matrix,mc_state.potential_variables.g_matrix
end