# Not a module, just supplementary code for the swap_config function, this is essential given the inclusion of the new energy types.

function swap_config!(mc_state,i_atom,trial_pos)
    mc_state.config.pos[i_atom] = trial_pos
    mc_state.dist2_mat[i_atom,:] = mc_state.new_dist2_vec
    mc_state.dist2_mat[:,i_atom] = mc_state.new_dist2_vec
    mc_state.en_tot = mc_state.new_en 
    mc_state.count_atom[1] += 1
    mc_state.count_atom[2] += 1
    
    swap_vars!(i_atom,mc_state.potential_variables)
    
end

function swap_vars!(i_atom,potential_variables::V) where V <: DimerPotentialVariables

end
function swap_vars!(i_atom,potential_variables::ELJPotentialBVariables)
    potential_variables.tan_mat[i_atom,:] = potential_variables.new_tan_vec
    potential_variables.tan_mat[:,i_atom] = potential_variables.new_tan_vec

end
function swap_vars!(i_atom,potential_variables::EmbeddedAtomVariables)
    potential_variables.component_vector = potential_variables.new_component_vector
end
function swap_vars!(i_atom,potential_variables::NNPVariables)
    potential_variables.en_atom_vec = potential_variables.new_en_atom 
    potential_variables.g_matrix = potential_variables.new_g_matrix 

    potential_variables.f_matrix[i_atom,:] = potential_variables.new_f_vec
    potential_variables.f_matrix[:,i_atom] = potential_variables.new_f_vec
end