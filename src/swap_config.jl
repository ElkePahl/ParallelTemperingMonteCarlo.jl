# Not a module, just supplementary code for the swap_config function, this is essential given the inclusion of the new energy types.

"""
    swap_config!(mc_state,movetype::atommove)
    swap_config!(mc_state,movetype::volumemove)

Basic function for replacing the existing mc_state values with the updated values assuming the metropolis condition is met. 
    - First method applies where the `movetype` is an atommove and distributes the new ensemblevariables such as `i_atom` and `trial_pos` into the second method, which actually swaps the variables. 
    - Second method applies where `movetype` is a volumemove, this splits the ensemble variables into the swap_config_v! function to replace the `trial_config` the `new_dist2_mat` the `en_vec_new` and the `new_en_tot` into their appropriate place in the mc_state struct
All methods also call the swap_vars! function which distributes the appropriate `mc_states.potential_variables` values into the current mc_state struct.
"""

function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::String) where {T,N,BC,P<:PotentialVariables,E<:EnsembleVariables}
    if movetype == "atommove"
        swap_atom_config!(mc_state, mc_state.ensemble_variables.index, mc_state.ensemble_variables.trial_move)
    else
        swap_config_v!(mc_state, mc_state.ensemble_variables.trial_config, mc_state.ensemble_variables.dist2_mat_new, mc_state.potential_variables.en_atom_vec, mc_state.new_en)
    end

end
# function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::atommove) where {T,N,BC,P<:PotentialVariables,E<:EnsembleVariables}
#     swap_atom_config!(mc_state, mc_state.ensemble_variables.index, mc_state.ensemble_variables.trial_move)
# end
# function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::volumemove) where {T,N,BC,P<:PotentialVariables,E<:EnsembleVariables}
#     swap_config_v!(mc_state, mc_state.ensemble_variables.trial_config, mc_state.ensemble_variables.dist2_mat_new, mc_state.potential_variables.en_atom_vec, mc_state.new_en)
# end
"""
    swap_atom_config(mc_state::MCState,i_atom,trial_pos)
"""
function swap_atom_config!(mc_state::MCState{T,N,BC,P,E},i_atom,trial_pos) where {T,N,BC,P<:PotentialVariables,E<:EnsembleVariables}
    mc_state.config.pos[i_atom] = trial_pos
    mc_state.dist2_mat[i_atom,:] = mc_state.new_dist2_vec
    mc_state.dist2_mat[:,i_atom] = mc_state.new_dist2_vec
    mc_state.en_tot = mc_state.new_en 
    mc_state.count_atom[1] += 1
    mc_state.count_atom[2] += 1
    
    swap_vars!(i_atom,mc_state.potential_variables)
    
end
"""
    swap_config_v!(mc_state,trial_config,dist2_mat_new,en_vec_new,new_en_tot)
secondary function where a volume move has been made, takes the new ensemble variables and puts them into the appropriate current state of the struct.
"""
function swap_config_v!(mc_state::MCState,trial_config::Config,dist2_mat_new,en_vec_new,new_en_tot)
    mc_state.config = Config(trial_config.pos,PeriodicBC(trial_config.bc.box_length))
    mc_state.dist2_mat = dist2_mat_new
    mc_state.potential_variables.en_atom_vec = en_vec_new
    mc_state.en_tot = new_en_tot
    mc_state.count_vol[1] += 1
    mc_state.count_vol[2] += 1
end
"""
    swap_vars!(i_atom,potential_variables::V)
    swap_vars!(i_atom,potential_variables::ELJPotentialBVariables)
    swap_vars!(i_atom,potential_variables::EmbeddedAtomVariables)
    swap_vars!(i_atom,potential_variables::NNPVariables)
Secondary function to swap_config! takes the appropriate `potential_variables` that are specific to the potential energy surface under consideration and replaces the current values with the new values such as:
    - Under magnetic fields, the new tan matrix replaces the old
    - In the EAM, we replace the rho and phi vectors with the new updated versions
    - Using an NNP we require the new G matrix and F matrix to replace the old versions. 
"""
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