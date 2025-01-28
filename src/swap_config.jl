# Not a module, just supplementary code for the swap_config function, this is essential given the inclusion of the new energy types.
export swap_config!, swap_atom_config!, swap_config_v!, swap_vars!
"""
    swap_config!(mc_state,movetype)

Function for replacing the MC state and potential variables values with the updated values when metropolis condition is met. 
Implemented for the following `move_type`:
-   `atommove` 
-   `volumemove`
All methods also call the [`swap_vars!`](@ref) function which distributes the appropriate `mc_states.potential_variables` values into the current `mc_state` struct.
"""
function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::String) where {T,N,BC,P<:AbstractPotentialVariables,E<:AbstractEnsembleVariables}
    if movetype == "atommove"
        swap_atom_config!(mc_state, mc_state.ensemble_variables.index, mc_state.ensemble_variables.trial_move)
    else
        swap_config_v!(mc_state, mc_state.config.bc, mc_state.ensemble_variables.trial_config, mc_state.ensemble_variables.new_dist2_mat, mc_state.potential_variables.en_atom_vec, mc_state.new_en)
    end

end
# function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::atommove) where {T,N,BC,P<:PotentialVariables,E<:AbstractEnsembleVariables}
#     swap_atom_config!(mc_state, mc_state.ensemble_variables.index, mc_state.ensemble_variables.trial_move)
# end
# function swap_config!(mc_state::MCState{T,N,BC,P,E},movetype::volumemove) where {T,N,BC,P<:PotentialVariables,E<:AbstractEnsembleVariables}
#     swap_config_v!(mc_state, mc_state.ensemble_variables.trial_config, mc_state.ensemble_variables.dist2_mat_new, mc_state.potential_variables.en_atom_vec, mc_state.new_en)
# end
"""
    swap_atom_config(mc_state::MCState,i_atom,trial_pos)
"""
function swap_atom_config!(mc_state::MCState{T,N,BC,P,E},i_atom::Int,trial_pos::PositionVector) where {T,N,BC,P<:AbstractPotentialVariables,E<:AbstractEnsembleVariables}
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
Swaps `mc_state`s and ensemble variables in case of accepted volume move for NPT ensemble.
Implemented for [`CubicBC`](@ref Main.ParallelTemperingMonteCarlo.BoundaryConditions.CubicBC) and [`RhombicBC`](@ref Main.ParallelTemperingMonteCarlo.BoundaryConditions.RhombicBC)
"""
function swap_config_v!(mc_state::MCState,bc::CubicBC,trial_config::Config,new_dist2_mat::Matrix{N},en_vec_new::VorS,new_en_tot::Number) where N <: Number
    mc_state.config = Config(trial_config.pos,CubicBC(trial_config.bc.box_length))
    mc_state.dist2_mat = new_dist2_mat
    mc_state.potential_variables.en_atom_vec = en_vec_new
    mc_state.en_tot = new_en_tot
    mc_state.count_vol[1] += 1
    mc_state.count_vol[2] += 1

    mc_state.ensemble_variables.r_cut = mc_state.ensemble_variables.new_r_cut
end

function swap_config_v!(mc_state::MCState,bc::RhombicBC,trial_config::Config,new_dist2_mat::Matrix{N},en_vec_new::VorS,new_en_tot::Number) where N <: Number
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
-   Under magnetic fields, the new tan matrix replaces the old
-   In the EAM, we replace the rho and phi vectors with the new updated versions
-   Using an NNP we require the new G matrix and F matrix to replace the old versions. 
Implemented for potential variables:
-   [`DimerPotentialVariables`](@ref)
-   [`ELJPotentialBVariables`](@ref)
-   [`EmbeddedAtomVariables`](@ref)
-   [`NNPVariables`](@ref)
"""
function swap_vars!(i_atom::Int,potential_variables::V) where V <: DimerPotentialVariables
end

function swap_vars!(i_atom::Int,potential_variables::ELJPotentialBVariables)
    potential_variables.tan_mat[i_atom,:] = potential_variables.new_tan_vec
    potential_variables.tan_mat[:,i_atom] = potential_variables.new_tan_vec

end

function swap_vars!(i_atom::Int,potential_variables::EmbeddedAtomVariables)
    potential_variables.component_vector = potential_variables.new_component_vector
end

function swap_vars!(i_atom::Int,potential_variables::NNPVariables)
    potential_variables.en_atom_vec = potential_variables.new_en_atom 
    potential_variables.g_matrix = potential_variables.new_g_matrix 

    potential_variables.f_matrix[i_atom,:] = potential_variables.new_f_vec
    potential_variables.f_matrix[:,i_atom] = potential_variables.new_f_vec
end