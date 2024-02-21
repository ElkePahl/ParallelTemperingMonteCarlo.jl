module MCMoves


export atom_displacement,volume_change
export generate_move!

using StaticArrays

using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles

"""
    atom_displacement(pos, max_displacement, bc)
    atom_displacement(mc_states,index)
    atom_displacement(mc_state)

Generates trial position for atom, moving it from `pos` by some random displacement 
Random displacement determined by `max_displacement`
These variables are additionally contained in `mc_state` where the pos is determined by `index`.
Implemented for:
    
    - `SphericalBC`: trial move is repeated until moved atom is within binding sphere
    - `PeriodicBC`: periodic boundary condition enforced, an atom is moved into the box from the other side when it tries to get out.

The final method is a wrapper function which unpacks mc_states, which contains all the necessary arguments for the two methods above. When we have correctly implemented move_strat this wrapper will be expanded to include other methods
"""
function atom_displacement(pos, max_displacement, bc::SphericalBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    count = 0
    while check_boundary(bc, trial_pos)         #displace the atom until it's inside the binding sphere
        count += 1
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = pos + delta_move
        count == 100 && error("Error: too many moves out of binding sphere")
    end
    return trial_pos
end
function atom_displacement(pos, max_displacement, bc::PeriodicBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    trial_pos -= bc.box_length*[round(trial_pos[1]/bc.box_length), round(trial_pos[2]/bc.box_length), round(trial_pos[3]/bc.box_length)]
    return trial_pos
end
function atom_displacement(mc_state::MCState)

    mc_state.ensemble_variables.trial_move = atom_displacement(mc_state.config.pos[mc_state.ensemble_variables.index],mc_state.max_displ[1],mc_state.config.bc)


    return mc_state
end 

"""
    volume_change(conf::Config, max_vchange, bc::PeriodicBC) 
scale the whole configuration, including positions and the box length.
returns the trial configuration as a struct. 
"""
function volume_change(conf::Config, max_vchange, max_length)
    scale = exp((rand()-0.5)*max_vchange)^(1/3)
    if conf.bc.box_length >= max_length && scale > 1.
        scale=1.
    end

    trial_config = Config(conf.pos * scale,PeriodicBC(conf.bc.box_length * scale))
    return trial_config
end
function volume_change(mc_state)
    #change volume
    mc_state.ensemble_variables.trial_config = volume_change(mc_state.config,mc_state.max_displ[2],mc_state.max_boxlength)
    #change r_cut
    mc_state.ensemble_variables.new_r_cut = mc_state.ensemble_variables.trial_config.bc.box_length^2/4
    #get the new dist2 matrix
    mc_state.ensemble_variables.dist2_mat_new = get_distance2_mat(mc_state.ensemble_variables.trial_config)

    return mc_state
end

"""
    generate_move!(mc_state,movetype::atommove)
    generate_move!(mc_state,movetype::volumemove)
generate move is the currying function that takes mc_state and a movetype and generates the variables required inside of the ensemblevariables struct within mc_state. 
"""
function generate_move!(mc_state,movetype::atommove)
    return atom_displacement(mc_state)
end
function generate_move!(mc_state,movetype::volumemove)
    return volume_change(mc_state)
end


end 
