module MCMoves


export atom_displacement,volume_change
export generate_move!

using StaticArrays

using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles
using ..CustomTypes
#using ..MCRun

"""
    atom_displacement(pos::PositionVector, max_displacement::Number, bc::SphericalBC)
    atom_displacement(pos::PositionVector, max_displacement::Number, bc::CubicBC)
    atom_displacement(pos::PositionVector, max_displacement::Number, bc::RhombicBC)
    atom_displacement(mc_state::MCState{T, N, BC}) where {T, N, BC <: PeriodicBC}
    atom_displacement(mc_state::MCState{T, N, BC}) where {T, N, BC <: SphericalBC}

Generates trial position for atom, moving it from `pos` by some random displacement.
Random displacement determined by `max_displacement`.
These variables are additionally contained in `mc_state` where the pos is determined by `index`.
Implemented for:
-   `SphericalBC`: trial move is repeated until moved atom is within binding sphere
-   `CubicBC`; `RhombicBC`: periodic boundary condition enforced, an atom is moved into the box from the other side when it tries to get out.


The final method is a wrapper function which unpacks `mc_states`, which contains all the necessary arguments for the two methods above. When we have correctly implemented `move_strat` this wrapper will be expanded to include other methods.
"""
function atom_displacement(pos::PositionVector, max_displacement::Number, bc::SphericalBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    # count = 0
    # while check_boundary(bc, trial_pos)         #displace the atom until it's inside the binding sphere
    #     count += 1
    #     delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    #     trial_pos = pos + delta_move
    #     count == 100 && error("Error: too many moves out of binding sphere")
    # end
    return trial_pos
end

function atom_displacement(pos::PositionVector, max_displacement::Number, bc::CubicBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    trial_pos -= bc.box_length*[round(trial_pos[1]/bc.box_length), round(trial_pos[2]/bc.box_length), round(trial_pos[3]/bc.box_length)]
    return trial_pos
end

function atom_displacement(pos::PositionVector, max_displacement::Number, bc::RhombicBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    trial_pos -= [bc.box_length*round((trial_pos[1]-trial_pos[2]/3^0.5-bc.box_length/2)/bc.box_length)+bc.box_length/2*round((trial_pos[2]-bc.box_length*3^0.5/4)/(bc.box_length*3^0.5/2)), bc.box_length*3^0.5/2*round((trial_pos[2]-bc.box_length*3^0.5/4)/(bc.box_length*3^0.5/2)), bc.box_height*round((trial_pos[3]-bc.box_height/2)/bc.box_height)]
    return trial_pos
end
function atom_displacement(mc_state::MCState{T,N,BC}) where {T,N,BC<:PeriodicBC}
    mc_state.ensemble_variables.trial_move = atom_displacement(mc_state.config.pos[mc_state.ensemble_variables.index],mc_state.max_displ[1],mc_state.config.bc)
    for (i, b) in enumerate(mc_state.config.pos)
        mc_state.new_dist2_vec[i] = distance2(mc_state.ensemble_variables.trial_move,b,mc_state.config.bc)
    end
    mc_state.new_dist2_vec[mc_state.ensemble_variables.index] = 0.
    return mc_state
end 

function atom_displacement(mc_state::MCState{T,N,BC}) where {T,N,BC<:SphericalBC}
    count = 0.
    trial_pos = atom_displacement(mc_state.config.pos[mc_state.ensemble_variables.index],mc_state.max_displ[1],mc_state.config.bc)
    while check_boundary(mc_state.config.bc, trial_pos)
        count += 1 
        if count == 50
            recentre!(mc_state.config)
        else
            trial_pos = atom_displacement(mc_state.config.pos[mc_state.ensemble_variables.index],mc_state.max_displ[1],mc_state.config.bc)
            count == 100 && error("Error: too many moves out of binding sphere")
        end
    end
    mc_state.ensemble_variables.trial_move = trial_pos

    # mc_state.ensemble_variables.trial_move = atom_displacement(mc_state.config.pos[mc_state.ensemble_variables.index],mc_state.max_displ[1],mc_state.config.bc)
    for (i, b) in enumerate(mc_state.config.pos)
        mc_state.new_dist2_vec[i] = distance2(mc_state.ensemble_variables.trial_move,b,mc_state.config.bc)
    end
    mc_state.new_dist2_vec[mc_state.ensemble_variables.index] = 0.
    return mc_state
end 

"""
    volume_change(conf::Config, bc::CubicBC, max_vchange::Number, max_length::Number)
    volume_change(conf::Config, bc::RhombicBC, max_vchange::Number, max_length::Number)
    volume_change(mc_state::MCState) 
Scale the whole configuration, including positions and the box length.
Returns the trial configuration as a struct. 
"""
function volume_change(conf::Config, bc::CubicBC, max_vchange::Number, max_length::Number)
    scale = exp((rand()-0.5)*max_vchange)^(1/3)
    if conf.bc.box_length >= max_length && scale > 1.
        scale=1.
    end
    trial_config = Config(conf.pos * scale,CubicBC(conf.bc.box_length * scale))
    return trial_config,scale
end

function volume_change(conf::Config, bc::RhombicBC, max_vchange::Number, max_length::Number)
    scale = exp((rand()-0.5)*max_vchange)^(1/3)
    if conf.bc.box_length >= max_length && scale > 1.
        scale=1.
    end

    trial_config = Config(conf.pos * scale,RhombicBC(conf.bc.box_length * scale, conf.bc.box_height * scale))
    return trial_config,scale
end
function volume_change(mc_state::MCState)
    #change volume
    mc_state.ensemble_variables.trial_config, scale = volume_change(mc_state.config, mc_state.config.bc, mc_state.max_displ[2], mc_state.max_boxlength)
    #change r_cut
    mc_state.ensemble_variables.new_r_cut = get_r_cut(mc_state.ensemble_variables.trial_config.bc)
    #get the new dist2 matrix
    mc_state.ensemble_variables.new_dist2_mat = mc_state.dist2_mat .* scale
    return mc_state
end

"""  
    (swap_atoms(mc_state::MCState{T, N, BC, PV, EV}) where {T, N, BC, PV, EV <: NNVTVariables{tee, n, N1, N2}}) where {tee, n, N1, N2}
Swaps two atoms in the configuration.
"""
function swap_atoms(mc_state::MCState{T,N,BC,PV,EV}) where {T,N,BC,PV,EV<:NNVTVariables{tee,n,N1,N2}} where {tee,n,N1,N2}
    i1,i2 = rand(1:N1),rand(N1+1:N)
    mc_state.ensemble_variables.swap_indices = SVector{2}(i1,i2)
   return mc_state
end

"""
    generate_move!(mc_state::MCState,movetype::String)
[`generate_move!`](@ref) is the currying function that takes `mc_state` and a `movetype` 
and generates the variables required inside of the `ensemblevariables` struct within `mc_state`. 
"""
function generate_move!(mc_state::MCState,movetype::String)
    if movetype == "atommove"
        return atom_displacement(mc_state)
    elseif movetype == "atomswap"
        return swap_atoms(mc_state)
    else
        return volume_change(mc_state)
    end
end



end 
