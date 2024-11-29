module MCMoves


export atom_displacement,volume_change
export scale_xy,scale_z,volume_change_xy,volume_change_z,volume_change_xyz
export generate_move!

using StaticArrays

using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..Ensembles
using ..EnergyEvaluation

"""
    atom_displacement(pos, max_displacement, bc)
    atom_displacement(mc_states,index)
    atom_displacement(mc_state)

Generates trial position for atom, moving it from `pos` by some random displacement 
Random displacement determined by `max_displacement`
These variables are additionally contained in `mc_state` where the pos is determined by `index`.
Implemented for:
    
    - `SphericalBC`: trial move is repeated until moved atom is within binding sphere
    - `CubicBC`; `RhombicBC`: periodic boundary condition enforced, an atom is moved into the box from the other side when it tries to get out.


The final method is a wrapper function which unpacks mc_states, which contains all the necessary arguments for the two methods above. When we have correctly implemented move_strat this wrapper will be expanded to include other methods
"""
function atom_displacement(pos, max_displacement, bc::SphericalBC)
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

function atom_displacement(pos, max_displacement, bc::CubicBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    trial_pos -= bc.box_length*SVector(round(trial_pos[1]/bc.box_length), round(trial_pos[2]/bc.box_length), round(trial_pos[3]/bc.box_length))
    return trial_pos
end

function atom_displacement(pos, max_displacement, bc::RhombicBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = pos + delta_move
    trial_pos -= SVector(bc.box_length*round((trial_pos[1]-trial_pos[2]/3^0.5-bc.box_length/2)/bc.box_length)+bc.box_length/2*round((trial_pos[2]-bc.box_length*3^0.5/4)/(bc.box_length*3^0.5/2)), bc.box_length*3^0.5/2*round((trial_pos[2]-bc.box_length*3^0.5/4)/(bc.box_length*3^0.5/2)), bc.box_height*round((trial_pos[3]-bc.box_height/2)/bc.box_height))
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
    volume_change(conf::Config, max_vchange, bc::PeriodicBC) 
scale the whole configuration, including positions and the box length.
returns the trial configuration as a struct. 
"""
function volume_change(conf::Config, bc::CubicBC, max_vchange, max_length)
    scale = exp((rand()-0.5)*max_vchange)^(1/3)
    if conf.bc.box_length >= max_length && scale > 1.
        scale=1.
    end
    trial_config = Config(conf.pos * scale,CubicBC(conf.bc.box_length * scale))
    return trial_config,scale
end

function volume_change(conf::Config, bc::RhombicBC, max_vchange, max_length)
    scale = exp((rand()-0.5)*max_vchange)^(1/3)
    if conf.bc.box_length >= max_length && scale > 1.
        scale=1.
    end

    trial_config = Config(conf.pos * scale,RhombicBC(conf.bc.box_length * scale, conf.bc.box_height * scale))
    return trial_config,scale
end


function scale_xy(pos::Vector,scale)
    new_pos=Array{SVector}(undef,length(pos))
    for i=1:length(pos)
        #pos[i] = setindex(pos[i], pos[i][1]*scale, 1)
        #pos[i] = setindex(pos[i], pos[i][2]*scale, 2)
        new_pos[i]=SVector(pos[i][1]*scale,pos[i][2]*scale,pos[i][3])
    end
    return new_pos
end
function scale_z(pos::Vector,scale)
    new_pos=Array{SVector}(undef,length(pos))
    for i=1:length(pos)
        #pos[i] = setindex(pos[i], pos[i][3]*scale, 3)
        new_pos[i]=SVector(pos[i][1],pos[i][2],pos[i][3]*scale)
    end
    return new_pos
end

function volume_change_xy(conf::Config, bc::RhombicBC, max_vchange, max_length, lh_ratio)
    scale = exp((rand()-0.5)*max_vchange)^(1/2)
    if conf.bc.box_length/conf.bc.box_height >= lh_ratio*1.2 && scale > 1.
        scale=1/scale
    elseif conf.bc.box_length/conf.bc.box_height <= lh_ratio*0.83 && scale < 1.
        scale=1/scale
    end
    if conf.bc.box_length>=max_length && scale > 1.
        scale=1/scale
    end
    trial_config = Config(scale_xy(conf.pos,scale),RhombicBC(conf.bc.box_length * scale, conf.bc.box_height))

    return trial_config,scale
end
function volume_change_z(conf::Config, bc::RhombicBC, max_vchange, max_height, lh_ratio)
    scale = exp((rand()-0.5)*max_vchange)
    if conf.bc.box_length/conf.bc.box_height <= lh_ratio*1.2 && scale > 1.
        scale=1/scale
    elseif conf.bc.box_length/conf.bc.box_height >= lh_ratio*0.83 && scale < 1.
        scale=1/scale
    end
    if conf.bc.box_height>=max_height && scale > 1.
        scale=1/scale
    end
    
    trial_config = Config(scale_z(conf.pos,scale),RhombicBC(conf.bc.box_length, conf.bc.box_height * scale))
    return trial_config,1/scale
end

function mat_scale!(new_dist2_mat::Matrix{Float64}, dist2_mat::Matrix{Float64}, scale::Float64)
    for i=1:32
        for j=1:i
            new_dist2_mat[i,j] = dist2_mat[i,j]*scale^2
            new_dist2_mat[j,i] = new_dist2_mat[i,j]
        end
    end
    return new_dist2_mat
end


function volume_change(mc_state::MCState)
    #change volume
    mc_state.ensemble_variables.trial_config, scale = volume_change(mc_state.config, mc_state.config.bc, mc_state.max_displ[2], mc_state.max_boxlength)
    #change r_cut
    mc_state.ensemble_variables.new_r_cut = get_r_cut(mc_state.ensemble_variables.trial_config.bc)

    
    #get the new dist2 matrix
    mc_state.ensemble_variables.new_dist2_mat .= mc_state.dist2_mat .* scale^2
    #mc_state.ensemble_variables.new_dist2_mat = mc_state.dist2_mat * scale^2
    #mc_state.ensemble_variables.new_dist2_mat = mat_scale!(mc_state.ensemble_variables.new_dist2_mat, mc_state.dist2_mat, scale)


    return mc_state
end

function volume_change_xyz(mc_state::MCState)
    #change volume
    ra=rand(1:3)
    if ra==1  # Choose z-direction volume change
        mc_state.ensemble_variables.xy_or_z=1        # Specify the type of volume moves chosen, z-direction
        mc_state.ensemble_variables.trial_config, scale = volume_change_z(mc_state.config, mc_state.config.bc, mc_state.max_displ[3], mc_state.max_boxheight,mc_state.lh_ratio)
    else
        mc_state.ensemble_variables.xy_or_z=0        # xy-direction
        mc_state.ensemble_variables.trial_config, scale = volume_change_xy(mc_state.config, mc_state.config.bc, mc_state.max_displ[2], mc_state.max_boxlength,mc_state.lh_ratio)
    end
        #change r_cut
    mc_state.ensemble_variables.new_r_cut = get_r_cut(mc_state.ensemble_variables.trial_config.bc)

    #get the new dist2 matrix
    mc_state.ensemble_variables.new_dist2_mat = get_distance2_mat(mc_state.ensemble_variables.trial_config)
    #println(typeof(mc_state.potential_variables))
    if typeof(mc_state.potential_variables) == ELJPotentialBVariables{Float64} || typeof(mc_state.potential_variables) == LookupTableVariables{Float64}
        mc_state.potential_variables.new_tan_mat = mc_state.potential_variables.tan_mat * scale
    end
    return mc_state
end

function volume_change(mc_state::MCState,separated_volume::Bool)
   if separated_volume==false
        mc_state=volume_change(mc_state)
    else
        mc_state=volume_change_xyz(mc_state)
    end
    return mc_state
end

"""
    generate_move!(mc_state,movetype::atommove)
    generate_move!(mc_state,movetype::volumemove)
generate move is the currying function that takes mc_state and a movetype 
and generates the variables required inside of the ensemblevariables struct within mc_state. 
"""
function generate_move!(mc_state::MCState,movetype::String,ensemble)
    if movetype == "atommove"
        return atom_displacement(mc_state)
    else
        return volume_change(mc_state,ensemble.separated_volume)
    end
end
# function generate_move!(mc_state,movetype::atommove)
#     return atom_displacement(mc_state)
# end
# function generate_move!(mc_state,movetype::volumemove)
#     return volume_change(mc_state)
# end


end 
