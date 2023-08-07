module MCMoves

export MoveStrategy, atom_move_frequency, vol_move_frequency, rot_move_frequency
export AbstractMove, StatMove
export AtomMove
export atom_displacement,generate_displacements #, update_max_stepsize!

using StaticArrays

using ..MCStates
using ..BoundaryConditions
using ..Configurations

"""
    MoveStrategy(atom_moves, vol_moves, rot_moves)
    MoveStrategy(;atom_moves=1, vol_moves=0, rot_moves=0)
Type that implements move strategy containing information of frequencies of moves:
 - atom_moves:  frequency of atom moves
 - vol_moves:  frequency of volume moves
 - rot_moves:  frequency of rotation moves
"""
struct MoveStrategy{A,V,R}
end 

function MoveStrategy(a,v,r)
    return MoveStrategy{a,v,r}()
end

function MoveStrategy(;atom_moves=1, vol_moves=0, rot_moves=0)
    return MoveStrategy(atom_moves,vol_moves,rot_moves)
end 

"""
    atom_move_frequency(ms::MoveStrategy{A,V,R})
gives frequency of atom moves    
"""
atom_move_frequency(ms::MoveStrategy{A,V,R}) where {A,V,R} = A 
"""
    vol_move_frequency(ms::MoveStrategy{A,V,R})
gives frequency of volume moves    
"""
vol_move_frequency(ms::MoveStrategy{A,V,R}) where {A,V,R} = V
"""
    rot_move_frequency(ms::MoveStrategy{A,V,R})
gives frequency of rotation moves    
"""
rot_move_frequency(ms::MoveStrategy{A,V,R}) where {A,V,R} = R

"""
    AbstractMove
abstract type for possible Monte Carlo moves
implemented: 
    - AtomMove
"""
abstract type AbstractMove end

"""
    AtomMove
implements type for atom move (random displacement of randomly selected atom)
field name: frequency: number of moves per Monte Carlo cycle
            max_displacement: max. displacement for move per temperature (updated during MC run)
            n_update_stepsize: number of MC cycles between update of max. displacement
            count_acc: total number of accepted atom moves
            count_acc_adj: number of accepted moves between stepsize adjustments 
"""
mutable struct AtomMove{T} <: AbstractMove
    frequency::Int
    max_displacement::T
    n_update_stepsize::Int
    count_acc::Int       
    count_acc_adj::Int 
end

function AtomMove(frequency, displ; update_stepsize=100, count_acc=0, count_acc_adj=0)
    T = eltype(displ)
    return AtomMove{T}(frequency, displ, update_stepsize, count_acc, count_acc_adj)
end

"""
    atom_displacement(pos, max_displacement, bc)
    atom_displacement(mc_states,index)

Generates trial position for atom, moving it from `pos` by some random displacement 
Random displacement determined by `max_displacement`
These variables are additionally contained in `mc_state` where the pos is determined by `index`.
Implemented for:
    
    - `SphericalBC`: trial move is repeated until moved atom is within binding sphere
    - `PeriodicBC`: periodic boundary condition enforced, an atom is moved into the box from the other side when it tries to get out.

The final method is a wrapper function which unpacks mc_states, which contains all the necessary arguments for the two methods above. When we have correctly implemented move_strat this wrapper will be expanded to include other methods
"""
function atom_displacement(state, index, max_displacement, bc::SphericalBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = state.config.pos[index] + delta_move
    count = 0
    while check_boundary(bc, trial_pos)         #displace the atom until it's inside the binding sphere
        count += 1
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = state.config.pos[index] + delta_move
        count == 100 && error("Error: too many moves out of binding sphere")
    end
    return trial_pos
end
function atom_displacement(state, index, max_displacement, bc::AdjacencyBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = state.config.pos[index] + delta_move
    count = 0
    trial_dist2mat = copy(state.dist2_mat)
    for (ind,i) in enumerate(state.config.pos)
        trial_dist2mat[index,ind] = distance2(trial_pos,i)
        trial_dist2mat[ind,index] = trial_dist2mat[index,ind]
    end
    while check_boundary(bc, trial_dist2mat, trial_pos)         #displace the atom until it's inside the binding sphere
        count += 1
        delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
        trial_pos = state.config.pos[index] + delta_move
        for (ind,i) in enumerate(state.config.pos)
            trial_dist2mat[index,ind] = distance2(trial_pos,i)
            trial_dist2mat[ind,index] = trial_dist2mat[index,ind]
        end
        count == 10000 && error("Error: too many moves out of boundary")
    end
    return trial_pos
end
function atom_displacement(state, index, max_displacement, bc::PeriodicBC)
    delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)
    trial_pos = state.config.pos[index] + delta_move
    trial_pos -= [round(trial_pos[1]/bc.box_length), round(trial_pos[2]/bc.box_length), round(trial_pos[3]/bc.box_length)]
    return trial_pos
end
function atom_displacement(mc_state,index)
    #if index <= length(mc_state.config.pos)
        trial_pos = atom_displacement(mc_state,index,mc_state.max_displ[1],mc_state.config.bc)
    #end

    return trial_pos
end 
"""
    function generate_displacements(mc_states,mc_params)
generates a perturbed atom per trajectory. Accepts `mc_states` as a vector of mc_state structs and `mc_params` to define the vector lengths. Output is a vector of `indices` indicating which atom per trajectory has been used to generate `trial positions` 
"""
function generate_displacements(mc_states,mc_params)
    indices=rand(1:mc_params.n_atoms,mc_params.n_traj)
    trial_positions = atom_displacement.(mc_states,indices)
    return indices,trial_positions
end


# #This method will supercede the above when multiple moves are implemented. Perhaps a more dynamic move strategy system will improve this?

# function generate_displacements(mc_states,mc_params,a,v,r)
#     indices=rand(1:(a+v+r),mc_params.n_traj)
#     trial_positions = atom_displacement.(mc_states,indices)
#     return indices,trial_positions
# end


"""
    function volume_change(conf::Config, max_vchange, bc::PeriodicBC) 
scale the whole configuration, including positions and the box length.
returns the trial configuration as a struct. 
"""
function volume_change(conf::Config, max_vchange)
    scale = exp((rand()-0.5)*max_vchange)^(1/3)
    trial_config = Config(conf.pos * scale, PeriodicBC(conf.bc.box_length * scale))
    return trial_config
end

# """ ----- OLD VERSION NOT USED---------#
#     update_max_stepsize!(displ, n_update, count_accept, n_atom)
# update of maximum step size of atom moves after n_update MC cyles (n_atom moves per cycle)
# depends on ratio of accepted moves - as given in count_accept - 
# for acceptance ratio < 40% max_displacement is reduced to 90% of its value,
# for acceptance ratio > 60% max_displacement is increased to 110% of its value.
# count_accept is set back to zero at end.      
# """
# function update_max_stepsize!(displ, n_update, count_accept, n_atom)
#     for i in 1:length(count_accept)
#         acc_rate =  count_accept[i] / (n_update * n_atom)
#         if acc_rate < 0.4
#             displ[i] *= 0.9
#         elseif acc_rate > 0.6
#             displ[i] *= 1.1
#         end
#         count_accept[i] = 0
#     end
#     return displ, count_accept
# end

end #module
