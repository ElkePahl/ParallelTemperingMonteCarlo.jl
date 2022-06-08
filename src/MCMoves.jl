module MCMoves

export atom_displacement

using StaticArrays

using ..BoundaryConditions

"""
    atom_displacement(pos, max_displacement, bc)

Generates trial position for atom, moving it from `pos` by some random displacement 
Random displacement determined by `max_displacement`

Implemented for:
    
    - `SphericalBC`: trial move is repeated until moved atom is within binding sphere
    - `CubicBC`: (to be added) periodic boundary condition enforced
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



end #module