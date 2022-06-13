module MCMoves

export AbstractMove
export AtomMove
export atom_displacement, update_max_stepsize!

using StaticArrays

using ..BoundaryConditions

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
"""
struct AtomMove{T} <: AbstractMove
    frequency::Int
    max_displacement::Vector{T}
    n_update_stepsize::Int
end

function AtomMove(frequency, displ, tgrid; update_stepsize=100)
    T = eltype(displ)
    N = length(tgrid)
    #initialize displacement vector
    max_displacement = [0.1*sqrt(displ*tgrid[i]) for i in 1:N]
    return AtomMove{T}(frequency, max_displacement, update_stepsize)
end

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

"""
    update_max_stepsize!(displ, n_update, count_accept, n_atom)
update of maximum step size of atom moves after n_update MC cyles (n_atom moves per cycle)
depends on ratio of accepted moves - as given in count_accept - 
for acceptance ratio < 40% max_displacement is reduced to 90% of its value,
for acceptance ratio > 60% max_displacement is increased to 110% of its value.
count_accept is set back to zero at end.      
"""
function update_max_stepsize!(displ, n_update, count_accept, n_atom)
    for i in 1:length(count_accept)
        acc_rate =  count_accept[i] / (n_update * n_atom)
        if acc_rate < 0.4
            displ[i] *= 0.9
        elseif acc_rate > 0.6
            displ[i] *= 1.1
        end
        count_accept[i] = 0
    end
    return displ, count_accept
end



end #module