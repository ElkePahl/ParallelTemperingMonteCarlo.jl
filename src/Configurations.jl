"""
    module Configurations

This module defines types and functions for working with atomic configurations of N atoms.

## Exported types  
- [`Config`](@ref) 

## Exported functions
- [`distance2`](@ref)
- [`get_distance_mat`](@ref)
- [`move_atom!`](@ref)
"""
module Configurations

using StaticArrays, LinearAlgebra

using ..BoundaryConditions

export Config
export distance2, get_distance2_mat, move_atom!

# """
#     Point(x::T,y::T,z::T)
# Generates a point with x,y, and z coordinate.

# See [`dist2`](@ref).
# """
# struct Point{T} #<: AbstractVector 
#     x::T
#     y::T
#     z::T
# end

# Base.size(::Point) = Tuple(3)
# Base.IndexStyle(::Point}) = IndexLinear()
# Base.getindex(::Point, i::Int) = 


# import Base: +

# (+)(p1::Point, p2::Point) = Point(p1.x+p2.x, p1.y+p2.y, p1.z+p2.z)
# (+)(p1::Point, v::Union{Tuple,AbstractVector}) = Point(p1.x+v[1], p1.y+v[2], p1.z+v[3])

# """
#     dist2(p1::Point,p2::Point) 
    
# Computes squared distance between two [`Point`](@ref)s.
# """
# dist2(p1::Point, p2::Point) = ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 +(p1.z - p2.z)^2)

# struct for configurations
# """
#     Config(points::Vector{Point}, boundary::AbstractBC)
#     Config{N}(positions::Vector{Point}, boundary::AbstractBC)
# Generate a configuration of `N` points.
# """
# struct Config{N, BC, T} 
#     pos::Vector{Point{T}}
#     bc::BC
# end

# #type constructors - not type stable! (as N can only be determined during run time when positions known)
# #one can check with @code_warntype
# function Config(positions::Vector{Point{T}}, boundary::BC) where {T,BC<:AbstractBC}
#     N = length(positions)
#     return Config{N,BC,T}(positions,boundary)
# end

# #type stable constructor as N is passed along (for compiler)
# function Config{N}(positions::Vector{Point{T}}, boundary::BC) where {N,T,BC<:AbstractBC}
#     @boundscheck length(positions) == N || error("number of atoms and number of positions not the same")
#     return Config{N,BC,T}(positions,boundary)
# end

# #overloads Base function length
# Base.length(::Config{N}) where N = N

#struct for configurations
"""
    Config(pos::Vector{SVector{3,T}}, bc::AbstractBC)
    Config{N}(positions::Vector{SVector{3,T}}, bc::AbstractBC)
    Config(pos, bc::BC) where {BC<:AbstractBC}
Generates a configuration of `N` atomic positions, each position saved as SVector of length 3.
Fieldnames: 
- `pos`: vector of x,y, and z coordinates of every atom 
- `bc`: boundary condition
"""
struct Config{N, BC, T} 
    pos::Vector{SVector{3,T}}
    bc::BC
end

#type constructors:

#not type stable! (as N can only be determined during run time when positions known)
#one can check with @code_warntype
function Config(pos::Vector{SVector{3,T}}, bc::BC) where {T,BC<:AbstractBC}
    N = length(pos)
    return Config{N,BC,T}(pos,bc)
end

#type stable constructor as N is passed along 
function Config{N}(pos::Vector{SVector{3,T}}, bc::BC) where {N,T,BC<:AbstractBC}
    @boundscheck length(pos) == N || error("number of atoms and number of positions not the same")
    return Config{N,BC,T}(pos,bc)
end

#not type stable, allows for input of positions as vector or tuples
function Config(pos, bc::BC) where {BC<:AbstractBC}
    poss = [SVector{3}(p[i] for i in 1:3) for p in pos]
    N = length(poss)
    T = eltype(poss[1])
    return Config{N,BC,T}(poss, bc)
end

#overloads Base function length
Base.length(::Config{N}) where N = N


#?pos as matrix

"""
    move_atom!(config::Config, n_atom, delta_move)

Moves `n_atom`-th atom in configuration by `delta_move`.  
"""
function move_atom!(config::Config, n_atom, delta_move)
    config.pos[n_atom] += delta_move
    return config
end

function move_atom!(pos,delta_move)
    pos += delta_move
    return pos
end


IF ( MOD(MCstep,100) == 0 ) THEN                        !100 is the number of steps after which step size is updated
        Adjust=REAL(Maccepted(t))/REAL(100*NMolecules)       !Denominator is multiplied by NMolecules because the EXTRA loop over the number of molecules
        IF ( Adjust < 0.4 ) THEN                             !number of accepted steps/number of total steps should
           maxDisplacement(t)=maxDisplacement(t)*0.9         !lie in [0.4,0.6]
        ELSEIF ( Adjust > 0.6 ) THEN
           maxDisplacement(t)=maxDisplacement(t)*1.1
        ENDIF
        Maccepted(t)=0                                       !It restarts to count only the conf accepted every 100 steps
     ENDIF

function update_max_stepsize!(displ, count_accept, NAtoms)
    for i in 1:length(count_acc)
        acc_rate =  count_accept[i] / (displ.update_step * NAtoms)
        if acc_rate < 0.4
            displ.max_displacement[i] *= 0.9
        elseif acc_rate > 0.6
            displ.max_displacement[i] *= 1.1
        end
        count_accept[i] = 0
    end
    return displ, count_accept
end

"""
    distance2(a,b) 
    
Finds the distance between two positions a and b.
"""
distance2(a,b) = (a-b)⋅(a-b)

#distance matrix
"""
    get_distance_mat(conf::Config{N})

Builds the matrix of squared distances between positions of configuration.
"""
get_distance2_mat(conf::Config{N}) where N = [distance2(a,b) for a in conf.pos, b in conf.pos]

end