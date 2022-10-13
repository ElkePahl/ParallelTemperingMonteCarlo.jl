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
export distance2, get_distance2_mat, move_atom!, check_bc

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


#= """
    move_atom!(pos, delta_move, bc)

Moves an atom at position `pos` by `delta_move`.
has to be defined for given boundary conditions
implemented for:
    - `SphericalBC`: trial move is repeated until moved atom is within binding sphere
    - `CubicBC`: (to be added) periodic boundary condition implemented

""" =#
#function move_atom!(config::Config, n_atom, delta_move)
#    config.pos[n_atom] += delta_move
#    return config
#end

#function move_atom!(pos,delta_move)
#    pos += delta_move
#    return pos
#end

#function move_atom!(config::Config, n_atom, delta_move,bc::SphericalBC)
#    config.pos[n_atom] += delta_move
#    return config
#end

"""
    distance2(a,b) 
    
Finds the distance between two positions a and b.
"""
distance2(a,b) = (a-b)⋅(a-b)

"""
    distance2(a,b,bc::CubicBC) 
    
Finds the distance between two positions a and the nearest image of b in a cubic box.
"""
distance2(a,b,bc::PeriodicBC) = distance2(a,b+[round((a[1]-b[1])/bc.box_length), round((a[2]-b[2])/bc.box_length), round((a[3]-b[3])/bc.box_length)]*bc.box_length)


#distance matrix
"""
    get_distance_mat(conf::Config{N})

Builds the matrix of squared distances between positions of configuration.
"""
get_distance2_mat(conf::Config{N}) where N = [distance2(a,b) for a in conf.pos, b in conf.pos]

get_distance2_mat(positions::Vector) where N = [distance2(a,b) for a in positions, b in positions]
"""
    checker(adjmat, config, trialpos ,atom_index,r2_cut)
This checks the boundary of Adjacency BC types 
"""
function check_bc(config, trialpos ,atom_index)
    adj_temp = copy(config.bc.adj_mat)
    bc_flag  = false

    dist2_new = [distance2(trialpos,b) for b in config.pos]
    dist2_new[atom_index] = 0.

    new_adj = [ifelse(a <=config.bc.r2_cut ,1,0) for a in dist2_new]

    adj_temp[:,atom_index] = new_adj
    adj_temp[atom_index,:] = new_adj

    for col in eachcol(adj_temp)
        dummysum = sum(col)
        if dummysum < 4
            bc_flag = true

            break
        end
    end
    
    return bc_flag, adj_temp, dist2_new
end
end
