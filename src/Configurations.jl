"""
    module Configurations

This module defines types and functions for working with atomic configurations of N atoms.

## Exported types  
- [`Point`](@ref) 
- [`Config`](@ref) 

## Exported functions
- [`dist2`](@ref)
- [`move_atom!`](@ref)
"""
module Configurations

export Config, Config1, Point
export dist2, move_atom!

using StaticArrays

using ..BoundaryConditions

"""
    Point(x::T,y::T,z::T)
Generates a point with x,y, and z coordinate.

See [`dist2`](@ref).
"""
struct Point{T} #<: AbstractVector 
    x::T
    y::T
    z::T
end

# Base.size(::Point) = Tuple(3)
# Base.IndexStyle(::Point}) = IndexLinear()
# Base.getindex(::Point, i::Int) = 


import Base: +

(+)(p1::Point, p2::Point) = Point(p1.x+p2.x, p1.y+p2.y, p1.z+p2.z)
(+)(p1::Point, v::Union{Tuple,AbstractVector}) = Point(p1.x+v[1], p1.y+v[2], p1.z+v[3])

"""
    dist2(p1::Point,p2::Point) 
    
Computes squared distance between two [`Point`](@ref)s.
"""
dist2(p1::Point, p2::Point) = ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 +(p1.z - p2.z)^2)

# struct for configurations
"""
    Config(points::Vector{Point}, boundary::AbstractBC)
    Config{N}(positions::Vector{Point}, boundary::AbstractBC)
Generate a configuration of `N` points.
"""
struct Config{N, BC, T} 
    pos::Vector{Point{T}}
    bc::BC
end

#type constructors - not type stable! (as N can only be determined during run time when positions known)
#one can check with @code_warntype
function Config(positions::Vector{Point{T}}, boundary::BC) where {T,BC<:AbstractBC}
    N = length(positions)
    return Config{N,BC,T}(positions,boundary)
end

#type stable constructor as N is passed along (for compiler)
function Config{N}(positions::Vector{Point{T}}, boundary::BC) where {N,T,BC<:AbstractBC}
    @boundscheck length(positions) == N || error("number of atoms and number of positions not the same")
    return Config{N,BC,T}(positions,boundary)
end

#overloads Base function length
Base.length(::Config{N}) where N = N

struct Config1{N, BC, T} 
    pos::Vector{SVector{3,T}}
    bc::BC
end

function Config1(pos::Vector{SVector{3,T}}, bc::BC) where {T,BC<:AbstractBC}
    N = length(pos)
    return Config1{N,BC,T}(pos,bc)
end

#type stable constructor as N is passed along (for compiler)
function Config1{N}(pos::Vector{SVector{3,T}}, bc::BC) where {N,T,BC<:AbstractBC}
    @boundscheck length(pos) == N || error("number of atoms and number of positions not the same")
    return Config1{N,BC,T}(pos,bc)
end

"""
    move_atom!(config::Config, n_atom, delta_move)

Moves n_atom-th atom in configuration by delta_move  
"""
function move_atom!(config, n_atom, delta_move)
    config.pos[n_atom] += delta_move
    return config
end

end