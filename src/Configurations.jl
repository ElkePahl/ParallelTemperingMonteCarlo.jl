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

using Test

using StaticArrays, LinearAlgebra

using ..BoundaryConditions

export Config
export distance2, distance2_dv, get_distance2_mat, move_atom!
export distance2_cbc, get_distance2_mat_cbc

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
function move_atom!(config::Config, n_atom, delta_move,bc::SphericalBC)
    config.pos[n_atom] += delta_move
    return config
end

function move_atom!(pos,delta_move, bc::SphericalBC)
    pos += delta_move
    return pos
end

function move_atom!(config::Config, n_atom, delta_move,bc::CubicBC)
    config.pos[n_atom] += delta_move
    config.pos[n_atom] -= [round(config,pos[n_atom][1]/config.bc.length), round(config,pos[n_atom][2]/config.bc.length), round(config,pos[n_atom][2]/config.bc.length)]
    return config
end

function move_atom!(pos,delta_move, bc::CubicBC)
    pos += delta_move
    pos -= [round(pos[1]/bc.length), round(pos[2]/bc.length), round(pos[3]/bc.length)]
    return pos
end

"""
    distance2(a,b) 
    
Finds the distance between two positions a and b.
"""
distance2(a,b) = (a-b)⋅(a-b)

function distance2_dv(a::SVector{3,Float64},b::SVector{3,Float64})
    d2=0
    for i=1:3
        d2+=(a[i]-b[i])^2
    end
    return d2
end

function distance2_dv(a::Vector,b::Vector)
    d2=0
    for i=1:3
        d2+=(a[i]-b[i])^2
    end
    return d2
end


function distance2_cbc(a,b,l)
    b_shift=zeros(3)
    b_shift=b+[round((a[1]-b[1])/l), round((a[2]-b[2])/l), round((a[3]-b[3])/l)]*l
    d2=distance2(a,b_shift)
    return d2
end

#a=[1.,1.,1.]
#b=[3.,3.,3.]
#println(distance2_cbc(a,b,3.5))

#distance matrix
"""
    get_distance_mat(conf::Config{N})

Builds the matrix of squared distances between positions of configuration.
"""
get_distance2_mat(conf::Config{N}) where N = [distance2(a,b) for a in conf.pos, b in conf.pos]

#get_distance2_mat_cbc(conf::Config{N}) where N = [distance2_cbc(a,b,conf.bc.length) for a in conf.pos, b in conf.pos]







end