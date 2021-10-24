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


#default configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne13 =[[2.825384495892464, 0.928562467914040, 0.505520149314310],
[2.023342172678102,	-2.136126268595355, 0.666071287554958],
[2.033761811732818,	-0.643989413759464, -2.133000349161121],
[0.979777205108572,	2.312002562803556, -1.671909307631893],
[0.962914279874254,	-0.102326586625353, 2.857083360096907],
[0.317957619634043,	2.646768968413408, 1.412132053672896],
[-2.825388342924982, -0.928563755928189, -0.505520471387560],
[-0.317955944853142, -2.646769840660271, -1.412131825293682],
[-0.979776174195320, -2.312003751825495, 1.671909138648006],
[-0.962916072888105, 0.102326392265998,	-2.857083272537599],
[-2.023340541398004, 2.136128558801072,	-0.666071089291685],
[-2.033762834001679, 0.643989905095452, 2.132999911364582],
[0.000002325340981,	0.000000762100600, 0.000000414930733]]

bc_ne13 = SphericalBC(radius=5.32)   #Angstrom

conf_ne13 = Config(pos_ne13, bc_ne13)


bc_ar32 = SphericalBC(radius=14.5)  #Angstrom

#Ar32 starting config (fcc, pbc)
# pos_ar32 = 0.2000000000E+02  0.2000000000E+02  0.2000000000E+02
# -0.1000000000E+02 -0.1000000000E+02 -0.1000000000E+02
# -0.5000000000E+01 -0.5000000000E+01 -0.1000000000E+02
# -0.5000000000E+01 -0.1000000000E+02 -0.5000000000E+01
# -0.1000000000E+02 -0.5000000000E+01 -0.5000000000E+01
# -0.1000000000E+02 -0.1000000000E+02  0.0000000000E+00
# -0.5000000000E+01 -0.5000000000E+01  0.0000000000E+00
# -0.5000000000E+01 -0.1000000000E+02  0.5000000000E+01
# -0.1000000000E+02 -0.5000000000E+01  0.5000000000E+01
# -0.1000000000E+02  0.0000000000E+00 -0.1000000000E+02
# -0.5000000000E+01  0.5000000000E+01 -0.1000000000E+02
# -0.5000000000E+01  0.0000000000E+00 -0.5000000000E+01
# -0.1000000000E+02  0.5000000000E+01 -0.5000000000E+01
# -0.1000000000E+02  0.0000000000E+00  0.0000000000E+00
# -0.5000000000E+01  0.5000000000E+01  0.0000000000E+00
# -0.5000000000E+01  0.0000000000E+00  0.5000000000E+01
# -0.1000000000E+02  0.5000000000E+01  0.5000000000E+01
#  0.0000000000E+00 -0.1000000000E+02 -0.1000000000E+02
#  0.5000000000E+01 -0.5000000000E+01 -0.1000000000E+02
#  0.5000000000E+01 -0.1000000000E+02 -0.5000000000E+01
#  0.0000000000E+00 -0.5000000000E+01 -0.5000000000E+01
#  0.0000000000E+00 -0.1000000000E+02  0.0000000000E+00
#  0.5000000000E+01 -0.5000000000E+01  0.0000000000E+00
#  0.5000000000E+01 -0.1000000000E+02  0.5000000000E+01
#  0.0000000000E+00 -0.5000000000E+01  0.5000000000E+01
#  0.0000000000E+00  0.0000000000E+00 -0.1000000000E+02
#  0.5000000000E+01  0.5000000000E+01 -0.1000000000E+02
#  0.5000000000E+01  0.0000000000E+00 -0.5000000000E+01
#  0.0000000000E+00  0.5000000000E+01 -0.5000000000E+01
#  0.0000000000E+00  0.0000000000E+00  0.0000000000E+00
#  0.5000000000E+01  0.5000000000E+01  0.0000000000E+00
#  0.5000000000E+01  0.0000000000E+00  0.5000000000E+01
#  0.0000000000E+00  0.5000000000E+01  0.5000000000E+01

end