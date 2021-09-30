module Configurations
# structs for configurations

export Config, Point
export dist2

using ..BoundaryConditions

"""
    Point(x::T,y::T,z::T)
Generates a point with x,y, and z coordinate.

See [`dist2`](@ref).
"""
struct Point{T}
    x::T
    y::T
    z::T
end

"""
    dist2(p1::Point,p2::Point) 
    
Computes squared distance between two [`Point`](@ref)s.
"""
dist2(p1::Point,p2::Point) = ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 +(p1.z - p2.z)^2)

# struct for configurations
"""
    Config(points::Vector{Point}, boundary::AbstractBC)
    Config{N}(positions::Vector{Point}, boundary::AbstractBC)
Generate a configuration of `N` points.
"""
struct Config{N, BC, T} 
    points::Vector{Point{T}}
    bc::BC
end

#type constructors - not type stable! (as N can only be determined during run time when positions known)
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


end