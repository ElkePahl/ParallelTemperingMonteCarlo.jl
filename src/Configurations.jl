module Configurations
# structs for configurations

export Config, Point

using ..BoundaryConditions

struct Point{T}
    x::T
    y::T
    z::T
end

# struct 
"""
    Config(points::Vector{Point}, boundary::AbstractBC)
    Config{N}(positions::Vector{Point}, boundary::AbstractBC)
Generate a configuration of `N` points.
"""
struct Config{N, BC, T} 
    points::Vector{Point{T}}
    bc::BC
end

#type constructor
function Config(positions::Vector{Point{T}}, boundary::BC) where {T,BC<:AbstractBC}
    N = length(positions)
    return Config{N,BC,T}(positions,boundary)
end

function Config{N}(positions::Vector{Point{T}}, boundary::BC) where {N,T,BC<:AbstractBC}
    @boundscheck length(positions) == N || error("number of atoms and number of positions not the same")
    return Config{N,BC,T}(positions,boundary)
end

Base.length(::Config{N}) where N = N


end