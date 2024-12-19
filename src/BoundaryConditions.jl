""" 
    module BoundaryConditions

This module provides structs and methods for different kinds of boundary conditions.
        
"""
module BoundaryConditions
using StaticArrays
export SphericalBC, AbstractBC, PeriodicBC, CubicBC, RhombicBC
export check_boundary, PositionVector, PositionArray
"""
    PositionVector = Union{SVector{3, T}, Vector{T}} where T <: Number
Type alias for all kinds of acceptable position vectors.
"""
const PositionVector = Union{SVector{3, T}, Vector{T}} where T <: Number
"""
    PositionArray = Union{Vector{Vector{T}}, Vector{SVector{3, T}}} where T <: Number
Type alias for a list of positions.
"""
const PositionArray = Union{Vector{Vector{T}}, Vector{SVector{3, T}}} where T <: Number
# include("SphericalBC.jl")
# could be named SphericalBC.jl

"""   
    AbstractBC{T} 
Encompasses possible boundary conditions; implemented: 
-   [`SphericalBC`](@ref)
-   [`PeriodicBC`](@ref)

Needs methods implemented for
-   [`atom_displacement`](@ref ParallelTemperingMonteCarlo.MCMoves.atom_displacement)
"""
abstract type AbstractBC{T} end

"""
    SphericalBC{T}(;radius::Number)
Implements type for spherical boundary conditions; subtype of [`AbstractBC`](@ref).
Needs radius of binding sphere as keyword argument.
Fieldname: `radius2`: squared radius of binding sphere
"""
struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius of binding sphere squared
    SphericalBC(; radius::T) where T <: Number = new{T}(radius*radius)
end

"""
    PeriodicBC{T}
Overarching type of boundary condition for simulating the infinite bulk
-   Implemented types:
    -   [`CubicBC`](@ref)
    -   [`RhombicBC`](@ref)
"""
abstract type PeriodicBC{T} <: AbstractBC{T} end
"""
    CubicBC{T}(; side_length::Number)
Subtype of periodic boundary conditions where the `box_length` is isotropic.
"""
struct CubicBC{T} <: PeriodicBC{T}
    box_length::T
    CubicBC(; side_length::T) where T <: Number = new{T}(side_length)
    CubicBC{T}(x::T) where T <: Number = new{T}(x)
    CubicBC(x::T) where T <: Number = new{T}(x)
end
"""
    RhombicBC{T}(; length::Number, height::Number)
Subtype of periodic boundary condition where the `box_length` and `box_height` are not the same. The projection of the box on the xy-plane is a rhombus, `box_length` applies to all four sides.
"""
struct RhombicBC{T} <: PeriodicBC{T}
    box_length::T
    box_height::T
    RhombicBC(; length::T, height::T) where T <: Number = new{T}(length, height)
    RhombicBC{T}(x::T, y::T) where T <: Number = new{T}(x, y)
    RhombicBC(x::T, y::T) where T <: Number = new{T}(x, y)
end

"""
    check_boundary(bc::SpericalBC,pos::PositionVector) where T <: Number

Returns `true` when atom outside of spherical boundary
(squared norm of position vector < radius^2 of binding sphere).
"""
check_boundary(bc::SphericalBC,pos::PositionVector) = sum(x->x^2,pos) > bc.radius2


"""
    test_cluster_inside(pos::Vector{SVector{3,T}},bc::SphericalBC) where T <: Number
Tests if whole cluster lies in the binding sphere.
"""
test_cluster_inside(pos::Vector{SVector{3,T}},bc::SphericalBC) where T <: Number = sum(x->check_boundary(bc,x),pos) == 0

end
