"""
    module BoundaryConditions

Provides structs and methods for different boundary conditions.

"""
module BoundaryConditions

using StaticArrays
using ..CustomTypes

export SphericalBC, AbstractBC, PeriodicBC, CubicBC, RhombicBC, RectangularBC
export check_boundary

"""
    AbstractBC{T}

Is abstract type for boundary conditions. 

Implemented boundary conditions:
-   `SphericalBC`
-   `PeriodicBC` with subtypes:
    -   `CubicBC`
    -   `RhombicBC`
    -   `RectangularBC`

Needs methods implemented for
-   [`atom_displacement`](@ref Main.ParallelTemperingMonteCarlo.MCMoves.atom_displacement)
"""
abstract type AbstractBC{T} end

"""
    SphericalBC{T}(;radius::Number)

Implements type for spherical boundary conditions; subtype of [`AbstractBC`](@ref).

# Keywords:
- radius of binding sphere

# Fields: 
- radius2: squared radius of binding sphere

"""
struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius of binding sphere squared
    SphericalBC(; radius::T) where T <: Number = new{T}(radius*radius)
end

"""
    PeriodicBC{T}

Is abstract type for periodic boundary conditions to simulate bulk systems.

- Implemented types:
    - `CubicBC`
    - `RhombicBC`
    - `RectangularBC`
"""
abstract type PeriodicBC{T} <: AbstractBC{T} end

"""
    CubicBC{T}(; side_length::Number)

Is subtype of [`PeriodicBC`](@ref) for systems with cubic symmetry.

Keyword argument:
-    `side_length`: length of side of the cubic box

Field name:
-    `box_length`:  length of side of the cubic box
"""
struct CubicBC{T} <: PeriodicBC{T}
    box_length::T
    CubicBC(; side_length::T) where T <: Number = new{T}(side_length)
    CubicBC{T}(x::T) where T <: Number = new{T}(x)
    CubicBC(x::T) where T <: Number = new{T}(x)
end

"""
    RectangularBC{T}

Is subtype of [`PeriodicBC`](@ref) for systems with rectangular symmetry
(orthogonal axes with length of box in ``x,y`` direction differs from height of box in ``z``-direction).

# Fields:
- `box_length`: length of side of square in ``x,y`` direction
- `box_height`: height of the box in ``z`` direction
"""
struct RectangularBC{T} <: PeriodicBC{T}
    box_length::T
    box_height::T
end

# TODO  check how exactly implemented (height is length of side or projection on z-axis?)
"""
    RhombicBC{T}(; length::Number, height::Number)

Is subtype of [`PeriodicBC`](@ref) for systems with rhombic symmetry
(length of box in ``x,y`` direction differs from height of box in ``z``-direction).
The projection of the box on the ``xy``-plane is a rhombus with four equal sides. 

# Keywords
- `length`: length of box in ``x,y`` direction
- `height`: height of the box in ``z`` direction

# Fields:
- `box_length`: length of side of the cubic box
- `box_height`: height of the box in ``z`` direction
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

Checks if atom moved outside of spherical boundary
(squared norm of position vector smaller than squared radius of binding sphere).
Returns `true` if atom lies outside.

# Arguments
- [`SphericalBC`](@ref)
- `pos`: position of moved atom

"""
check_boundary(bc::SphericalBC,pos::PositionVector) = sum(x->x^2,pos) > bc.radius2

"""
    test_cluster_inside(pos::Vector{SVector{3,T}},bc::SphericalBC) where T <: Number

Tests if whole cluster lies in the binding sphere.

# Arguments
- atomic positions
- [`SphericalBC`](@ref)
"""
test_cluster_inside(pos::Vector{SVector{3,T}},bc::SphericalBC) where {T <: Number} = sum(x->check_boundary(bc,x),pos) == 0
# TO DO: check if this function is used at all? If so, make consistent with check_boundary

end
