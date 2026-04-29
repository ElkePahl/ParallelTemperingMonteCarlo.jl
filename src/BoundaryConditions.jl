"""
    module BoundaryConditions

Provides structs and methods for different boundary conditions.

"""
module BoundaryConditions

using StaticArrays

export SphericalBC, AbstractBC, PeriodicBC, CubicBC, RhombicBC, RectangularBC
export check_boundary, long_range_correction, volume

"""
    check_boundary(bc::AbstractBC, position)

Check if `position` is within the boundaries of `bc` and move it back into the boundary (in
case of [`PeriodicBC`](@ref)), or return `nothing` if the position is invalid.
"""
check_boundary

"""
    long_range_correction(bc::AbstractBC, potential, num_atoms, r_cut)

Compute correction to energy from atoms outside the boundary condition. It is the integral
of all interaction outside the cutoff distance, using uniform density approximation. Returns
zero for [`SphericalBC`](@ref).
"""
long_range_correction

"""
    AbstractBC{T}

Is abstract type for boundary conditions.

# Implemented boundary conditions

- [`SphericalBC`](@ref)
- [`PeriodicBC`](@ref) with subtypes:
    - [`CubicBC`](@ref)
    - [`RhombicBC`](@ref)
    - [`RectangularBC`](@ref)

All subtypes should implement [`check_boundary`](@ref).
"""
abstract type AbstractBC{T} end

"""
    SphericalBC{T}(;radius::Real)

Implements type for spherical boundary conditions; subtype of [`AbstractBC`](@ref).

# Keywords:
- radius of binding sphere

# Fields:
- radius2: squared radius of binding sphere

"""
struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius of binding sphere squared
    SphericalBC(; radius::T) where {T<:Real} = new{T}(radius * radius)
end
function check_boundary(bc::SphericalBC, position)
    if sum(abs2, position) > bc.radius2
        return nothing
    else
        return position
    end
end
long_range_correction(::SphericalBC, _, _, _) = 0.0

"""
    PeriodicBC{T}

Is abstract type for periodic boundary conditions to simulate bulk systems.

# Implemented types
- [`CubicBC`](@ref)
- [`RhombicBC`](@ref)
- [`RectangularBC`](@ref)

In addition to the methods required by [`AbstractBC`](@ref), a `PeriodicBC` should
implement
- [`volume`](@ref)
- [`scale_xyz`](@ref)
- [`scale_xy`](@ref)
- [`scale_z`](@ref)
"""
abstract type PeriodicBC{T} <: AbstractBC{T} end

"""
    volume(::PeriodicBC)

Returns the volume of a box according to its geometry for use where the ensemble does not
imply a fixed `V`.
"""
volume

"""
    CubicBC{T}(; side_length::Real)

Is subtype of [`PeriodicBC`](@ref) for systems with cubic symmetry.

Keyword argument:
-    `side_length`: length of side of the cubic box

Field name:
-    `box_length`:  length of side of the cubic box
"""
struct CubicBC{T} <: PeriodicBC{T}
    box_length::T
    CubicBC(; side_length::T) where {T<:Real} = new{T}(side_length)
    CubicBC{T}(x::T) where {T<:Real} = new{T}(x)
    CubicBC(x::T) where {T<:Real} = new{T}(x)
end
function volume(bc::CubicBC)
    return bc.box_length^3
end
function check_boundary(bc::CubicBC, position)
    return position - bc.box_length * SVector(
        round(position[1] / bc.box_length),
        round(position[2] / bc.box_length),
        round(position[3] / bc.box_length),
    )
end
function long_range_correction(bc::CubicBC, potential, num_atoms, r_cut)
    return long_range_correction(args, potential, num_atoms, r_cut)
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
function volume(bc::RectangularBC)
    return bc.box_length^2 * bc.box_height
end
function check_boundary(bc::RectangularBC, position)
    return position - SVector(
        bc.box_length * round(position[1] / bc.box_length),
        bc.box_length * round(position[2] / bc.box_length),
        bc.box_height * round(position[3] / bc.box_height),
    )
end
function long_range_correction(bc::RectangularBC, potential, num_atoms, r_cut)
    lrc = long_range_correction(potential, num_atoms, r_cut)
    if bc.box_length < bc.box_height
        return lrc * bc.box_length / bc.box_height
    else
        return lrc * bc.box_height^2 / bc.box_length^2
    end
end

"""
    RhombicBC{T}(; length::Real, height::Real)

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
    RhombicBC(; length::T, height::T) where {T<:Real} = new{T}(length, height)
    RhombicBC{T}(x::T, y::T) where {T<:Real} = new{T}(x, y)
    RhombicBC(x::T, y::T) where {T<:Real} = new{T}(x, y)
end
function volume(bc::RhombicBC)
    return bc.box_length^2 * bc.box_height * 3^0.5 / 2
end
function check_boundary(bc::RhombicBC, position)
    return position - SVector(
        bc.box_length *
        round((position[1] - position[2] / 3^0.5 - bc.box_length / 2) / bc.box_length) +
        bc.box_length / 2 *
        round((position[2] - bc.box_length * 3^0.5 / 4) / (bc.box_length * 3^0.5 / 2)),
        bc.box_length * 3^0.5 / 2 *
        round((position[2] - bc.box_length * 3^0.5 / 4) / (bc.box_length * 3^0.5 / 2)),
        bc.box_height * round((position[3] - bc.box_height / 2) / bc.box_height),
    )
end
function long_range_correction(bc::RhombicBC, potential, num_atoms, r_cut)
    return long_range_correction(potential, num_atoms, r_cut) *
        3bc.box_length / 4bc.box_height
end

end
