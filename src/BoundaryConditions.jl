"""
    module BoundaryConditions

Provides structs and methods for different boundary conditions.

"""
module BoundaryConditions

using StaticArrays

export SphericalBC, AbstractBC, PeriodicBC, CubicBC, RhombicBC, RectangularBC
export check_boundary, long_range_correction, volume, r_cut

"""
    AbstractBC{T}

Abstract type for boundary conditions.

# Implemented boundary conditions

- [`SphericalBC`](@ref)
- [`PeriodicBC`](@ref) with subtypes:
    - [`CubicBC`](@ref)
    - [`RhombicBC`](@ref)
    - [`RectangularBC`](@ref)

# Interface

Basic interface:
- [`check_boundary`](@ref) (required)
- [`r_cut`](@ref) (optional for non-periodic boundaries, defaults to returning `Inf`)
- [`long_range_correction`](@ref) (optional for non-periodic boundaries, defaults to
  returning zero)

Required for use with the [`NPT`](@ref Main.ParallelTemperingMonteCarlo.Ensembles.NPT)
ensemble:
- [`volume`](@ref)
- [`scale_xyz`](@ref)
- [`scale_xy`](@ref)
- [`scale_z`](@ref)

"""
abstract type AbstractBC{T} end

"""
    check_boundary(bc::AbstractBC, position)

Check if `position` is within the boundaries of `bc` and move it back into the boundary (in
case of [`PeriodicBC`](@ref)), or return `nothing` if the position is invalid.
"""
check_boundary

"""
    long_range_correction(bc::AbstractBC, potential, num_atoms)
    long_range_correction(potential, num_atoms, r_cut)

Compute correction to energy from atoms outside the boundary condition, e.g. an integral
of all interaction outside the cutoff distance using uniform density approximation.

The first method should call the second and multiply it with an appropriate factor (for
periodic boundary conditions) or return zero (for boundary conditions where a long range
correction is not necessary).

The second method only needs to be defined for a given potential for it to be usable with
periodic boundary conditions.
"""
long_range_correction(::AbstractBC, _, _) = 0.0

"""
    r_cut(::AbstractBC)

The square of the cut-off radius `r_cut` that is implied by periodic boundary conditions to
avoid double-counting. Defaults to returning `Inf`, and as such only needs to be implemented
for periodic boundary conditions.
"""
r_cut(::AbstractBC) = Inf

"""
    volume(::AbstractBC)

Returns the volume of a box according to its geometry for use where the ensemble does not
imply a fixed `V`.
"""
volume(::AbstractBC) = missing # TODO: avoid computing volume for non NPT ensembles

"""
    scale_xyz(::AbstractBC, α)
    scale_xyz(::Vector{<:SVector}, α)
    scale_xyz(::Config, α)

Scale boundary condition, vector, or configuration in all three dimensions by factor `α`.
"""
scale_xyz

"""
    scale_xy(::AbstractBC, α)
    scale_xy(::Vector{<:SVector}, α)
    scale_xy(::Config, α)

Scale boundary condition, vector, or configuration in all ``x`` and ``y`` dimensions by
factor `α`.
"""
scale_xy

"""
    scale_z(::AbstractBC, α)
    scale_z(::Vector{<:SVector}, α)
    scale_z(::Config, α)

Scale boundary condition, vector, or configuration in the ``z`` dimension by factor `α`.
"""
scale_z

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

"""
    PeriodicBC{T}

Is abstract type for periodic boundary conditions to simulate bulk systems.

# Implemented types
- [`CubicBC`](@ref)
- [`RhombicBC`](@ref)
- [`RectangularBC`](@ref)

A `PeriodicBC` should implement [`long_range_correction`](@ref) and [`r_cut`](@ref).
"""
abstract type PeriodicBC{T} <: AbstractBC{T} end

# Override defaults, so they are necessary to implement for PeriodicBC.
long_range_correction(bc::PeriodicBC, pot, n, r_cut) = throw(MethodError(bc, pot, n, r_cut))
r_cut(bc::PeriodicBC) = throw(MethodError(r_cut, bc))

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
    return position .- bc.box_length .* round.(position ./ bc.box_length)
end
function long_range_correction(bc::CubicBC, potential, num_atoms)
    return long_range_correction(potential, num_atoms, r_cut(bc))
end

scale_xyz(bc::CubicBC, α) = CubicBC(α * bc.box_length)

r_cut(bc::CubicBC) = bc.box_length^2 / 4

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
    box_size = SVector(bc.box_length, bc.box_length, bc.box_height)
    return position .- box_size .* round.(position ./ box_size)
end
function long_range_correction(bc::RectangularBC, potential, num_atoms)
    lrc = long_range_correction(potential, num_atoms, r_cut(bc))
    if bc.box_length < bc.box_height
        return lrc * bc.box_length / bc.box_height
    else
        return lrc * bc.box_height^2 / bc.box_length^2
    end
end

scale_xyz(bc::RectangularBC, α) = RectangularBC(α * bc.box_length, α * bc.box_height)
scale_xy(bc::RectangularBC, scale) = RectangularBC(bc.box_length * scale, bc.box_height)
scale_z(bc::RectangularBC, scale) = RectangularBC(bc.box_length, bc.box_height * scale)

r_cut(bc::RectangularBC) = min(bc.box_length^2 / 4, bc.box_height^2 / 4)

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
    return bc.box_length^2 * bc.box_height * √3 / 2
end
function check_boundary(bc::RhombicBC, position)
    return position - SVector(
        bc.box_length *
        round((position[1] - position[2] / √3 - bc.box_length / 2) / bc.box_length) +
        bc.box_length / 2 *
        round((position[2] - bc.box_length * √3 / 4) / (bc.box_length * √3 / 2)),
        bc.box_length * √3 / 2 *
        round((position[2] - bc.box_length * √3 / 4) / (bc.box_length * √3 / 2)),
        bc.box_height * round((position[3] - bc.box_height / 2) / bc.box_height),
    )
end
function long_range_correction(bc::RhombicBC, potential, num_atoms)
    return long_range_correction(potential, num_atoms, r_cut(bc)) * 3bc.box_length /
           4bc.box_height
end

scale_xyz(bc::RhombicBC, α) = RhombicBC(α * bc.box_length, α * bc.box_height)
scale_xy(bc::RhombicBC, scale) = RhombicBC(bc.box_length * scale, bc.box_height)
scale_z(bc::RhombicBC, scale) = RhombicBC(bc.box_length, bc.box_height * scale)

r_cut(bc::RhombicBC) = min(bc.box_length^2 * 3 / 16, bc.box_height^2 / 4)

end
