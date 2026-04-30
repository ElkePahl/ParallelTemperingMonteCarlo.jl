"""
    module Configurations

This module defines types and functions for working with atomic configurations of N atoms.

## Exported types
-   [`Config`](@ref)

## Exported functions
-   [`distance2`](@ref)
-   [`get_distance2_mat`](@ref)
"""
module Configurations

using StaticArrays, LinearAlgebra, Statistics

using ..BoundaryConditions
using ..CustomTypes

import ..BoundaryConditions: scale_xyz, scale_xy, scale_z
export scale_xyz, scale_xy, scale_z

export Config
export distance2,
    get_distance2_mat, get_distance2_mat!, get_tan, get_tantheta_mat, get_tantheta_mat!
export get_centre, recentre!

"""
    struct Config{T<:Real,B}(positions, boundary_conditions) <: AbstractVector{SVector{3,T}}

Configuration of atoms with a boundary condition. `positions` can be given as vector of
vectors or 3-tuples.

Behaves as a vector of positions.

# Fields:
- `positions::Vector{SVector{3,<:Real}}`: vector of x,y, and z coordinates of every atom.
- `boundary_condition::AbstractBC`: boundary condition.
"""
struct Config{T<:AbstractFloat,B} <: AbstractVector{SVector{3,T}}
    positions::Vector{SVector{3,T}}
    boundary_condition::B
end
function Config(positions, boundary_condition)
    return Config(
        [SVector{3}(float(p[1]), float(p[2]), float(p[3])) for p in positions],
        boundary_condition,
    )
end

Base.size(config::Config, args...) = size(config.positions, args...)
Base.getindex(config::Config, key) = getindex(config.positions, key)
function Base.setindex!(config::Config{T}, val, key) where {T}
    return config.positions[key] = SVector{3,T}(val[1], val[2], val[3])
end
function Base.summary(io::IO, config::Config)
    return print(io, length(config), "-element Config with ", config.boundary_condition)
end

"""
    get_centre(positions)

Find the centre of mass of a configuration.
"""
function get_centre(positions)
    return SVector(
        mean(pos[1] for pos in positions),
        mean(pos[2] for pos in positions),
        mean(pos[3] for pos in positions),
    )
end
"""
    recentre!(positions)

Change the centre of mass of a configuration `positions` to [0,0,0] in Cartesian space.
"""
function recentre!(positions)
    centre = get_centre(positions)
    for i in eachindex(positions)
        positions[i] = positions[i] - centre
    end
    return positions
end
"""
    distance2(a::PositionVector,b::PositionVector)
    distance2(a::PositionVector,b::PositionVector,bc::SphericalBC)
    distance2(a::PositionVector,b::PositionVector,bc::CubicBC)
    distance2(a::PositionVector,b::PositionVector,bc::RhombicBC)
    distance2(a::PositionVector,b::PositionVector,bc::RectangularBC)
Method 1&2 -
Finds the distance between two positions a and b.
Method 3 -
Finds the distance between two positions a and the nearest image of b in a cubic box.
Method 4 -
Finds the distance between two positions a and the nearest image of b in a rhombic box.
Minimum image convension in the z-direction is the same as the cubic box.
In x and y-direction, first the box is transformed into a rectangular box, then MIC is done, finally the new coordinates are transformed back.
"""
distance2(a::PositionVector, b::PositionVector) = (a - b) ⋅ (a - b)

distance2(a::PositionVector, b::PositionVector, bc::SphericalBC) = distance2(a, b)
function distance2(a::PositionVector, b::PositionVector, bc::CubicBC)
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_length * round((a[3] - b[3]) / bc.box_length)
    return distance2(a, SVector(b_x, b_y, b_z))
end
function distance2(a::PositionVector, b::PositionVector, bc::RhombicBC)
    b_y =
        b[2] +
        (3^0.5 / 2 * bc.box_length) * round((a[2] - b[2]) / (3^0.5 / 2 * bc.box_length))
    b_x =
        b[1] - b[2] / 3^0.5 +
        bc.box_length * round(((a[1] - b[1]) - 1 / 3^0.5 * (a[2] - b[2])) / bc.box_length) +
        1 / 3^0.5 * b_y
    b_z = b[3] + bc.box_height * round((a[3] - b[3]) / bc.box_height)
    return distance2(a, SVector(b_x, b_y, b_z))
end
function distance2(a, b, bc::RectangularBC)
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_height * round((a[3] - b[3]) / bc.box_height)
    return distance2(a, SVector(b_x, b_y, b_z))
end

#distance matrix

"""
    get_distance2_mat(conf::Config)

Builds the matrix of squared distances between positions of configuration.
"""
function get_distance2_mat(config::Config)
    mat = zeros(length(config), length(config))
    get_distance2_mat!(mat, config)
    return mat
end

"""
    get_distance2_mat(conf::Config)

In-place version of [`get_distance2_mat`](@ref).
"""
function get_distance2_mat!(dest, config::Config)
    @boundscheck if size(dest) ≠ (length(config), length(config))
        throw(
            DimensionMismatch(
                "invalid dimension of destination $(size(dest)) for" *
                " a Config of length $(length(config))",
            ),
        )
    end
    @inbounds for i in 1:length(config), j in (i + 1):length(config)
        dest[i, j] = dest[j, i] = distance2(config[i], config[j], config.boundary_condition)
    end
    @inbounds for i in eachindex(config)
        dest[i, i] = 0
    end
    return dest
end

"""
    get_tan(a::PositionVector,b::PositionVector)
    get_tan(a::PositionVector,b::PositionVector,bc::SphericalBC)
    get_tan(a::PositionVector,b::PositionVector,bc::CubicBC)
    get_tan(a::PositionVector,b::PositionVector,bc::RhombicBC)
    get_tan(a::PositionVector,b::PositionVector,bc::RectangularBC)
Method 1&2 :
tan of the angle between the line connecting two points a and b, and the z-direction
Method 3:
tan of the angle between the line connecting two points a and the nearest image of b, and the z-direction in a cubic boundary
Method 4:
tan of the angle between the line connecting two points a and the nearest image of b, and the z-direction in a rhombic boundary
"""
function get_tan(a, b)
    tan = ((a[1] - b[1])^2 + (a[2] - b[2])^2)^0.5 / (a[3] - b[3])
    return abs(tan)
end
function get_tan(a, b, bc::SphericalBC)
    return get_tan(a, b)
end
function get_tan(a, b, bc::CubicBC)
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_length * round((a[3] - b[3]) / bc.box_length)
    tan = ((a[1] - b_x)^2 + (a[2] - b_y)^2)^0.5 / (a[3] - b_z)
    return abs(tan)
end
function get_tan(a, b, bc::RhombicBC)
    b_y =
        b[2] +
        (3^0.5 / 2 * bc.box_length) * round((a[2] - b[2]) / (3^0.5 / 2 * bc.box_length))
    b_x =
        b[1] - b[2] / 3^0.5 +
        bc.box_length * round(((a[1] - b[1]) - 1 / 3^0.5 * (a[2] - b[2])) / bc.box_length) +
        1 / 3^0.5 * b_y
    b_z = b[3] + bc.box_height * round((a[3] - b[3]) / bc.box_height)
    tan = ((a[1] - b_x)^2 + (a[2] - b_y)^2)^0.5 / (a[3] - b_z)
    return abs(tan)
end
function get_tan(a, b, bc::RectangularBC)
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_height * round((a[3] - b[3]) / bc.box_height)
    tan = ((a[1] - b_x)^2 + (a[2] - b_y)^2)^0.5 / (a[3] - b_z)
    return abs(tan)
end

"""
    get_tantheta_mat(conf::Config)

Builds the matrix of tan of angles between positions of configuration.
"""
function get_tantheta_mat(config::Config)
    mat = zeros(length(config), length(config))
    get_tantheta_mat!(mat, config)
    return mat
end

function get_tantheta_mat!(dest, config::Config)
    @boundscheck if size(dest) ≠ (length(config), length(config))
        throw(
            DimensionMismatch(
                "invalid dimension of destination $(size(dest)) for" *
                " a Config of length $(length(config))",
            ),
        )
    end
    @inbounds for i in 1:length(config), j in (i + 1):length(config)
        dest[i, j] = dest[j, i] = get_tan(config[i], config[j], config.boundary_condition)
    end
    @inbounds for i in eachindex(config)
        dest[i, i] = 0
    end
    return dest
end

scale_xyz(vector, α) = α * vector
function scale_xyz(config::Config, α)
    return Config(scale_xyz(config.positions, α), scale_xyz(config.boundary_condition, α))
end
function scale_xy(pos, scale)
    new_pos = map(pos) do p
        SVector(p[1] * scale, p[2] * scale, p[3])
    end
    return new_pos
end
function scale_xy(config::Config, scale)
    return Config(
        scale_xy(config.positions, scale), scale_xy(config.boundary_condition, scale)
    )
end
function scale_z(pos, scale)
    new_pos = map(pos) do p
        SVector(p[1], p[2], p[3] * scale)
    end
    return new_pos
end
function scale_z(config::Config, scale)
    return Config(
        scale_z(config.positions, scale), scale_z(config.boundary_condition, scale)
    )
end

end
