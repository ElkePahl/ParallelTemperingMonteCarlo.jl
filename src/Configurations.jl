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
export distance2, get_distance2_mat, get_tan, get_tantheta_mat, get_volume
export get_centre,recentre!

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

"""
    get_centre(position,N)
function to find the centre of mass of a configuration. Accepts the positions and number of positions and calculates the xyz coordinates of their centre.
"""
function get_centre(position,N)
    return [sum([pos[1] for pos in position]),sum([pos[2] for pos in position]),sum([pos[3] for pos in position])]./N
end
"""
    recentre!(conf::Config{N,BC,T}) where {N,BC,T}
function to change the centre of mass of a configuration `conf' to [0,0,0] in cartesian space
"""
function recentre!(conf::Config{N,BC,T}) where {N,BC,T}
    cofm = get_centre(conf.pos,N)
    for index in 1:N
        conf.pos[index] = SVector{3}(conf.pos[index] .- cofm)
    end

end
"""
    distance2(a,b) 
    distance2(a,b,bc::SphericalBC)
    distance2(a,b,bc::CubicBC) 
    distance2(a,b,bc::RhombicBC)
method 1&2 -
Finds the distance between two positions a and b.
method 3 -
Finds the distance between two positions a and the nearest image of b in a cubic box.
method 4 - 
Finds the distance between two positions a and the nearest image of b in a rhombic box.
Minimum image convension in the z-direction is the same as the cubic box.
In x and y-direction, first the box is transformed into a rectangular box, then MIC is done, finally the new coordinates are transformed back.
"""
distance2(a,b) = (a-b)⋅(a-b)

distance2(a,b,bc::SphericalBC) = distance2(a,b)
function distance2(a,b,bc::CubicBC)
    b_x=b[1]+bc.box_length*round((a[1]-b[1])/bc.box_length)
    b_y=b[2]+bc.box_length*round((a[2]-b[2])/bc.box_length)
    b_z=b[3]+bc.box_length*round((a[3]-b[3])/bc.box_length)
    return distance2(a,SVector(b_x,b_y,b_z))
end
function distance2(a,b,bc::RhombicBC)
    b_y=b[2]+(3^0.5/2*bc.box_length)*round((a[2]-b[2])/(3^0.5/2*bc.box_length))
    b_x=b[1]-b[2]/3^0.5 + bc.box_length*round(((a[1]-b[1])-1/3^0.5*(a[2]-b[2]))/bc.box_length) + 1/3^0.5*b_y
    b_z=b[3]+bc.box_height*round((a[3]-b[3])/bc.box_height)
    return distance2(a,SVector(b_x,b_y,b_z))
end

#distance matrix
"""
    get_distance2_mat(conf::Config{N})

Builds the matrix of squared distances between positions of configuration.
"""

#get_distance2_mat(conf::Config{N}) where N = [distance2(a,b,conf.bc) for a in conf.pos, b in conf.pos]

function get_distance2_mat(conf::Config{N}) where N
    mat=zeros(N,N)
    for i=1:N
        for j=i+1:N
            mat[i,j]=mat[j,i]=distance2(conf.pos[i],conf.pos[j],conf.bc)
        end
    end
    return mat
end

"""
    get_tan(a,b)
    get_tan(a,b,bc::SphericalBC)
    get_tan(a,b,bc::CubicBC)
    get_tan(a,b,bc::RhombicBC)
method 1&2 :
tan of the angle between the line connecting two points a and b, and the z-direction
method 3:
tan of the angle between the line connecting two points a and the nearest image of b, and the z-direction in a cubic boundary
method 4: 
tan of the angle between the line connecting two points a and the nearest image of b, and the z-direction in a rhombic boundary
"""
function get_tan(a,b)
    tan=((a[1]-b[1])^2+(a[2]-b[2])^2)^0.5/(a[3]-b[3])
    return tan
end
function get_tan(a,b,bc::SphericalBC)
    tan=((a[1]-b[1])^2+(a[2]-b[2])^2)^0.5/(a[3]-b[3])
    return tan
end
function get_tan(a,b,bc::CubicBC)
    b_x = b[1] + bc.box_length*round((a[1]-b[1])/bc.box_length)
    b_y = b[2] + bc.box_length*round((a[2]-b[2])/bc.box_length)
    b_z = b[3] + bc.box_length*round((a[3]-b[3])/bc.box_length)
    tan=((a[1]-b_x)^2+(a[2]-b_y)^2)^0.5/(a[3]-b_z)
    return tan
end
function get_tan(a,b,bc::RhombicBC)
    b_y=b[2]+(3^0.5/2*bc.box_length)*round((a[2]-b[2])/(3^0.5/2*bc.box_length))
    b_x=b[1]-b[2]/3^0.5 + bc.box_length*round(((a[1]-b[1])-1/3^0.5*(a[2]-b[2]))/bc.box_length) + 1/3^0.5*b_y
    b_z=b[3]+bc.box_height*round((a[3]-b[3])/bc.box_height)
    tan=((a[1]-b_x)^2+(a[2]-b_y)^2)^0.5/(a[3]-b_z)
    return tan
end

"""
    get_theta_mat(conf::Config{N},conf.bc::SphericalBC)

Builds the matrix of tan of angles between positions of configuration in a spherical boundary.
"""

function get_tantheta_mat(conf::Config,bc::SphericalBC)
    N=length(conf.pos)
    mat=zeros(N,N)
    for i=1:N
        for j=i+1:N
            mat[i,j]=mat[j,i] = get_tan(conf.pos[i],conf.pos[j])
        end
    end
    return mat
end

"""
    get_theta_mat(conf::Config{N},conf.bc::CubicBC)

Builds the matrix of tan of angles between positions of configuration in a cubic boundary.
"""
function get_tantheta_mat(conf::Config,bc::CubicBC)
    N=length(conf.pos)
    mat=zeros(N,N)
    for i=1:N
        for j=i+1:N
            mat[i,j]=mat[j,i] = get_tan(conf.pos[i],conf.pos[j],bc)
        end
    end
    return mat
end

"""
    get_theta_mat(conf::Config{N},conf.bc::CubicBC)

Builds the matrix of tan of angles between positions of configuration in a rhombic boundary.
"""
function get_tantheta_mat(conf::Config,bc::RhombicBC)
    N=length(conf.pos)
    mat=zeros(N,N)
    for i=1:N
        for j=i+1:N
            mat[i,j]=mat[j,i] = get_tan(conf.pos[i],conf.pos[j],bc)
        end
    end
    return mat
end
"""
    get_volume(bc::CubicBC)
    get_volume(bc::RhombicBC)
returns the volume of a box according to its geometry for use where the ensemble does not imply a fixed V. 
"""
function get_volume(bc::CubicBC)
    return bc.box_length^3
end

function get_volume(bc::RhombicBC)
    return bc.box_length^2 * bc.box_height * 3^0.5/2
end

end
