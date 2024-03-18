""" 
    module BoundaryConditions

    module providing structs and methods for different types of boundary conditions
        
"""
module BoundaryConditions

export SphericalBC, AbstractBC, PeriodicBC, CubicBC, RhombicBC
export check_boundary


"""   
    AbstractBC{T} 
abstract type that encompasses possible boundary conditions; implemented: 
    - SphericalBC [`SphericalBC`](@ref)
    - PeriodicBC [`PeriodicBC`](@ref)

needs methods implemented for
    - atom_displacement [`atom_displacement`](@ref)
"""
abstract type AbstractBC{T} end

"""
    SphericalBC{T}(;radius)
Implements spherical boundary conditions; subtype of [`AbstractBC`](@ref).
Key word argument:
    - `radius`: radius of binding sphere
Field:
    -  `radius2`: squared radius of binding sphere
"""
struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius of binding sphere squared
    SphericalBC(; radius::T) where T = new{T}(radius*radius)
end

"""
    PeriodicBC{T}
Abstract type of periodic boundary conditions 
Implemented types:
    - [`CubicBC`](@ref): cubic simulation cell
    - [`RhombicBC``](@ref): rhombic simulation cell
"""
abstract type PeriodicBC{T} <: AbstractBC{T} end

"""
    CubicBC{T}
Subtype of periodic boundary conditions for cubic simulation cell 
Field name:
    - `box_length`: side length of cubic cell 
"""
struct CubicBC{T} <: PeriodicBC{T}
    box_length::T
end

"""
    RhombicBC{T}
Subtype of periodic boundary condition for rhombic cell (60 degree angle between base and height)
Field names:
    - `box_length`: side length of base
    - `box_height`: height of cell
"""
struct RhombicBC{T} <: PeriodicBC{T}
    box_length::T
    box_height::T
end

"""
    check_boundary(bc::SpericalBC,pos)
Checks if moved atom stayed within binding sphere
Arguments:
    - bc: boundary condition
    - pos: position of moved atom
Returns `true` when moved atom is outside of spherical boundary.
(squared norm of position vector < radius^2 of binding sphere).
"""
check_boundary(bc::SphericalBC,pos) = sum(x->x^2,pos) > bc.radius2


"""
    test_cluster_inside(conf,bc::SphericalBC)
Tests if whole cluster lies in the binding sphere
Arguments:
    - conf: configuration     
    - bc: boundary condition
"""
test_cluster_inside(conf,bc::SphericalBC) = sum(x->outside_of_boundary(bc,x),conf.pos) == 0

end
