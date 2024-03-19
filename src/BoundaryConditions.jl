""" 
    module BoundaryConditions

    this module provides structs and methods for different kinds of boundary conditions
        
"""
module BoundaryConditions

export SphericalBC, AbstractBC, PeriodicBC, CubicBC, RhombicBC
export check_boundary

# include("SphericalBC.jl")
# could be named SphericalBC.jl

"""   
    AbstractBC{T} 
Encompasses possible boundary conditions; implemented: 
- SphericalBC [`SphericalBC`](@ref)
- PeriodicBC [`PeriodicBC`](@ref)

needs methods implemented for
    - atom_displacement [`atom_displacement`](@ref)
"""
abstract type AbstractBC{T} end

"""
    SphericalBC{T}(;radius)
Implements type for spherical boundary conditions; subtype of [`AbstractBC`](@ref).
Needs radius of binding sphere as keyword argument   
fieldname: `radius2`: squared radius of binding sphere
"""
struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius of binding sphere squared
    SphericalBC(; radius::T) where T = new{T}(radius*radius)
end

"""
    PeriodicBC{T}
Overarching type of boundary condition for simulating the infinite bulk
    Implemented types:
    - CubicBC
    - RhombicBC
"""
abstract type PeriodicBC{T} <: AbstractBC{T} end
"""
    CubicBC{T}
Subtype of periodic boundary conditions where the `box_length` is isotropic.
"""
struct CubicBC{T} <: PeriodicBC{T}
    box_length::T
end
"""
    RhombicBC{T}
Subtype of periodic boundary condition where the `box_length` and `box_height` are not the same. The projection of the box on the xy-plane is a rhombus, box_length applies to all four sides.
"""
struct RhombicBC{T} <: PeriodicBC{T}
    box_length::T
    box_height::T
end

"""
    check_boundary(bc::SpericalBC,pos)

Returns `true` when atom outside of spherical boundary
(squared norm of position vector < radius^2 of binding sphere).
"""
check_boundary(bc::SphericalBC,pos) = sum(x->x^2,pos) > bc.radius2


"""
    test_cluster_inside(conf,bc::SphericalBC)
Tests if whole cluster lies in the binding sphere     
"""
test_cluster_inside(conf,bc) = sum(x->outside_of_boundary(bc,x),conf.pos) == 0

end
