""" 
    module BoundaryConditions

    this module provides structs and methods for different kinds of boundary conditions
        
"""
module BoundaryConditions

export SphericalBC, AbstractBC
export outside_of_boundary

# include("SphericalBC.jl")
# could be named SphericalBC.jl

"""   
    AbstractBC{T} 
Encompasses possible boundary conditions; implemented: 
- SphericalBC [`SphericalBC`](@ref)
#- PeriodicBC [`PeriodicBC`](@ref)
"""
abstract type AbstractBC{T} end

"""
    SphericalBC{T}
Implements type for spherical boundary conditions; subtype of [`AbstractBC`](@ref).
fieldname: `radius2`: squared radius of binding sphere
"""
struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius of binding sphere squared
end

"""
    outside_of_boundary(bc::SpericalBC,pos)

Returns `true` when atom outside of spherical boundary
(squared norm of position vector < radius^2 of binding sphere).
"""
outside_of_boundary(bc::SphericalBC,pos) = sum(x->x^2,pos) > bc.radius2

end