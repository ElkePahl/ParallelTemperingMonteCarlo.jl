""" 
    module BoundaryConditions

    this module provides strcts for different kinds of boundary conditions
        
"""
module BoundaryConditions

export SphericalBC, AbstractBC

# include("SphericalBC.jl")
# could be named SphericalBC.jl

"documentation - one line"
abstract type AbstractBC{T} end

struct SphericalBC{T} <: AbstractBC{T}
    radius2::T   #radius squared
end

function move_atom!(config,bc::SphericalBC)
end

    

end