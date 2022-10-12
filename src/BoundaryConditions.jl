""" 
    module BoundaryConditions

    this module provides structs and methods for different kinds of boundary conditions
        
"""
module BoundaryConditions

export SphericalBC, AbstractBC, PeriodicBC,AdjacencyBC
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

struct PeriodicBC{T} <: AbstractBC{T}
    box_length::T
end

mutable struct AdjacencyBC{T} <: AbstractBC{T}
    r2_cut::T
    adj_mat::Matrix{T}
end
function AdjacencyBC(r2_cut, pos)
    adj_mat = find_adjmat(pos,r2_cut)

    AdjacencyBC(r2_cut,adj_mat)
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
"""
    find_adjmat(dist2_matrix, r2_cut)
    find_adjmat(config::Config{N},r2_cut)
creates an Adjacency Matrix from either a matrix of distance squared or from a configuration. Both require a fieldname r2_cut, the square distace of the bonding radius
"""
function find_adjmat(dist2_matrix::Matrix, r2_cut)
    adjmat = map(dist2_matrix -> ifelse(dist2_matrix <= r2_cut, 1, 0), dist2_matrix )
    
    return adjmat
end
find_adjmat(pos::Vector,r2_cut) = [ifelse(distance2(a,b)<=r2_cut,1,0) for a in pos, b in pos]

# function check_boundary(bc::AdjacencyBC,dist2_matrix)

#     bcflag = false

#     for col in eachcol(find_adjmat(dist2_matrix, bc.r2_cut))

#       dummysum = sum(col)
      
#         if dummysum < 4
#             bcflag = true
#              # break
#         end
#         #push!(SumVec, dummysum)
#     end
#     return bcflag
# end


end
