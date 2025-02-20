module CustomTypes
using StaticArrays
"""
    PositionVector = Union{SVector{3, T}, Vector{T}} where T <: Number
Type alias for all kinds of acceptable position vectors.
"""
const PositionVector = Union{SVector{3,T},Vector{T}} where T <: Number
export PositionVector
"""
    PositionArray = Union{Vector{Vector{T}}, Vector{SVector{3, T}}} where T <: Number
Type alias for a list of positions.
"""
const PositionArray = Union{Vector{Vector{T}}, Vector{SVector{3, T}}} where T <: Number
export PositionArray

"""
    VorS = T where T <: AbstractArray{Z, 1} where Z <: Number
Type alias for a collection of numbers. The name derives from most collections being a Vector or StaticVector of numbers.
"""
const VorS = T where T <: AbstractArray{Z, 1} where Z <: Number
export VorS


end