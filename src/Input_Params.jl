""" 
    module Input

    this module provides structs and methods to arrange input parameters
"""
module Input

using StaticArrays

export MCParams, TempGrid

struct MCParams
    mc_cycles::Int
    eq_cycles::Int
end 

function MCParams(cycles;eq_percentage = 0.2)
    mc_cycles = Int(cycles)
    eq_cycles = round(Int, eq_percentage * mc_cycles)
    return MCParams(mc_cycles,eq_cycles)
end

struct TempGrid{N,T} 
    t_grid::SVector{N,T}
end

function TempGrid{N}(ti, tf; tdistr=:geometric) where {N}
    if tdistr == :equally_spaced
        delta = (tf-ti)/(N-1)
        tgrid = [ti + (i-1)*delta for i in 1:N]
    elseif tdistr == :geometric
        tgrid =[ti*(tf/ti)^((i-1)/(N-1)) for i in 1:N]
    else
        throw(ArgumentError("chosen temperature distribution $tdistr does not exist"))
    end
    return TempGrid{N,eltype(tgrid)}(SVector{N}(tgrid))
end 

end