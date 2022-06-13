""" 
    module Input

    this module provides structs and methods to arrange input parameters
"""
module InputParams

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..EnergyEvaluation

export InputParameters
export MCParams, TempGrid
export AbstractDisplacementParams, DisplacementParamsAtomMove
export StatMoves, StatMovesInit

const kB = 3.16681196E-6  # in Hartree/K (3.166811429E-6)

struct MCParams
    mc_cycles::Int
    eq_cycles::Int
end 

function MCParams(cycles; eq_percentage = 0.2)
    mc_cycles = Int(cycles)
    eq_cycles = round(Int, eq_percentage * mc_cycles)
    return MCParams(mc_cycles,eq_cycles)
end

struct TempGrid{N,T} 
    t_grid::SVector{N,T}
    beta_grid::SVector{N,T}
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
    betagrid = 1. /(kB*tgrid)
    return TempGrid{N,eltype(tgrid)}(SVector{N}(tgrid), SVector{N}(betagrid))
end 

TempGrid(ti, tf, N; tdistr=:geometric) = TempGrid{N}(ti, tf; tdistr)

abstract type AbstractDisplacementParams{T} end

struct DisplacementParamsAtomMove{T} <: AbstractDisplacementParams{T}
    max_displacement::Vector{T} #maximum atom displacement in Angstrom
    update_step::Int
end 

function DisplacementParamsAtomMove(displ,tgrid; update_stepsize=100)
    T = eltype(displ)
    N = length(tgrid)
    #initialize displacement vector
    max_displ = [0.1*sqrt(displ*tgrid[i]) for i in 1:N]
    return DisplacementParamsAtomMove{T}(max_displ, update_stepsize)
end

function update_max_stepsize!(displ::DisplacementParamsAtomMove, count_accept, n_atom)
    for i in 1:length(count_acc)
        acc_rate =  count_accept[i] / (displ.update_step * n_atom)
        if acc_rate < 0.4
            displ.max_displacement[i] *= 0.9
        elseif acc_rate > 0.6
            displ.max_displacement[i] *= 1.1
        end
        count_accept[i] = 0
    end
    return displ, count_accept
end

struct InputParameters
    mc_parameters::MCParams
    temp_parameters::TempGrid
    starting_conf::Config
    random_seed::Int
    potential::AbstractPotential
    max_displacement::AbstractDisplacementParams
end

mutable struct StatMoves{N}
    count_acc::Vector{Int}       #total count of acceptance of atom moves
    count_acc_adj::Vector{Int}    #acceptance used for stepsize adjustment for atom moves, will be reset to 0 after each adjustment

    count_exc::Vector{Int}       #number of proposed exchanges 
    count_exc_acc::Vector{Int}  #number of accepted exchanges

    count_v_acc::Vector{Int}     #total count of acceptance of volume moves
    count_v_acc_adj::Vector{Int}#acceptance used for stepsize adjustment for volume moves, will be reset to 0 after each adjustment
end

function StatMovesInit(N)
    count_acc=zeros(N)
    count_acc_adj=zeros(N)
    count_exc=zeros(N)
    count_exc_acc=zeros(N)
    count_v_acc=zeros(N)
    count_v_acc_adj=zeros(N) 
    return StatMoves{N}(count_acc,count_acc_adj,count_exc,count_exc_acc,count_v_acc,count_v_acc_adj)
end

#bc_ar32 = SphericalBC(radius=14.5)  #Angstrom

#Ar32 starting config (fcc, pbc)
# pos_ar32 = 0.2000000000E+02  0.2000000000E+02  0.2000000000E+02
# -0.1000000000E+02 -0.1000000000E+02 -0.1000000000E+02
# -0.5000000000E+01 -0.5000000000E+01 -0.1000000000E+02
# -0.5000000000E+01 -0.1000000000E+02 -0.5000000000E+01
# -0.1000000000E+02 -0.5000000000E+01 -0.5000000000E+01
# -0.1000000000E+02 -0.1000000000E+02  0.0000000000E+00
# -0.5000000000E+01 -0.5000000000E+01  0.0000000000E+00
# -0.5000000000E+01 -0.1000000000E+02  0.5000000000E+01
# -0.1000000000E+02 -0.5000000000E+01  0.5000000000E+01
# -0.1000000000E+02  0.0000000000E+00 -0.1000000000E+02
# -0.5000000000E+01  0.5000000000E+01 -0.1000000000E+02
# -0.5000000000E+01  0.0000000000E+00 -0.5000000000E+01
# -0.1000000000E+02  0.5000000000E+01 -0.5000000000E+01
# -0.1000000000E+02  0.0000000000E+00  0.0000000000E+00
# -0.5000000000E+01  0.5000000000E+01  0.0000000000E+00
# -0.5000000000E+01  0.0000000000E+00  0.5000000000E+01
# -0.1000000000E+02  0.5000000000E+01  0.5000000000E+01
#  0.0000000000E+00 -0.1000000000E+02 -0.1000000000E+02
#  0.5000000000E+01 -0.5000000000E+01 -0.1000000000E+02
#  0.5000000000E+01 -0.1000000000E+02 -0.5000000000E+01
#  0.0000000000E+00 -0.5000000000E+01 -0.5000000000E+01
#  0.0000000000E+00 -0.1000000000E+02  0.0000000000E+00
#  0.5000000000E+01 -0.5000000000E+01  0.0000000000E+00
#  0.5000000000E+01 -0.1000000000E+02  0.5000000000E+01
#  0.0000000000E+00 -0.5000000000E+01  0.5000000000E+01
#  0.0000000000E+00  0.0000000000E+00 -0.1000000000E+02
#  0.5000000000E+01  0.5000000000E+01 -0.1000000000E+02
#  0.5000000000E+01  0.0000000000E+00 -0.5000000000E+01
#  0.0000000000E+00  0.5000000000E+01 -0.5000000000E+01
#  0.0000000000E+00  0.0000000000E+00  0.0000000000E+00
#  0.5000000000E+01  0.5000000000E+01  0.0000000000E+00
#  0.5000000000E+01  0.0000000000E+00  0.5000000000E+01
#  0.0000000000E+00  0.5000000000E+01  0.5000000000E+01

end