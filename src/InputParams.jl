""" 
    module InputParams

    this module provides structs and methods to arrange input parameters
"""
module InputParams

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..EnergyEvaluation

export InputParameters
export MCParams, TempGrid
export Output
#export Results
#export AbstractDisplacementParams, DisplacementParamsAtomMove

const kB = 3.16681196E-6  # in Hartree/K (3.166811429E-6)

struct MCParams
    mc_cycles::Int
    eq_cycles::Int
    n_traj::Int
    n_atoms::Int
end 

function MCParams(cycles, n_traj, n_atoms; eq_percentage = 0.2)
    mc_cycles = Int(cycles)
    eq_cycles = round(Int, eq_percentage * mc_cycles)
    return MCParams(mc_cycles,eq_cycles,n_traj,n_atoms)
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

struct Output{T}
    ham::Vector{T}
    max_displ::Vector{T}
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_rot::Vector{Int}
    count_exc::Vector{Int}
    en_ave::Vector{T}
    heat_cap::Vector{T}
    #en_histogram::EnHist{T}
    rdf::Vector{T}
end

function Output(max_displ,n_bin)
    T=eltype(max_displ)
    ham = en_ave = heat_cap = rdf = T[]
    count_atom = count_vol = count_rot = count_exc = [0,0]
    en_hist = EnHist(n_bin)
    return Output{T}(ham,max_displ,count_atom,count_vol,count_rot,count_exc,en_ave,heat_cap,en_hist,rdf)
end

#struct Results{T}
#    ham::Vector{T}
#    max_displ::Vector{T}
#    count_atom::Vector{Int}
#    count_vol::Vector{Int}
#   count_rot::Vector{Int}
#    count_exc::Vector{Int}
#    en_ave::Vector{T}
#    heat_cap::Vector{T}
#    en_histogram::EnHist{T}
#    rdf::Vector{T}
#end

#= function Results(;n_bin = 100,ham=[],max_displ::Vector{T}=[0.1,0.1,1.],count_atom=[0,0],count_vol=[0,0],count_rot=[0,0],count_exc=[0,0], en_ave=[], heat_cap=[],rdf=[]) where {T}
    en_histogram = EnHist(n_bin)
    return Results{T}(ham,max_displ,count_atom,count_vol,count_rot,count_exc,en_ave,heat_cap,en_histogram,rdf)
end

function Results(n_bin, max_displ::Vector{T}; ham=[],count_atom=[0,0],count_vol=[0,0],count_rot=[0,0],count_exc=[0,0], en_ave=[], heat_cap=[],rdf=[]) where {T}
    en_histogram = EnHist(n_bin)
    return Results{T}(ham,max_displ,count_atom,count_vol,count_rot,count_exc,en_ave,heat_cap,en_histogram,rdf)
end
 =#
#abstract type AbstractDisplacementParams{T} end

#struct DisplacementParamsAtomMove{T} <: AbstractDisplacementParams{T}
#    max_displacement::Vector{T} #maximum atom displacement in Angstrom
#    update_step::Int
#end 

#function DisplacementParamsAtomMove(displ,tgrid; update_stepsize=100)
#    T = eltype(displ)
#    N = length(tgrid)
    #initialize displacement vector
#    max_displ = [0.1*sqrt(displ*tgrid[i]) for i in 1:N]
#    return DisplacementParamsAtomMove{T}(max_displ, update_stepsize)
#end

#function update_max_stepsize!(displ::DisplacementParamsAtomMove, count_accept, n_atom)
#    for i in 1:length(count_acc)
#       acc_rate =  count_accept[i] / (displ.update_step * n_atom)
#        if acc_rate < 0.4
#            displ.max_displacement[i] *= 0.9
#        elseif acc_rate > 0.6
#            displ.max_displacement[i] *= 1.1
#        end
#        count_accept[i] = 0
#    end
#    return displ, count_accept
#end

struct InputParameters
    mc_parameters::MCParams
    temp_parameters::TempGrid
    starting_conf::Config
    random_seed::Int
    potential::AbstractPotential
#    max_displacement::AbstractDisplacementParams
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