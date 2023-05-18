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

const JtoEh=2.2937104486906E17

const A3tom3=10.0E-30

const Bohr3tom3=1.4818474345E-31

"""
    MCParams(cycles, n_traj, n_atoms; eq_percentage = 0.2, mc_sample = 1, n_adjust = 100)
Type that collects MC specific data: 
- number of MC cycles `cycles`, temperatures `n_traj` and atoms `n_atom`
- percentage of equilibration: default 20%
- frequency of sampling energy (adding to Markov chain): 
`mc_sample` gives number of MC cycles after which energy is saved (default: 1)
- step size of moves is automatically adjusted after `n_adjust` MC cycles (default: 100)
"""
struct MCParams
    mc_cycles::Int
    eq_cycles::Int
    mc_sample::Int
    n_traj::Int
    n_atoms::Int
    n_adjust::Int
end 

function MCParams(cycles, n_traj, n_atoms; eq_percentage = 0.2, mc_sample = 1, n_adjust = 100)
    mc_cycles = Int(cycles)
    eq_cycles = round(Int, eq_percentage * mc_cycles)
    return MCParams(mc_cycles, eq_cycles, mc_sample, n_traj, n_atoms, n_adjust)
end

"""
    TempGrid{N}(ti, tf; tdistr) 
    TempGrid(ti, tf, N; tdistr=:geometric)
Generates grid of `N` temperatures and inverse temperatures for MC calculation
between initial and final temperatures `ti` and `tf`
Field names:
- `t_grid`: temperatures (in K)
- `beta_grid`: inverse temperatures (in atomic units)
keyword argument `tdistr`:
- :geometric (default): generates geometric temperature distribution
- :equally_spaced: generates equally spaced temperature grid
"""
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
    betagrid = 1. ./ (kB .* tgrid)
    return TempGrid{N,eltype(tgrid)}(SVector{N}(tgrid), SVector{N}(betagrid))
end 

TempGrid(ti, tf, N; tdistr=:geometric) = TempGrid{N}(ti, tf; tdistr)

"""
    Output{T}(n_bin; en_min = 0)
Collects output of MC calculation:
- `n_bin`: number of energy bins for histograms
- `en_min`: minimum energy found during calculation
- `max_displ`: final maximum displacements
- `en_avg`: inner energy U(T) (as average over sampled energies)
- `heat_cap`: heat capacities C(T) 
- `rdf`: radial distribution information
- `count_stat_*`: statistics of accepted atom, volume and rotation moves 
and attempted and successful parallel-tempering exchanges
"""
mutable struct Output{T}
    n_bin::Int
    en_min::T
    en_max::T
    v_min::T
    v_max::T
    max_displ::Vector{T}
    en_avg::Vector{T}
    heat_cap::Vector{T}
    en_histogram::Vector{Vector{T}}
    ev_histogram::Vector{Matrix{T}}
    rdf::Vector{Vector{T}}
    count_stat_atom::Vector{T}
    count_stat_vol::Vector{T}
    count_stat_rot::Vector{T}
    count_stat_exc::Vector{T}
end

function Output{T}(n_bin; en_min = 0) where T
    en_min = 0.
    en_max = 0.
    v_min = 0.
    v_max = 0.
    max_displ = T[]
    en_avg = T[]
    heat_cap = T[]
    en_histogram = []
    ev_histogram = []
    rdf = []
    count_stat_atom = T[]
    count_stat_vol = T[]
    count_stat_rot = T[]
    count_stat_exc = T[]
    return Output{T}(n_bin, en_min, en_max, v_min, v_max, max_displ, en_avg, heat_cap, en_histogram, ev_histogram, rdf, count_stat_atom, count_stat_vol, count_stat_rot, count_stat_exc)
end

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

#= struct InputParameters
    mc_parameters::MCParams
    temp_parameters::TempGrid
    starting_conf::Config
    random_seed::Int
    potential::AbstractPotential
#    max_displacement::AbstractDisplacementParams
end =#

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
