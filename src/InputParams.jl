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

const kB = 3.16681196E-6  # in Hartree/K (3.166811429E-6)
#const JtoEh=2.2937104486906E17
#const A3tom3=10.0E-30
#const Bohr3tom3=1.4818474345E-31

"""
    MCParams(cycles, n_traj, n_atoms; eq_percentage = 0.2, mc_sample = 1, n_adjust = 100, n_bin = 100)
Type that collects MC specific data; field names: 
    - `mc_cycle`: number of MC cycles
    - `eq_cycles`: number of equilibration cycles (default 20% of `mc_cycle`)
    - `mc_sample`: gives number of MC cycles after which energy is saved (default: 1)
    - `n_traj`: number of trajectories (ie. temperatures) propagated in parallel
    - `n_atoms`: number N of atoms in configuration
    - `n_adjust`: number of moves after which step size of atom/volume moves is adjusted (default: 100)
    - `n_bin`: number of histogram bins (default: 100)
"""
struct MCParams
    mc_cycles::Int
    eq_cycles::Int
    mc_sample::Int
    n_traj::Int
    n_atoms::Int
    n_adjust::Int
    n_bin::Int 
end 

function MCParams(cycles, n_traj, n_atoms; eq_percentage = 0.2, mc_sample = 1, n_adjust = 100, n_bin = 100)
    mc_cycles = Int(cycles)
    eq_cycles = round(Int, eq_percentage * mc_cycles)
    return MCParams(mc_cycles, eq_cycles, mc_sample, n_traj, n_atoms, n_adjust,n_bin)
end

"""
    TempGrid{N}(ti, tf; tdistr) 
    TempGrid(ti, tf, N; tdistr=:geometric)
Generates grid of `N` temperatures and inverse temperatures for MC calculation
between initial and final temperatures `ti` and `tf`
Field names:
    - `t_grid`: temperatures (in K)
    - `beta_grid`: inverse temperatures (in atomic units)
Keyword argument `tdistr`:
    - :geometric (default): generates geometric temperature distribution
    - :equally_spaced: generates equally spaced temperature grid (not implemented presently)
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
Collects output of MC calculation; field names:
    - `n_bin`: number of energy bins for histograms
    - `en_min`: minimum energy found during calculation
    - `en_max`: maximum energy found during calculation
    - `v_min`: minimum volume
    - `v_max`: maximum volume
    - `delta_en_hist`: the step size associated with the energy histogram 
    - `delta_v_hist`: step size associated with volume histogram
    - `delta_r2`: step size associated with the rdf histogram
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
    delta_en_hist::T
    delta_v_hist::T
    delta_r2::T
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
    delta_en_hist=0.
    delta_v_hist=0.
    delta_r2=0.
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
    return Output{T}(n_bin, en_min, en_max, v_min, v_max,delta_en_hist,delta_v_hist,delta_r2 , max_displ, en_avg, heat_cap, en_histogram, ev_histogram, rdf, count_stat_atom, count_stat_vol, count_stat_rot, count_stat_exc)
end

function Output{T}(n_bin, en_min, en_max, v_min, v_max, max_displ, en_avg, heat_cap, en_histogram, ev_histogram, rdf, count_stat_atom, count_stat_vol, count_stat_rot, count_stat_exc) where T
    delta_en_hist = (en_max-en_min)/(n_bin-1)
    delta_v_hist = (v_max - v_min)/n_bin
    delta_r2 = 0.
    return Output{T}(n_bin, en_min, en_max, v_min, v_max,delta_en_hist,delta_v_hist,delta_r2 , max_displ, en_avg, heat_cap, en_histogram, ev_histogram, rdf, count_stat_atom, count_stat_vol, count_stat_rot, count_stat_exc)
end

end
