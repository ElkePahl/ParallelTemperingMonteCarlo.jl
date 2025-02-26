""" 
    module InputParams

This module provides structs and methods to arrange input parameters.
"""
module InputParams

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..EnergyEvaluation
using ..CustomTypes

export InputParameters
export MCParams, TempGrid
export Output

const kB = 3.16681196E-6  # in Hartree/K (3.166811429E-6)
#const JtoEh=2.2937104486906E17
#const A3tom3=10.0E-30
#const Bohr3tom3=1.4818474345E-31

"""
    MCParams(cycles::Int, n_traj::Int, n_atoms::Int; eq_percentage = 0.2, mc_sample = 1, n_adjust = 100, n_bin = 100)
Type that collects MC specific data; field names: 
-   `mc_cycle::Int`: number of MC cycles
-   `eq_cycles::Int`: number of equilibration cycles (default 20% of `mc_cycle`)
-   `mc_sample::Int`: gives number of MC cycles after which energy is saved (default: 1)
-   `n_traj::Int`: number of trajectories (ie. temperatures) propagated in parallel
-   `n_atoms::Int`: number N of atoms in configuration
-   `n_adjust::Int`: number of moves after which step size of atom/volume moves is adjusted (default: 100)
-   `n_bin::Int`: number of histogram bins (default: 100)
"""
struct MCParams
    mc_cycles::Int
    eq_cycles::Int
    mc_sample::Int
    n_traj::Int
    n_atoms::Int
    n_adjust::Int
    n_bin::Int
    min_acc::Float64
    max_acc::Float64
end 

function MCParams(cycles::Int, n_traj::Int, n_atoms::Int; eq_percentage = 0.2, mc_sample = 1, n_adjust = 100, n_bin = 100, min_acc=0.4, max_acc=0.6)
    mc_cycles = Int(cycles)
    eq_cycles = round(Int, eq_percentage * mc_cycles)
    return MCParams(mc_cycles, eq_cycles, mc_sample, n_traj, n_atoms, n_adjust,n_bin,min_acc,max_acc)
end

"""
    TempGrid{N}(ti::Number, tf::Number; tdistr) 
    TempGrid(ti::Number, tf::Number, N::Int; tdistr=:geometric)
Generates grid of `N` temperatures and inverse temperatures for MC calculation
between initial and final temperatures `ti` and `tf`.
-   Field names:
    -   `t_grid::SVector{N,T}`: temperatures (in K)
    -   `beta_grid::SVector{N,T}`: inverse temperatures (in atomic units)
-   Keyword argument `tdistr`:
    -   `:geometric` (default): generates geometric temperature distribution
    -   `:equally_spaced`: generates equally spaced temperature grid (not implemented presently)
"""
struct TempGrid{N,T} 
    t_grid::SVector{N,T}
    beta_grid::SVector{N,T}
end

function TempGrid{N}(ti::Number, tf::Number; tdistr=:geometric) where {N}
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

TempGrid(ti::Number, tf::Number, N::Int; tdistr=:geometric) = TempGrid{N}(ti, tf; tdistr)

"""
    Output{T}(n_bin; en_min = 0) where T <: Number
Collects output of MC calculation; field names:
-   `n_bin::Int`: number of energy bins for histograms
-   `en_min::T`: minimum energy found during calculation
-   `en_max::T`: maximum energy found during calculation
-   `v_min::T`: minimum volume
-   `v_max::T`: maximum volume
-   `delta_en_hist::T`: the step size associated with the energy histogram 
-   `delta_v_hist::T`: step size associated with volume histogram
-   `delta_r2::T`: step size associated with the rdf histogram
-   `max_displ::Vector{T}`: final maximum displacements
-   `en_avg::Vector{T}`: inner energy U(T) (as average over sampled energies)
-   `heat_cap::Vector{T}`: heat capacities C(T) 
-   `rdf::Vector{Vector{T}}`: radial distribution information
-   `count_stat_*::Vector{T}`: statistics of accepted atom, volume and rotation moves and attempted and successful parallel-tempering exchanges
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

function Output{T}(n_bin::Int; en_min = 0) where T <: Number
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

const VV = Vector{A} where A <: Vector{B} where B <: Number
const VM = Vector{A} where A <: Matrix{B} where B <: Number
function Output{T}(n_bin::Int, en_min::Number, en_max::Number, v_min::Number, v_max::Number, max_displ::VorS, en_avg::VorS, heat_cap::VorS, en_histogram::VV, ev_histogram::VM, rdf::VV, count_stat_atom::VorS, count_stat_vol::VorS, count_stat_rot::VorS, count_stat_exc::VorS) where T <: Number
    delta_en_hist = (en_max-en_min)/(n_bin-1)
    delta_v_hist = (v_max - v_min)/n_bin
    delta_r2 = 0.
    return Output{T}(n_bin, en_min, en_max, v_min, v_max,delta_en_hist,delta_v_hist,delta_r2 , max_displ, en_avg, heat_cap, en_histogram, ev_histogram, rdf, count_stat_atom, count_stat_vol, count_stat_rot, count_stat_exc)
end

end
