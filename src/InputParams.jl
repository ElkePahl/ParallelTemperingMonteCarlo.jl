""" 
    module Input

    this module provides structs and methods to arrange input parameters
"""
module InputParams

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..EnergyEvaluation

export pos_ne13, bc_ne13, conf_ne13
export InputParameters
export MCParams, TempGrid
export AbstractDisplacementParams, DisplacementParamsAtomMove
export update_max_stepsize!

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
    betagrid = 1 ./(kB*tgrid)
    return TempGrid{N,eltype(tgrid)}(SVector{N}(tgrid), SVector{N}(betagrid))
end 

TempGrid(ti, tf, N; tdistr=:geometric) = TempGrid{N}(ti, tf; tdistr)

T=TempGrid(1,20,10; tdistr=:geometric)
println(T)


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
    for i in 1:length(count_accept)
        acc_rate =  count_accept[i] / (displ.update_step * n_atom)
        #println(acc_rate)
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

#default configurations
#icosahedral ground state of Ne13 (from Cambridge cluster database) in Angstrom
pos_ne13 = [[2.825384495892464, 0.928562467914040, 0.505520149314310],
[2.023342172678102,	-2.136126268595355, 0.666071287554958],
[2.033761811732818,	-0.643989413759464, -2.133000349161121],
[0.979777205108572,	2.312002562803556, -1.671909307631893],
[0.962914279874254,	-0.102326586625353, 2.857083360096907],
[0.317957619634043,	2.646768968413408, 1.412132053672896],
[-2.825388342924982, -0.928563755928189, -0.505520471387560],
[-0.317955944853142, -2.646769840660271, -1.412131825293682],
[-0.979776174195320, -2.312003751825495, 1.671909138648006],
[-0.962916072888105, 0.102326392265998,	-2.857083272537599],
[-2.023340541398004, 2.136128558801072,	-0.666071089291685],
[-2.033762834001679, 0.643989905095452, 2.132999911364582],
[0.000002325340981,	0.000000762100600, 0.000000414930733]]

pos_ne13=pos_ne13*1.88973

#define boundary conditions starting configuration
#bc_ne13 = SphericalBC(radius=5.32)   #Angstrom
bc_ne13 = SphericalBC(radius=10.)

#starting configuration
conf_ne13 = Config(pos_ne13, bc_ne13)

max_displacement=1
delta_move = SVector((rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement,(rand()-0.5)*max_displacement)

move_atom!(conf_ne13::Config, 1, delta_move)

println(conf_ne13)
println()


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