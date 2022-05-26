""" 
    module Input

    this module provides structs and methods to arrange input parameters
"""
module InputParams

using StaticArrays

using ..BoundaryConditions
using ..Configurations
using ..EnergyEvaluation

export kB
export JtoEh
export A3tom3, Bohr3tom3
export pos_ne13, bc_ne13, conf_ne13
export pos_ne25, bc_ne25, conf_ne25
export pos_ne32, bc_ne32, conf_ne32
export pos_ne32_pbc, bc_ne32_pbc, conf_ne32_pbc
export InputParameters
export MCParams, TempGrid
export AbstractDisplacementParams, DisplacementParamsAtomMove
export update_max_stepsize!

const kB = 3.16681196E-6  # in Hartree/K (3.166811429E-6)

const JtoEh=2.2937104486906*10^17

const A3tom3=10^(-30)

const Bohr3tom3=1.4818474345*10^-31

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

bc_ne13_1 = CubicBC(10.)

#starting configuration
conf_ne13 = Config(pos_ne13, bc_ne13)




pos_ne25=   [[0.6884429539,       -0.4902635898,       -1.4414708801],
[1.5396551009,       -0.2813139477,       -0.5852156965],
[0.2992192805,       -0.8759083275,        1.3497840799],
[-1.0925595292,       -1.2175594744,       -0.0502651686],
[0.9522409835,        0.1104583918,        1.2955305437],
[-1.2508551561,       -0.4303582211,       -0.9206118147],
[0.6892741454,        1.3963220508,       -0.1856296613],
[-0.1265862220,        1.1960629687,       -1.0063264757],
[-1.5531183915,       -0.1668315742,        0.1733133237],
[-0.1828336473,        0.1695371326,        1.5517325405],
[-0.4407747908,        1.4576313083,        0.0824813739],
[1.0886129477,       -0.7118181353,        0.4907160372],
[-0.2698891373,       -1.0452995947,       -0.8758256486],
[-0.7517788323,       -0.6095115347,        0.8960769718],
[0.8115876015,        0.5108174607,       -0.9314602554],
[-0.9972021004,        0.6156520803,       -0.4896383964],
[0.2948002720,        0.9328016953,        0.8100352760],
[0.0141198443,       -1.1671917814,        0.2707936990],
[-0.2290140421,        0.0618638128,       -1.1168902166],
[1.0617600945,        0.3787136168,        0.1815357335],
[-0.7275568920,        0.4876290257,        0.6042713445],
[0.4945173151,       -0.4224563127,       -0.3885123336],
[-0.5457016294,       -0.3137559291,       -0.1342207556],
[0.1856536312,       -0.1342268229,        0.6014654858],
[0.0479861996,        0.5490057018,       -0.1816691063]]

pos_ne25=pos_ne25*5.27
bc_ne25 = SphericalBC(radius=15.)
conf_ne25 = Config(pos_ne25, bc_ne25)



pos_ne32= [[-0.5118327432,        1.5200481266,        1.0813725133],
[-0.6190350946,       -1.5750424193,       -0.9369717974],
[-1.6256918835,        0.2357227253,       -0.2751151647],
[-1.4745083363,       -0.3866259062,        0.6711964388],
[ 0.4885765728,       -1.4875196994,       -1.0563354891],
[0.5934532170,        1.5401904807,        0.9180939542],
[1.3502915242,       -0.5487354338,        0.7697569517],
[1.0552472919,        0.6658098274,       -1.0770570582],
[0.0639893710,       -1.7361386921,       -0.0497372639],
[0.1497545492,        0.7404287481,        1.5652767895],
[-0.1775530982,       -0.7417516846,       -1.5617370735],
[-0.0917526113,        1.7347829221,        0.0532527687],
[0.4953414545,       -1.2627572031,        0.8712619429],
[0.5282333540,       -0.3129362014,        1.4906528819],
[0.1379791660,        1.2935710279,       -0.9521729470],
[0.1050718541,        0.3437728612,       -1.5715478759],
[-1.2096309503,       -0.8362798316,       -0.3567187866],
[-1.1556832707,        0.7211409560,        0.6589142264],
[0.7720399412,       -0.4111102371,       -1.0693857317],
[0.8245781570,        1.1055544408,       -0.0803402530],
[0.9726327415,        0.4960238283,        0.8465143019],
[0.9201060910,       -1.0206528588,       -0.1425372613],
[1.2020715832,        0.0585012795,       -0.1535592917],
[-0.4942101210,       -0.0573389250,        1.1481429905],
[-0.7695269053,       -0.0041779903,       -0.9866843449],
[-0.5269493969,       -1.0027161915,        0.5316387984],
[-0.7367730802,        0.9411960611,       -0.3701868286],
[-0.6523929124,       -0.0317534193,        0.0833460057],
[-0.0402214881,        0.7366414278,        0.4908776999],
[-0.0914715932,       -0.7430505962,       -0.4740567993],
[0.1863769549,        0.3112862636,       -0.4872447736],
[0.3314896614,       -0.2860836867,        0.4210904765]]

pos_ne32=pos_ne32*5.27
bc_ne32 = SphericalBC(radius=20.)
conf_ne32 = Config(pos_ne32, bc_ne32)

#bc_ar32 = SphericalBC(radius=14.5)  #Angstrom

#Ar32 starting config (fcc, pbc)
pos_ne32_pbc = [[-0.1000000000E+02, -0.1000000000E+02, -0.1000000000E+02],
[-0.5000000000E+01, -0.5000000000E+01, -0.1000000000E+02],
[-0.5000000000E+01, -0.1000000000E+02, -0.5000000000E+01],
[-0.1000000000E+02, -0.5000000000E+01, -0.5000000000E+01],
[-0.1000000000E+02, -0.1000000000E+02,  0.0000000000E+00],
[-0.5000000000E+01, -0.5000000000E+01,  0.0000000000E+00],
[-0.5000000000E+01, -0.1000000000E+02,  0.5000000000E+01],
[-0.1000000000E+02, -0.5000000000E+01,  0.5000000000E+01],
[-0.1000000000E+02,  0.0000000000E+00, -0.1000000000E+02],
[-0.5000000000E+01,  0.5000000000E+01, -0.1000000000E+02],
[-0.5000000000E+01,  0.0000000000E+00, -0.5000000000E+01],
[-0.1000000000E+02,  0.5000000000E+01, -0.5000000000E+01],
[-0.1000000000E+02,  0.0000000000E+00,  0.0000000000E+00],
[-0.5000000000E+01,  0.5000000000E+01,  0.0000000000E+00],
[-0.5000000000E+01,  0.0000000000E+00,  0.5000000000E+01],
[-0.1000000000E+02,  0.5000000000E+01,  0.5000000000E+01],
[0.0000000000E+00, -0.1000000000E+02, -0.1000000000E+02],
[0.5000000000E+01, -0.5000000000E+01, -0.1000000000E+02],
[0.5000000000E+01, -0.1000000000E+02, -0.5000000000E+01],
[0.0000000000E+00, -0.5000000000E+01, -0.5000000000E+01],
[0.0000000000E+00, -0.1000000000E+02,  0.0000000000E+00],
[0.5000000000E+01, -0.5000000000E+01,  0.0000000000E+00],
[0.5000000000E+01, -0.1000000000E+02,  0.5000000000E+01],
[0.0000000000E+00, -0.5000000000E+01,  0.5000000000E+01],
[0.0000000000E+00,  0.0000000000E+00, -0.1000000000E+02],
[0.5000000000E+01,  0.5000000000E+01, -0.1000000000E+02],
[0.5000000000E+01,  0.0000000000E+00, -0.5000000000E+01],
[0.0000000000E+00,  0.5000000000E+01, -0.5000000000E+01],
[0.0000000000E+00,  0.0000000000E+00,  0.0000000000E+00],
[0.5000000000E+01,  0.5000000000E+01,  0.0000000000E+00],
[0.5000000000E+01,  0.0000000000E+00,  0.5000000000E+01],
[0.0000000000E+00,  0.5000000000E+01,  0.5000000000E+01]]

pos_ne32_pbc=pos_ne32_pbc*0.828
bc_ne32_pbc = CubicBC(16.56)
conf_ne32_pbc = Config(pos_ne32_pbc, bc_ne32_pbc)

#println(conf_ne32_pbc)
#println(conf_ne32_pbc.bc)
#println(conf_ne32_pbc.bc.length)

#println("distances in neon 32 is ", get_distance2_mat_cbc(conf_ne32_pbc))

#println("ne32 pbc initial energy = ", dimer_energy_config(get_distance2_mat_cbc(conf_ne32_pbc), 32, elj_ne))

end