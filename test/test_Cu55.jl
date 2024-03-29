using ParallelTemperingMonteCarlo
using Random, DelimitedFiles

# takes about 100 seconds on 1 thread on CI

script_folder = "../scripts/" # folder where the scripts are located
data_path = joinpath(script_folder, "data") # path to data files, so "./data/"

## When moving this script to a different folder it may be necessary to adjust the
## location of the shared library "librunner.so". Uncomment and adjust the next line for this.
# ParallelTemperingMonteCarlo.MachineLearningPotential.ForwardPass.lib_path() = joinpath(@__DIR__, "../MachineLearningPotential/lib/")


#set random seed - for reproducibility
Random.seed!(1234)

# number of atoms

n_atoms = 55


# temperature grid
ti = 550
tf = 900

n_traj = 20


temp = TempGrid{n_traj}(ti,tf)

# MC simulation details

mc_cycles = 5 #default 20% equilibration cycles on top


mc_sample = 1  #sample every mc_sample MC cycles

#move_atom=AtomMove(n_atoms) #move strategy (here only atom moves, n_atoms per MC cycle)
displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

#moves - allowed at present: atom, volume and rotation moves (volume,rotation not yet implemented)
move_strat = MoveStrategy(atom_moves = n_atoms)

#ensemble
ensemble = NVT(n_atoms)




#starting configurations
#icosahedral ground state of Cu55 (from Cambridge cluster database) converted to angstrom
ico_13  = [[-0.0000000049,       -0.0000000044,       -0.0000000033],
[-0.0000007312,       -0.0000000014,        0.6554619119],
 [0.1811648930,      -0.5575692094,        0.2931316798],
[-0.4742970242,       -0.3445967289,        0.2931309525],
[-0.4742970303,        0.3445967144,        0.2931309494],
 [0.1811648830,        0.5575692066,        0.2931316748],
 [0.5862626299,        0.0000000022,        0.2931321262],
[-0.1811648928,       -0.5575692153,       -0.2931316813],
[-0.5862626397,       -0.0000000109,       -0.2931321327],
[-0.1811649028,        0.5575692007,       -0.2931316863],
 [0.4742970144,        0.3445967202,       -0.2931309590],
 [0.4742970205,       -0.3445967231,       -0.2931309559],
 [0.0000007214,       -0.0000000073,       -0.6554619185]]


ico_55 = [[0.0000006584,       -0.0000019175,        0.0000000505],
[-0.0000005810,       -0.0000004871,        0.6678432175],
[0.1845874248,       -0.5681026047,        0.2986701538],
[-0.4832557457,       -0.3511072166,        0.2986684497],
[-0.4832557570,        0.3511046452,        0.2986669456],
[0.1845874064,        0.5681000550,        0.2986677202],
[0.5973371920,       -0.0000012681,        0.2986697030],
[-0.1845860897,       -0.5681038901,       -0.2986676192],
[-0.5973358752,       -0.0000025669,       -0.2986696020],
[-0.1845861081,        0.5680987696,       -0.2986700528],
[0.4832570624,        0.3511033815,       -0.2986683486],
[0.4832570738,       -0.3511084803,       -0.2986668445],
[0.0000018978,       -0.0000033480,       -0.6678431165],
[-0.0000017969,        0.0000009162,        1.3230014650],
[0.1871182835,       -0.5758942175,        0.9797717078],
[-0.4898861924,       -0.3559221410,       0.9797699802],
[-0.4898862039,        0.3559224872,        0.9797684555],
[0.1871182648,        0.5758945856,        0.9797692407],
[0.6055300485,        0.0000001908,        0.9797712507],
[0.7926501864,       -0.5758950093,        0.6055339635],
[0.3656681761,       -1.1254128670,        0.5916673591],
[-0.3027660545,       -0.9318173412,        0.6055326929],
[-0.9573332453,       -0.6955436707,        0.5916639831],
[-0.9797705418,       -0.0000006364,        0.6055294407],
[-0.9573332679,        0.6955423392,        0.5916610035],
[-0.3027660847,        0.9318160902,        0.6055287012],
[0.3656681396,        1.1254115783,        0.5916625380],
[0.7926501677,        0.5758937939,        0.6055314964],
[1.1833279992,       -0.0000006311,        0.5916664660],
[0.6770051458,       -0.9318186223,        0.0000033028],
[0.0000006771,       -1.1517907207,        0.0000025175],
[-0.6770037988,       -0.9318186442,        0.0000007900],
[-1.0954155825,       -0.3559242494,       -0.0000012200],
[-1.0954155940,        0.3559203788,       -0.0000027447],
[-0.6770038290,        0.9318147872,       -0.0000032017],
[0.0000006397,        1.1517868856,       -0.0000024165],
[0.6770051155,        0.9318148091,       -0.0000006889],
[1.0954168993,        0.3559204143,        0.0000013211],
[1.0954169108,       -0.3559242139,        0.0000028458],
[0.3027674014,       -0.9318199253,       -0.6055286002],
[-0.3656668229,       -1.1254154134,       -0.5916624370],
[-0.7926488510,       -0.5758976290,       -0.6055313954],
[-1.1833266824,       -0.0000032040,       -0.5916663649],
[-0.7926488697,        0.5758911742,       -0.6055338624],
[-0.3656668594,        1.1254090319,       -0.5916672580],
[0.3027673712,        0.9318135061,       -0.6055325919],
[0.9573345621,        0.6955398357,       -0.5916638820],
[0.9797718586,       -0.0000031986,       -0.6055293396],
[0.9573345846,       -0.6955461743,       -0.5916609025],
[-0.1871169480,       -0.5758984207,       -0.9797691397],
[-0.6055287318,       -0.0000040259,       -0.9797711497],
[-0.1871169667,        0.5758903824,       -0.9797716067],
[0.4898875091,        0.3559183059,       -0.9797698792],
[0.4898875207,       -0.3559263223,       -0.9797683545],
[0.0000031136,       -0.0000047513,       -1.3230013639]]
#convert to Bohr
nmtobohr = 18.8973
copperconstant = 0.36258*nmtobohr
pos_cu55 = copperconstant*ico_55
pos_cu13 = copperconstant*ico_13*1.5
AtoBohr = 1.8897259886

length(pos_cu55) == n_atoms || error("number of atoms and positions not the same - check starting config")


#boundary conditions
bc_cu55 = SphericalBC(radius=14*AtoBohr)   #5.32 Angstrom
bc_cu13 = SphericalBC(radius=8*AtoBohr)
#starting configuration

start_config = Config(pos_cu55, bc_cu55)

#histogram information
n_bin = 100
#-------------------------------------------#
#--------Vector of radial symm values-------#
#-------------------------------------------#
X = [ 1    1              0.001   0.000  11.338
 1    0              0.001   0.000  11.338
 1    1              0.020   0.000  11.338
 1    0              0.020   0.000  11.338
 1    1              0.035   0.000  11.338
 1    0              0.035   0.000  11.338
 1    1              0.100   0.000  11.338
 1    0              0.100   0.000  11.338
 1    1              0.400   0.000  11.338
 1    0              0.400   0.000  11.338]

radsymmvec = []


#--------------------------------------------#
#--------Vector of angular symm values-------#
#--------------------------------------------#
V = [[0.0001,1,1,11.338],[0.0001,-1,2,11.338],[0.003,-1,1,11.338],[0.003,-1,2,11.338],[0.008,-1,1,11.338],[0.008,-1,2,11.338],[0.008,1,2,11.338],[0.015,1,1,11.338],[0.015,-1,2,11.338],[0.015,-1,4,11.338],[0.015,-1,16,11.338],[0.025,-1,1,11.338],[0.025,1,1,11.338],[0.025,1,2,11.338],[0.025,-1,4,11.338],[0.025,-1,16,11.338],[0.025,1,16,11.338],[0.045,1,1,11.338],[0.045,-1,2,11.338],[0.045,-1,4,11.338],[0.045,1,4,11.338],[0.045,1,16,11.338],[0.08,1,1,11.338],[0.08,-1,2,11.338],[0.08,-1,4,11.338],[0.08,1,4,11.338]]

T = [[1.,1.,1.],[1.,1.,0.],[1.,0.,0.]]

angularsymmvec = []
#-------------------------------------------#
#-----------Including scaling data----------#
#-------------------------------------------#
file = open(joinpath(data_path,"scaling.data")) # full path "./data/scaling.data"
scalingvalues = readdlm(file)
close(file)
G_value_vec = []
for row in eachrow(scalingvalues[1:88,:])
    max_min = [row[4],row[3]]
    push!(G_value_vec,max_min)
end


for symmindex in eachindex(eachrow(X))
    row = X[symmindex,:]
    radsymm = RadialType2{Float64}(row[3],row[5],[row[1],row[2]],G_value_vec[symmindex])
    push!(radsymmvec,radsymm)
end


let n_index = 10

for element in V
    for types in T

        n_index += 1

        symmfunc = AngularType3{Float64}(element[1],element[2],element[3],11.338,types,G_value_vec[n_index])

        push!(angularsymmvec,symmfunc)
    end
end
end
#---------------------------------------------------#
#------concatenating radial and angular values------#
#---------------------------------------------------#

totalsymmvec = vcat(radsymmvec,angularsymmvec)


#--------------------------------------------------#
#-----------Initialising the nnp weights-----------#
#--------------------------------------------------#
num_nodes::Vector{Int32} = [88, 20, 20, 1]
activation_functions::Vector{Int32} = [1, 2, 2, 1]
file = open(joinpath(data_path, "weights.029.data"), "r+") # "./data/weights.029.data"
weights=readdlm(file)
close(file)
weights = vec(weights)
nnp = NeuralNetworkPotential(num_nodes,activation_functions,weights)

runnerpotential = RuNNerPotential(nnp,totalsymmvec)
#------------------------------------------------------------#
#============================================================#
#------------------------------------------------------------#

mc_states = [MCState(temp.t_grid[i], temp.beta_grid[i], start_config, runnerpotential; max_displ=[max_displ_atom[i],0.01,1.]) for i in 1:n_traj]



#results = Output(n_bin, max_displ_vec)
results = Output{Float64}(n_bin; en_min = mc_states[1].en_tot)

@time ptmc_run!((mc_states, move_strat, mc_params, runnerpotential, ensemble, results));

# plot(temp.t_grid,results.heat_cap)

 data = [results.en_histogram[i] for i in 1:n_traj]
# plot(data)
# histplot = plot(data)
# savefig(histplot,"histograms.png")
