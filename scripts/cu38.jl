using ParallelTemperingMonteCarlo

using Random,DelimitedFiles

script_folder = @__DIR__ # folder where this script is located
data_path = joinpath(script_folder, "data") # path to data files, so "./data/"

Random.seed!(1234)


#-------------------------------------------------------#
#-----------------------MC Params-----------------------#
#-------------------------------------------------------#


n_atoms = 38

ti = 200.
tf = 1300.
n_traj = 28

temp = TempGrid{n_traj}(ti,tf) 

mc_cycles = 10000
mc_sample = 1

displ_atom = 0.1 # Angstrom
n_adjust = 100

max_displ_atom = [0.1*sqrt(displ_atom*temp.t_grid[i]) for i in 1:n_traj]

mc_params = MCParams(mc_cycles, n_traj, n_atoms, mc_sample = mc_sample, n_adjust = n_adjust)

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

evtohartree = 0.0367493
nmtobohr = 18.8973

#parameters taken from L Vocadlo etal J Chem Phys V120N6 2004
n = 8.482
m = 4.692
ϵ = evtohartree*0.0370
a = 0.25*nmtobohr
C = 27.561

pot = EmbeddedAtomPotential(n,m,ϵ,C,a)
#-------------------------------------------------------------#
#------------------RuNNer Potential---------------------------#
#-------------------------------------------------------------#
#-------------------------------------------#
#--------Vector of radial symm values-------#
#-------------------------------------------#
X = [ 11              0.001   0.000  11.338
 10              0.001   0.000  11.338
 11              0.020   0.000  11.338
 10              0.020   0.000  11.338
 11              0.035   0.000  11.338
 10              0.035   0.000  11.338
 11              0.100   0.000  11.338
 10              0.100   0.000  11.338
 11              0.400   0.000  11.338
 10              0.400   0.000  11.338]

radsymmvec = []


#--------------------------------------------#
#--------Vector of angular symm values-------#
#--------------------------------------------#
V = [[0.0001,1,1,11.338],[0.0001,-1,2,11.338],[0.003,-1,1,11.338],[0.003,-1,2,11.338],[0.008,-1,1,11.338],[0.008,-1,2,11.338],[0.008,1,2,11.338],[0.015,1,1,11.338],[0.015,-1,2,11.338],[0.015,-1,4,11.338],[0.015,-1,16,11.338],[0.025,-1,1,11.338],[0.025,1,1,11.338],[0.025,1,2,11.338],[0.025,-1,4,11.338],[0.025,-1,16,11.338],[0.025,1,16,11.338],[0.045,1,1,11.338],[0.045,-1,2,11.338],[0.045,-1,4,11.338],[0.045,1,4,11.338],[0.045,1,16,11.338],[0.08,1,1,11.338],[0.08,-1,2,11.338],[0.08,-1,4,11.338],[0.08,1,4,11.338]]

T = [111,110,100]

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
    radsymm = RadialType2{Float64}(row[2],row[4],Int(row[1]),G_value_vec[symmindex])
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

runnerpotential = RuNNerPotential(nnp,radsymmvec,angularsymmvec)
#-------------------------------------------------------------#
#------------------------Move Strategy------------------------#
#-------------------------------------------------------------#
ensemble = NVT(n_atoms)
move_strat = MoveStrategy(ensemble)
#-------------------------------------------------------------#
#-----------------------Starting Config-----------------------#
#-------------------------------------------------------------#


pos_cu38 = [[2.3603476948363165, 2.3603476948363165, 0.0],
[6.994369407022418, 2.33998871029911, 0.0],
[2.33998871029911, 6.994369407022418, 0.0],
[-2.3603476948363165, 2.3603476948363165, 0.0],
[-6.994369407022418, 2.33998871029911, 0.0],
[-2.33998871029911, 6.994369407022418, 0.0],
[-2.3603476948363165, -2.3603476948363165, 0.0],
[-6.994369407022418, -2.33998871029911, 0.0],
[-2.33998871029911, -6.994369407022418, 0.0],
[2.3603476948363165, -2.3603476948363165, 0.0],
[6.994369407022418, -2.33998871029911, 0.0],
[2.33998871029911, -6.994369407022418, 0.0],
[0.0, 0.0, 3.3380357219419614],
[4.84532317769689, 0.0, 3.4261608756649893],
[-4.84532317769689, 0.0, 3.4261608756649893],
[0.0, 4.84532317769689, 3.4261608756649893],
[0.0, -4.84532317769689, 3.4261608756649893],
[4.667179058660764, 4.667179058660764, 3.2911441531516483],
[-4.667179058660764, 4.667179058660764, 3.2911441531516483],
[-4.667179058660764, -4.667179058660764, 3.2911441531516483],
[4.667179058660764, -4.667179058660764, 3.2911441531516483],
[0.0, 0.0, -3.3380357219419614],
[4.84532317769689, 0.0, -3.4261608756649893],
[-4.84532317769689, 0.0, -3.4261608756649893],
[0.0, 4.84532317769689, -3.4261608756649893],
[0.0, -4.84532317769689, -3.4261608756649893],
[4.667179058660764, 4.667179058660764, -3.2911441531516483],
[-4.667179058660764, 4.667179058660764, -3.2911441531516483],
[-4.667179058660764, -4.667179058660764, -3.2911441531516483],
[4.667179058660764, -4.667179058660764, -3.2911441531516483], 
[2.327190348361654, 2.327190348361654, 6.600387922922003],
[-2.327190348361654, 2.327190348361654, 6.600387922922003],
[-2.327190348361654, -2.327190348361654, 6.600387922922003],
[2.327190348361654, -2.327190348361654, 6.600387922922003],
[2.327190348361654, 2.327190348361654, -6.600387922922003],
[-2.327190348361654, 2.327190348361654, -6.600387922922003],
[-2.327190348361654, -2.327190348361654, -6.600387922922003],
[2.327190348361654, -2.327190348361654, -6.600387922922003]]

n_bin = 100
AtoBohr = 1.8897259886

bc_cu38 = SphericalBC(radius=14*AtoBohr) 
start_config = Config(pos_cu38, bc_cu38)


states,results = ptmc_run!(mc_params,temp,start_config,pot,ensemble;save=1000)
