using ParallelTemperingMonteCarlo,DelimitedFiles


script_folder = @__DIR__
data_path = joinpath(script_folder, "testing_data") 

X = [ 2              0.001   0.000  11.338
 2              0.020   0.000  11.338
 2              0.035   0.000  11.338
 2              0.100   0.000  11.338
 2              0.400   0.000  11.338
 ]

radsymmvec = []
#--------------------------------------------#
#--------Vector of angular symm values-------#
#--------------------------------------------#
V = [[0.0001,1,1,11.338],
[0.0001,-1,2,11.338],
[0.003,-1,1,11.338],
[0.003,-1,2,11.338],
[0.008,-1,1,11.338],
[0.008,-1,2,11.338],
[0.008,1,2,11.338],
[0.015,1,1,11.338],
[0.015,-1,2,11.338],
[0.015,-1,4,11.338],
[0.015,-1,16,11.338],
[0.025,-1,1,11.338],
[0.025,1,1,11.338],
[0.025,1,2,11.338],
[0.025,-1,4,11.338],
[0.025,-1,16,11.338],
[0.025,1,16,11.338],
[0.045,1,1,11.338],
[0.045,-1,2,11.338],
[0.045,-1,4,11.338],
[0.045,1,4,11.338],
[0.045,1,16,11.338],
[0.08,1,1,11.338],
[0.08,-1,2,11.338],
[0.08,-1,4,11.338],
[0.08,1,4,11.338]]

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
G_value_vec_b = []
for row in eachrow(scalingvalues[89:end,:])
    max_min = [row[4],row[3]]
    push!(G_value_vec_b,max_min)
end
for symmindex in eachindex(eachrow(X))
    row = X[symmindex,:]
    radsymm = RadialType2a{Float64}(row[2],row[4],Int(row[1]),[G_value_vec[(symmindex-1)*2 + 1],G_value_vec[(symmindex-1)*2 + 2]],[G_value_vec_b[(symmindex-1)*2 + 1],G_value_vec_b[(symmindex-1)*2 + 2]])
    push!(radsymmvec,radsymm)
end
let n_index = 10
let j_index = 0
for element in V
    #for types in T

        j_index += 1

        symmfunc = AngularType3a{Float64}(element[1],element[2],element[3],11.338,2,[G_value_vec[n_index + (j_index-1)*3 + 1],G_value_vec[n_index + (j_index-1)*3 + 2],G_value_vec[n_index + (j_index-1)*3 + 3] ],[G_value_vec_b[n_index + (j_index-1)*3 + 1],G_value_vec_b[n_index + (j_index-1)*3 + 2],G_value_vec_b[n_index + (j_index-1)*3 + 3] ])

        push!(angularsymmvec,symmfunc)
    #end
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
nnpcu = NeuralNetworkPotential(num_nodes,activation_functions,weights)

file2=open(joinpath(data_path, "weights.030.data"),"r+") #./data/weights.030.data
weights2=readdlm(file2)
close(file2)
weights2 = vec(weights2)
nnpzn= NeuralNetworkPotential(num_nodes,activation_functions,weights2)
ensemble = NNVT([32,6];natomswaps=2)

runnerpotential = RuNNerPotential2Atom(nnpcu,nnpzn,radsymmvec,angularsymmvec,4,2)