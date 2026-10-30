using ParallelTemperingMonteCarlo
using Random, DelimitedFiles

data_path = joinpath(@__DIR__, "data")

n_atoms = 55
ti = 250.0
tf = 900.0
n_traj = 10

temp = TempGrid{n_traj}(ti, tf)
mc_cycles = 20
mc_params = MCParams(mc_cycles, n_traj, n_atoms)

# Potential
evtohartree = 0.0367493
nmtobohr = 18.8973

X = [
    2 0.001 0.000 11.338
    2 0.020 0.000 11.338
    2 0.035 0.000 11.338
    2 0.100 0.000 11.338
    2 0.400 0.000 11.338
]

#--------------------------------------------#
#--------Vector of angular symm values-------#
#--------------------------------------------#
V = [
    [0.0001, 1, 1, 11.338],
    [0.0001, -1, 2, 11.338],
    [0.003, -1, 1, 11.338],
    [0.003, -1, 2, 11.338],
    [0.008, -1, 1, 11.338],
    [0.008, -1, 2, 11.338],
    [0.008, 1, 2, 11.338],
    [0.015, 1, 1, 11.338],
    [0.015, -1, 2, 11.338],
    [0.015, -1, 4, 11.338],
    [0.015, -1, 16, 11.338],
    [0.025, -1, 1, 11.338],
    [0.025, 1, 1, 11.338],
    [0.025, 1, 2, 11.338],
    [0.025, -1, 4, 11.338],
    [0.025, -1, 16, 11.338],
    [0.025, 1, 16, 11.338],
    [0.045, 1, 1, 11.338],
    [0.045, -1, 2, 11.338],
    [0.045, -1, 4, 11.338],
    [0.045, 1, 4, 11.338],
    [0.045, 1, 16, 11.338],
    [0.08, 1, 1, 11.338],
    [0.08, -1, 2, 11.338],
    [0.08, -1, 4, 11.338],
    [0.08, 1, 4, 11.338],
]
T = [111, 110, 100]
#-------------------------------------------#
#-----------Including scaling data----------#
#-------------------------------------------#
scalingvalues = readdlm(joinpath(data_path, "scaling.data"))[1:(end - 1), :]
G_value_vec = Vector{Float64}[]
for row in eachrow(scalingvalues[1:88, :])
    max_min = [row[4], row[3]]
    push!(G_value_vec, max_min)
end
G_value_vec_b = Vector{Float64}[]
for row in eachrow(scalingvalues[89:end, :])
    max_min = [row[4], row[3]]
    push!(G_value_vec_b, max_min)
end

radsymmvec = RadialType2a{Float64}[]

for symmindex in eachindex(eachrow(X))
    row = X[symmindex, :]
    radsymm = RadialType2a{Float64}(
        row[2],
        row[4],
        Int(row[1]),
        [G_value_vec[(symmindex - 1) * 2 + 1], G_value_vec[(symmindex - 1) * 2 + 2]],
        [G_value_vec_b[(symmindex - 1) * 2 + 1], G_value_vec_b[(symmindex - 1) * 2 + 2]],
    )
    push!(radsymmvec, radsymm)
end

angularsymmvec = AngularType3a{Float64}[]

let n_index = 10
    let j_index = 0
        for element in V
            j_index += 1

            symmfunc = AngularType3a{Float64}(
                element[1],
                element[2],
                element[3],
                11.338,
                2,
                [
                    G_value_vec[n_index + (j_index - 1) * 3 + 1],
                    G_value_vec[n_index + (j_index - 1) * 3 + 2],
                    G_value_vec[n_index + (j_index - 1) * 3 + 3],
                ],
                [
                    G_value_vec_b[n_index + (j_index - 1) * 3 + 1],
                    G_value_vec_b[n_index + (j_index - 1) * 3 + 2],
                    G_value_vec_b[n_index + (j_index - 1) * 3 + 3],
                ],
            )

            push!(angularsymmvec, symmfunc)
        end
    end
end
#---------------------------------------------------#
#------concatenating radial and angular values------#
#---------------------------------------------------#
totalsymmvec = vcat(radsymmvec, angularsymmvec)
#--------------------------------------------------#
#-----------Initialising the nnp weights-----------#
#--------------------------------------------------#
num_nodes = Int32[88, 20, 20, 1]
activation_functions = Int32[1, 2, 2, 1]
weights = vec(readdlm(joinpath(data_path, "weights.029.data")))
nnp_copper = NeuralNetworkPotential(num_nodes, activation_functions, weights)

weights2 = vec(readdlm(joinpath(data_path, "weights.030.data")))
nnp_zinc = NeuralNetworkPotential(num_nodes, activation_functions, weights2)
ensemble = NNVT([50, 5]; n_atom_swaps=2)

runnerpotential = RuNNerPotential2Atom(
    nnp_copper, nnp_zinc, radsymmvec, angularsymmvec, 50, 5
)

config = magic_cluster(2; r_min=3.13)
shuffle!(config.positions)

states, results, stats = ptmc_run!(
    mc_params, temp, config, runnerpotential, ensemble; save=100
)
