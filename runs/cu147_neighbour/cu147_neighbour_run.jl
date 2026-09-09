script_folder = @__DIR__
repo_folder = normpath(joinpath(script_folder, "..", ".."))

Pkg.activate(repo_folder)
Pkg.instantiate()

using ParallelTemperingMonteCarlo
using Random
using DelimitedFiles
using StaticArrays
using LinearAlgebra
using Statistics

data_path = joinpath(repo_folder, "scripts", "data")

function mackay_icosahedron(shell::Int)
    φ = (1 + sqrt(5.0)) / 2

    verts = SVector{3,Float64}[]

    for s1 in (-1.0, 1.0), s2 in (-1.0, 1.0)
        push!(verts, SVector(0.0, s1, s2 * φ))
        push!(verts, SVector(s1, s2 * φ, 0.0))
        push!(verts, SVector(s1 * φ, 0.0, s2))
    end

    edge = minimum(
        norm(verts[i] - verts[j])
        for i in eachindex(verts), j in eachindex(verts)
        if i < j
    )

    faces = Tuple{Int,Int,Int}[]

    for i in 1:length(verts)-2
        for j in i+1:length(verts)-1
            for k in j+1:length(verts)
                if abs(norm(verts[i] - verts[j]) - edge) < 1e-8 &&
                   abs(norm(verts[i] - verts[k]) - edge) < 1e-8 &&
                   abs(norm(verts[j] - verts[k]) - edge) < 1e-8
                    push!(faces, (i, j, k))
                end
            end
        end
    end

    pointset = Set{NTuple{3,Float64}}()

    for (a, b, c) in faces
        v1, v2, v3 = verts[a], verts[b], verts[c]

        for i in 0:shell
            for j in 0:(shell - i)
                for k in 0:(shell - i - j)
                    p = (i * v1 + j * v2 + k * v3) / shell
                    push!(
                        pointset,
                        (
                            round(p[1], digits=10),
                            round(p[2], digits=10),
                            round(p[3], digits=10),
                        ),
                    )
                end
            end
        end
    end

    positions = [SVector{3,Float64}(p) for p in pointset]

    centre = sum(positions) / length(positions)

    return [p - centre for p in positions]
end

function deduplicate_positions(positions; tol=1e-8)
    seen = Set{NTuple{3,Int}}()
    unique_positions = SVector{3,Float64}[]

    for p in positions
        key = (
            round(Int, p[1] / tol),
            round(Int, p[2] / tol),
            round(Int, p[3] / tol)
        )

        if !(key in seen)
            push!(seen, key)
            push!(unique_positions, p)
        end
    end

    return unique_positions
end

function find_central_atom(positions)
    return argmin([norm(p) for p in positions])
end

function scale_by_central_neighbours(positions; target_nn_bohr)
    centre_index = find_central_atom(positions)
    x0 = positions[centre_index]

    dists = sort([
        norm(positions[i] - x0)
        for i in eachindex(positions)
        if i != centre_index
    ])

    current_nn = mean(dists[1:12])
    scale = target_nn_bohr / current_nn

    return [scale * p for p in positions]
end

Random.seed!(1234)

println("Julia threads: ", Threads.nthreads())
println("Running Cu147 neighbour PTMC 10,000-cycle timing run")


ti = 400.0
tf = 1200.0

# Retain the real number of replicas initially.
n_traj = 28

temp = TempGrid{n_traj}(ti, tf)

mc_cycles = 10000
mc_sample = 1
n_adjust = 100

#-------------------------------------------------------------#
#----------------------Potential------------------------------#
#-------------------------------------------------------------#

evtohartree = 0.0367493
nmtobohr = 18.8973


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

radsymmvec = RadialType2{Float64}[]
angularsymmvec = AngularType3{Float64}[]

#--------------------------------------------#
#--------Vector of angular symm values-------#
#--------------------------------------------#
V = [[0.0001,1,1,11.338],[0.0001,-1,2,11.338],[0.003,-1,1,11.338],[0.003,-1,2,11.338],[0.008,-1,1,11.338],[0.008,-1,2,11.338],[0.008,1,2,11.338],[0.015,1,1,11.338],[0.015,-1,2,11.338],[0.015,-1,4,11.338],[0.015,-1,16,11.338],[0.025,-1,1,11.338],[0.025,1,1,11.338],[0.025,1,2,11.338],[0.025,-1,4,11.338],[0.025,-1,16,11.338],[0.025,1,16,11.338],[0.045,1,1,11.338],[0.045,-1,2,11.338],[0.045,-1,4,11.338],[0.045,1,4,11.338],[0.045,1,16,11.338],[0.08,1,1,11.338],[0.08,-1,2,11.338],[0.08,-1,4,11.338],[0.08,1,4,11.338]]

T = [111,110,100]

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

#--------------------------------------------------#
#-----------Initialising the nnp weights-----------#
#--------------------------------------------------#
num_nodes::Vector{Int32} = [88, 20, 20, 1]
activation_functions::Vector{Int32} = [1, 2, 2, 1]

file = open(joinpath(data_path, "weights.029.data"), "r")
weights = readdlm(file)
close(file)

weights = vec(weights)

nnp = NeuralNetworkPotential(
    num_nodes,
    activation_functions,
    weights,
)

runnerpotential = RuNNerPotentialWithNeighbourhood(
    nnp,
    radsymmvec,
    angularsymmvec,
)

# ------------------------------------------------------------
# Generate Cu147 Mackay icosahedron
# ------------------------------------------------------------

target_nn_bohr = 4.575916480942233

positions_raw = mackay_icosahedron(3)

positions_unique = deduplicate_positions(positions_raw)

println("Atoms before deduplication = ", length(positions_raw))

n_atoms = length(positions_unique)
println("Atoms after deduplication = ", n_atoms)

@assert length(positions_unique) == 147

positions = scale_by_central_neighbours(
    positions_unique;
    target_nn_bohr=target_nn_bohr,
)
centre_index_scaled = find_central_atom(positions)

scaled_central_nn = minimum(
    norm(positions[i] - positions[centre_index_scaled])
    for i in eachindex(positions)
    if i != centre_index_scaled
)

println("Scaled shortest central-neighbour distance = ", scaled_central_nn)
println("Target central-neighbour distance = ", target_nn_bohr)
println("Scaling difference = ", abs(scaled_central_nn - target_nn_bohr))

@assert n_atoms == 147

println("Generated Cu147")
println("Central atom index = ", find_central_atom(positions))

mc_params = MCParams(
    mc_cycles,
    n_traj,
    n_atoms;
    mc_sample=mc_sample,
    n_adjust=n_adjust,

)
println("mc_params.eq_cycles = ", mc_params.eq_cycles)

cluster_radius = maximum(norm(p) for p in positions)

bc_cu147 = SphericalBC(
    radius = cluster_radius + 5.0,
)

start_config = Config(positions, bc_cu147)

println("Potential type: ", typeof(runnerpotential))
println("Neural-network type: ", typeof(nnp))
println("Production cycles: ", mc_params.mc_cycles)
println("Equilibration cycles: ", mc_params.eq_cycles)
println("Julia threads: ", Threads.nthreads())


ensemble = NVT(n_atoms)

# ------------------------------------------------------------
# Cu147 neighbour PTMC timing / production-style test
# ------------------------------------------------------------

println()
println("Starting Cu147 neighbour PTMC run")
println("Potential type: ", typeof(runnerpotential))
println("Number of atoms: ", n_atoms)
println("Production cycles: ", mc_params.mc_cycles)
println("Equilibration cycles: ", mc_params.eq_cycles)
println("Julia threads: ", Threads.nthreads())
println("Boundary radius (Bohr): ", cluster_radius + 5.0)

elapsed = @elapsed begin
    global states, results = ptmc_run!(
        mc_params,
        temp,
        start_config,
        runnerpotential,
        ensemble;
        rdfsave=false,
        restart=false,
        save=500,
        saveconfigs=500,
        configsname="cu147_10000_",
        workingdirectory=script_folder,
    )
end

println()
println("Cu147 neighbour PTMC timing run completed successfully")
println("PTMC elapsed time: ", elapsed, " seconds")
println("Number of returned states: ", length(states))
println("Final energies: ", [state.en_tot for state in states])
