# A wrapper script around a Fortran library for the RuNNer forward pass.

using Test
using ParallelTemperingMonteCarlo
using ParallelTemperingMonteCarlo.MachineLearningPotential.ForwardPass: lib_path

function forward_pass(
    input::Matrix{Float64},
    num_layers::Int32,
    num_nodes::Vector{Int32},
    batchsize::Int32,
    activation_functions::Vector{Int32},
    num_parameters::Int32,
    parameters::Vector{Float64}
)
    """Perform a forward pass using a Fortran library."""
    eatom = zeros(Float64, batchsize)
    return ccall(
        (:forward, joinpath(lib_path(), "librunnerjulia.so")),
        Float64,
        (Ref{Float64},Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32},
         Ref{Int32}, Ref{Float64}, Ref{Int32}, Ref{Float64}),
        input, num_nodes[1], num_layers, num_nodes, batchsize,
        num_parameters, parameters, activation_functions, eatom
    )
end

function test_forward(num_atoms, batchsize)
    """Run a forward pass through a neural network for testing purposes."""
    # Define constant variables.
    num_layers::Int32 = 4
    num_nodes::Vector{Int32} = [88, 20, 20, 1]
    activation_functions::Vector{Int32} = [1, 2, 2, 1]

    # Calculate the number of parameters in the defined NN.
    num_parameters::Int32 = 0
    for i in 2:num_layers
        num_parameters += num_nodes[i - 1] * num_nodes[i]
        num_parameters += num_nodes[i]
    end

    # Set parameters and input to one for testing.
    parameters = ones(Float64, num_parameters)
    input = ones(Float64, (num_nodes[1], batchsize))

    # Predict atomic energy for fictious `num_atoms` atoms.
    for i in 1:num_atoms
        res = forward_pass(
            input,
            num_layers,
            num_nodes,
            batchsize,
            activation_functions,
            num_parameters,
            parameters
        )
        @test res == 21
    end
end

@testset "Runner forward pass" begin
    # Run forward pass multiple times.
    num_atoms = 55 * 1e2
    batchsize::Int32 = 2
    num_repeats = 2
    timings = [@elapsed test_forward(num_atoms, batchsize) for i in 1:num_repeats]
    @test length(timings) == num_repeats

    # Print average runtime. Disregard first run due to JIT compilation.
    avg_elapsed_time = sum(timings[2:num_repeats]) / num_atoms / num_repeats / batchsize
    println("Average Elapsed Time [μs/atom]:", avg_elapsed_time * 1e6)
end
