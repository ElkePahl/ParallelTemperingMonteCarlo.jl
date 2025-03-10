module ForwardPass

export NeuralNetworkPotential, forward_pass

"""
    lib_path()
Returns path to binary library. E.g. this is where to find "librunnerjulia.so" for
computing a runner forward pass. Redefine this function if locally compiled library is
needed.


### Example
```julia
ParallelTemperingMonteCarlo.MachineLearningPotential.ForwardPass.lib_path() = "my/path/"
```
"""
lib_path() = joinpath(@__DIR__, "lib") # path to folder where to find "librunnerjulia.so"

"""
    NeuralNetworkPotential
The basic struct containing the parameters of the neural network itself. `n_layers` and `n_params` define the length of the vectors, these are required by the Fortran program. `num_nodes` is a vector containing the number of nodes per layer, also required to appropriately assign the parameters to the correct node. `activation_functions` should usually be `[1 2 2 1]` meaning "linear, tanh, tanh, linear" last is the vector of parameters assigned to each connexion.
"""
struct NeuralNetworkPotential
    n_layers::Int32
    n_params::Int32
    num_nodes::Vector{Int32}
    activation_functions::Vector{Int32}
    parameters::Vector{Float64}
end

"""
    NeuralNetworkPotential(num_nodes::Vector,activation_functions::Vector, parameters)
Unpacks the `num_nodes` vector and `parameters` and assigns their lengths to the missing struct parameters.
"""
function NeuralNetworkPotential(num_nodes::Vector,activation_functions::Vector, parameters)
    return NeuralNetworkPotential(length(num_nodes),length(parameters),num_nodes,activation_functions,parameters)
end
"""
    forward_pass( input::AbstractArray, batchsize, num_layers, num_nodes, activation_functions, num_parameters, parameters)
    forward_pass(input::AbstractArray,batchsize,nnparams::NeuralNetworkPotential)
    forward_pass(eatom,input::AbstractArray,batchsize,nnparams::NeuralNetworkPotential; directory = pwd())
    forward_pass( eatom,input::AbstractArray, batchsize, num_layers, num_nodes, activation_functions, num_parameters, parameters,dir)


Calls the RuNNer forward pass module written by A. Knoll located in `directory`. This self-defines the `eatoms` output, a vector of the atomic energies. `batchsize` is based on the number of atoms whose energies we want to determine. The remaining inputs are contained in `nnparams.` Details of this struct can be found in the definition of the [`NeuralNetworkPotential`](@ref) struct.
The last two definitions are identical except eatoms is an input rather than a vector determined during the calculation. This can save memory in the long run.
"""
function forward_pass( input::AbstractArray, batchsize, num_layers, num_nodes, activation_functions, num_parameters, parameters)
    eatom = Vector{Float64}(undef,batchsize)
    ccall( (:forward, joinpath(lib_path(), "librunnerjulia.so")),
    Float64,  (Ref{Float64},Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32},
    Ref{Int32}, Ref{Float64}, Ref{Int32}, Ref{Float64}),
    input, num_nodes[1], num_layers, num_nodes, batchsize,
    num_parameters, parameters, activation_functions, eatom
    )
    return eatom
end
function forward_pass(input::AbstractArray,batchsize,nnparams::NeuralNetworkPotential)
    return forward_pass(input, batchsize, nnparams.n_layers, nnparams.num_nodes, nnparams.activation_functions,nnparams.n_params, nnparams.parameters)
end
function forward_pass(eatom,input::AbstractArray,batchsize,nnparams::NeuralNetworkPotential)
    return forward_pass(eatom,input, batchsize, nnparams.n_layers, nnparams.num_nodes, nnparams.activation_functions,nnparams.n_params, nnparams.parameters)
end
function forward_pass( eatom,input::AbstractArray, batchsize, num_layers, num_nodes, activation_functions, num_parameters, parameters)
    ccall( (:forward, "./librunnerjulia.so"),
    Float64,  (Ref{Float64},Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32},
    Ref{Int32}, Ref{Float64}, Ref{Int32}, Ref{Float64}),
    input, num_nodes[1], num_layers, num_nodes, batchsize,
    num_parameters, parameters, activation_functions, eatom
    )
    return eatom
end

end
