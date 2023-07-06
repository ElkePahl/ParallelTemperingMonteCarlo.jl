"""
   MachineLearningPotential
Developed for use with the ParallelTemperingMonteCarlo package (https://github.com/ElkePahl/ParallelTemperingMonteCarlo.jl.git) in order to calculate the energies of atomic structures using the RuNNer programme. Currently this module requires librunnerjulia.so -- a FORTRAN-based program to actually calculate the atomic energies. 
"""

module MachineLearningPotential

using Reexport

include("Cutoff.jl")
include("SymmFunc.jl")
include("DeltaMatrix.jl")
include("ForwardPass.jl")

# Write your package code here.
@reexport using .Cutoff
@reexport using .SymmetryFunctions
@reexport using .DeltaMatrix
@reexport using .ForwardPass


end
