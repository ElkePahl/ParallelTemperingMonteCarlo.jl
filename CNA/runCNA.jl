# Author: AJ Tyler
# Date: 24/01/22

# This script calls the compare and classify function from the main module.

include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/main.jl")
using .main
compare("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/DianaConfigs")
classify("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/DianaConfigs")
