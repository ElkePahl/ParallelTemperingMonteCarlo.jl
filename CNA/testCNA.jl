# Author: AJ Tyler
# Date: 24/11/21

# This script calls the readConfigs function from the .IO module
include("readConfigs.jl")
using .IO
readConfigs("DianaConfigurations",(1.5*3.1227)^2)