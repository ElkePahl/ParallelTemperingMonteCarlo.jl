# Author: AJ Tyler
# Date: 24/01/22

# This script calls the compare and classify function from the main module.

include("main.jl")
using .main
compare("DianaConfigs")
classify("DianaConfigs")
