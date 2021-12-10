# Author: AJ Tyler
# Date: 24/11/21

# This script calls the readConfigs function from the .IO module
include("main.jl")
using .IO
rCutRange = round.(LinRange(4.0/3,5.0/3,10);digits = 3)
#main("GrayConfigs",rCutRange,3.6, similarityMeasure = "total")
main("DianaConfigs",rCutRange,3.1227)
lc = snn = 4.07
fnn = lc/(2.0^0.5)
rCut = (fnn+snn)/2.0
#rCut = lc*sqrt(3/2)
#readConfigs("GeoffConfigs",rCut^2)