# Author: AJ Tyler
# Date: 24/11/21

# This script calls the readConfigs function from the .IO module
include("readConfigs.jl")
using .IO
rCutRange = LinRange(1,2,10)
readConfigs("GrayConfigs",(rCutRange[3]*3.6)^2)
#readConfigs("DianaConfigurations",(rCutRange[5]*3.1227)^2)
lc = snn = 4.07
fnn = lc/(2.0^0.5)
rCut = (fnn+snn)/2.0
#rCut = lc*sqrt(3/2)
#readConfigs("GeoffConfigs",rCut^2)