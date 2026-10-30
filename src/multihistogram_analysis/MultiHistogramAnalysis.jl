module MultiHistogramAnalysis

using ..InputParams: TempGrid
using ProgressMeter: ProgressMeter, ProgressThresh, Progress
using LogExpFunctions: logsumexp
using DataFrames: DataFrame, metadata
using LinearAlgebra: dot
using Arrow
using Statistics: std

const kB = 3.16681196e-6 # in atomic units

include("multihistogram.jl")
include("blocked_multihistogram.jl")

end
