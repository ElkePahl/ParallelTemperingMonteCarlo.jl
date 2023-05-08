module CC
# This module can read a file of cluster configurations and classify their sturctures.

include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/CNA.jl")
include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/comparison.jl")
include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/cluster classification.jl")


using .CommonNeighbourAnalysis
using .Comparison
using .Cluster_Classification
using Optim
using LinearAlgebra
using Graphs

N_c=10   #Number of configurations
N=13    #Number of atoms in each configuration
cut=1.3549

#println()
open("/Users/tiantianyu/Downloads/13_B_configuration.txt") do f
    for i=1:N_c
	    configuration = Array{Float64}(undef,N,3) # Initialise 3D array for the atom coordinates for each configuration in the file.
	    totalProfiles = Array{Dict{String,Int}} # Initialise array to hold total CNA profiles
	    atomicProfiles = Any[] # Initialise array to hold atomic CNA profiles
	    for j=1:2              # Read the first two lines
            line = readline(f) 
        end
        for j=1:N              # Read the coordinates
            line = readline(f) 
            for (k,x) in enumerate((split(line, r" +"))[2:4]) # For each component of the coordinate
		    	configuration[j,k] = parse(Float64,x) # Store the component
		    end
        end
        #println(configuration)
        totalProfiles, atomicProfiles = CNA(configuration/2.782, N, cut, true, 1)
        #println(totalProfiles)
        #println(atomicProfiles)
        open("/Users/tiantianyu/Downloads/13_10000_classify.txt","a") do io
            println(io,cluster_classify(atomicProfiles,N))
        end
        #println()
    end
end

end