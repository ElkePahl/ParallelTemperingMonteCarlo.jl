module CC
# This module can read a file of cluster configurations and classify their sturctures.
dir = pwd()
include("CNA.jl")
include("comparison.jl")
include("cluster classification.jl")


using .CommonNeighbourAnalysis
using .Comparison
using .Cluster_Classification
using Optim
using LinearAlgebra
using Graphs

N_c=10   #Number of configurations
N=13    #Number of atoms in each configuration
cut=7.0

M=32
tgrid = [5. *(16. /5.)^((i-1)/(M-1)) for i in 1:M]

#println()
filedir = "$dir/configs"
savedir = "$filedir/classify"
for i in eachindex(tgrid)
    temp = tgrid[i]
    open("$filedir/$temp.xyz") do f
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
            totalProfiles, atomicProfiles = CNA(configuration, N, cut, true, 1)
            #println(totalProfiles)
            #println(atomicProfiles)
            open("$savedir/$temp-classify.txt","a") do io
                println(io,cluster_classify(atomicProfiles,N))
            end
        
            #println()
        end
    end
end

end