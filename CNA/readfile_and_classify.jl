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
sigma=2.782

M=32
tgrid = [5. *(16. /5.)^((i-1)/(M-1)) for i in 1:M]

"
Lennard-Jones dimer potential
The /100000 is to prevent the energy gradient being too large.
Otherwise the minimisation might push atoms apart at the first step.
"
function lj2(d)
    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
    if d>=0.9
        e=0
        dummy = 6
            for i in c
            e += i * d^(dummy)
            dummy += 2
        end
    else
        e=0.50993431+7.9720358*(0.9-d)
    end
    return 4e/100000
end

"Lennard-Jones potential for the whole configuration"
function elj_ne(x)
    
    E=0
    for i=1:N-1
        for j=i+1:N
            E+=lj2((x[3*j-2]-x[3*i-2])^2+(x[3*j-1]-x[3*i-1])^2+(x[3*j]-x[3*i])^2)
        end
    end
    return E
end

#println()
filedir = "$dir/configs"
savedir = "$filedir/classify"
for i in eachindex(tgrid)
    temp = tgrid[i]
    classifyfile = open("$savedir/$temp-classify.txt","w+")
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
            
            totalProfiles_i, atomicProfiles_i = CNA(configuration, N, cut, true, 1)
            #println(totalProfiles)
            #println(atomicProfiles)

            x = configuration*2

            classi = cluster_classify(atomicProfiles_i,N)
            
            
            od=OnceDifferentiable(elj_ne,x; autodiff=:forward);     #gradient
            result = Optim.minimizer(optimize(od, x, ConjugateGradient(),Optim.Options(g_tol=1e-8)))

            totalProfiles_f, atomicProfiles_f = CNA(result/2, N, cut, true, 1)
            classf = cluster_classify(atomicProfiles_f,N)

            #println(result/2)
            
            write(classifyfile, "$classi => $classf \n")
        end
    end
    close(classifyfile)
end


end