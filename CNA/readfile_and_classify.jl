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

N_c=1000   #Number of configurations
N=38    #Number of atoms in each configuration
cut=7.5
sigma=2.782

M=32
tgrid = [5. *(16. /5.)^((i-1)/(M-1)) for i in 1:M]

#ico = [5.294682606677612, 1.74009716698944, 0.9473289631755355, 3.791679084443151, -4.003030763557157, 1.2481962327125617, 3.81120559618501, -1.2068145890118294, -3.997172736460349, 1.8360724389860341, 4.332618355353545, -3.1331031921298242, 1.8044723389262654, -0.19175598302143088, 5.354081496111285, 0.5958439732022384, 4.959959644124769, 2.6462898025669164, -5.2946853312914826, -1.7400980998293896, -0.9473288849184892, -0.5958409506997832, -4.959961299680371, -2.6462896120220933, -1.836070528605722, -4.332620441173307, 3.133103314441934, -1.8044747891440631, 0.19175586145771703, -5.3540817785904125, -3.791676430898504, 4.003034841710014, -1.2481959566841079, -3.8112070445964297, 1.2068156228076719, 3.997172524612168, -9.63184319775816e-7, -3.161696747439155e-7, -1.7197856982327948e-7]



"
Lennard-Jones dimer potential
The /100000 is to prevent the energy gradient being too large.
Otherwise the minimisation might push atoms apart at the first step.
"
function lj2(d)
    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
    #if d>=0.9
        e=0
        dummy = 6
        x = sqrt(d)
        for i in c
            e += i *  (1/x)^(dummy)
            dummy += 2
        end
    #else
    #    e=0.50993431+7.9720358*(0.9-d)
    #end
    #return 4e/100000
    return e
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
for t in eachindex(tgrid)
    temp = tgrid[t]
    classifyfile = open("$savedir/$temp-classify.txt","w+")
    profilefile = open("$savedir/$temp-profiles_and_energies.txt", "w+")
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
            
            
            
            #println(totalProfiles)
            #println(atomicProfiles)
            #println(configuration)
            x = zeros(3N)
            for i = 1:N
                x[3i-2] = configuration[i,1]
                x[3i-1] = configuration[i,2]
                x[3i] = configuration[i,3]

            end
            #println(x)
            
            
            
            
            od=OnceDifferentiable(elj_ne,x; autodiff=:forward);     #gradient
            result = Optim.minimizer(optimize(od, x, ConjugateGradient(),Optim.Options(g_tol=1e-8)))
            config_f = Array{Float64}(undef, N, 3)
            for i = 1:N
                config_f[i,1] = result[3i-2]
                config_f[i,2] = result[3i-1]
                config_f[i,3] = result[3i]
            end
            #totalProfiles_i, atomicProfiles_i = CNA(configuration, N, cut, true, 1)
            #classi = cluster_classify(atomicProfiles_i,N)
            totalProfiles_f, atomicProfiles_f = CNA(config_f, N, cut, true, 1)
            
            en = elj_ne(result)
            write(profilefile, "$en \n")
            write(profilefile, "$totalProfiles_f \n")
            bonds = length(atomicProfiles_f)
            write(profilefile, "$bonds \n")
            for i in eachindex(atomicProfiles_f)
                prof = atomicProfiles_f[i]
                write(profilefile, "$prof \n")
            end
            classf = cluster_classify(atomicProfiles_f,N)

            #println(elj_ne(result))

            write(classifyfile, "$classf \n")
            
        end
    end
    close(classifyfile)
    close(profilefile)
end


end