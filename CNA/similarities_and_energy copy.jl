module Similarities

dir = pwd()
include("CNA.jl")
include("$dir/CNA/comparison.jl")
include("cluster classification.jl")


using .CommonNeighbourAnalysis
using .Comparison
using .Cluster_Classification
using Optim
using LinearAlgebra
using Plots



N_c=1000   #Number of configurations
N=38    #Number of atoms in each configuration
cut=7.5
sigma=2.782

M=32
tgrid = [5. *(16. /5.)^((i-1)/(M-1)) for i in 1:M]


function lj2(d)
    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
        e=0
        dummy = 6
        x = sqrt(d)
        for i in c
            e += i *  (1/x)^(dummy)
            dummy += 2
        end

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


config_1=Array{Float64}(undef,N,3)
#category=Array{String}(undef,N_c)

ico = [2.825384495892464, 0.928562467914040, 0.505520149314310, 2.023342172678102, -2.136126268595355, 0.666071287554958, 2.033761811732818, -0.643989413759464, -2.133000349161121, 0.979777205108572, 2.312002562803556, -1.671909307631893, 0.962914279874254, -0.102326586625353, 2.857083360096907, 0.317957619634043, 2.646768968413408, 1.412132053672896, -2.825388342924982, -0.928563755928189, -0.505520471387560, -0.317955944853142, -2.646769840660271, -1.412131825293682, -0.979776174195320, -2.312003751825495, 1.671909138648006, -0.962916072888105, 0.102326392265998, -2.857083272537599, -2.023340541398004, 2.136128558801072, -0.666071089291685, -2.033762834001679, 0.643989905095452, 2.132999911364582, 0.000002325340981, 0.000000762100600, 0.000000414930733]
ico = ico*sigma
#ico = [2.825384495892464, 0.928562467914040, 0.505520149314310, 2.023342172678102,	-2.136126268595355, 0.666071287554958, 2.033761811732818,	-0.643989413759464, -2.133000349161121, 0.979777205108572,	2.312002562803556, -1.671909307631893, 0.962914279874254,	-0.102326586625353, 2.857083360096907, 0.317957619634043,	2.646768968413408, 1.412132053672896, -2.825388342924982, -0.928563755928189, -0.505520471387560, -0.317955944853142, -2.646769840660271, -1.412131825293682, -0.979776174195320, -2.312003751825495, 1.671909138648006, -0.962916072888105, 0.102326392265998,	-2.857083272537599, -2.023340541398004, 2.136128558801072,	-0.666071089291685, -2.033762834001679, 0.643989905095452, 2.132999911364582, 0.000002325340981,	0.000000762100600, 0.000000414930733]
#ico = ico*1.8897259886

for i=1:N
	for j=1:3
		config_1[i,j]=ico[3i-(3-j)]
	end
end

totalProfile_global = Dict{String,Int}()         #CNA profile of global minimum
atomicProfile_global = [Dict{String,Int}() for i in 1:N]

totalProfile_global, atomicProfile_global = CNA(config_1, N, cut, true, 1)
#println(cluster_classify(atomicProfile_global,N))
# = CNA(config_1, N, 1.3549, false, 1.)[2]

println("CNA global mininum")
println(totalProfile_global)
println(atomicProfile_global)

filedir = "$dir/configs"
savedir = "$filedir/classify"
savefile = open("$savedir/ne38.txt", "w+")
plt = plot(legend=:none, title="Bird Poo Plot of Minimised Configurations from Basin Hopping (Purple) and Monte Carlo (Green)")
plot!(xlabel="Similarity (a.u)")
plot!(ylabel="Configurational Energy (a.u.)")

for temp in tgrid
    energies=Array{Float64}(undef,N_c)
    similarity=Array{Float64}(undef,N_c)
    open("$savedir/$temp-profiles_and_energies.txt") do f
        for i=1:N_c
            line = readline(f)
            energies[i] = parse(Float64, line)
            totalProfiles = Dict{String,Int}()
            line = readline(f)
            len = length(line)
            dummy = 7
            while dummy <= len
                string = line[dummy:dummy+6]
                try
                    int = parse(Int64, line[dummy+12:dummy+13])
                    totalProfiles[string] = int
                    dummy += 17
                catch
                    int = parse(Int64, line[dummy+12])
                    totalProfiles[string] = int 
                    dummy += 16
                end

            end
            bonds = parse(Int64, readline(f))
            atomicProfiles = []
            for i = 1:bonds
                line = readline(f)
                len = length(line)
                #println(len)
                str = line[len-9:len-3]
                arr = []
                
                if len == 20
                    push!(arr, parse(Int64, line[3]))
                    push!(arr, parse(Int64, line[6]))
                elseif len == 21
                    push!(arr, parse(Int64, line[3]))
                    push!(arr, parse(Int64, line[6:7]))
                elseif len == 22
                    push!(arr, parse(Int64, line[3:4]))
                    push!(arr, parse(Int64, line[7:8]))
                end
                prof = (arr, str)
                push!(atomicProfiles, prof)
            end
            similarity[i]= similarityScore_one(1, 2, [totalProfile_global,totalProfiles],[atomicProfile_global,atomicProfiles],"total",N)
            
        end
        write(savefile, "$temp \n")
        write(savefile, "$similarity \n")
        write(savefile, "$energies \n")

        scatter!(similarity, energies; mc=:yellowgreen, markersize=11)
    end

end

close(savefile)

display(plt)
end