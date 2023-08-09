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
tgrid = [5. *(25. /5.)^((i-1)/(M-1)) for i in 1:M]
templist = [tgrid[1], tgrid[10], tgrid[11], tgrid[12], tgrid[24], tgrid[25], tgrid[26], tgrid[32]]


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

#ico = [2.825384495892464, 0.928562467914040, 0.505520149314310, 2.023342172678102, -2.136126268595355, 0.666071287554958, 2.033761811732818, -0.643989413759464, -2.133000349161121, 0.979777205108572, 2.312002562803556, -1.671909307631893, 0.962914279874254, -0.102326586625353, 2.857083360096907, 0.317957619634043, 2.646768968413408, 1.412132053672896, -2.825388342924982, -0.928563755928189, -0.505520471387560, -0.317955944853142, -2.646769840660271, -1.412131825293682, -0.979776174195320, -2.312003751825495, 1.671909138648006, -0.962916072888105, 0.102326392265998, -2.857083272537599, -2.023340541398004, 2.136128558801072, -0.666071089291685, -2.033762834001679, 0.643989905095452, 2.132999911364582, 0.000002325340981, 0.000000762100600, 0.000000414930733]
#ico = ico*sigma
#ico = [2.825384495892464, 0.928562467914040, 0.505520149314310, 2.023342172678102,	-2.136126268595355, 0.666071287554958, 2.033761811732818,	-0.643989413759464, -2.133000349161121, 0.979777205108572,	2.312002562803556, -1.671909307631893, 0.962914279874254,	-0.102326586625353, 2.857083360096907, 0.317957619634043,	2.646768968413408, 1.412132053672896, -2.825388342924982, -0.928563755928189, -0.505520471387560, -0.317955944853142, -2.646769840660271, -1.412131825293682, -0.979776174195320, -2.312003751825495, 1.671909138648006, -0.962916072888105, 0.102326392265998,	-2.857083272537599, -2.023340541398004, 2.136128558801072,	-0.666071089291685, -2.033762834001679, 0.643989905095452, 2.132999911364582, 0.000002325340981,	0.000000762100600, 0.000000414930733]
#ico = ico*1.8897259886

ico = [0.1947679907, 0.3306365642, 1.7069272101, 1.1592174250, -1.1514615100, -0.6254746298, 1.4851406793, -0.0676273830, 0.9223060046, -0.1498046416, 1.4425168343, -0.9785553065, 1.4277261305, 0.3530265376, -0.9475378022, -0.6881246261, -1.5737014419, -0.3328844168, -1.4277352637, -0.3530034531, 0.9475270683, 0.6881257085, 1.5736904826, 0.3329032458, -1.1592204530, 1.1514535263, 0.6254777879, 0.1498035273, -1.4424985165, 0.9785685322, -1.4851196066, 0.0676193562, -0.9223231092, -0.7057028384, 0.6207073550, -1.4756523155, -0.8745359533, 0.4648140463, 1.4422103492, -0.9742077067, -0.8837261792, -1.1536019836, -0.1947765396, -0.3306358487, -1.7069179299, 0.3759933035, -1.7072373106, -0.0694439840, -1.7124296000, 0.3336352522, 0.1307959669, 0.9143159284, 1.3089975397, -0.7151210582, -0.3759920260, 1.7072300336, 0.0694634263, 1.7124281219, -0.3336312342, -0.1308207313, -0.9143187026, -1.3089785474, 0.7151290509, 0.9742085109, 0.8837023041, 1.1536069633, 0.7057104439, -0.6206907639, 1.4756502961, 0.8745319670, -0.4648127187, -1.4422106957, -1.1954804901, -0.6171923123, -0.1021449363, 0.0917363053, -1.0144887859, -0.8848410405, 0.9276243144, -0.8836123311, 0.4234140820, 1.1954744473, 0.6171883800, 0.1021399054, -0.9276176774, 0.8836123556, -0.4234173533, -0.3595942315, -0.4863167551, 1.2061133825, 0.3595891589, 0.4863295901, -1.2061152849, -0.0917352078, 1.0144694592, 0.8848400639, 0.6410702480, -0.1978633363, -0.3898095439, -0.4162942817, -0.0651798741, -0.6515502084, 0.1334019604, 0.7474406294, -0.1600033264, -0.6410732823, 0.1978593218, 0.3898012337, 0.4162968444, 0.0651733322, 0.6515490914, -0.1333998872, -0.7474445984, 0.1600019961]

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

for temp in templist
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