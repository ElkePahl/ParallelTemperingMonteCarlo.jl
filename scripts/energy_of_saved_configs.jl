using ParallelTemperingMonteCarlo
using Random

N=216   #number of atoms
N_c=70  #number of configurations
AtoBohr=1.0
pressure=101325
ensemble = NPT(N,pressure*2.2937122783969076e-13/AtoBohr^3,true)

link="/Users/tiantianyu/Downloads/look-up_table.txt"
potlut=LookuptablePotential(link)

potential=potlut
energy_temp=[]
energy_list=[]

folder="/Users/tiantianyu/Downloads/216_B3/216_small_range_mixed/216_hcp"


function TempGrid(ti, tf,N)
    tgrid =[ti*(tf/ti)^((i-1)/(N-1)) for i in 1:N]
    return tgrid
end 

N_traj=24
T_min=45.0
T_max=50.0
T=TempGrid(T_min,T_max,N_traj)
println(T)

for t=1:N_c
    global energy_temp
    open(folder*"/1atm/216/configuration_$(T[t]).txt") do f

        for i=1:N
            println("Config Number.",i)
	        configuration = Array{Array}(undef,N)
            for j=1:N
                configuration[j]=zeros(3)
            end
	        totalProfiles = Array{Dict{String,Int}} # Initialise array to hold total CNA profiles
	        atomicProfiles = Any[] # Initialise array to hold atomic CNA profiles
	        line = readline(f) 
            line = readline(f) 
            box_length = parse(Float64,line)
            line = readline(f) 
            box_height = parse(Float64,line)
            println(box_length)
            println(box_height)


            for j=1:N              # Read the coordinates
                line = readline(f) 
                for (k,x) in enumerate((split(line, r" +"))[2:4]) # For each component of the coordinate
		    	    configuration[j][k] = parse(Float64,x) # Store the component
		        end
                #print("[")
                #print(configuration[j][1],",",configuraiton[j][2],",",configuraiton[j][3])
                #print("]")

            end
            

            bc=RhombicBC(box_length,box_height)

            for j=1:N
                print("[")
                print(configuration[j][1],",",configuration[j][2],",",configuration[j][3])
                println("],")
            end
            #for j=1:N
                #print("[")
                #print(configuration[j][1],",",configuraiton[j][2],",",configuraiton[j][3])
                #print("]")
            #end

            config = Config(configuration, bc)

            mc_state=MCState(1.0, 1.0, config,ensemble,potential)
            println(mc_state.en_tot)
            #println(mc_state.dist2_mat)
            push!(energy_temp,mc_state.en_tot)
            println()




            line = readline(f) 
        end

        
    end
    push!(energy_list,energy_temp)
    energy_temp=[]
end

open(folder*"/energy_list.txt","w") do io
    println(io,energy_list)
end