module RuNNer

using DelimitedFiles,StaticArrays

using ..Configurations

#using 
export initialiseconfiguration,writeinit,writeconfig #,initialisetrajectories
#export edit_init,write_init

export updateconfiguration!
export getenergy,getenergy!,getRuNNerenergy
#----------------------------------------------------------------------#
#----------------------RuNNer output and read functions----------------#
#----------------------------------------------------------------------#
"""

    read_RuNNer(inputdir::String)
Opens the energy.out file produced by the RuNNer program. Input is a directory containing the energy.out file.
"""
function read_RuNNer(inputdir::String)

    readfile = open("$(inputdir)energy.out") #reads the energy out file from RuNNer
    contents = readdlm(readfile,skipstart=1) #ignores the header
    return contents
end
"""
    findRuNNerenergy(inputdir::String,NTraj)
returns the energies given by RuNNer, accepts a directory and a number of trajectories and returns a vector of energy.
"""
function findRuNNerenergy(inputdir::String,NTraj)

    contents= read_RuNNer(inputdir)

    energyvector = Vector{Float64}(undef,NTraj)
    ###------This is a bugfix that ultimately needs a more elegant solution---------###
    if length(contents[:,4]) < NTraj
        #RuNNer couldn't find one or more energy without saying which
        println("error in RuNNer")
        #So we set the energy too high to accept
        energyvector[:] = 1000*ones(NTraj)
    else
        for i in 1:NTraj
            energyvector[i] = contents[i,4] # the fourth column is the output energy of the NN
        end
    end
    
    return energyvector
end
function findRuNNerenergy(inputdir,NTraj,input_idx)

    readfile = open("$(inputdir)energy$(input_idx).out") #reads the energy out file from RuNNer
    contents = readdlm(readfile,skipstart=1)

    energyvector = Vector{Float64}(undef,NTraj)
    ###------This is a bugfix that ultimately needs a more elegant solution---------###
    if length(contents[:,4]) < NTraj
        #RuNNer couldn't find one or more energy without saying which
        println("error in RuNNer")
        #So we set the energy too high to accept
        energyvector[:] = 1000*ones(NTraj)
    else
        for i in 1:NTraj
            energyvector[i] = contents[i,4] # the fourth column is the output energy of the NN
        end
    end
    
    return energyvector
end

"""
    getRuNNerenergy(dir::String,NTraj)]
Function to run RuNNer and read the output. This represents the total output function. Point to a directory dir containing the RuNNer.serial.x and input files, and specify the number of trajectories NTraj. Output is an NTraj vector of energies
"""
function getRuNNerenergy(dir::String,NTraj; input_idx = 0)
    cd(dir)
    if input_idx == 0
        run(`./RuNNer.x`);
    
        E = findRuNNerenergy(dir,NTraj)
    else
        run(`./RuNNer$input_idx.x $input_idx`)

        E = findRuNNerenergy(dir,NTraj,input_idx)
    end

    return E
end
#--------------------------------------------------------------#
#-------------------------RuNNer Input-------------------------#
#--------------------------------------------------------------#
"""
    writefile(dir::String, config::Config, atomtype::String)

    writefile(dir::String, config::Config, atomtype::Vector)
    writefile(dir::String, config::Config, atomtype::String, ix, pos::SVector)

Function to write the input file for RuNNer. This accepts the directory containing RuNNer.serial.x and the requisite trained Neural Network, the configuration in question and a string or vector named atomtype containing the type of atom, either one name for monoatomic or many for pluriatomic. 

    The third method is used for measuring the difference in energy for a displaced configuration. It accepts an index and a position S_vector and prints the configuration itself and then the updated configuration.
"""
function writefile(dir::String, config::Config, atomtype::String)
    file = open("$(dir)input.data","w+")
    write(file, "begin \n")
    for atom in config.pos
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end")
    close(file)
end
function writefile(dir::String, config::Config, atomtype::String,input_index)
    file = open("$(dir)input$(input_index).data","w+")
    write(file, "begin \n")
    for atom in config.pos
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end")
    close(file)
end
function writefile(dir::String,config::Config, atomtype::Vector)
    i = 0
    file = open("$(dir)input.data","w+")
    write(file, "begin \n")
    for atom in config.pos
        i += 1
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $(atomtype[i])  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end")
    close(file)
end
function writefile(dir::String,config::Config,atomtype::String,ix,pos::SVector)
    testconfig = copy(config.pos)
    testconfig[ix] = pos

    file = open("$(dir)input.data","w+")
    #write config one
    write(file, "begin \n")
    for atom in config.pos
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end \n")
    #write config 2
    write(file, "begin \n")
    for atom in testconfig
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end")
    close(file)
end


"""
    writeinit(dir::String)
for writing a series of trajectories in a single runner input.data file. Output is an IOStream for use in writing configs.

"""
function writeinit(dir::String;input_idx=0)
    if input_idx == 0
        inputfile = open("$(dir)input.data","w+")
    else
        inputfile = open("$(dir)input$(input_idx).data","w+")
    end

    return inputfile
end
"""
    writeconfig(file::IOStream, config::Config, atomtype)

    writeconfig(file::IOStream, config::Config,index, test_pos, atomtype)
    

two methods for a function to write out a configuration into an open IOStream called file. Both inputs require a configuration and string labelled atomtype which is written in the standard RuNNer format. 
    The second method also requires an index (integer) and SVector containing a new atomic position. It writes out config, but exchanges atom [index] with this new atomic position [test_pos]
"""
function writeconfig(file::IOStream,config::Config,atomtype)
    write(file,"begin \n")
    for atom in config.pos
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end \n")

end
function writeconfig(file::IOStream,config::Config,index,test_pos, atomtype)
    write(file,"begin \n")
    i=0
    for atom in config.pos
        i+=1
        if i == index
            write(file, "atom  $(test_pos[1])  $(test_pos[2])  $(test_pos[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
        else
            write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
        end
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end \n")
end

# function edit_init(dir::String)
#     editfile = open("$(dir)edit.sh", "w+")
#     write(editfile, "#! /usr/bin/bash \n")
#     # for j = 1:3
#     #     write(editfile, "sed -i \"$(line_number)s/$(vec_old[j])/$(vec_new[j])\" ")
#     # end
#     close(editfile)
#     editfile = open("$(dir)edit.sh", "a")

#     return editfile
# end
# function writeedit(editfile::IOStream,line_number,vec_old,vec_new)
#     for j = 1:3
#         write(editfile, "sed -i \"$(line_number)s/$(vec_old[j])/$(vec_new[j])/\" input.data \n")
#     end

# end
#--------------------------------------------------------------#
#------------------------RuNNer Complete-----------------------#
#--------------------------------------------------------------#
"""
    getenergy(dir,config::Config,atomtype)
    getenergy(dir,config::Config,atomtype,ix,pos::SVector)

A function to find the energy of a configuration using RuNNer from start to finish. It writes either one configuration and returns the total energy, or it writes a configuration and a slightly perturbed version. This second method returns a vector with the UN-perturbed energy first, and the perturbed energy second.
"""
function getenergy(dir,config::Config,atomtype)

    writefile(dir,config,atomtype)
    
    E = getRuNNerenergy(dir,1)[1];

    return E
end
function getenergy(dir,config::Config,atomtype,input_index)

    writefile(dir,config,atomtype,input_index)
    
    E = getRuNNerenergy(dir,1,input_idx=input_index)[1];

    return E
end
function getenergy(dir,config::Config,atomtype,ix,pos::SVector)

    writefile(dir,config,atomtype,ix,pos)
    
    E = getRuNNerenergy(dir,2);

    return E
end

# function getenergy(dir,mc_states,atomtype,mc_params)
# # a parallelised version of getenergy which accepts a series of states rather than one perturbed state
#     file = writeinit(dir)
#     for state in mc_states
#         writeconfig(file, state.config,atomtype)
#     end
#     close(file)

#     energyvector = getRuNNerenergy(dir,mc_params.n_traj)

#     return energyvector
# end

end