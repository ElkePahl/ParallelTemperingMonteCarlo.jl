module RuNNer

using DelimitedFiles,StaticArrays

using ..Configurations

#using 
export initialiseconfiguration,writeinit,writeconfig #,initialisetrajectories
export updateconfiguration!
export getenergy,getenergy!
#----------------------------------------------------------------------#
#----------------------RuNNer output and read functions----------------#
#----------------------------------------------------------------------#
"""
    RunnerReader(inputdir::String)
Opens the energy.out file produced by the RuNNer program. Input is a directory containing the energy.out file.
"""
function RunnerReader(inputdir::String)
    file = open("$(inputdir)energy.out") #reads the energy out file from RuNNer
    contents = readdlm(file,skipstart=1) #ignores the header
    return contents
end
"""
    findRuNNerenergy(inputdir::String,NTraj)
returns the energies given by RuNNer, accepts a directory and a number of trajectories and returns a vector of energy.
"""
function findRuNNerenergy(inputdir::String,NTraj)
    contents= RunnerReader(inputdir)
    energyvector = Vector{Float64}(undef,NTraj)

    for i in 1:NTraj
        energyvector[i] = contents[i,4] # the fourth column is the output energy of the NN
    end
    
    return energyvector
end
"""
    getRuNNerenergy(dir::String,NTraj)]
Function to run RuNNer and read the output. This represents the total output function. Point to a directory dir containing the RuNNer.serial.x and input files, and specify the number of trajectories NTraj. Output is an NTraj vector of energies
"""
function getRuNNerenergy(dir::String,NTraj)
    cd(dir)
    run(`./RuNNer.x`);
    E = findRuNNerenergy(dir,NTraj)

    return E
end
#--------------------------------------------------------------#
#-------------------------RuNNer Input-------------------------#
#--------------------------------------------------------------#
"""
    writefile(dir::String, config::Config, atomtype::String)
    writefile(dir::String, config::Config, atomtype::Vector)
    writefile(dir::String,config::Config,atomtype::String,ix,pos::SVector)
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



function writeinit(dir::String)
    file = open("$(dir)input.data","w+")

    return file
end
function writeconfig(file::IOStream,config::Config,atomtype)
    write(file,"begin \n")
    for atom in config.pos
        write(file, "atom  $(atom[1])  $(atom[2])  $(atom[3])  $atomtype  0.0  0.0  0.0  0.0  0.0 \n")
    end
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end \n")

end
function writeconfig(file::IOStream,config::Config,index,test_pos)
    write(file,"begin \n")
    for atom in config.pos
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