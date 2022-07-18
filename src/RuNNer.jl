module RuNNer

using DelimitedFiles,DataFrames,StaticArrays

using ..Configurations
#using 
export initialiseconfiguration,initialisetrajectories
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
    contents= runnerreader(inputdir)
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
    run(`./RuNNer.serial.x`);
    E = findRuNNerenergy(dir,NTraj)

    return E
end
#----------------------------------------------------------------------#
#----------------------RuNNer input and write functions----------------#
#----------------------------------------------------------------------#
"""
    initialiseconfiguration(Natoms::Int64,atomtype::String)
    initialiseconfiguration(Natoms::Int64,atomtype::String,configuration)
    initialiseconfiguration(Natoms::Int64,atomtype::Vector,configuration)

Three variations of the initialise function, this creates a dataframe designed for easy writing of the RuNNer input file. 
    -The first method accepts one atom type and initialises a blank dataframe of Natom atoms of this type at [0,0,0].
    -The second method accepts one atom type and an Natom X 3 array with the positions of the atoms.
    -The third method only differs from the first in that it accepts a _vector_ of atom types, this is used for molecular systems where the atom types may vary.

"""
function initialiseconfiguration(Natoms::Int64,atomtype::String)
    df = DataFrame(header = String[], x=Float64[], y=Float64[], z=Float64[], type=String[] , aenergy=Float64[], acharge=Float64[] , fx=Float64[], fy=Float64[], fz=Float64[])

    for i in 1:Natoms
        push!(df, ["atom", 0., 0., 0., atomtype, 0.000, 0.000, 0., 0., 0.]) 
        #this creates blank rows to populate the dataframe later
    end
    return df
end
function initialiseconfiguration(Natoms::Int64,atomtype::String,configuration)
    df = DataFrame(header = String[], x=Float64[], y=Float64[], z=Float64[], type=String[] , aenergy=Float64[], acharge=Float64[] , fx=Float64[], fy=Float64[], fz=Float64[])

    for i in 1:Natoms
        push!(df, ["atom", configuration[i,1], configuration[i,2], configuration[i,3], atomtype, 0.000, 0.000, 0., 0., 0.]) 
        #this creates blank rows to populate the dataframe later
    end
    return df
end
function initialiseconfiguration(Natoms::Int64,atomtype::Vector,configuration)
    df = DataFrame(header = String[], x=Float64[], y=Float64[], z=Float64[], type=String[] , aenergy=Float64[], acharge=Float64[] , fx=Float64[], fy=Float64[], fz=Float64[])

    for i in 1:Natoms
        push!(df, ["atom", configuration[i,1], configuration[i,2], configuration[i,3], atomtype[i], 0.000, 0.000, 0., 0., 0.]) 
        #this creates blank rows to populate the dataframe later
    end
    return df
end
"""
    initialisetrajectories(Natoms,Ntraj,atomtype,configurations)
Function to initialise a vector of dataframes ready for output via RuNNer. This leverages the initialiseconfiguration function NTraj times and pushes the resulting frames to a vector. 
    
    dev note: At the moment this only works for monoatomic systems. In future I will add a field to the Config struct to include the atom type. In this way we can make systems where the atomtype field is linked to the correct line.
"""
function initialisetrajectories(Natoms, atomtype, configurations)
    trajectories = []
    for config in configurations
        frame = initialiseconfiguration(Natoms,atomtype,config)
        push!(trajectories,frame)
    end

    return trajectories
end
"""
    updateconfiguration!(df, ix::Int64, )
    updateconfiguration!(df, config, Natoms)
A function to update the positions of atoms in a dataframe. Supply the dataframe to update and the configuration.
    -Method one accepts an index ix of the moved atom as its final field and adjusts the correct row of the dataframe. Natoms remains a required field to differentiate the two methods. A better system may be needed in the future?
    -Method two accepts the total number of atoms and updates all rows accordingly

"""
function updateconfiguration!(df,Natoms, config, ix::Int64)
    df.x[ix],df.y[ix],df.z[ix] = config[ix,1],config[ix,2],config[ix,3]
return df
end
function updateconfiguration!(df, Natoms::Int64, config)

for ix in 1:Natoms

    updateconfiguration!(df, Natoms, config, ix)
    
end    

return df

end
"""
    writefile(dir::String, df)
    writefile(dir::String,NTraj::Int64,trajframes)
File designed to write the input for RuNNer using a dataframe. 
    - Method one writes a single configuration df in a file input.data in directory dir
    - Method two writes NTraj configurations {df ∈ trajframes} in $dir/input.data
"""
function writefile(dir::String, df)
    file = open("$(dir)input.data","w+")
    write(file, "begin \n")
    writedlm(file,eachrow(df))
    write(file, "energy  0.000 \n")
    write(file, "charge  0.000 \n")
    write(file,"end")
    close(file)

end
function writefile(dir::String, NTraj::Int64, trajframes)
    file = open("$(dir)input.data","w+")
    

    for frame in trajframes
        
        write(file, "begin \n")
        writedlm(file,eachrow(frame))
        write(file, "energy 0.000 \n")
        write(file, "charge  0.000 \n")
        write(file,"end \n")

    end
    close(file)

end

#----------------------------------------------------------------------#
#----------------------------RuNNer Complete---------------------------#
#----------------------------------------------------------------------#
"""
    getenergy!(dir,dataframes,NTraj,Natoms)
    getenergy!(dir,dataframes,Natoms)
Two methods for the end-use get energy function. 
    -Method one accepts a directory containing RuNNer.serial.x, writes $dir/input.data containing NTraj configurations, runs the program and reads the energy.
    -Method two is much like the above except it only writes and runs a single trajectory.
"""
function getenergy(dir,dataframes,NTraj,Natoms)

    writefile(dir,NTraj,dataframes)

    E = getRuNNerenergy(dir,NTraj)

    return E
end
function getenergy(dir,dataframes,Natoms)

    writefile(dir,dataframes)
    
    E = getRuNNerenergy(dir,1)

    return E
end
"""
    getenergy!(dir,dataframes,energyvector,configs,NTraj,Natoms)
A function designed to accept a _new_ set of configurations configs, update the corresponding dataframes containing the old configurations, print them in $dir/input.data, run RuNNer and read the energy, storing the dataframe and the energy in energyvector
"""
function getenergy!(dir,dataframes,energyvector,configs,NTraj,Natoms)
    for i in 1:NTraj
        updateconfiguration!(dataframes[i],configs[i],Natoms)
    end
    #having updated the configurations we write the input file
    writefile(dir,NTraj,dataframes)

    energyvector = getRuNNerenergy(dir,NTraj)

    return dataframes, energyvector
end