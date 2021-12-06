# Author: AJ Tyler
# Date: 24/11/21
module IO # This module reads in .xyz files in a directory and computes the CNA profile of each configuration
include("CNA.jl") # This file contains the CommonNeighbourAnalysis module
include("similarity.jl")
using BenchmarkTools
using .CommonNeighbourAnalysis # This module contains the CNA function
using .Similarity
export readConfigs # This function reads in each .xyz file computes the CNA profile of each configuration


"""
This function reads in all the .xyz files in a directory and performs CNA on each of the configurations within each of the files.
For each input file, and output file will be created in the 'profiles' directory with the prefix profile_

Inputs:
configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files
rcutSquared: (Float64) The squared cut-off radius, which determines which atoms are 'bonded'.
"""
function readConfigs(configDir::String,rcutSquared::Float64)
	if (!isdir("$configDir\\output")) # If output directory doesn't exist
		mkdir("$configDir\\output") # Create the output directory
	end
	files = readdir(configDir) # Create list of all files/directories in the configuration directory
	for file in files # For each file
		if (last(split(file,'.')) == "xyz") # If is an .xyz file
			fp = open("$configDir\\$file") # Open the file
			energies = Vector{Float64}()
			profiles = Vector{Dict{String,Int}}()
			coordinatesNext = false # Does next line contain coordinates?
			headerNext = true # Does next line contain the header for a new configuration?
			skipLine = false # Does next line contain useless/no information?
			N, configuration, E, atomNum, configNum::Int = 0, Array{Float64}(undef,0,2), 0.0, 0, 0 # Initialise variables 
			while (true) # While not at end of file
				if (headerNext) # If next line is the start of a new configuration
					atomNum = 0 # Initialise number of atoms processed in current configuration
					line = readline(fp) # Read first line
                    configNum = parse(Int,line) # Get configuration number
                    line = readline(fp) # Read blank line
                    line = readline(fp) # Read next line
					N = parse(Int,line) # Get the number of atoms in the configuration
					configuration = Array{Float64}(undef, (N,3)) # Preallocate space for the configuration coordinates
                    line = readline(fp) # Read blank line
					line = readline(fp) # Read next line
					E = round(parse(Float64,line);digits = 6) # Get the number of atoms in the configuration
                    line = readline(fp) # Read blank line
                    headerNext = false # Have completed reading the header
					coordinatesNext = true # Cooridinate information is on the next line		
				elseif (coordinatesNext) # If next line is coordinates
					line = readline(fp) # Read the next line
					for (j,x) in enumerate((split(line, r" +"))[2:4]) # For each component of the coordinate
						configuration[atomNum+1,j] = parse(Float64,x) # Store the component
					end
					atomNum += 1 # Increment the number of atoms processed in the current configuration
					if (atomNum==N) # If have processed all the atoms in the current configuration
						coordinatesNext = false # Have finished reading in coordinates
						skipLine = true # Either eof or another configuration
					end
				elseif (skipLine) # If need to determine if at eof or there is another configuration
					line = readline(fp) # Read next line
					#@btime profile = CNA($configuration,$N,$rcutSquared) # Compute the CNA profile of the current configuration
					profile = CNA(configuration,N,rcutSquared) # Compute the CNA profile of the current configuration
					push!(profiles,profile)
					push!(energies,E)
					if (eof(fp)) # If at end of file
						break # end while loop
					elseif (isempty(line)) # If just a blank line -> another configuration follows
						line = readline(fp) # Read a blank line
						headerNext = true # New configuration header on next line
					end
				end
			end

			# Find most similar configurations for each
			numConfigs = length(profiles)
			similarConfigs = Vector{Vector{Int64}}(undef,numConfigs)
			maxSims = zeros(Float64,1,numConfigs)
			for configA in 1:numConfigs-1
				for configB in configA+1:numConfigs
					similarity = round(JacardIndex(profiles[configA],profiles[configB]); digits = 2)
					if (similarity > maxSims[configA])
						similarConfigs[configA] = Vector{Int64}([configB])
						maxSims[configA] = similarity
					elseif (similarity == maxSims[configA])
						push!(similarConfigs[configA],configB)
					end

					if (similarity > maxSims[configB])
						similarConfigs[configB] = Vector{Int64}([configA])
						maxSims[configB] = similarity
					elseif (similarity == maxSims[configB])
						push!(similarConfigs[configB],configA)
					end
				end
			end
			maxLength = 0
			if (numConfigs>1)
				for configs in similarConfigs
					if (length(configs) > maxLength)
						maxLength = length(configs)
					end
				end
			else
				similarConfigs = ["None"]
			end
			open("$configDir\\output\\profile_$file","a") do io # Open the output file corresponding to the input file
				println(io,"Config #    Energy        $(rpad("Most similar to", max(maxLength*4,15), " "))    Similarity Score     CNA Profile")
				for i in 1:length(profiles)
					println(io,"$(rpad(i,8," "))    $(rpad(energies[i], 10, " "))    $(rpad(similarConfigs[i],max(maxLength*4,15), " "))    $(rpad(maxSims[i],17, " "))    $(profiles[i])")
				end
			end
		end
	end
end

end # End of module