# Author: AJ Tyler
# Date: 24/01/22
module IO # This module contains functions that read in and write out files.
include("CNA.jl") # This file contains the CommonNeighbourAnalysis module.
include("atomicShells.jl") # This file contains the AtomicShells module.
using .CommonNeighbourAnalysis # This module computes the total and atomic CNA profile of a configuration.
using .AtomicShells # This module computes the atomic shell number of a cluster.
using Graphs # This module contains graph theory functions which allow for a graphical representation of configurations.
# These functions either read in or write out data.
export processFileName, processFile, writeClassification, writeComparison, writeProfiles, writeUnique

"""
This function extracts the information embedded in the .xyz filenames.

Inputs:
configDir: (String) The path to the directory containing the .xyz files.
fileName: (String) A filename.

Outputs:
isXYZ: (Boolean) Whether or not the file is a .xyz file
prefix: (String) The unique prefix of the fileName.
fp: (IOStream) The file pointer for the file.
L: (Int64) The number of configurations in the file.
N: (Int64) The number of atoms in each configuration.
B :(Bool) Whether configurations were found under a magnetic field of 0.3 au or not.
"""
function processFileName(configDir,fileName)
	parts = split(fileName,".xy") # Process filename to be able to tell if .xyz file
	if (last(parts) == "z") # If is an .xyz file
		fp = open("$configDir\\$fileName") # Open the file
		parts = split(fileName,'_') # Process filename
		prefix = parts[1] # Get filename prefix
		N,L = (parse.(Int64,parts[2:3]))[:] # Number of configurations in the file (part of file name)
		B = false
		if parts[4] == "BField"
			B = true
		end

		return true,prefix,fp,L,N,B
	else
		return false,nothing,nothing,nothing,nothing,nothing
	end
end

"""
This function reads in all the .xyz files in a directory and performs CNA on each of the configurations for a range of rCut values.
The atomic shell numbers for configurations that are to be classified.

Inputs:
fp: (String) The file path to the .xyz file to process.
L: (Int64) The number of configurations in the file.
N: (Int64) The number of atoms in each configuration.
B: (Bool) Whether configurations were found under a magnetic field of 0.3 au or not.
compare: (Bool) Whether the configurations are to be compared or classified.
rCutRange: (Vector{Float64}) A range of cut-off radii to test, in units of equilibriumBondLength.
M: (Int64) The number of rCut values to test. Default of 1 for classification.

Outputs:
energies: (Vector{Float64}) The energy of each configuration in each the file.
totalProfiles: Array{Dict{String,Int}} 2D array containg the totalCNA profiles of each configuration, for each rCut value.
atomicProfiles: Array{Dict{String,Int}} 3D array containing the CNA profile of each configuration, for each rCut value, for each atom.
configurations: (Array{Float64}) 3D array containing the atom coordinates for each configuration in the file. Only used if classifying.
shells: (Vector{Vector{Int64}}) A vector of shell numbers of each atom in each configuration. Empty if comparing.
bondGraphs: Vector{SimpleGraph} A vector of graphical representation of each configuration. Empty if comparing.
"""
function processFile(fp,L,N,B,compare,EBL,rCutRange;M=1)
	
	energies = Vector{Float64}() # Initialise list of configuration energies
	configurations = Array{Float64}(undef, (L,N,3)) # Initialise 3D array for the atom coordinates for each configuration in the file.
	totalProfiles = Array{Dict{String,Int}}(undef,M,L) # Initialise array to hold total CNA profiles
	atomicProfiles = Array{Dict{String,Int}}(undef,M,L,N) # Initialise array to hold atomic CNA profiles
	shells = Vector{Vector{Int64}}() # Initialise array to hold shell number of each atom in cluster
	bondGraphs = Vector{SimpleGraph}() # Initialise vector of graphical representation of each configuration.
	coordinatesNext = false # Does next line contain coordinates?
	headerNext = true # Does next line contain the header for a new configuration?
	skipLine = false # Does next line contain useless/no information?
	configuration, E, atomNum, configNum::Int = Array{Float64}(undef, (N,3)), 0.0, 0, 0 # Initialise variables 
	while (true) # While not at end of file
		if (headerNext) # If next line is the start of a new configuration
			atomNum = 0 # Initialise number of atoms processed in current configuration
			line = readline(fp) # Read first line
			configNum = parse(Int,line) # Get configuration number
			line = readline(fp) # Read blank line
			line = readline(fp) # Skip over number of atoms in the configuration as known from filename
			line = readline(fp) # Read blank line
			line = readline(fp) # Read next line
			E = round(parse(Float64,line); digits = 15) # Get the number of atoms in the configuration
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
			for (i,rCut) in enumerate(rCutRange) # For each rCut value
				if !compare # If classifying
					labels,graph = ShellLabelling(configuration,N,B,rCut,EBL) # Compute atomic shell numbers
					push!(shells,labels) # Add shell numbers for current configuration to list.
					push!(bondGraphs,graph) # Add graph for current configuration to list.
				end
				totalProfile,atomicProfile = CNA(configuration,N,rCut,B,EBL) # Compute the CNA profile of the current configuration
				totalProfiles[i,configNum] = totalProfile # Add the total CNA profile to vector
				atomicProfiles[i,configNum,:] = atomicProfile[:] # Store the atomic Profile
			end
			push!(energies,E) # Add the configuration energy to vector
			configurations[configNum,:,:] = configuration
			if (eof(fp)) # If at end of file
				break # end while loop
			elseif (isempty(line)) # If just a blank line -> another configuration follows
				line = readline(fp) # Read a blank line
				headerNext = true # New configuration header on next line
				configuration = Array{Float64}(undef, (N,3)) # Preallocate space for the configuration coordinates
			end
		end
	end
	return energies, totalProfiles, atomicProfiles, configurations, shells, bondGraphs # Return outputs
end

"""
Write out symmetry classifications for all configurations.

Inputs:
configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files.
filename: (String) The prefix of the file.
classifications: Vector{Dict{Vector{Int64},String}} Vector of dictionaries mapping a list of atoms to their symmetry.
capSymmetries: Vector{Dict{Dict{Int64,Vector{Int64}},String}} Vector of dictionaries mapping list of atoms forming a cap to their symmetry.
L: (Int64) The number of configurations in the file. 
"""
function writeClassification(configDir,filename,classifications,capSymmetries,L)
	# Write output file contatining classifications of all configurations in the file
	open("$configDir\\classification\\$(filename)_classification.txt","a") do io # Open the output file corresponding to the input file
		# Write file headers
		println(io,"Config #    classification")
		for j in 1:L # all the configurations in the file
			# Print out the classifcations
			println(io,"$(rpad(j,8," "))    $(classificationToString(classifications[j],capSymmetries[j]))")
		end
	end
end

"""
This function converts the classification and capSymmetry dictionaries to a string.

Inputs:
classification: Dict{Vector{Int64},String} Dictionary mapping a list of atoms to their symmetry.
capSymmetries: Dict{Dict{Int64,Vector{Int64}},String} Dictionary mapping list of atoms forming a cap to their symmetry.

Outputs:
string: (String) Formatted string.
"""
function classificationToString(classification,capSymmetry)
	if isempty(classification) # If no symmetries identified
		string = "No core or symmetries detected"
	else # If symmetries detected
		if isempty(capSymmetry) # If no caps detected
			capSymmetry = "None dectected"
		end
		string = "Symmetries: $classification; Caps: $capSymmetry" # Format string
	end	
	return string # Return string
end

"""
Write out total CNA profiles for all configurations for a particular rCut value.

Inputs:
rCutRange: (Vector{Float64}) A range of cut-off radii to test, in units of equilibriumBondLength.
configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files.
filename: (String) The prefix of the file.
energies: (Vector{Float64}) The energy of each configuration in each the file.
totalProfiles: Array{Dict{String,Int}} 2D array containg the totalCNA profiles of each configuration, for each rCut value.
i: (Int64) The index of the current rCut value.
L: (Int64) The number of configurations in the file. 
"""
function writeProfiles(rCutRange,configDir,filename,energies,totalProfiles,i,L)
	if !isdir("$configDir\\comparison\\rCut_$(rCutRange[i])")
		mkdir("$configDir\\comparison\\rCut_$(rCutRange[i])") # Create the output directory
	end
	# Write output files for each rCut value
	open("$configDir\\comparison\\rCut_$(rCutRange[i])\\profile_$(filename).txt","a") do io # Open the output file corresponding to the input file
		# Write file headers
		println(io,"Config #    Energy                 Total CNA Profile")
		for j in 1:L # all the configurations in the file
			# Print out the results
			println(io,"$(rpad(j,8," "))    $(rpad(energies[j], 19, " "))    $(totalProfiles[i,j])")
		end
	end
end

"""
Write out the configuration comparision results.

	Inputs:
	configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files.
	filename: (String) The prefix of the file.
	sortingArray: Vector{Int64} A vector of configuration numbers, sorted by maximum similarity score.
	sortedEnergies: (Vector{Float64}) The energy of each configuration, sorted by maximum similarity score.
	maxSims: Vector{Float64} A Vector of the maximum similarity score for each configuration when compared to every other configuration
							 over all rCut values, in decreasing order.
	similarConfigs: Vector{Dict{Int64,Vector{Float64}}} A vector of dictionaries which map the most similar configurations to the rCut values
														for which it is maximised, for each configuraiton, sorted by similarity.
	L: (Int64) The number of configurations in the file.
"""
function writeComparison(configDir,filename, sortingArray, sortedEnergies, maxSims, similarConfigs,L) # Write comparison output file
	open("$configDir\\comparison\\comparison_$(filename).txt","a") do io # Open the output file corresponding to the input file
		# Write file headers
		println(io,"Config #    Energy                 Similarity Score    $("Most similar to (Config #; rCut)")")
		for j in 1:L # all the configurations in the file
			# Print out the results
			println(io,"$(rpad(sortingArray[j],8," "))    $(rpad(sortedEnergies[j], 19, " "))    $(rpad(maxSims[j],17, " "))    $(similarConfigs[j])")
		end
	end
end

"""
Write out the distinguisable configuration set results.

	Inputs:
	configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files.
	filename: (String) The prefix of the file.
	numUnique (Int64) Number of distinguiable sets of configurations.
	uniqueConfigs: Vector{Vector{Int64}} Vector of configuration numbers of the distinguisable sets of configurations.
	uniqueSimilarities: Vector{Float64} Vector of similarity scores of each of the distinguiable sets.
	uniqueEnergyStats: Array{Float64} 2D array of energy statistics. First row is the mean, second row is standard deviation.
"""
function writeUnique(configDir,filename,numUnique,uniqueConfigs,uniqueSimilarities, uniqueEnergyStats)
	# Write unique configurations output file
	open("$configDir\\comparison\\unique_$(filename).txt","a") do io # Open the output file corresponding to the input file
		# Write file headers
		println(io,"Unique Config    Similarity Score     Energy Mean            Energy Std")
		for j in 1:numUnique # all the configurations in the file
			# Print out the results
			println(io,"$(rpad(uniqueConfigs[j],13," "))    $(rpad(uniqueSimilarities[j],17, " "))    $(rpad(uniqueEnergyStats[1,j], 19, " "))    $(uniqueEnergyStats[2,j])")
		end
	end
end

end