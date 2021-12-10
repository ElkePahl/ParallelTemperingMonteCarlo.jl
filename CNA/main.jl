# Author: AJ Tyler
# Date: 10/12/21
module IO # This module reads in .xyz files in a directory and computes the CNA profile of each configuration
include("CNA.jl") # This file contains the CommonNeighbourAnalysis module
include("similarity.jl")
#using BenchmarkTools
using Statistics
using .CommonNeighbourAnalysis # This module contains the CNA function
using .Similarity
export main # This function reads in each .xyz file computes the CNA profile of each configuration

"""
This function reads in all the .xyz files in a directory and performs CNA on each of the configurations for each rCut value.
For each input file, a .txt file is generated inside a rCut_XXX directory for each rCut value containing the total CNA
profiles of each configuration. A XXX_comparison_XXX.txt file is also generated for each input file, which lists which
configurations were most similar to each configuration and for which rCuts. Finally a XXX_unique_XXX.txt file is created which
lists each set of distinguishable configurations and their associated energy statistics.

Inputs:
configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files.
					The .xyz files must have the following format name_N_L.xyz, where name is different for all the files in the
					directory, N is the number of atoms in each configuration and L is the number of configurations in the file.
rCutRange: (Vector{Float64}) A range of cut-off radii to test, in units of equilibriumBondLength.
equilibriumBondLength: (Float64) The equilibrium bond length in the same units as the configuration coordinates.
Keyword Arguments:
similarityMeasure: (String) The method of comparing configurations, either "atomic" or "total" CNA.
similarityThreshold: (Float64) Similarity threshold above which two configurations are considered identical.
"""
function main(configDir::String,rCutRange::Vector{Float64},equilibriumBondLength::Float64; similarityMeasure = "atomic", similarityThreshold = 0.95)
	if (!isdir("$configDir\\output")) # If output directory doesn't exist
		mkdir("$configDir\\output") # Create the output directory
	end
	rcutSquared = (equilibriumBondLength.*rCutRange).^2
	M = length(rCutRange) # Number of rCut values to test
	files = readdir(configDir) # Create list of all files/directories in the configuration directory
	for file in files # For each file
		parts = split(file,".xy") # Process filename to be able to tell if .xyz file
		if (last(parts) == "z") # If is an .xyz file
			fp = open("$configDir\\$file") # Open the file
			parts = split(parts[1],'_') # Process filename
			filename = parts[1] # Get filename prefix
			N,L = (parse.(Int64,parts[2:3]))[:] # Number of configurations in the file (part of file name)
			# Compute CNA profiles of all configurations in file
			energies,totalProfiles, atomicProfiles = processFile(fp,M,L,N,rcutSquared)
			if (L==1) # If file only has 1 configuration
				for i in 1:M # For each rCut value
					writeProfiles(rCutRange,configDir,filename,energies,totalProfiles,i,L) # Write out total CNA profile
				end
				continue # Go to next file
			end
			# Find most similar configurations for each configuration based on CNA profiles over all rCut values
			similarConfigs, maxSims = compareConfigs(rCutRange,energies,totalProfiles, atomicProfiles, configDir,filename, similarityMeasure,M,L,N)
			sortingArray = sortperm(maxSims[1,:],rev=true) # Sort configurations by maximum similarity values in decreasing order
			# Sort arrays
			maxSims, sortedEnergies, similarConfigs = maxSims[sortingArray], energies[sortingArray], similarConfigs[sortingArray]
			# Write out configuration comparison results
			writeComparison(configDir,filename, sortingArray, sortedEnergies, maxSims, similarConfigs,L)
			# Group configurations into distinguishable sets
			uniqueConfigs,uniqueEnergies,uniqueSimilarities = groupConfigs(energies,sortingArray,maxSims, similarConfigs, similarityThreshold)
			numUnique = length(uniqueSimilarities) # Compute how many distinguishable sets are in the file
			uniqueStats = configStats(uniqueEnergies,numUnique) # Compute statistics on energies of each distinguishable set
			writeUnique(configDir,filename,numUnique,uniqueConfigs,uniqueSimilarities, uniqueStats) # Write out distinguishable sets file
		end
	end
end

"""
This function reads in all the .xyz files in a directory and performs CNA on each of the configurations within each of the files.

Inputs:
fp: (String) The file path to the .xyz file to process.
M: (Int64) The number of rCut values to test.
L: (Int64) The number of configurations in the file.
N: (Int64) The number of atoms in each configuration.
rcutSquared: (Vector{Float64}) A range of squared cut-off radii to test, which determines which atoms are 'bonded'.

Outputs:
energies: (Vector{Float64}) The energy of each configuration in each the file.
totalProfiles: Array{Dict{String,Int}} 2D array containg the totalCNA profiles of each configuration, for each rCut value.
atomicProfiles: Array{Dict{String,Int}} 3D array containing the CNA profile of each configuration, for each rCut value, for each atom.
"""
function processFile(fp,M,L,N,rcutSquared)
	energies = Vector{Float64}() # Initialise list of configuration energies
	totalProfiles = Array{Dict{String,Int}}(undef,M,L) # Initialise array to hold total CNA profiles
	atomicProfiles = Array{Dict{String,Int}}(undef,M,L,N) # Initialise array to hold atomic CNA profiles
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

			for (i,rCut) in enumerate(rcutSquared)
				#@btime profile = CNA($configuration,$N,$rcutSquared) # Compute the CNA profile of the current configuration
				totalProfile,atomicProfile = CNA(configuration,N,rCut) # Compute the CNA profile of the current configuration
				totalProfiles[i,configNum] = totalProfile # Add the total CNA profile to vector
				atomicProfiles[i,configNum,:] = atomicProfile[:] # Store the atomic Profile
			end
			push!(energies,E) # Add the configuration energy to vector
			if (eof(fp)) # If at end of file
				break # end while loop
			elseif (isempty(line)) # If just a blank line -> another configuration follows
				line = readline(fp) # Read a blank line
				headerNext = true # New configuration header on next line
				configuration = Array{Float64}(undef, (N,3)) # Preallocate space for the configuration coordinates
			end
		end
	end
	return energies, totalProfiles, atomicProfiles # Return outputs
end

"""
Converts a 2D array of containing the CNA profile of each configuration, for each atom, to a vector of dictionaries,
mapping atom CNA profiles to their frequency, for each configuration.

Inputs:
atomicProfiles: Array{Dict{String,Int}} 3D array containing the CNA profile of each configuration, for each rCut value, for each atom.
i: (Int64) The index of the current rCut value.
L: (Int64) The number of configurations in the file.

Outputs:
atomicProfileDict: Vector{Dict{String,Int64}} ector of dictionaries, mapping atom CNA profiles to their frequency, for each configuration.
"""
function convertAtomicProfiles(atomicProfiles,i,L)
	# Convert atomicProfiles into format better for computing similarity
	atomicProfilesDict = [Dict{String,Int64}() for j in 1:L] # Initialise new format
	for j in 1:L # For all configurations
		atomicProfile = atomicProfiles[i,j,:] # Get atom profile
		for atomProfile in atomicProfile # For each atom
			# Increments atomic CNA profile frequency by 1
			push!(atomicProfilesDict[j], "$atomProfile" => get!(atomicProfilesDict[j],"$atomProfile",0)+1)
		end
	end

	return atomicProfilesDict # Return output
end

"""
Finds the most similar configuration to every configuration over all rCut values based on their CNA profile.

Inputs:
rCutRange: (Vector{Float64}) A range of cut-off radii to test, in units of equilibriumBondLength.
energies: (Vector{Float64}) The energy of each configuration in each the file.
totalProfiles: Array{Dict{String,Int}} 2D array containg the totalCNA profiles of each configuration, for each rCut value.
atomicProfiles: Array{Dict{String,Int}} 3D array containing the CNA profile of each configuration, for each rCut value, for each atom.
configDir: (String) The path to (or name of if in same location as this program) the directory containing the .xyz files.
filename: (String) The prefix of the file.
similarityMeasure: (String) The method of comparing configurations, either "atomic" or "total" CNA.
M: (Int64) The number of rCut values to test.
L: (Int64) The number of configurations in the file.
N: (Int64) The number of atoms in each configuration.

Outputs:
similarConfigs: Vector{Dict{Int64,Vector{Float64}}} A vector of dictionaries which map the most similar configurations to the rCut values
													for which it is maximised, for each configuraiton.
maxSims: Vector{Float64} A Vector of the maximum similarity score for each configuration when compared to every other configuration
						 over all rCut values.
"""
function compareConfigs(rCutRange,energies,totalProfiles,atomicProfiles,configDir,filename,similarityMeasure,M,L,N)
	# Initialise vector to contain most similar configurations for each configuration
	similarConfigs = Vector{Dict{Int64,Vector{Float64}}}(undef,L)
	maxSims = zeros(Float64,1,L) # Initialise the maximum similarity score for each configuration
	for i in 1:M # For all rCut values
		writeProfiles(rCutRange,configDir,filename,energies,totalProfiles,i,L) # Write out the total CNA profiles
		atomicProfilesDict = convertAtomicProfiles(atomicProfiles,i,L) # Convert atomic CNA profile format
		# For each pair of configurations
		for configA in 1:L-1
			for configB in configA+1:L
				# Compute the similarity for the cluster pair
				similarity = similarityScore(configA, configB, totalProfiles,atomicProfilesDict,similarityMeasure,N,i)
				# Update the most similar configurations to configA
				if (similarity > maxSims[configA]) # If new similarity score is greater than previous highest
					similarConfigs[configA] = Dict{Int64,Vector{Float64}}(configB=>Vector{Float64}([rCutRange[i]])) # Create new vector, containing configB
					maxSims[configA] = similarity # Update maximum similarity score
				elseif (similarity == maxSims[configA]) # If the new similarity is the same as the previous highest
					if (haskey(similarConfigs[configA],configB)) # If the same configuration pair is already most similar
						# Add the current rCut value to the dictionary
						push!(similarConfigs[configA], configB => push!(get!(similarConfigs[configA],configB,0),rCutRange[i]))
					else
						push!(similarConfigs[configA], configB => Vector{Float64}([rCutRange[i]])) # Add the configuration to the dictionary
					end
				end

				# Same for configB
				if (similarity > maxSims[configB]) # If new similarity score is greater than previous highest
					similarConfigs[configB] = Dict{Int64,Vector{Float64}}(configA=>Vector{Float64}([rCutRange[i]])) # Create new vector, containing configB
					maxSims[configB] = similarity # Update maximum similarity score
				elseif (similarity == maxSims[configB]) # If the new similarity is the same as the previous highest
					if (haskey(similarConfigs[configB],configA))
						push!(similarConfigs[configB], configA => push!(get!(similarConfigs[configB],configA,0),rCutRange[i]))
					else
						push!(similarConfigs[configB], configA => Vector{Float64}([rCutRange[i]]))
					end
				end
			end
		end
	end
	return similarConfigs, maxSims # Return outputs
end

"""
This function returns the similarity score of two configurations based on their CNA profiles.

Inputs:
configA: (Int64) The configuration number of one of the configurations to compare.
configB: (Int64) The configuration number of the other of the configurations to compare.
totalProfiles: Array{Dict{String,Int}} 2D array containg the totalCNA profiles of each configuration, for each rCut value.
atomicProfilesDict: Vector{Dict{String,Int64}} ector of dictionaries, mapping atom CNA profiles to their frequency, for each configuration.
similarityMeasure: (String) The method of comparing configurations, either "atomic" or "total" CNA.
N: (Int64) The number of atoms in each configuration.
i: (Int64) The index of the current rCut value.

Outputs:
similarity: (Float64) The similarity score between the two configurations.
"""
function similarityScore(configA, configB, totalProfiles,atomicProfilesDict,similarityMeasure,N,i)
	similarity = 0.0 # Initialise similarity score
	if (similarityMeasure == "atomic") # If want to compare using atomic CNA profile
		# Compute similarity of atomic CNA profiles
		for key in keys(atomicProfilesDict[configA]) # For each atom CNA profile
			# Add the number of atoms with the same atom CNA profile
			similarity += min(get(atomicProfilesDict[configA],key,0),get(atomicProfilesDict[configB],key,0))
		end
		similarity /= N # Normalise the similarity score
		
	elseif (similarityMeasure == "total") # If want to compare using total CNA profile
		# Compute Jacard index of their total CNA profiles
		similarity = JacardIndex(totalProfiles[i,configA],totalProfiles[i,configB])
	else # If none of the valid similarity measures were used
		throw(DomainError(similarityMeasure, "Invalid similarity measure")) # Throw an error
	end

	return round(similarity; digits = 3) # Return rounded similarity score
end

"""
Write out total CNA profiles for all configurations for a particule rCut value.

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
	if (!isdir("$configDir\\output\\rCut_$(rCutRange[i])")) # If output directory doesn't exist
		mkdir("$configDir\\output\\rCut_$(rCutRange[i])") # Create the output directory
	end
	# Write output files for each rCut value
	open("$configDir\\output\\rCut_$(rCutRange[i])\\profile_$(filename).txt","a") do io # Open the output file corresponding to the input file
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
	open("$configDir\\output\\comparison_$(filename).txt","a") do io # Open the output file corresponding to the input file
		# Write file headers
		println(io,"Config #    Energy                 Similarity Score    $("Most similar to (Config #; rCut)")")
		for j in 1:L # all the configurations in the file
			# Print out the results
			println(io,"$(rpad(sortingArray[j],8," "))    $(rpad(sortedEnergies[j], 19, " "))    $(rpad(maxSims[j],17, " "))    $(similarConfigs[j])")
		end
	end
end

"""
This function groups configurations into distinguisable sets based on their similarity scores.

Inputs:
energies: (Vector{Float64}) The energy of each configuration in each the file.
sortingArray: Vector{Int64} A vector of configuration numbers, sorted by maximum similarity score.
maxSims: Vector{Float64} A Vector of the maximum similarity score for each configuration when compared to every other configuration
						 over all rCut values, in decreasing order.
similarConfigs: Vector{Dict{Int64,Vector{Float64}}} A vector of dictionaries which map the most similar configurations to the rCut values
						 							for which it is maximised, for each configuraiton, sorted by similarity.
similarityThreshold: (Float64) Similarity threshold above which two configurations are considered identical.

Outputs:
uniqueConfigs: Vector{Vector{Int64}} Vector of configuration numbers of the distinguisable sets of configurations.
uniqueEnergies: Vector{Vector{Float64}} Vector of configuration energies of the distinguisable sets of configurations.
uniqueSimilarities: Vector{Float64} Vector of similarity scores of each of the distinguiable sets.
"""
function groupConfigs(energies,sortingArray,maxSims,similarConfigs,similarityThreshold)
	# Initialise arrays
	uniqueConfigs = Vector{Vector{Int64}}()
	uniqueEnergies = Vector{Vector{Float64}}()
	uniqueSimilarities = Vector{Float64}()
	classified = Vector{Int64}() # Vector of configuration numbers that have already been grouped
	for (j,config) in enumerate(sortingArray) # For all configurations
		if (!(config in classified)) # If the configuration has not already been grouped
			if (maxSims[j] > similarityThreshold) # If the maximum similarity of the configuration exceeds a threshold
				 # Get vector of configuration numbers of configurations that are similar
				same = push!(collect(keys(similarConfigs[j])),config)
				push!(uniqueConfigs,same) # Add the set of configurations to the list
				push!(uniqueEnergies,energies[same]) # Add the energies of the configurations to the list
				push!(uniqueSimilarities,maxSims[j]) # Add the similarity score to the list
				classified = vcat(classified,same) # Add the configurations to the list of already classified
			else # If similarity score not high enough, then it it's own distinguiable set
				# Add configuration to lists
				push!(uniqueConfigs,Vector{Int64}([config]))
				push!(uniqueEnergies,Vector{Float64}([energies[config]]))
				push!(uniqueSimilarities,NaN)
				push!(classified,config)
			end
		end
	end

	return uniqueConfigs, uniqueEnergies, uniqueSimilarities # Return outputs
end

"""
This function returns some basic statistics of the energies of each of the distinguisbale configuration sets.

Inputs:
uniqueEnergies: (Vector{Vector{Float64}}) Vector of configuration energies of the distinguisable sets of configurations.
numUnique (Int64) Number of distinguiable sets of configurations.

Outputs:
uniqueEnergyStats: Array{Float64} 2D array of energy statistics. First row is the mean, second row is standard deviation.
"""
function configStats(uniqueEnergies,numUnique)
	uniqueEnergyStats = Array{Float64}(undef,2,numUnique)
	for i in 1:numUnique
		uniqueEnergyStats[1,i] = Statistics.mean(uniqueEnergies[i])
		uniqueEnergyStats[2,i] = Statistics.std(uniqueEnergies[i])
	end

	return uniqueEnergyStats
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
	open("$configDir\\output\\unique_$(filename).txt","a") do io # Open the output file corresponding to the input file
		# Write file headers
		println(io,"Unique Config    Similarity Score     Energy Mean            Energy Std")
		for j in 1:numUnique # all the configurations in the file
			# Print out the results
			println(io,"$(rpad(uniqueConfigs[j],13," "))    $(rpad(uniqueSimilarities[j],17, " "))    $(rpad(uniqueEnergyStats[1,j], 19, " "))    $(uniqueEnergyStats[2,j])")
		end
	end
end

end # End of module