# Author: AJ Tyler
# Date: 24/01/22
module Comparison # This module contains functions which compare configurations using their CNA profiles.
include("io.jl") # This file contains the IO module.
using .IO # This module contains functions that read in and write out files.
using Statistics # This module allows easy mean and standard deviation.
export compareConfigs,groupConfigs,configStats, similarityScore, similarityScore_one # These functions ared used to compare configurations.

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
		if L == 1
			continue
		end
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

function similarityScore_one(configA, configB, totalProfiles,atomicProfilesDict,similarityMeasure,N)
	similarity = 0.0 # Initialise similarity score
	if (similarityMeasure == "atomic") # If want to compare using atomic CNA profile
		# Compute similarity of atomic CNA profiles
		println("atomic")
		for key in keys(atomicProfilesDict[configA]) # For each atom CNA profile
			# Add the number of atoms with the same atom CNA profile
			println(get(atomicProfilesDict[configA],key,0))
			println(get(atomicProfilesDict[configB],key,0))
			similarity += min(get(atomicProfilesDict[configA],key,0),get(atomicProfilesDict[configB],key,0))
		end
		similarity /= N # Normalise the similarity score
		
	elseif (similarityMeasure == "total") # If want to compare using total CNA profile
		# Compute Jacard index of their total CNA profiles
		similarity = JacardIndex(totalProfiles[configA],totalProfiles[configB])
	else # If none of the valid similarity measures were used
		throw(DomainError(similarityMeasure, "Invalid similarity measure")) # Throw an error
	end

	return round(similarity; digits = 3) # Return rounded similarity score
end

"""
This function calculates the Jacard Index for two dictionaries,
with their keys being the unique elements of a multiset and the values are their frequencies

Inputs:
A: (Dictionary) Multiset A
B: (Dictionary) Multiset B

Outputs:
similarity: (Float64) The Jacard Index of multsets A & B
"""
function JacardIndex(A,B)
	intersection = 0.0 # Initialise the number of elements in the intersection of A & B
	union = 0.0 # Initialise the number of elements in the union of A & B
	for key in keys(A) # For all the unique elements of A
		intersection += min(get(A,key,0),get(B,key,0)) # Add the minimum number of occurances of the key across both A & B
		union += get(A,key,0) # Add all the elements of A
	end

	for key in keys(B) # For all the unique elements of B
		union += get(B,key,0) # Add all the elements of B
	end

	union -= intersection # union = A + B - intersection
	similarity = intersection/union # Comput the Jacard index
	return similarity # Return the Jacard Index
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
M: (Int64) The number of rCut values tested.

Keyword Arguments:
rCutThreshold: (Float64) Proportion of rCut values that have to meet the similarity threshold for two configs to be considered identical.

Outputs:
uniqueConfigs: Vector{Vector{Int64}} Vector of configuration numbers of the distinguisable sets of configurations.
uniqueEnergies: Vector{Vector{Float64}} Vector of configuration energies of the distinguisable sets of configurations.
uniqueSimilarities: Vector{Float64} Vector of similarity scores of each of the distinguiable sets.
"""
function groupConfigs(energies,sortingArray,maxSims,similarConfigs,similarityThreshold,M;rCutThreshold=0.9)
	# Initialise arrays
	uniqueConfigs = Vector{Vector{Int64}}()
	uniqueEnergies = Vector{Vector{Float64}}()
	uniqueSimilarities = Vector{Float64}()
	classified = Vector{Int64}() # Vector of configuration numbers that have already been grouped
	for (j,config) in enumerate(sortingArray) # For all configurations
		if (!(config in classified)) # If the configuration has not already been grouped
			same = Vector{Int64}([config])
			if (maxSims[j] > similarityThreshold) # If the maximum similarity of the configuration exceeds a threshold
				for key in keys(similarConfigs[j])
					if (length(get!(similarConfigs[j],key,0)) > rCutThreshold*M && !(key in classified))
						# Get vector of configuration numbers of configurations that are similar
						push!(same,key)
					end
				end
			end
			push!(uniqueConfigs,same) # Add the set of configurations to the list
			push!(uniqueEnergies,energies[same]) # Add the energies of the configurations to the list
			push!(uniqueSimilarities,maxSims[j]) # Add the similarity score to the list
			classified = vcat(classified,same) # Add the configurations to the list of already classified
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
	uniqueEnergyStats = Array{Float64}(undef,2,numUnique) # Initialse array
	for i in 1:numUnique # For each grouped set of configurations
		uniqueEnergyStats[1,i] = Statistics.mean(uniqueEnergies[i]) # Compute mean
		uniqueEnergyStats[2,i] = Statistics.std(uniqueEnergies[i]) # Compute standard deviation
	end

	return uniqueEnergyStats # Return statistics
end

end # End of Module
