# Author: AJ Tyler
# Date: 24/01/22
module main # This module contains the two callable functions, compare() and classify().
include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/io.jl") # This file contains the IO module.
include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/comparison.jl") # This file contains the Comparison module
include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/visualisation.jl") # This file contains the Visualisation module.
include("/Users/tiantianyu/Downloads/ParallelTemperingMonteCarlo.jl-2/CNA/classify.jl") # This file contains the Classification module.
using .Comparison # This module the contains functions which compare configurations.
using .IO # This module contains functions that read in and write out files.
using .Visualisation # This module contains functions which create a .vesta file with helpful colourisations.
using .Classify # This module contains functions which classify configurations based on their symmetries.
using Graphs # This module contains graph theory functions which allow for a graphical representation of configurations.
export classify # This function uses the atomic CNA profile of a configuration to identify symmetries.
export compare # This function computed the CNA profiles of configurations oevr a range of rCut values.

"""
This function computed the CNA profiles of configurations over a range of rCut values. These configurations are then compared between
eachother using a similarity score. If these similarity score is high enough over enough rCut values, then the configurations are
considered to be the same. For each input .xyz file, a .txt file is generated inside a rCut_XXX directory for each rCut value (XXX)
containing the total CNA profiles of each configuration. A comparison_NAME.txt file is also generated inside the comparison directory
for each input file, which lists which configurations were most similar to each configuration and for which rCuts. Finally
a unique_NAME.txt file is also created inside the comparison directory which lists each set of distinguishable configurations and their
associated energy statistics.

Inputs:
configDir: (String) The path to (or name of, if in the local directory) the directory containing the .xyz files.
					The .xyz files must have the following format NAME_N_L.xyz, where NAME is unique for all the files in the
					directory, N is the number of atoms in each configuration and L is the number of configurations in the file.
Keyword Arguments:
rCutRange: (Vector{Float64}) A range of cut-off radii to test, in units of EBL. Default is 10 values between 4/3 and 2.
similarityMeasure: (String) The method of comparing configurations, either "atomic" or "total" CNA. Default is total.
similarityThreshold: (Float64) Similarity threshold above which two configurations are considered identical.
EBL: (Float64) Equilibrium Bond Length. Default is 3.1227 Angstroms (from MP2 data for Neon2).
"""
function compare(configDir::String; rCutRange::Vector{Float64} = round.(LinRange(4/3,2,10);digits = 3), similarityMeasure = "total", similarityThreshold = 0.95,EBL=3.1227)
	rm("$configDir/comparison";force = true,recursive = true) # Delete comparision directory if exists from previous run
	mkdir("$configDir/comparison") # Make comparision directory
	M = length(rCutRange) # Number of rCut values to test
	files = readdir(configDir) # Create list of all files/directories in the configuration directory
	for file in files # For each file
		isXYZ,fileName,fp,L,N,B = processFileName(configDir,file) # Extract information from filename
		if isXYZ # If file is a .xyz file
			# Compute CNA profiles of all configurations in file
			energies,totalProfiles, atomicProfiles,_,_,_ = processFile(fp,L,N,B,true,EBL,rCutRange;M)
			# Find most similar configurations for each configuration based on CNA profiles over all rCut values
			similarConfigs, maxSims = compareConfigs(rCutRange,energies,totalProfiles, atomicProfiles, configDir,fileName, similarityMeasure,M,L,N)
			if L == 1 # If only have one configuration in the file
				continue # Don't need to do any comparisons
			end
			sortingArray = sortperm(maxSims[1,:],rev=true) # Sort configurations by maximum similarity values in decreasing order
			# Sort arrays
			maxSims, sortedEnergies, similarConfigs = maxSims[sortingArray], energies[sortingArray], similarConfigs[sortingArray]
			# Write out configuration comparison results
			writeComparison(configDir,fileName, sortingArray, sortedEnergies, maxSims, similarConfigs,L)
			# Group configurations into distinguishable sets
			uniqueConfigs,uniqueEnergies,uniqueSimilarities = groupConfigs(energies,sortingArray,maxSims, similarConfigs, similarityThreshold,M)
			numUnique = length(uniqueSimilarities) # Compute how many distinguishable sets are in the file
			uniqueStats = configStats(uniqueEnergies,numUnique) # Compute statistics on energies of each distinguishable set
			writeUnique(configDir,fileName,numUnique,uniqueConfigs,uniqueSimilarities, uniqueStats) # Write out distinguishable sets file
		end
	end
end

"""
This function uses the atomic CNA profile of a configuration to identify symmetries. These symmetries include icosahedral (ICO),
body centred cubic (BCC), face centered cubic (FCC), hexagonal close packed (HCP) and OTHER. Futhermore, if a core symmetry can be
identified, then cap and capping atoms are also identified. For each input .xyz file, a NAME_classification.txt file is generated inside
the classification directory, with which atoms have which symmertry and if they are part of the cluster's core, a cap or a capping atom.
A NAME_i.vesta file is also generated in the visulisation directory, where i is the configuration number in the .xyz file, which uses the
classification information to which configurations were most similar to each configuration and for which rCuts. Finally a unique_NAME.txt
file is created which colour the atoms so aid visualisation of the cluster's structure.

Inputs:
configDir: (String) The path to (or name of, if in the local directory) the directory containing the .xyz files.
					The .xyz files must have the following format NAME_N_L.xyz, where NAME is unique for all the files in the
					directory, N is the number of atoms in each configuration and L is the number of configurations in the file.
Keyword Arguments:
rCut: (Float64) The cut-off radii used in the CNA analysis in units of EBL. Default is 4/3.
EBL: (Float64) Equilibrium Bond Length. Default is 3.1227 Angstroms (from MP2 data for Neon2).
"""
function classify(configDir::String; rCut = 4/3,EBL=3.1227)
	rm("$configDir/classification";force = true,recursive = true) # Delete classification directory if exists from previous run
	mkdir("$configDir/classification") # Create the classification directory
	rm("$configDir/visualisation";force = true,recursive = true) # Delete visualisation directory if exists from previous run
	mkdir("$configDir/visualisation") # Create the visualisation directory
	files = readdir(configDir) # Create list of all files/directories in the configuration directory
	for file in files # For each file
		isXYZ,fileName,fp,L,N,B = processFileName(configDir,file) # Extract information from filename
		if isXYZ # If file is a .xyz file
			# Compute CNA profiles and atom shell numbers of all configurations in file
			_,_, atomicProfiles,configurations, shells, bondGraphs = processFile(fp,L,N,B,false,EBL,[rCut])
			# Identify symmetries of the clusters using CNA profiles
			classifications, capSymmetries = findSymmetries(shells,atomicProfiles,bondGraphs,L)
			writeClassification(configDir,fileName,classifications,capSymmetries,L) # Output found symmetries
			vestaFile(configDir,fileName,configurations,classifications,capSymmetries,rCut,EBL,L,N) # Create colourised .vesta file
		end
	end
end

end # End of module