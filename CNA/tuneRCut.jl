# Author: AJ Tyler
# Date: 25/11/21

# This Module tunes the r cut-off parameter for CNA
module RCut
include("compare.jl") 
include("readConfigs.jl")
using .Compare # This module allows for comparisons between configurations
using .IO # This module reads in configurations and computes the CNA profile
using Plots
export tuneRCut

"""
This function computes the similarity between two configurations for a number of RCut values.
	This is useful for tuning the RCut parameter,
		so want a RCut value with a similarity of 1 for configurations which have the same structure.
	
	Inputs:
	dir: (String) Path to directory containing the .xyz file with the two configurations.
	fileName: (String) Name of .xyz file containing the configurations to compare.
	testSize: (Int64) The number of different RCut values to test within thr range 1-2 equilibrium distance.
	equilibriumDistance: (Float64) The equilibrium bond length in angstroms. Default value is for Leonard Jones Potential.

	Outputs:
	same: (Vector{Int64}) Indicies of RCut test values for which the similarity is 1.
"""
function tuneRCut(dir,fileName,testSize,equilibriumDistance=2.0^(1/6))
	fp = "$dir\\output\\profile_$fileName" # Filepath for configuration file
	configA = 1 # Configuration number in the configuration file for one of the configurations.
	configB = 2 # Configuration number in the configuration file for the other configuration.
	testRange = [1,2] # Range of RCut values in terms of equilibrium distance.
	similarity = Array{Float64}(undef, testSize) # Initialise array for similarity scores for RCut values.
	RCutRange = equilibriumDistance.*LinRange(testRange[1],testRange[2],testSize) # Create range of RCut values in Angstroms
	same = Vector{Int64}() # Intialise vector of Indicies of RCut test values for which the similarity is 1.
	for (i,RCut) in enumerate(RCutRange) # For all RCut values
		readConfigs(dir,RCut^2) # Compute the total CNA profile of both configurations
		similarity[i] = compareConfigs(fp, configA, configB,i) # Compute the similarity of the configurations
		if (similarity[i] == 1.0) # If the configurations were indistinguishable
			push!(same,i) # Add the index of the RCut value
		end
		rm(fp) # Remove the configuration file.
	end
	
	# Plot similarity against RCut
	display(plot(RCutRange,similarity, title = "R cutoff calibration"))
	xlabel!("R cutoff (Angstroms)")
	ylabel!("Jacard Index")
	png("output") # Save the plot
	return same # Return the similarity 1 RCut indicies
end

end # End of module