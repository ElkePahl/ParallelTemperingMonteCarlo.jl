# Author: AJ Tyler
# Date: 29/11/21

# This script randomly moves each atom of a cluster by increasing amounts until the there is
# sufficient difference in similarity between them. This allows RCut to be calibrated and determine
# the sensitivity of the CNA method.

include("tuneRCut.jl")
using .RCut # This module computes the similarity of two configurations for a range of RCut values.

dir = "RCutConfigs" # Directory where the created configuration file will be created.
fileName = "config_20.xyz" # Filename of the configuration file to perturb
fp  = "$dir\\testConfigs\\$fileName" # Filepath of the configuration file to perturb
testSize = 10 # Number of RCut values to test
perturb = 0.01 # Initial relative amount to move each atom, relative so outer shell atoms move more

while true # Until the pertrubation causes the configurations to be significantly different.
	lines = readlines(fp) # Read the configuration file
	if (isfile("$dir\\TuneRCut.xyz")) # Check if the file wanting to be outputted already exists.
		GC.gc() # Run the garbage cleaner (Windows problem when deleting files)
		rm("$dir\\TuneRCut.xyz") # Remove the file we want to output
	end
	open("$dir\\TuneRCut.xyz","a") do io # Open the output file corresponding to the input file
		for i in 1:2 # Do twice, once for unperturbed configuration and once for the perturbed
			for (j,line) in enumerate(lines) # For each line in input configuration file
				if (i==2 && j == 1) # If writing perturbed configuration and first line
					println(io,2) # Write the configuration number as 2
				elseif (i==1 || j < 7) # Else if the unperturbed configuration or not a coordinate
					println(io,line) # Copy the line from the input file
				else
					print(io,"10") # Print atomic number
					for x in (split(line, r" +"))[2:4] # For each component of the coordinate
						print(io," ") # Print space
						print(io,(1+perturb*(2*rand()-1))*parse(Float64,x)) # Write out the randomly perturbed coordinate
					end
					print(io,"\n") # Print a new line
				end
			end
			if i == 1 # If just finished the unperturbed configuration
				println(io,"\n") # Add a new line
			end
		end
	end
	same = tuneRCut(dir,"TuneRCut.xyz",testSize) # Find the indicies of the RCut values for which had a similarity of 1.
	println(same) # Print out the indicies
	if (length(same) > testSize/3) # If configurations are not sufficiently different
		global perturb += 0.02 # Increase the perturbation amount
	else
		print(perturb) # Print the amount of pertrubation required to differentiate the configurations.
		break # End the loop (script).
	end
end
