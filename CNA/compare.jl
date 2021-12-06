# Author: AJ Tyler
# Date: 25/11/21
module Compare # This module compares two configurations based on their energy and total CNA profile
include("similarity.jl") # This file computes the Jacard index of two multisets
using .Similarity # This module contains the JacardIndex function
export stringToDict # This function converts a string representation of a dictionary to a dictionary object
export compareConfigs # This function retuns the similarity index of two configurations

"""
This function converts a string representation of a dictionary to a dictionary object.

Inputs:
string: (String) String representation of a dictionary

Outputs:
dict: (Dictionary) converted dictionary
"""
function stringToDict(string)
	dict = Dict{String,Int}() # Create dictionary object
	seq = split(chop(string,head=5,tail=1),", ") # Split string into dictionary elements, removing excess characters
	for element in seq # For each dictionary element
		parts = split(element,"\" => ") # Split up the key and value
		key = chop(parts[1],head = 1,tail=0) # Extract key
		value = parse(Int,parts[2]) # Extract value
		push!(dict, key => value) # Add element to dictionary object
	end

	return dict # Return dictionary
end

"""
This function reads in a readConfigs output file and compares two of the configurations based on their energy and CNA profile.
This function returns -1 if invalid configuration numbers were input, 0 if the two configuration's energies differ by more than 1%,
otherwise returns their Jacard Index.

Inputs:
filePath: (String) File path to the readConfigs output file, profile_XXX.xyz
configA: (Int64) The configuration number in the readConfigs output file, profile_XXX.xyz, of the first configuration to compare
configB: (Int64) The configuration number in the readConfigs output file, profile_XXX.xyz, of the second configuration to compare.
tol: (Float64) fractional energy difference tolerance. Default is 0.01.
"""
function compareConfigs(fp, configA, configB, tol=0.01)
	lines = readlines(fp) # Read in readConfigs output file
	lastConfig = length(lines-1) # Extract largest configuration number in the file
	if max(configA, configB) > lastConfig # If either input configuration number exceeds this maxmimum
		return -1
	end

	energyA = parse(Float64,(split(lines[configA+1], r" +"))[2]) # Extract the energy of configA
	energyB = parse(Float64,(split(lines[configB+1], r" +"))[2]) # Extract the energy of configB

	if (abs(energyA-energyB)/abs(energyA)>tol) # If fractional energy difference exceeds the tolerance
		return 0
	end

	profileA = stringToDict((split(lines[configA+1], r" +"))[5]) # Extract the CNA profile of configA
	profileB = stringToDict((split(lines[configB+1], r" +"))[5]) # Extract the CNA profile of configB
	return JacardIndex(profileA,profileB) # Return their Jacard Index
end

end # End of Module
	