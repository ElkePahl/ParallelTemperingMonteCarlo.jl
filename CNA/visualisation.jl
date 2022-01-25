# Author: AJ Tyler
# Date: 25/01/22
module Visualisation # This module converts a configuration and its classifications into a coloursied .vesta file
export vestaFile # This function outputs a colourised .vesta file

"""
This function converts a classified configuration into a colourised .vesta file. It does this by using a reference .vesta file,
'blueprint.vesta', and only changes the lines corresponding to atom coordinates and colours based on detected symmetries.

Inputs:
configDir: (String) The path to (or name of, if in the local directory) the directory containing the .xyz files.
fileName: (String) The unique prefix of the fileName.
configurations: (Array{Float64}) 3D array containing the atom coordinates for each configuration in the file.
classifications: Vector{Dict{Vector{Int64},String}} Vector of dictionaries mapping a list of atoms to their symmetry.
capSymmetries: Vector{Dict{Dict{Int64,Vector{Int64}},String}} Vector of dictionaries mapping list of atoms forming a cap to their symmetry.
L: (Int64) The number of configurations in the file.
N: (Int64) The number of atoms in each configuration.
"""
function vestaFile(configDir,fileName,configurations,classifications,capSymmetries,L,N)
	oldFP = open("blueprint.vesta") # Obtain file pointer to reference .vesta file
	lines = readlines(oldFP) # Get vector of lines of the reference file
	for i in 1:L # For all configurations in the file
		open("$configDir\\visualisation\\$(fileName)_$i.vesta","a") do newFP # Create new .vesta file
			c = 1 # Initialise line counter
			while c <= length(lines) # While have not reached eof
				if (lines[c] == "STRUC") # If have reached structure section of the .vesta file
					println(newFP,"STRUC") # write header to new file
					for j in 1:N # For each atom in the configuration
						# Add the atom's coordinate to the file (aswell as other required values for Vesta)
						println(newFP,"  $j  1  $j  1.0000  $(configurations[i,j,1])  $(configurations[i,j,2])  $(configurations[i,j,3])    1")
						 # Not why this line is neccessary in the Vesta file
						println(newFP,"                    0.000000   0.000000   0.000000  0.00")
					end
					println(newFP,"  0 0 0 0 0 0 0") # Not why this line is neccessary in the Vesta file
					println(newFP,"THERI 1") # Write header to new file
					for j in 1:N # For each atom in the configuration
						println(newFP,"  $j  $j  0.000000") # Not why this line is neccessary in the Vesta file
					end
					c+= 42 # Skip lines to get to next section in lines
				elseif (lines[c] == "SITET") # If have reached colour section of the .vesta file
					println(newFP,"SITET") # Write header to new file
					for j in 1:N # For each atom in the configuration
						atomColour = colourAtom(j,classifications[i],capSymmetries[i]) # Colour the atom according to its identified symmetries
						println(newFP,"  $j  $j  0.8000  $atomColour  76  76  76 204 0") # Add colour to file
					end
					c += 14 # Skip lines to get to next section in lines
				else
					println(newFP,lines[c]) # Print same line from reference file to new file
					c+= 1 # Increment line counter
				end
			end
		end
	end
end

"""
This function determines a colour (RGB values) for an atom given its indentified symmetries.
The colours have the following meanings:
Red -> icosahedral, Orange -> BCC, Green -> FCC, Blue -> HCP, Purple -> OTHER,Black -> No identifiable structure.
The shades of the colour also have meanings. Darker shades mean the atom is part of an identified atom core. Light shades mean the atom is a
'cap' atom. An inbetween shade means the atom is either a capped atom or if core classification failed but the atom still has an identified symmetry.

Inputs:
atomNum: (Int64) The atom number to be colourised based off the order the atoms were listed in the .xyz file.
classifications: Dict{Vector{Int64},String} Dictionary mapping a list of atoms to their symmetry.
capSymmetries: Dict{Dict{Int64,Vector{Int64}},String} Dictionary mapping list of atoms forming a cap to their symmetry.

Outputs:
colour: (String) String representation of the RGB values that make up the desired colour for the input atom.
"""
function colourAtom(atomNum,classification,capSymmetries)
	for key in keys(classification) # For each list of atoms that have a symmetry
		if atomNum in key # If this atom is in that list
			symmetry = get!(classification,key,nothing) # Find what symmetry it has
			if occursin("Core",symmetry) # If symmetry is a core symmetry
				if occursin("ICO",symmetry) # If symmetry is icosahedral
					return "153  0  0" # Return dark red
				elseif occursin("BCC",symmetry)
					return "153  76  0" # Return dark orange
				elseif occursin("FCC",symmetry)
					return "76  153  0" # Return dark green
				elseif occursin("HCP",symmetry)
					return "0  0  153" # Return dark blue
				else # IF OTHER
					return "76  0  153" # Return dark purple
				end
			else # If not a coe symmetry
				if occursin("ICO",symmetry)
					return "255  0  0" # Return red
				elseif occursin("BCC",symmetry)
					return "255  128  0" # Return orange
				elseif occursin("FCC",symmetry)
					return "128  255  0" # Return green
				elseif occursin("HCP",symmetry)
					return "0  0  255" # Return blue
				else # If OTHER
					return "127  0  255" # Return purple
				end
			end
		end
	end

	for dict in keys(capSymmetries) # For each cap symmetry
		symmetry = get!(capSymmetries,dict,nothing) # Get symmetry type
		for key in keys(dict) # For each capping atom
			if atomNum == key # If this atom is the capping atom
				if occursin("ICO",symmetry)
					return "255  102  102" # Return light red
				elseif occursin("BCC",symmetry)
					return "255  178  102" # Return light orange
				elseif occursin("FCC",symmetry)
					return "178  255  102" # Return light green
				elseif occursin("HCP",symmetry)
					return "102  102  255" # Return light blue
				else
					return "178  102  255" # Return light purple
				end
			elseif atomNum in get!(dict,key,nothing) # If atom is a capped atom
				if occursin("ICO",symmetry)
					return "255  0  0" # Return red
				elseif occursin("BCC",symmetry)
					return "255  128  0" # Return orange
				elseif occursin("FCC",symmetry)
					return "128  255  0" # Return green
				elseif occursin("HCP",symmetry)
					return "0  0  255" # Return blue
				else
					return "127  0  255" # Return purple
				end
			end
		end
	end

	return "76  76  76" # IF atom does not have a symmetry, return grey
end

end