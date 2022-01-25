# Author: AJ Tyler
# Date: 24/01/22
module Classify # This module contains functions which classify configurations based on their symmetries.
using Graphs # This module allows graphical representation of configurations.
export findSymmetries # This function identifies symmetries in a configuration.

"""
This function identifies symmetries in a configuration. First the core atoms of the configuration are tried to be identified. If a core
is found, the caps are attempted to be identified. If no core was found, then symmetric components throughout the configurations are
investigated.

Inputs:
shells: (Vector{Vector{Int64}}) A vector of shell numbers of each atom in each configuration.
atomicProfiles: Array{Dict{String,Int}} An array containing the CNA profile of each configuration.
bondGraphs: Vector{SimpleGraph} A vector of graphical representation of each configuration. Empty if comparing.
L: (Int64) The number of configurations in the file.

Outputs:
classifications: Vector{Dict{Vector{Int64},String}} Vector of dictionaries mapping a list of atoms to their symmetry.
capSymmetries: Vector{Dict{Dict{Int64,Vector{Int64}},String}} Vector of dictionaries mapping list of atoms forming a cap to their symmetry.
"""
function findSymmetries(shells,atomicProfiles,bondGraphs,L)
	classifications = Vector{Dict{Vector{Int64},String}}(undef,L) # Initialise vector of classifications
	capSymmetries = Vector{Dict{Dict{Int64,Vector{Int64}},String}}(undef,L) # Initialise vector of capSymmetries
	for i in 1:L # For each of the configurations in the file
		hasCore, classifications[i] = classifyCore(shells[i],atomicProfiles[1,i,:],bondGraphs[i]) # Try classify cluster core
		if (hasCore) # Check if cluster has identifiable core
			capSymmetries[i] = classifyCaps(shells[i],atomicProfiles[1,i,:],bondGraphs[i]) # Identify caps
		else
			classifications[i] = classifySymmetricComponents(atomicProfiles[1,i,:],bondGraphs[i]) # Classify symmetric components
			capSymmetries[i] = Dict{Dict{Int64,Vector{Int64}},String}() # No caps identifiable
		end
	end

	return classifications,capSymmetries # Return symmetries
end

"""
This function determines the symmetry contained in a CNA profile.

Inputs:
profile: (Vector{Dict{String,Int}}) 1D vector containing atomic CNA profiles.

Ouputs:
classification (String) Detected symmetry
"""
function classifySymmetry(profile)
	n421,n422,n444,n666,n555 = 0,0,0,0,0 # Initialise counts of CNA profiles
	for bond in keys(profile) # For each CNA profile of the current atom
		if bond == "(4,2,1)" # If has the (4,2,1) profile
			n421 += get!(profile,bond,0) # Add to the counter
		elseif bond == "(4,2,2)" # etc
			n422 += get!(profile,bond,0)
		elseif bond == "(4,4,4)"
			n444 += get!(profile,bond,0)
		elseif bond == "(6,6,6)"
			n666 += get!(profile,bond,0)
		elseif bond == "(5,5,5)"
			n555 += get!(profile,bond,0)
		end
	end

	numBonds = sum([n421,n422,n444,n666,n555]) # Compute how many bonds the atom has
	if numBonds == 12 # If has coordination number of 12
		if n421 == 12 # If has 12 (4,2,1) bonds
			return "FCC" # The atom's bonds have FCC symmetry
		elseif (n421 == 6 && n422 == 6) # If has 6 (4,2,1) and 6 (4,2,2) bonds
			return "HCP" # The atom's bonds have HCP symmetry
		elseif n555 == 12 # If has 12 (5,5,5) bonds
			return "ICO" # The atom's bonds have icosahedral symmetry
		else
			return "OTHER"
		end
	elseif (numBonds == 14 && n444 == 6 && n666 ==8) # If has coordination number of 14 and 6 (4,4,4) and 8 (6,6,6) bonds
		return "BCC" # The atom's bonds have BCC symmetry
	else # If doesn't have any of the above symmetries
		return "OTHER" # Then classify as OTHER
	end
end

"""
This function first determines which shell numbers contain the core and then checks if all those atoms have the same CNA profile. If they
do, then a core has been identified. Otherwise, non-cental symmetric components are attempted to be identified instead.

Inputs:
shells: (Vector{Int64}) Shell numbers of each atom in the configuration.
atomicProfiles: Dict{String,Int} A dictionary containing the CNA profile of the configuration.
bondGraph: SimpleGraph A graphical representation of the configuration.

Outputs:
hasCore: (Bool) Whether or not a core has been identified.
classification: (Dict) Dictionary mapping a list of atoms to their symmetry.
"""
function classifyCore(shells,atomicProfiles,bondGraph)
	coreAtoms = findall(x -> x<=floor(0.2*(maximum(shells))),shells) # Determine which atoms may form a core
	coreProfile = atomicProfiles[coreAtoms] # Obtain CNA profiles of the core atoms
	for profile in coreProfile # For each of the core atom CNA profiles
		if (profile != coreProfile[1]) # If the CNA profiles arn't all the same
			return false, Dict{Vector{Int64},String}() # Then core cannot be identified.
		end
	end
	classification = Dict(coreAtoms => classifySymmetry(coreProfile[1])*"(Core)") # Construct core classification
	
	return true,classification # Found core and return classification
end

"""
This function first finds the atoms in the outer and second most outer shells. If there are more atoms in the second outer shell, then the
outermost shell atoms are cap atoms and all their neighbours are capped atoms.

Inputs:
shells: (Vector{Int64}) Shell numbers of each atom in the configuration.
atomicProfiles: Dict{String,Int} A dictionary containing the CNA profile of the configuration.
bondGraph: SimpleGraph A graphical representation of the configuration.

capSymmetries: (Dict) Dictionary mapping list of atoms forming a cap to their symmetry. The keys are themselves dictionaries, mapping
the cap atom to a vector of the atoms it caps.
"""
function classifyCaps(shells,atomicProfiles,bondGraph)
	capSymmetries = Dict{Dict{Int64,Vector{Int64}},String}() # Initialise dictionary of cap symmetries
	maxShell = maximum(shells) # Find outer shell number
	outerShell = findall(x -> x == maxShell,shells) # Find outer shell atoms
	secondOuterShell = findall(x -> x == maxShell-1,shells) # Find 2nd most outer shell atoms
	if length(outerShell) < length(secondOuterShell) # If there are more atoms in the second outer shell than the outer shell
		for capAtom in outerShell # Each of the outer shell atoms are cap atoms
			cappedAtoms = neighbors(bondGraph,capAtom) # The cap atom caps all of its neighbours
			for cappedAtom in cappedAtoms # For each of the capped atoms
				symmetry = classifySymmetry(atomicProfiles[cappedAtom]) # Identify the capped atom's symmetry
				if symmetry != "OTHER" # If the symmetry is not OTHER
					push!(capSymmetries,Dict(capAtom=>cappedAtoms)=>symmetry) # Then the cap has a known symmetry
					break
				end
			end
			push!(capSymmetries,Dict(capAtom=>cappedAtoms)=>"OTHER") # The cap has an OTHER symmetry
		end
	end

	return capSymmetries # Return cap symmetries
end

"""
This function classifies components of the bondgraph which have the same symmetry via a breadth-first-search.

Inputs:
atomicProfiles: Dict{String,Int} A dictionary containing the CNA profile of the configuration.
bondGraph: SimpleGraph A graphical representation of the configuration.

Outputs:
Classification: (Dict) Dictionary mapping a list of atoms to their symmetry.
"""
function classifySymmetricComponents(atomicProfiles,bondGraph)
	atomsToProcess = Set([1]) # Start search from first atom
	atomsProcessed = Set{Int64}() # Initialise set of processed atoms
	classification = Dict{Vector{Int64},String}() # Initialise found symmetries
	while !isempty(atomsToProcess) # While have atoms left to process
		atom = pop!(atomsToProcess) # Pop off atom to process
		push!(atomsProcessed,atom) # Add atom to list of processed atoms
		symmetry = classifySymmetry(atomicProfiles[atom]) # Classify symmetry of atom
		if (symmetry != "OTHER") # If symmetry is not OTHER
			key = Vector{Int64}([atom]) # Initialise list of atoms that have the same symmetry
			# Obtain list of neighbours of the atom that have not already been searched
			neighboursToSearch = setdiff(Set(neighbors(bondGraph,atom)),atomsProcessed) 
			while (!isempty(neighboursToSearch)) # While have neighbours left to search
				neighbour = pop!(neighboursToSearch) # Pop off atom to process
				neighbourSymmetry = classifySymmetry(atomicProfiles[neighbour]) # Classify the neighbour's symmetry
				if (neighbourSymmetry == symmetry) # If neighbour has the same symmetry
					push!(key,neighbour) # Add neighbour to key
					# Add the neighbour's neighbours (that haven't already been processed) to search
					neighboursToSearch = union(neighboursToSearch,setdiff(Set(neighbors(bondGraph,neighbour)),atomsProcessed))
					push!(atomsProcessed,neighbour) # Add neighbour to processed atoms
					delete!(atomsToProcess,neighbour) # Remove neighbour from atoms to process
				else # If neighbour has different symmetry
					push!(atomsToProcess,neighbour) # Add neighbour to atomsToProcess
				end
			end

			push!(classification, key => symmetry) # Add component to classifications
		else # If atom's symmetry is OTHER
			# Add the atom's neighbours (that haven't already been processed) to search
			atomsToProcess= union(atomsToProcess,setdiff(Set(neighbors(bondGraph,atom)),atomsProcessed))
		end
	end
	
	return classification # Return classifications
end

end