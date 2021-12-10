# Author: AJ Tyler
# Date: 24/11/21
module CommonNeighbourAnalysis # This module implements the common neighbour analysis algorithm
export CNA # This function returns the total and atomic CNA profiles
using Graphs # Use Graphs package to represent cluster structure and LongestPaths to calculate longest bond path

"""
This function implements the Common Neighbour Analysis (CNA) algorithm and retuns the total and atomic CNA profile.

Inputs:
configuration: (Matrix{Float64}) x,y,z positions of the cluster's atoms.
N: (Int64) Number of atoms in the cluster
rcutSquared: (Float64) The squared cut-off radius, which determines which atoms are 'bonded'.

Outputs:
totalProfile: (Dictionary) Maps the bond type triplet identifier (i,j,k) to their frequency for the entire cluster
atomicProfile: (Dictionary) Maps the bond type triplet identifier (i,j,k) to their frequency for each atom
"""
function CNA(configuration,N,rcutSquared::Float64)
	
	# Create graph representation of bonded atoms, where the atoms are vertices are the vertices and the edges are bonds

	bondGraph = SimpleGraph(N) # Constructs a SimpleGraph object with N atoms(vertices)
	# For all pairs of atoms
	for j in 1:N-1
		for i in j+1:N
			diff = configuration[j,:]-configuration[i,:] # calculate vector between atoms
			 # If the squared distance between atoms is less than squared cut-off radius
			if (sum(diff.*diff) < rcutSquared)
				add_edge!(bondGraph,i,j) # Add bond (edge) to graph
			end
		end
	end

	# Compute triplet identifiers for each pair of bonded atoms (neighbours)

	totalProfile = Dict{String,Int}() # Create Dictionary to store triplet frequencies
	atomicProfile = [Dict{String,Int}() for i in 1:N] # Create vector of dictionaries to store triplet frequencies
	for atom1 in 1:N # For all atoms
		# neighbourhood1: The subgraph which only contains the bonds between the neighbours of atom1
		# map1: map1[i] is the vertex number in bondgraph that vertex i in neighbourhood1 corresponds to
		neighbourhood1, map1 = induced_subgraph(bondGraph, neighbors(bondGraph,atom1))
		for atom2 in map1 # For each of atom1's neighbours
			if atom2 > atom1 # Check not double counting bonds
				# neighbourhood2: The subgraph which only contains the bonds between the neighbours of atom2
				# map2: map2[i] is the vertex number in bondgraph that vertex i in neighbourhood2 corresponds to
				neighbourhood2, map2 = induced_subgraph(bondGraph, neighbors(bondGraph,atom2))

				commonNeighbourhood = bondGraph[intersect(map1,map2)] # Obtain the subgraph with the common neighbours of atom1 and atom2
				i = size(commonNeighbourhood,1) # Compute the number of common neighbours
				bondsToProcess = Set(edges(commonNeighbourhood))
				j = length(bondsToProcess) # Compute the number of bonds between the common neighbours
				kMax = 0 # Initialise longest bond path length
				while (!isempty(bondsToProcess)) # While still bonds left to process
					#thisCluster = Graph(i) # Useful for debugging to see longest path construction
					atomsToProcess = Set{Int64}() # Initialise list of atoms left to process in current longest path construction
					k = 1 # Initialise current longest path length
					nextBond = pop!(bondsToProcess) # Remove a bond to process and start processing it
					#add_edge!(thisCluster,nextBond) # Useful for debugging to see longest path construction
					# Add both atoms of the bond to be processed
					push!(atomsToProcess,nextBond.src)
					push!(atomsToProcess,nextBond.dst)
					while (!isempty(atomsToProcess)) # While have atoms to process
						atom = pop!(atomsToProcess) # Remove an atom to process and start processing it
						for bond in bondsToProcess # For all the bonds left to process
							if (bond.src == atom) # If the source of the bond was the atom to process
								push!(atomsToProcess,bond.dst) # Add it's bonded atom to be processed
							elseif (bond.dst == atom) # If the source of the bond was the atom to process
								push!(atomsToProcess,bond.src) # Add it's bonded atom to be processed
							else # If atom being processed is not in the current bond
								continue # Skip to next bond
							end
							#add_edge!(thisCluster,pop!(bondsToProcess,bond)) Useful for debugging to see longest path construction
							pop!(bondsToProcess) # Finished processing the bond
							k += 1 # Increment the longest bond path length
						end
					end
					kMax = max(k,kMax)
				end
				
				key = "($i,$j,$(Int(kMax)))" # Create triplet identifier key for dictionary
				push!(totalProfile, key => get!(totalProfile,key,0)+1) # Increments total CNA triplet frequency by 1
				push!(atomicProfile[atom1], key => get!(atomicProfile[atom1],key,0)+1) # Increments atom1 triplet frequency by 1
				push!(atomicProfile[atom2], key => get!(atomicProfile[atom2],key,0)+1) # Increments atom2 triplet frequency by 1
			end
		end
	end
	return totalProfile, atomicProfile # Return the total and atomic CNA profiles
end

end # End of module