# Author: AJ Tyler
# Date: 24/11/21
module CommonNeighbourAnalysis # This module implements the common neighbour analysis algorithm
export CNA # This function returns the total CNA profile
using Graphs, LongestPaths # Use Graphs package to represent cluster structure and LongestPaths to calculate longest bond path

"""
This function implements the Common Neighbour Analysis (CNA) algorithm and retuns the total CNA profile.

Inputs:
configuration: (Matrix{Float64}) x,y,z positions of the cluster's atoms.
N: (Int64) Number of atoms in the cluster
E: (Float64) Configuration Energy (kJ/mol)
rcutSquared: (Float64) The squared cut-off radius, which determines which atoms are 'bonded'.

Outputs:
profile: (Dictionary) Maps the bond type triplet identifier (i,j,k) to their frequency
"""
function CNA(configuration,N,E,rcutSquared::Float64)
	
	# Create graph representation of bonded atoms, where the atoms are vertices are the vertices and the edges are bonds

	bondGraph = SimpleGraph(N) # Constructs a SimpleGraph object with N atoms(vertices)
	# For all pairs of atoms
	for j in 1:N
		for i in j:N
			diff = configuration[j,:]-configuration[i,:] # calculate vector between atomsObtai
			 # If the squared distance between atoms is less than squared cut-off radius and not same atom
			if (sum(diff.*diff) < rcutSquared && i!=j)
				add_edge!(bondGraph,i,j) # Add bond (edge) to graph
			end
		end
	end

	# Compute triplet identifiers for each pair of bonded atoms (neighbours)

	profile = Dict{String,Int}() # Create Dictionary to store triplet frequencies
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
				j = length(edges(commonNeighbourhood)) # Compute the number of bonds between the common neighbours
				if (i<2)
					k = 0
				else
					k = 1 # Intialise the number of atoms in the longest bond chain between common neighbours
					# Find the longest bond chain between common neighbours assuming there is no cycle
					for atom in 1:i-1 # For each atom the longest chain could start from
						# Find the longest simple path in commonNeighbourhood starting at atom
						search = find_longest_path(commonNeighbourhood,atom,0,log_level = 0)
						(low ,up) = (search.lower_bound,search.upper_bound) # Extract the lower and upper limits 
						if low != up # If the limits disagree
							print("Longest path search did not converge!")
						end

						if low > k # If a longer path was found
							k = low # Update k
						end
					end
				end
				# Find the longest bond cycle between common neighbours (as may be longer than the longest path)
				search = find_longest_cycle(commonNeighbourhood,0,log_level=0)
				(low ,up) = (search.lower_bound,search.upper_bound) # Extract the lower and upper limits 
				if low != up # If the limits disagree
					print("Longest cycle search did not converge!")
				end
				# If the longest cycle is longer than the longest path 
				# Also need to check is greater than 2 since we don't want to count a single bond as a cycle of length 2
				# Note that the longest cycle can be at most one more than longest path and at least two
				if ((low > k) && (low > 2)) 
					k = low # Update k
				end

				key = "($i,$j,$(Int(k)))" # Create triplet identifier key for dictionary
				push!(profile, key => get!(profile,key,0)+1) # Increments triplet frequency by 1
			end
		end
	end
	return profile # Return the total CNA profile
end

end # End of module