# Author: AJ Tyler
# Date: 24/01/22
module CommonNeighbourAnalysis # This module implements the common neighbour analysis algorithm
export CNA # This function returns the total and atomic CNA profiles
export EBLength
export adjacencyGraph,adjacencyGraph_B # This function returns a graphical representation of a configuration.
using Graphs # Use Graphs package to represent cluster structure and LongestPaths to calculate longest bond path

"""
This function implements the Common Neighbour Analysis (CNA) algorithm and retuns the total and atomic CNA profile.

Inputs:
configuration: (Matrix{Float64}) x,y,z positions of the cluster's atoms.
N: (Int64) Number of atoms in the cluster
rCut: (Float64) Cutoff radius in units of equilibrium bond length.
B: (Bool) Whether configurations were found under a magnetic field of 0.3 au or not.
EBL: (Float64) Equilibrium bond length (angstroms) for MP2 data for Neon2.

Outputs:
totalProfile: (Dictionary) Maps the bond type triplet identifier (i,j,k) to their frequency for the entire cluster
atomicProfile: (Dictionary) Maps the bond type triplet identifier (i,j,k) to their frequency for each atom
"""
function CNA(configuration,N,rCut,B,EBL)
	if B
		bondGraph = adjacencyGraph_B(configuration,N,rCut)
	else
		bondGraph = adjacencyGraph(configuration,N,rCut,B,EBL) # Compute graph representation of configuration.
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

x=[0.1947679907,        0.3306365642,        1.7069272101,1.1592174250,       -1.1514615100,       -0.6254746298,1.4851406793,       -0.0676273830,        0.9223060046,-0.1498046416,        1.4425168343,       -0.9785553065,1.4277261305,        0.3530265376,       -0.9475378022,-0.6881246261,       -1.5737014419,       -0.3328844168,-1.4277352637,       -0.3530034531,        0.9475270683,0.6881257085,        1.5736904826,        0.3329032458,-1.1592204530,        1.1514535263,        0.6254777879,0.1498035273,       -1.4424985165,        0.9785685322,-1.4851196066,        0.0676193562,       -0.9223231092,-0.7057028384,        0.6207073550,       -1.4756523155,-0.8745359533,        0.4648140463,        1.4422103492,-0.9742077067,       -0.8837261792,       -1.1536019836,-0.1947765396,       -0.3306358487,       -1.7069179299,0.3759933035,       -1.7072373106,       -0.0694439840,-1.7124296000,        0.3336352522,        0.1307959669,0.9143159284,        1.3089975397,       -0.7151210582,-0.3759920260,        1.7072300336,        0.0694634263,1.7124281219,       -0.3336312342,       -0.1308207313,-0.9143187026,       -1.3089785474,        0.7151290509,0.9742085109,        0.8837023041,        1.1536069633,0.7057104439,       -0.6206907639,        1.4756502961,0.8745319670,       -0.4648127187,       -1.4422106957,-1.1954804901,       -0.6171923123,       -0.1021449363,0.0917363053,       -1.0144887859,       -0.8848410405,0.9276243144,       -0.8836123311,        0.4234140820,1.1954744473,        0.6171883800,        0.1021399054,-0.9276176774,        0.8836123556,       -0.4234173533,-0.3595942315,       -0.4863167551,        1.2061133825,0.3595891589,        0.4863295901,       -1.2061152849,-0.0917352078,        1.0144694592,        0.8848400639,0.6410702480,       -0.1978633363,       -0.3898095439,-0.4162942817,       -0.0651798741,       -0.6515502084,0.1334019604,        0.7474406294,       -0.1600033264,-0.6410732823,        0.1978593218,        0.3898012337,0.4162968444,        0.0651733322,        0.6515490914, -0.1333998872,       -0.7474445984,        0.1600019961]

N=38
config=zeros(N,3)
for i=1:N
	for j=1:3
		config[i,j]=x[3i-(3-j)]
	end
end

println(config)

function EBLength(theta)
	if theta > pi/2 # If theta is greater than pi/2 radians
		theta = pi-theta # Change theta to be within [0,pi/2] radians for interpolation purposes
	end
	return (0.188*cos(2*theta)+0.01073*cos(4*theta)-0.00269*cos(8*theta)+3.06621)/3.06621 # Return prefitted harmonic function
end

"""
This function constructs an undirected graph representation of a configuration. The nodes are the atoms which have an edge between them if
they are bonded. Atoms are bonded if the distance between them is less than rCut*EBL.

Inputs:
configuration: (Matrix{Float64}) x,y,z positions of the cluster's atoms.
N: (Int64) Number of atoms in the cluster
rCut: (Float64) Cutoff radius in units of equilibrium bond length.
B: (Bool) Whether configurations were found under a magnetic field of 0.3 au or not.
EBL: (Float64) Equilibrium bond length (angstroms) for MP2 data for Neon2.

Output:
bondGraph: (SimpleGraph) Graph representation of the configuration.
"""
function adjacencyGraph(configuration,N,rCut,B,EBL)
	# Create graph representation of bonded atoms, where the atoms are vertices are the vertices and the edges are bonds
	bondGraph = SimpleGraph(N) # Constructs a SimpleGraph object with N atoms(vertices)
	# For all pairs of atoms
	for j in 1:N-1
		for i in j+1:N
			diff = configuration[j,:]-configuration[i,:] # calculate vector between atoms
			norm = sqrt(sum(diff.*diff)) # Compute length of bond
			if B # If strong magnetic field present
				theta = acos(diff[3]/norm) # Compute angle bond makes with B field (z-axis) in radians
				rCutScaled = EBLength(theta)*rCut # Compute rCut is terms of equilibrium bond length in the theta direction
			else # If no magnetic field
				rCutScaled = EBL*rCut # Use standard equilibrium Bond Length.
			end
			 # If the distance between atoms is less the cut-off radius
			if (norm < rCutScaled)
				add_edge!(bondGraph,i,j) # Add bond (edge) to graph
			end
		end
	end

	return bondGraph # Return graph
end


function adjacencyGraph_B(configuration,N,rCut)
	# Create graph representation of bonded atoms, where the atoms are vertices are the vertices and the edges are bonds
	bondGraph = SimpleGraph(N) # Constructs a SimpleGraph object with N atoms(vertices)
	# For all pairs of atoms
	for j in 1:N-1
		for i in j+1:N
			diff = configuration[j,:]-configuration[i,:] # calculate vector between atoms
			norm = sqrt(sum(diff.*diff)) # Compute length of bond
			theta = acos(diff[3]/norm) # Compute angle bond makes with B field (z-axis) in radians
			rCutScaled = EBLength(theta)*rCut # Compute rCut is terms of equilibrium bond length in the theta direction
			 # If the distance between atoms is less the cut-off radius
			if (norm < rCutScaled)
				add_edge!(bondGraph,i,j) # Add bond (edge) to graph
			end
		end
	end

	return bondGraph # Return graph
end

"""
This function computes the equilibrium bond length in angstroms. If a 0.3 au magentic field is present, then bond angle with the B field
direction is used in a harmonic function to output the altered equlibrium bond length due to magnetic effects.

Input:
theta: (Float64) Angle the bond makes with the magnetic field in radians.

Output:
EBL: (Float64) The equilibrium bond length at angle theta.
"""
#println(CNA(config, N, 1.3549, false, 1.122))

end # End of module