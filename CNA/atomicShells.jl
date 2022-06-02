# Author: AJ Tyler
# Date: 24/01/22
module AtomicShells # This module labels the shell number of the atoms in a clutser
export ShellLabelling # This function returns the shell numbers of the atoms in a cluster
using Graphs # This module is used to call dijkstra_shortest_paths()
include("CNA.jl") # This file contains the CommonNeighbourAnalysis module
using .CommonNeighbourAnalysis # This module contains the CNA algorithm

"""
This function returns the shell number of the atoms of a cluster. This is done by first identifying an innermost core of atoms and then
determining the smallest number of bonds required to get to each atom from the core.

Input:
configuration: (Matrix{Float64}) 2D matrix contatining the coordinates of the atoms of the cluster
N: (Int64) The number of atoms in the cluster
B :(Bool) Whether configurations were found under a magnetic field of 0.3 au or not.
rCut: (Float64) Cutoff radius in units of equilibrium bond length.
req: (Float64) Equilibrium bond length (angstroms).

Output:
shells: (Vector{Int64}) Vector containing the shell number of each atom, same ordering as configuration.
bondGraph: (SimpleGraph) Graph representation of the configuration.
"""
function ShellLabelling(configuration,N,B,rCut,req)
    core = innerCore(configuration,N,req) # Determine which atoms form the innermost core
    bondGraph = adjacencyGraph(configuration,N,rCut,B,req) # Construct the graph representation of the cluster
    shells = fill(N,N) # Initialise shell numbers
    for innerAtom in core # For each innermost atom
        # Update the shell number of each atom to be the shortest number of bonds from innerAtom if smaller than current shell number.
        shells = min.(shells,(dijkstra_shortest_paths(bondGraph,innerAtom)).dists)
    end

	return shells, bondGraph # Return the shell numbers
end

"""
This function identifies the innermost core of atoms.

Input:
configuration: (Matrix{Float64}) 2D matrix contatining the coordinates of the atoms of the cluster
N: (Int64) The number of atoms in the cluster
req: (Float64) Equilibrium bond length (angstroms)
coreSize: (Float64) A measure of how large the core is, i.e. how many equilibrium bond lengths away from the centre of mass contains
all the innermost atoms. Default value of 0.85 (0.75 is too small, 1.0 is too large). May need further tuning.

Output:
core: (Vector{Int64}) Vector containg the atom numbers which are part of the inner core.
"""

function innerCore(configuration,N,req,coreSize=0.85)
    CoM = centreOfMass(configuration,N) # Compute the CoM of the cluster
    distances = radialDistances(configuration,N,CoM) # Compute the dsitances between all the atoms and the CoM
    # Return a vector of the atoms which are within 'coreSize' equilibrium bond lengths of the CoM.
    return findall(x-> x<coreSize*req,distances)
end

"""
This function computes the radial distances of each atom from the centre of mass.

Inputs:
configuration: (Matrix{Float64}) x,y,z positions of the cluster's atoms.
N: (Int64) Number of points
CoM: (Float64) Centre of mass of the points

Outputs:
radialDistances: (Vector{Float64}) The ith element of the array is the distance the ith atom in configuration is from the centre of mass
"""
function radialDistances(configuration,N,CoM)
    distances = Vector{Float64}(undef,N) # Preallocate the distances vector
    for i in 1:N # For each of the atoms in the cluster
        diff = configuration[i,:]-CoM[:] # calculate the vector between it and the CoM
		distances[i] = sqrt(sum(diff.*diff)) # Compute the length of the vector
    end

    return distances # Return the distacnes vector
end

"""
This function computes the centre of mass of N points (with equal mass)

Input:
configuration: (Matrix{Float64}) x,y,z positions of the cluster's atoms.
N: (Int64) Number of points

Output:
CoM: (Vector{Float64}) 
"""
function centreOfMass(configuration,N)
    # Initialise the CoM
    CoM = zeros(Float64,3,1)
    for i in 1:N # For each point
        CoM .+= configuration[i] # Add to CoM component wise
    end
    return CoM ./ N # Average over all points
end

end # End of module