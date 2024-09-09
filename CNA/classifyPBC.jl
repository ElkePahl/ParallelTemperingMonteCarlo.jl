module ClassifyPBC

using Graphs

export findSymmetries, periodic_distance

# Periodic distance calculation function
function periodic_distance(point1::Vector{Float64}, point2::Vector{Float64}, box_length::Float64)
    delta = point1 .- point2
    for i in 1:length(delta)
        if abs(delta[i]) > box_length / 2
            delta[i] -= sign(delta[i]) * box_length
        end
    end
    return sqrt(sum(delta .^ 2))
end

# Use this distance instead
# function distance2(a,b,bc::RhombicBC)
#     b_y=b[2]+(3^0.5/2*bc.box_length)*round((a[2]-b[2])/(3^0.5/2*bc.box_length))
#     b_x=b[1]-b[2]/3^0.5 +
#     bc.box_length*round(((a[1]-b[1])-1/3^0.5*(a[2]-b[2]))/bc.box_length) +
#     1/3^0.5*b_y
#     b_z=b[3]+bc.box_height*round((a[3]-b[3])/bc.box_height)
#     return distance2(a,SVector(b_x,b_y,b_z))
# end
"""
This function identifies symmetries in a configuration with periodic boundary conditions. Only core atoms are considered as there is no surface.

Inputs:
    atomicProfiles: Array{Dict{String,Int}} - An array containing the CNA profile of each configuration.
    bondGraphs: Vector{SimpleGraph} - A vector of graphical representation of each configuration. Empty if comparing.
    coordinates: Vector{Vector{Vector{Float64}}} - Coordinates of each atom in each configuration.
    box_length: Float64 - Length of the simulation box.
    L: Int64 - The number of configurations in the file.

Outputs:
    classifications: Vector{Dict{Vector{Int64},String}} - Vector of dictionaries mapping a list of atoms to their symmetry.
"""
function findSymmetries(atomicProfiles, bondGraphs, coordinates, box_length, L)
    classifications = Vector{Dict{Vector{Int64},String}}(undef, L)
    for i in 1:L
        classifications[i] = classifySymmetries(atomicProfiles[i], bondGraphs[i])
    end
    return classifications
end

"""
This function determines the symmetry contained in a CNA profile.

Inputs:
    profile: Dict{String,Int} - A dictionary containing the CNA profile of the configuration.

Outputs:
    classification: String - Detected symmetry.
"""
function classifySymmetry(profile)
    n421, n422, n444, n666, n555 = 0, 0, 0, 0, 0
    for bond in keys(profile)
        if bond == "(4,2,1)"
            n421 += get!(profile, bond, 0)
        elseif bond == "(4,2,2)"
            n422 += get!(profile, bond, 0)
        elseif bond == "(4,4,4)"
            n444 += get!(profile, bond, 0)
        elseif bond == "(6,6,6)"
            n666 += get!(profile, bond, 0)
        elseif bond == "(5,5,5)"
            n555 += get!(profile, bond, 0)
        end
    end

    numBonds = sum([n421, n422, n444, n666, n555])
    if numBonds == 12
        if n421 == 12
            return "FCC"
        elseif n421 == 6 && n422 == 6
            return "HCP"
        elseif n555 == 12
            return "ICO"
        else
            return "OTHER"
        end
    elseif numBonds == 14 && n444 == 6 && n666 == 8
        return "BCC"
    else
        return "OTHER"
    end
end

"""
This function classifies symmetries directly using CNA profiles.

Inputs:
    atomicProfiles: Dict{String,Int} - A dictionary containing the CNA profile of the configuration.
    bondGraph: SimpleGraph - A graphical representation of the configuration.

Outputs:
    classification: Dict - Dictionary mapping a list of atoms to their symmetry.
"""
function classifySymmetries(atomicProfiles, bondGraph)
    classifications = Dict{Vector{Int64},String}()
    for atom in 1:nv(bondGraph)
        profile = atomicProfiles[atom]
        symmetry = classifySymmetry(profile)
        push!(classifications, [atom] => symmetry)
    end
    return classifications
end

end

using DataFrames
using Graphs
using .ClassifyPBC: periodic_distance

# Function to read the custom format file and extract coordinates and box length
function read_custom_file(file_path::String)
    open(file_path, "r") do file
        lines = filter(x -> !isempty(x), readlines(file))  # Filter out empty lines
        first_line = lines[1]
        box_length_str = match(r"Box Length: ([0-9.]+)", first_line)
        if box_length_str == nothing
            error("Box length not found in the file")
        end
        box_length = parse(Float64, box_length_str.captures[1])
        element_coordinates = []
        for line in lines[2:end]
            split_line = split(line)
            push!(element_coordinates, (split_line[1], parse.(Float64, split_line[2:4])))
        end
        return element_coordinates, box_length
    end
end

# Helper function to unzip elements and coordinates
function unzip(element_coordinates)
    elements = [x[1] for x in element_coordinates]
    coordinates = [x[2] for x in element_coordinates]
    return elements, coordinates
end

# Function to generate a bond graph based on a cutoff distance
function generate_bond_graph(coordinates, box_length, cutoff_distance)
    num_atoms = length(coordinates)
    g = SimpleGraph(num_atoms)
    for i in 1:num_atoms
        for j in i+1:num_atoms
            if periodic_distance(coordinates[i], coordinates[j], box_length) < cutoff_distance
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

# Function to calculate CNA profile for each atom
function calculate_cna_profile(bond_graph, coordinates, box_length)
    profiles = []
    for atom in 1:nv(bond_graph)
        atom_neighbors = neighbors(bond_graph, atom)
        profile = Dict{String, Int}()
        for neighbor in atom_neighbors
            neighbor_neighbors = neighbors(bond_graph, neighbor)
            common_neighbors = intersect(atom_neighbors, neighbor_neighbors)
            cna_key = "($(length(common_neighbors)), $(length(atom_neighbors)), $(length(neighbor_neighbors)))"
            profile[cna_key] = get!(profile, cna_key, 0) + 1
        end
        push!(profiles, profile)
    end
    return profiles
end

# Main function to process the custom format file and classify symmetries
function main(input_dir::String, cutoff_distance::Float64)
    # Load the custom format file
    element_coordinates, box_length = read_custom_file(input_dir)
    
    # Extract coordinates and elements
    elements, coordinates = unzip(element_coordinates)
    
    # Convert coordinates to a Vector of Vectors
    coordinates_matrix = [Vector{Float64}(coord) for coord in coordinates]
    
    # Generate bond graph
    bond_graph = generate_bond_graph(coordinates_matrix, box_length, cutoff_distance)
    
    # Calculate atomic profiles using CNA
    atomic_profiles = calculate_cna_profile(bond_graph, coordinates_matrix, box_length)
    
    # Classify symmetries
    classifications = ClassifyPBC.classifySymmetries(atomic_profiles, bond_graph)
    
    # Output the classifications
    println("Classifications:")
    for (key, value) in classifications
        println("Atom(s) $key: $value")
    end
end

# Define input directory and cutoff distance
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized/minimized_configurations_cycle_state_1.xyz"
cutoff_distance = 2.7700 * 1.8897259886 # 1+sqrt(2)/2 * equilibrium distance <- bottom of curve

# Run the main function
main(input_dir, cutoff_distance)
