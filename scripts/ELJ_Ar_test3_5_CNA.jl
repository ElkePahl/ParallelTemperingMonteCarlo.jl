module ClassifyPBC

using LinearAlgebra, StaticArrays, Glob, Printf, DataFrames, CSV, Graphs

export calculate_cna_profile, process_directory

# Define structure for cubic boundary conditions
struct CubicBC
    box_length::Float64
end

# Corrected minimum_image_distance function using the standard minimum image convention
function minimum_image_distance(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::CubicBC)
    d = MVector(a .- b)  # Use MVector to allow modification
    for i in 1:3
        if d[i] > bc.box_length / 2
            d[i] -= bc.box_length
        elseif d[i] < -bc.box_length / 2
            d[i] += bc.box_length
        end
    end
    return sqrt(sum(d .* d))  # Return the real distance using the minimum image convention
end

# Function to read .dat files and extract atom coordinates and box length
function read_dat(file_path::String)
    open(file_path, "r") do file
        lines = readlines(file)
        
        # Extract the box length from the second line
        box_length_line = lines[2]
        println("Second line (box length line): $box_length_line")  # Debugging

        # Updated regex to match integer and floating-point numbers
        m = match(r"Box Length: (\d+\.?\d*)", box_length_line)
        if m !== nothing
            box_length = parse(Float64, m.captures[1])
        else
            error("Box length not found or invalid in file: $file_path")
        end

        if box_length == 0.0
            error("Box length is zero. Check the data file and the regex pattern.")
        end

        # Print the extracted box length
        println("Box length from file $file_path: $box_length")

        # Read the atom coordinates and explicitly convert them to SVector{3, Float64}
        coordinates = Vector{SVector{3, Float64}}()
        for i in 3:length(lines)
            split_line = split(lines[i])
            coord = SVector{3, Float64}(parse.(Float64, split_line[2:4])...)
            push!(coordinates, coord)
        end

        # Debugging: Print the number of atoms and a sample of coordinates
        println("Number of atoms read: $(length(coordinates))")
        println("Sample coordinates: ", coordinates[1:min(5, length(coordinates))])

        return coordinates, box_length
    end
end

# Function to construct an undirected graph representation of a configuration using the minimum image convention for PBC
function adjacencyGraph_PBC(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, box_length::Float64)
    bondGraph = SimpleGraph(N)  # Initialize a graph with N atoms as vertices
    bc = CubicBC(box_length)

    edge_count = 0  # Counter for the number of edges added

    # Iterate over all atom pairs to determine bonds using PBC
    for j in 1:N-1
        for i in j+1:N
            dist = minimum_image_distance(configuration[j], configuration[i], bc)
            # Debugging: Print the distance between atoms (uncomment if needed)
            # println("Distance between atom $j and atom $i: $dist")
            if dist < rCut  # Use rCut directly
                add_edge!(bondGraph, i, j)
                edge_count += 1
                # Debugging: Print when an edge is added (uncomment if needed)
                # println("Edge added between atom $j and atom $i")
            end
        end
    end

    # Print the number of bonds (edges) in the graph
    println("Number of bonds (edges) in the graph: $(ne(bondGraph)) (Counted edges: $edge_count)")

    return bondGraph
end

# Function to implement the full CNA algorithm
function CNA(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, box_length::Float64)
    bondGraph = adjacencyGraph_PBC(configuration, N, rCut, box_length)  # Build the adjacency graph using PBC

    # Print information about the graph
    println("Processing CNA for configuration with $N atoms and $(ne(bondGraph)) bonds.")

    # Initialize dictionaries for total and atomic profiles
    totalProfile = Dict{String, Int}()
    atomicProfile = [Dict{String, Int}() for _ in 1:N]

    for atom1 in 1:N
        neighbors1 = neighbors(bondGraph, atom1)
        # Debugging: Print the neighbors of atom1
        println("Atom $atom1 has neighbors: $neighbors1")

        for atom2 in neighbors1
            if atom2 > atom1
                neighbors2 = neighbors(bondGraph, atom2)
                common_neighbors = intersect(neighbors1, neighbors2)
                i = length(common_neighbors)

                # Debugging: Print the neighbors of atom2 and common neighbors
                println("Atom $atom2 has neighbors: $neighbors2")
                println("Common neighbors between atom $atom1 and atom $atom2: $common_neighbors (i = $i)")

                # Get the induced subgraph and mapping
                if !isempty(common_neighbors)
                    commonNeighborhood, mapping = induced_subgraph(bondGraph, common_neighbors)
                    j = ne(commonNeighborhood)  # Number of bonds between common neighbors

                    # Compute kMax (maximum bond path length among common neighbors)
                    components = connected_components(commonNeighborhood)
                    kMax = maximum(map(length, components))

                    # Debugging: Print details about bonds and kMax
                    println("Number of bonds between common neighbors (j): $j")
                    println("Maximum bond path length among common neighbors (kMax): $kMax")
                else
                    j = 0
                    kMax = 0
                    println("No common neighbors between atom $atom1 and atom $atom2.")
                end

                # Create triplet identifier key for dictionary
                key = "($i,$j,$(Int(kMax)))"
                totalProfile[key] = get(totalProfile, key, 0) + 1
                atomicProfile[atom1][key] = get(atomicProfile[atom1], key, 0) + 1
                atomicProfile[atom2][key] = get(atomicProfile[atom2], key, 0) + 1

                # Debugging: Print when counts are incremented
                println("Incremented counts for key $key")
            end
        end
    end

    # Debugging: Print the total profile
    println("Total CNA profile: $totalProfile")

    return totalProfile, atomicProfile  # Return the total and atomic CNA profiles
end

# Function to classify symmetries based on the CNA profile
function classifySymmetry(profile::Dict{String, Int})
    n421, n422, n444, n666, n555 = 0, 0, 0, 0, 0

    for bond in keys(profile)
        count = profile[bond]
        # Debugging: Print bond types and counts
        # println("Bond type $bond has count $count")

        if bond == "(4,2,1)"
            n421 += count
        elseif bond == "(4,2,2)"
            n422 += count
        elseif bond == "(4,4,4)"
            n444 += count
        elseif bond == "(6,6,6)"
            n666 += count
        elseif bond == "(5,5,5)"
            n555 += count
        end
    end

    numBonds = n421 + n422 + n444 + n666 + n555
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

# Function to calculate the CNA profile for each atom and classify it
function calculate_cna_profile(coordinates::Vector{SVector{3, Float64}}, box_length::Float64, rCut::Float64)
    N = length(coordinates)
    println("Using cutoff distance rCut = $rCut Angstroms")  # Debugging

    # Perform CNA with PBC
    totalProfile, atomicProfile = CNA(coordinates, N, rCut, box_length)

    # Classify symmetries
    for atom in 1:N
        profile = atomicProfile[atom]
        classification = classifySymmetry(profile)
        
        # Use `get` with a default value of 0 for each triplet count
        triplet_counts = [
            get(profile, "(4,2,1)", 0),
            get(profile, "(4,2,2)", 0),
            get(profile, "(5,5,5)", 0)
        ]
        println("Atom $atom CNA profile: $(triplet_counts) -> Symmetry: $classification")
    end

    return atomicProfile
end

# Function to process all files in a directory, perform CNA, and generate the report
function process_directory(input_dir::String, output_dir::String, rCut::Float64)
    files = glob("*.dat", input_dir)
    report_file = joinpath(output_dir, "cna_report.txt")
    summary = DataFrame(FileName = String[], Structure = String[])

    open(report_file, "w") do f
        println(f, "CNA Report for structures in: $input_dir\n")

        # Process only the first file for testing
        if isempty(files)
            println("No .dat files found in the directory: $input_dir")
            return
        end

        for file in files[1:1]
            println("\nProcessing file: $file")
            
            # Read coordinates from the dat file
            coordinates, box_length = read_dat(file)
            
            # Perform CNA classification
            calculate_cna_profile(coordinates, box_length, rCut)
        end
    end
end

end  # End of module ClassifyPBC

# Example usage with provided directories
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized"
output_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized/minimized_results"
EBL = 3.7782  # Equilibrium bond length for Argon (Angstroms)

# Adjusted cutoff distance for testing
# You can try different values for rCut to see which one works best for your system
#rCut = 6.0  # Increased cutoff distance for testing
rCut = 10.0 #2.5 * (1 + sqrt(2)) / 2 * EBL

# Create output directory if it doesn't exist
if !isdir(output_dir)
    mkpath(output_dir)
end

println("Using cutoff distance rCut = $rCut Angstroms")  # Debugging

# Run the directory processing function
ClassifyPBC.process_directory(input_dir, output_dir, rCut)
