module ClassifyPBC

using LinearAlgebra, StaticArrays, Glob, Printf, DataFrames, CSV, Graphs

export calculate_cna_profile, process_directory

# Define structure for cubic boundary conditions
struct CubicBC
    box_length::Float64
end

# Define the distance squared function between two points
distance2(a::SVector{3, Float64}, b::SVector{3, Float64}) = sum((a .- b) .^ 2)

# Implement the distance2 function with periodic boundary conditions
function distance2(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::CubicBC)
    # Adjust coordinates of b to its nearest image to a
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_length * round((a[3] - b[3]) / bc.box_length)
    b_adjusted = SVector{3, Float64}(b_x, b_y, b_z)
    # Return the squared distance between a and the adjusted b
    return distance2(a, b_adjusted)
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

# Function to construct an undirected graph representation of a configuration using PBC
function adjacencyGraph(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, B::Bool, EBL::Float64, box_length::Float64)
    bondGraph = SimpleGraph(N) # Constructs a SimpleGraph object with N atoms (vertices)
    bc = CubicBC(box_length)

    # For all pairs of atoms
    for j in 1:N-1
        for i in j+1:N
            dist2 = distance2(configuration[j], configuration[i], bc) # Compute squared distance with PBC
            dist = sqrt(dist2)
            if B # If strong magnetic field present
                diff = configuration[j] .- configuration[i] # calculate vector between atoms
                diff -= bc.box_length .* round.(diff ./ bc.box_length) # Apply minimum image convention
                norm = sqrt(sum(diff .* diff))
                theta = acos(diff[3] / norm) # Compute angle bond makes with B field (z-axis) in radians
                # Define EBL(theta) function if needed
                # For now, we'll assume EBL is a constant
                rCutScaled = EBL * rCut # Adjust as necessary
            else # If no magnetic field
                rCutScaled = EBL * rCut # Use standard equilibrium bond length
            end
            # If the distance between atoms is less than the cut-off radius
            if dist < rCutScaled
                add_edge!(bondGraph, i, j) # Add bond (edge) to graph
            end
        end
    end

    return bondGraph # Return graph
end

# Implement your CNA function
function CNA(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, B::Bool, EBL::Float64, box_length::Float64)
    bondGraph = adjacencyGraph(configuration, N, rCut, B, EBL, box_length) # Compute graph representation of configuration.
    # Compute triplet identifiers for each pair of bonded atoms (neighbors)
    totalProfile = Dict{String, Int}() # Create Dictionary to store triplet frequencies
    atomicProfile = [Dict{String, Int}() for i in 1:N] # Create vector of dictionaries to store triplet frequencies

    for atom1 in 1:N # For all atoms
        # Get the neighbors of atom1
        neighbors1 = neighbors(bondGraph, atom1)
        # neighbourhood1: The subgraph which only contains the bonds between the neighbors of atom1
        # map1: map1[i] is the vertex number in bondGraph that vertex i in neighbourhood1 corresponds to
        neighbourhood1, map1 = induced_subgraph(bondGraph, neighbors1)
        for idx in 1:length(map1) # For each of atom1's neighbors
            atom2 = map1[idx]
            if atom2 > atom1 # Check not double counting bonds
                # Get the neighbors of atom2
                neighbors2 = neighbors(bondGraph, atom2)
                # neighbourhood2: The subgraph which only contains the bonds between the neighbors of atom2
                # map2: map2[i] is the vertex number in bondGraph that vertex i in neighbourhood2 corresponds to
                neighbourhood2, map2 = induced_subgraph(bondGraph, neighbors2)
                # Obtain the common neighbors
                common_indices = intersect(map1, map2)
                if isempty(common_indices)
                    i = 0 # Number of common neighbors
                    j = 0 # Number of bonds between common neighbors
                    kMax = 0 # Maximum bond path length among common neighbors
                else
                    # Obtain the subgraph with the common neighbors of atom1 and atom2
                    commonNeighbourhood, common_map = induced_subgraph(bondGraph, common_indices)
                    i = nv(commonNeighbourhood) # Compute the number of common neighbors

                    bondsToProcess = Set(edges(commonNeighbourhood))
                    j = ne(commonNeighbourhood) # Compute the number of bonds between the common neighbors

                    kMax = 0 # Initialize longest bond path length

                    while (!isempty(bondsToProcess)) # While still bonds left to process
                        atomsToProcess = Set{Int64}() # Initialize list of atoms left to process in current longest path construction
                        k = 1 # Initialize current longest bond path length
                        nextBond = pop!(bondsToProcess) # Remove a bond to process and start processing it
                        # Add both atoms of the bond to be processed
                        push!(atomsToProcess, nextBond.src)
                        push!(atomsToProcess, nextBond.dst)
                        processedAtoms = Set{Int64}()
                        while (!isempty(atomsToProcess)) # While have atoms to process
                            atom = pop!(atomsToProcess) # Remove an atom to process and start processing it
                            if atom in processedAtoms
                                continue
                            end
                            push!(processedAtoms, atom)
                            bondsToRemove = Set{Edge{Int64}}()
                            for bond in bondsToProcess # For all the bonds left to process
                                if (bond.src == atom)
                                    push!(atomsToProcess, bond.dst)
                                    push!(bondsToRemove, bond)
                                    k += 1 # Increment the longest bond path length
                                elseif (bond.dst == atom)
                                    push!(atomsToProcess, bond.src)
                                    push!(bondsToRemove, bond)
                                    k += 1 # Increment the longest bond path length
                                end
                            end
                            # Remove processed bonds
                            for bond in bondsToRemove
                                delete!(bondsToProcess, bond)
                            end
                        end
                        kMax = max(k, kMax)
                    end
                end
                key = "($i,$j,$(Int(kMax)))" # Create triplet identifier key for dictionary
                totalProfile[key] = get(totalProfile, key, 0) + 1 # Increment total CNA triplet frequency by 1
                atomicProfile[atom1][key] = get(atomicProfile[atom1], key, 0) + 1 # Increment atom1 triplet frequency by 1
                atomicProfile[atom2][key] = get(atomicProfile[atom2], key, 0) + 1 # Increment atom2 triplet frequency by 1
            end
        end
    end
    return totalProfile, atomicProfile # Return the total and atomic CNA profiles
end

# Implement the generalized classification function
function classifyStructure(profile::Dict{String, Int}, N::Int)
    # Actual counts from the profile
    n421 = get(profile, "(4,2,1)", 0)
    n422 = get(profile, "(4,2,2)", 0)
    n444 = get(profile, "(4,4,4)", 0)
    n666 = get(profile, "(6,6,6)", 0)
    n555 = get(profile, "(5,5,5)", 0)
    total_triplets = sum(values(profile))

    actual_counts = Dict(
        "(4,2,1)" => n421,
        "(4,2,2)" => n422,
        "(4,4,4)" => n444,
        "(6,6,6)" => n666,
        "(5,5,5)" => n555
    )

    # Expected counts for each structure
    expected_counts = Dict(
        "FCC" => Dict("(4,2,1)" => 6 * N),
        "HCP" => Dict("(4,2,1)" => 3 * N, "(4,2,2)" => 3 * N),
        "BCC" => Dict("(6,6,6)" => 4 * N, "(4,4,4)" => 3 * N),
        "ICO" => Dict("(5,5,5)" => 6 * N)
    )

    # Initialize variables
    best_match = "OTHER"
    best_similarity = 0.0

    # Compare actual counts to expected counts for each structure
    for (structure, expected) in expected_counts
        # Calculate similarity as the fraction of expected triplets found
        similarity = 0.0
        for (triplet, expected_count) in expected
            actual_count = get(actual_counts, triplet, 0)
            # Avoid division by zero
            if expected_count > 0
                similarity += (min(actual_count, expected_count) / expected_count)
            end
        end
        similarity /= length(expected)  # Average similarity over triplets

        # Update classification if this structure has a higher similarity
        if similarity > best_similarity
            best_similarity = similarity
            best_match = structure
        end
    end

    return best_match
end

# Function to calculate the CNA profile and classify the structure
function calculate_cna_profile(coordinates::Vector{SVector{3, Float64}}, box_length::Float64, rCut::Float64, B::Bool, EBL::Float64)
    N = length(coordinates)
    println("Using cutoff distance rCut = $rCut Angstroms")

    # Perform CNA with PBC
    totalProfile, atomicProfile = CNA(coordinates, N, rCut, B, EBL, box_length)

    # Classify the whole structure
    classification = classifyStructure(totalProfile, N)  # Pass N to the function

    # Print the total CNA profile and classification
    println("Total CNA profile: $totalProfile")
    println("Structure classification: $classification")

    return totalProfile, classification, atomicProfile
end

# Function to process all files in a directory, perform CNA, and generate the report
function process_directory(input_dir::String, output_dir::String, rCut::Float64, B::Bool, EBL::Float64)
    files = glob("*.dat", input_dir)
    report_file = joinpath(output_dir, "cna_report.txt")
    summary = DataFrame(FileName = String[], Structure = String[])

    open(report_file, "w") do f
        println(f, "CNA Report for structures in: $input_dir\n")

        if isempty(files)
            println("No .dat files found in the directory: $input_dir")
            return
        end

        for file in files
            println("\nProcessing file: $file")
            
            # Read coordinates from the dat file
            coordinates, box_length = read_dat(file)
            
            # Perform CNA classification
            totalProfile, classification, atomicProfile = calculate_cna_profile(coordinates, box_length, rCut, B, EBL)

            # Write to the report file
            println(f, "File: $file")
            println(f, "Total CNA profile: $totalProfile")
            println(f, "Structure classification: $classification\n")

            # Append to summary DataFrame
            push!(summary, (FileName = basename(file), Structure = classification))
        end
    end

    # Optionally, save the summary DataFrame to a CSV file
    CSV.write(joinpath(output_dir, "cna_summary.csv"), summary)
end

end  # End of module ClassifyPBC

# Example usage with provided directories
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized"
output_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized/minimized_results"
EBL = 3.7782  # Equilibrium bond length for Argon (Angstroms)
B = false     # Assuming no magnetic field for simplicity

# Adjusted cutoff distance
rCut = 1.88 * (1 + sqrt(2)) / 2   # Use a multiple of EBL for the cutoff

# Ensure rCut is Float64
rCut = Float64(rCut)

# Create output directory if it doesn't exist
if !isdir(output_dir)
    mkpath(output_dir)
end

println("Using cutoff distance rCut = $rCut Angstroms")  # Debugging

# Run the directory processing function
ClassifyPBC.process_directory(input_dir, output_dir, rCut, B, EBL)