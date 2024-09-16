module ClassifyPBC

using LinearAlgebra, StaticArrays, Glob, Printf, DataFrames, CSV, Graphs

export calculate_cna_profile, process_directory

# Define structure for cubic boundary conditions
struct CubicBC
    box_length::Float64
end

# Define the distance squared function between two points
distance21(a,b) = (a-b)⋅(a-b)

# Implement the distance2 function with periodic boundary conditions
# function distance2(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::CubicBC)
#     # Adjust coordinates of b to its nearest image to a
#     b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
#     b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
#     b_z = b[3] + bc.box_length * round((a[3] - b[3]) / bc.box_length)
#     b_adjusted = SVector{3, Float64}(b_x, b_y, b_z)
#     # Return the squared distance between a and the adjusted b
#     return distance2(a, b_adjusted)
# end

function distance2(a,b,bc::CubicBC)
    b_x=b[1]+bc.box_length*round((a[1]-b[1])/bc.box_length)
    b_y=b[2]+bc.box_length*round((a[2]-b[2])/bc.box_length)
    b_z=b[3]+bc.box_length*round((a[3]-b[3])/bc.box_length)
    return distance21(a,SVector(b_x,b_y,b_z))
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
function adjacencyGraph_PBC(configuration::Vector{SVector{3, Float64}}, N::Int, rCut2::Float64, box_length::Float64)
    bondGraph = SimpleGraph(N)
    bc = CubicBC(box_length)

    edge_count = 0

    # Iterate over all atom pairs to determine bonds using PBC
    for j in 1:N-1
        for i in j+1:N
            dist2 = distance2(configuration[j], configuration[i], bc)  # Get squared distance
            if dist2 < rCut2
                add_edge!(bondGraph, i, j)
                edge_count += 1
            end
        end
    end

    println("Number of bonds (edges) in the graph: $(ne(bondGraph)) (Counted edges: $edge_count)")

    return bondGraph
end

# Function to implement the full CNA algorithm
function CNA(configuration::Vector{SVector{3, Float64}}, N::Int, rCut2::Float64, box_length::Float64)
    bondGraph = adjacencyGraph_PBC(configuration, N, rCut2, box_length)  # Build the adjacency graph using PBC

    println("Processing CNA for configuration with $N atoms and $(ne(bondGraph)) bonds.")

    # Initialize the total profile dictionary
    totalProfile = Dict{String, Int}()

    for atom1 in 1:N
        neighbors1 = neighbors(bondGraph, atom1)
        for atom2 in neighbors1
            if atom2 > atom1
                neighbors2 = neighbors(bondGraph, atom2)
                common_neighbors = intersect(neighbors1, neighbors2)
                i = length(common_neighbors)

                if !isempty(common_neighbors)
                    commonNeighborhood, mapping = induced_subgraph(bondGraph, common_neighbors)
                    j = ne(commonNeighborhood)  # Number of bonds between common neighbors

                    # Compute kMax (maximum bond path length among common neighbors)
                    components = connected_components(commonNeighborhood)
                    kMax = maximum(map(length, components))
                else
                    j = 0
                    kMax = 0
                end

                key = "($i,$j,$(Int(kMax)))"
                totalProfile[key] = get(totalProfile, key, 0) + 1
            end
        end
    end

    println("Total CNA profile: $totalProfile")

    return totalProfile  # Return only the total CNA profile
end


# Function to classify symmetries based on the CNA profile
function classifySymmetry(profile::Dict{String, Int})
    n421, n422, n444, n666, n555 = 0, 0, 0, 0, 0

    for bond in keys(profile)
        count = profile[bond]
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

# Function to calculate the CNA profile and classify the structure
function calculate_cna_profile(coordinates::Vector{SVector{3, Float64}}, box_length::Float64, rCut::Float64)
    N = length(coordinates)
    println("Using cutoff distance rCut = $rCut Angstroms")

    rCut2 = rCut  # Use squared cutoff distance for efficiency

    # Perform CNA with PBC
    totalProfile = CNA(coordinates, N, rCut2, box_length)

    # Classify the whole structure
    classification = classifySymmetry(totalProfile)

    # Print the total CNA profile and classification
    println("Total CNA profile: $totalProfile")
    println("Structure classification: $classification")

    return totalProfile, classification
end

# Function to process all files in a directory, perform CNA, and generate the report
function process_directory(input_dir::String, output_dir::String, rCut::Float64)
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
            totalProfile, classification = calculate_cna_profile(coordinates, box_length, rCut)

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

# Adjusted cutoff distance
rCut = (1 + sqrt(2)) / 2 * EBL  # Approximately 4.594 Å

# Ensure rCut is Float64
rCut = Float64(rCut)

# Create output directory if it doesn't exist
if !isdir(output_dir)
    mkpath(output_dir)
end

println("Using cutoff distance rCut = $rCut Angstroms")  # Debugging

# Run the directory processing function
ClassifyPBC.process_directory(input_dir, output_dir, rCut)
