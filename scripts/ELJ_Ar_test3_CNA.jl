module ClassifyPBC

using LinearAlgebra, StaticArrays, Glob, Printf, DataFrames, CSV, Graphs

export calculate_cna_profile, process_directory

# Define structure for cubic boundary conditions
struct CubicBC
    box_length::Float64
end

# Apply periodic boundary conditions and compute minimum image distance using MVector
function minimum_image_distance(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::CubicBC)
    d = MVector(a .- b)  # Use MVector to allow modification
    for i in 1:3
        d[i] -= bc.box_length * round(d[i] / bc.box_length)  # Apply PBC via minimum image convention
    end
    return sqrt(sum(d .* d))  # Return the real distance using the minimum image convention
end

# Function to read .dat files and extract atom coordinates and box length
function read_dat(file_path::String)
    open(file_path, "r") do file
        lines = readlines(file)
        
        # Extract the box length from the second line
        box_length_line = lines[2]
        box_length = 0.0
        m = match(r"Box Length: (\d+\.\d+)", box_length_line)
        if m !== nothing
            box_length = parse(Float64, m.captures[1])
        else
            error("Box length not found or invalid in file: $file_path")
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
        return coordinates, box_length
    end
end

# Function to construct an undirected graph representation of a configuration using the minimum image convention for PBC
function adjacencyGraph_PBC(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, EBL::Float64, box_length::Float64)
    bondGraph = SimpleGraph(N)  # Initialize a graph with N atoms as vertices
    bc = CubicBC(box_length)

    # Iterate over all atom pairs to determine bonds using PBC
    for j in 1:N-1
        for i in j+1:N
            dist = minimum_image_distance(configuration[j], configuration[i], bc)
            rCutScaled = EBL * rCut  # Scale rCut by the equilibrium bond length

            if dist < rCutScaled  # Add an edge if distance is less than cutoff
                add_edge!(bondGraph, i, j)
            end
        end
    end

    return bondGraph
end

# Function to implement the full CNA algorithm
function CNA(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, EBL::Float64, box_length::Float64)
    bondGraph = adjacencyGraph_PBC(configuration, N, rCut, EBL, box_length)  # Build the adjacency graph using PBC

    # Initialize dictionaries for total and atomic profiles
    totalProfile = Dict{String, Int}()
    atomicProfile = [Dict{String, Int}() for _ in 1:N]

    for atom1 in 1:N
        neighborhood1, map1 = induced_subgraph(bondGraph, neighbors(bondGraph, atom1))

        for atom2 in map1
            if atom2 > atom1
                neighborhood2, map2 = induced_subgraph(bondGraph, neighbors(bondGraph, atom2))
                commonNeighborhood = bondGraph[intersect(map1, map2)]

                i = size(commonNeighborhood, 1)  # Number of common neighbors
                bondsToProcess = Set(edges(commonNeighborhood))
                j = length(bondsToProcess)  # Number of bonds between common neighbors

                # Compute the maximum bond path length (kMax)
                kMax = 0
                while !isempty(bondsToProcess)
                    atomsToProcess = Set{Int64}()
                    k = 1
                    nextBond = pop!(bondsToProcess)
                    push!(atomsToProcess, nextBond.src)
                    push!(atomsToProcess, nextBond.dst)

                    while !isempty(atomsToProcess)
                        atom = pop!(atomsToProcess)
                        for bond in bondsToProcess
                            if bond.src == atom || bond.dst == atom
                                push!(atomsToProcess, bond.src == atom ? bond.dst : bond.src)
                                pop!(bondsToProcess)
                                k += 1
                            end
                        end
                    end
                    kMax = max(k, kMax)
                end
                
                # Create triplet identifier key for dictionary
                key = "($i,$j,$(Int(kMax)))"
                totalProfile[key] = get(totalProfile, key, 0) + 1
                atomicProfile[atom1][key] = get(atomicProfile[atom1], key, 0) + 1
                atomicProfile[atom2][key] = get(atomicProfile[atom2], key, 0) + 1
            end
        end
    end

    return totalProfile, atomicProfile  # Return the total and atomic CNA profiles
end

# Function to classify symmetries based on the CNA profile
function classifySymmetry(profile::Dict{String, Int})
    n421, n422, n444, n666, n555 = 0, 0, 0, 0, 0

    for bond in keys(profile)
        if bond == "(4,2,1)"
            n421 += profile[bond]
        elseif bond == "(4,2,2)"
            n422 += profile[bond]
        elseif bond == "(4,4,4)"
            n444 += profile[bond]
        elseif bond == "(6,6,6)"
            n666 += profile[bond]
        elseif bond == "(5,5,5)"
            n555 += profile[bond]
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
function calculate_cna_profile(coordinates::Vector{SVector{3, Float64}}, box_length::Float64, rCut::Float64, EBL::Float64)
    N = length(coordinates)

    # Perform CNA with PBC and Argon EBL
    totalProfile, atomicProfile = CNA(coordinates, N, rCut, EBL, box_length)

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
function process_directory(input_dir::String, output_dir::String, rCut::Float64, EBL::Float64)
    files = glob("*.dat", input_dir)
    report_file = joinpath(output_dir, "cna_report.txt")
    summary = DataFrame(FileName = String[], Structure = String[])

    open(report_file, "w") do f
        println(f, "CNA Report for structures in: $input_dir\n")

        # Process each file
        for file in files
            println("\nProcessing file: $file")
            
            # Read coordinates from the dat file
            coordinates, box_length = read_dat(file)
            
            # Perform CNA classification
            calculate_cna_profile(coordinates, box_length, rCut, EBL)
        end
    end
end

end  # End of module ClassifyPBC

# Example usage with provided directories
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized"
output_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized/minimized_results"
rCut = (1 + sqrt(2)) / 2  #* 3.7782 # Cutoff radius in units of EBL
EBL = 3.7782  # Equilibrium bond length for Argon

# Run the directory processing function
ClassifyPBC.process_directory(input_dir, output_dir, rCut, EBL)
