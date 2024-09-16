module ClassifyPBC

using LinearAlgebra, StaticArrays, Glob, Printf, DataFrames, CSV, Graphs

export calculate_cna_profile, process_directory

# Define structure for cubic boundary conditions
struct CubicBC
    box_length::Float64
end

# Function to calculate the distance between two atoms using the minimum image convention
function distance(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::CubicBC)
    # Use the minimum image convention for periodic boundary conditions
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_length * round((a[3] - b[3]) / bc.box_length)

    # Return the direct distance, without extra vector operations
    dx = a[1] - b_x
    dy = a[2] - b_y
    dz = a[3] - b_z

    return sqrt(dx^2 + dy^2 + dz^2)  # Return actual distance
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

# Function to construct an undirected graph representation of a configuration with PBC
function adjacencyGraph(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, EBL::Float64, bc::CubicBC)
    bondGraph = SimpleGraph(N) # Constructs a SimpleGraph object with N atoms (vertices)
    
    # Iterate over all pairs of atoms to check if they are bonded
    for j in 1:N-1
        for i in j+1:N
            diff = configuration[j,:] - configuration[i,:] # Calculate vector between atoms
            dist = minimum_image_distance(configuration[j], configuration[i], bc)  # Use PBC
            
            # Calculate the bond length, check if within cutoff
            rCutScaled = EBL * rCut
            if dist < rCutScaled
                add_edge!(bondGraph, i, j)  # Add bond (edge) to graph if distance < cutoff
            end
        end
    end
    
    return bondGraph  # Return the graph representation of the configuration
end

# Function to implement the full CNA algorithm
# Function to implement the full CNA algorithm
function CNA(configuration::Vector{SVector{3, Float64}}, N::Int, rCut::Float64, EBL::Float64, box_length::Float64)
    bc = CubicBC(box_length)
    bondGraph = adjacencyGraph(configuration, N, rCut, EBL, bc)  # Build the adjacency graph using PBC

    # Initialize dictionaries for the total and atomic CNA profiles
    totalProfile = Dict{String, Int}()
    atomicProfile = [Dict{String, Int}() for _ in 1:N]

    for atom1 in 1:N
        neighborhood1, map1 = induced_subgraph(bondGraph, neighbors(bondGraph, atom1))

        for atom2 in map1
            if atom2 > atom1
                neighborhood2, map2 = induced_subgraph(bondGraph, neighbors(bondGraph, atom2))
                commonNeighborhood = bondGraph[intersect(map1, map2)]

                i = size(commonNeighborhood, 1)
                bondsToProcess = Set(edges(commonNeighborhood))
                j = length(bondsToProcess)

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
                
                key = "($i,$j,$(Int(kMax)))"
                # Increment the total CNA triplet frequency by 1 using push!
                totalProfile[key] = get!(totalProfile, key, 0) + 1
                
                # Increment the atomic CNA triplet frequency for both atom1 and atom2
                push!(atomicProfile[atom1], key => get!(atomicProfile[atom1], key, 0) + 1)
                push!(atomicProfile[atom2], key => get!(atomicProfile[atom2], key, 0) + 1)
            end
        end
    end

    return totalProfile, atomicProfile
end

# Function to calculate the CNA profile for each atom and classify it
function calculate_cna_profile(coordinates::Vector{SVector{3, Float64}}, box_length::Float64, rCut::Float64, EBL::Float64)
    N = length(coordinates)

    # Perform CNA with PBC
    totalProfile, atomicProfile = CNA(coordinates, N, rCut, EBL, box_length)

    # Classify symmetries
    for atom in 1:N
        profile = atomicProfile[atom]
        triplet_counts = [
            get(profile, "(4,2,1)", 0),
            get(profile, "(4,2,2)", 0),
            get(profile, "(5,5,5)", 0)
        ]
        println("Atom $atom CNA profile: $(triplet_counts)")
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
rCut = (1 + sqrt(2)) / 2 #* 3.7782
#rCut = 1+sqrt(2)/2 * 3.7782
EBL = 3.7782  # Equilibrium bond length for Ar2

# Run the directory processing function
ClassifyPBC.process_directory(input_dir, output_dir, rCut, EBL)
