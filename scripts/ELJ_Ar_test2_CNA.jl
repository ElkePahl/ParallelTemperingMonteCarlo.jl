module ClassifyPBC

using Graphs, LinearAlgebra, StaticArrays, Printf, Glob, DataFrames

export findSymmetries, distance2_rhombic, distance2_cubic, classify_structure, vestaFile, classifySymmetry, process_directory

# Define structures to hold boundary conditions
struct RhombicBC
    box_length::Float64
    box_height::Float64
end

struct CubicBC
    box_length::Float64
end

# Minimum image distance calculation considering cubic boundary conditions
function distance2_cubic(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::CubicBC)
    b_x = b[1] + bc.box_length * round((a[1] - b[1]) / bc.box_length)
    b_y = b[2] + bc.box_length * round((a[2] - b[2]) / bc.box_length)
    b_z = b[3] + bc.box_length * round((a[3] - b[3]) / bc.box_length)
    return norm(a - SVector(b_x, b_y, b_z))
end


# Function to read .xyz files and extract atom coordinates and box length
function read_xyz(file_path::String)
    open(file_path, "r") do file
        lines = readlines(file)
        num_atoms = parse(Int, lines[1])  # The number of atoms is on the first line
        # Try to extract the box length from the second line if it contains a valid number
        second_line_data = split(lines[2])
        box_length = 0.0  # Default box length in case we don't find it
        for item in second_line_data
            try
                box_length = parse(Float64, item)
                break  # Stop after finding the first valid number (box length)
            catch
                continue  # If the item is not a valid number, continue to the next item
            end
        end

        # Read the atom coordinates
        element_coordinates = []
        for i in 3:num_atoms+2
            split_line = split(lines[i])
            push!(element_coordinates, (split_line[1], parse.(Float64, split_line[2:4])))
        end
        return element_coordinates, box_length
    end
end

# Function to classify the entire structure based on the number of atoms with FCC, HCP, BCC, ICO, and OTHER symmetries.
function classify_structure(classifications::Dict{Int64, String})
    structure_classification = Dict("FCC" => 0, "HCP" => 0, "BCC" => 0, "ICO" => 0, "OTHER" => 0)

    # Increment counters based on atom symmetries
    for symmetry in values(classifications)
        if haskey(structure_classification, symmetry)
            structure_classification[symmetry] += 1
        else
            structure_classification["OTHER"] += 1  # Treat any unknown symmetries as "OTHER"
        end
    end

    return structure_classification
end

# Function to classify the symmetry based on the CNA profile
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

# Function to calculate the CNA profile for each atom and print it
function calculate_cna_profile(coordinates::Vector{SVector{3, Float64}}, box_length::Float64, rCut::Float64)
    num_atoms = length(coordinates)
    profiles = Dict{Int, Dict{String, Int}}()

    # Create the cubic boundary conditions object
    bc = CubicBC(box_length)

    for atom in 1:num_atoms
        # Find neighbors of the current atom within the cutoff distance
        neighbors = []
        for other_atom in 1:num_atoms
            if atom != other_atom && distance2_cubic(coordinates[atom], coordinates[other_atom], bc) < rCut
                push!(neighbors, other_atom)
            end
        end

        # Prepare to count CNA signatures
        cna_profile = Dict{String, Int}("(4,2,1)" => 0, "(4,2,2)" => 0, "(5,5,5)" => 0, "(6,6,6)" => 0, "(4,4,4)" => 0)

        # For each pair of neighbors, count the common neighbors
        for i in 1:length(neighbors)
            for j in i+1:length(neighbors)
                common_neighbors = 0
                ni = neighbors[i]
                nj = neighbors[j]
                for k in neighbors
                    if distance2_cubic(coordinates[ni], coordinates[k], bc) < rCut && distance2_cubic(coordinates[nj], coordinates[k], bc) < rCut
                        common_neighbors += 1
                    end
                end

                # Classify the local bond environment by the number of common neighbors
                if common_neighbors == 4
                    cna_profile["(4,2,1)"] += 1  # FCC-like
                elseif common_neighbors == 6
                    cna_profile["(4,2,2)"] += 1  # HCP-like
                elseif common_neighbors == 5
                    cna_profile["(5,5,5)"] += 1  # ICO-like
                elseif common_neighbors == 8
                    cna_profile["(6,6,6)"] += 1  # BCC-like
                elseif common_neighbors == 12
                    cna_profile["(4,4,4)"] += 1  # High coordination, possible distorted FCC
                end
            end
        end

        # Store the profile for the current atom
        profiles[atom] = cna_profile

        # Print the CNA profile for the current atom
        println("Atom $atom CNA profile: $(cna_profile)")
    end
    return profiles
end


# Function to generate VESTA files for visualization
function vestaFile(output_dir::String, fileName::String, configurations, classifications, rCut, EBL, bc::CubicBC)
    blueprint_path = "/Users/samuelcase/GIT/PTMC/ParallelTemperingMonteCarlo.jl/CNA/blueprint.vesta"
    if !isfile(blueprint_path)
        error("Blueprint VESTA file not found")
    end

    lines = readlines(blueprint_path)

    L = length(configurations)
    N = length(configurations[1])

    for i in 1:L
        output_file = joinpath(output_dir, "$(fileName)_$i.vesta")
        open(output_file, "w") do newFP
            c = 1
            while c <= length(lines)
                if lines[c] == "STRUC"
                    println(newFP, "STRUC")
                    for j in 1:N
                        # Add the atom's coordinate to the file
                        @printf(newFP, "  %d  1  %d  1.0000  %.6f  %.6f  %.6f    1\n", j, j, configurations[i][j]...)
                        println(newFP, "                    0.000000   0.000000   0.000000  0.00")
                    end
                    println(newFP, "  0 0 0 0 0 0 0")
                    println(newFP, "THERI 1")
                    for j in 1:N
                        println(newFP, "  $j  $j  0.000000")
                    end
                    c += 42
                elseif lines[c] == "SBOND"
                    println(newFP, "SBOND")
                    @printf(newFP, "  1     1     1    0.0000    %.5f  0  1  1  0  1  0.250  2.000 127 127 127\n", rCut * EBL)
                    println(newFP, "  0 0 0 0")
                    println(newFP, "SITET")
                    for j in 1:N
                        atomColour = colourAtom(j, classifications[i])
                        println(newFP, "  $j  $j  0.8000  $atomColour  76  76  76 204 0")
                    end
                    c += 17
                else
                    println(newFP, lines[c])
                    c += 1
                end
            end
        end
    end
end

# Helper function to color atoms based on classifications
function colourAtom(atomNum, classification)
    if classification[atomNum] == "FCC"
        return "128 255 0"  # Green for FCC
    elseif classification[atomNum] == "HCP"
        return "0 0 255"  # Blue for HCP
    elseif classification[atomNum] == "BCC"
        return "255 128 0"  # Orange for BCC
    elseif classification[atomNum] == "ICO"
        return "255 0 0"  # Red for ICO
    else
        return "127 0 255"  # Purple for OTHER
    end
end

# Main function to process all files in a directory, perform CNA, and generate the report
function process_directory(input_dir::String, output_dir::String, rCut::Float64, EBL::Float64)
    files = glob("*.xyz", input_dir)
    report_file = joinpath(output_dir, "cna_report.txt")
    summary = DataFrame(FileName = String[], Structure = String[])

    open(report_file, "w") do f
        println(f, "CNA Report for structures in: $input_dir\n")

        # Summary of counts
        total_FCC, total_HCP, total_BCC, total_ICO, total_OTHER = 0, 0, 0, 0, 0

        # Process each file
        for file in files
            # Read coordinates from the xyz file
            element_coordinates, box_length = read_xyz(file)
            coordinates = [SVector{3, Float64}(coord[2]) for coord in element_coordinates]

            # Perform CNA classification
            cna_profile = calculate_cna_profile(coordinates, box_length, rCut)
            classifications = Dict(i => classifySymmetry(cna_profile[i]) for i in 1:length(cna_profile))

            # Generate VESTA file for visualization
            fileName = basename(file)
            fileName = replace(fileName, ".xyz" => "")
            vestaFile(output_dir, fileName, [coordinates], [classifications], rCut, EBL, CubicBC(box_length))

            # Classify structure and update summary
            structure_summary = classify_structure(classifications)
            total_FCC += structure_summary["FCC"]
            total_HCP += structure_summary["HCP"]
            total_BCC += structure_summary["BCC"]
            total_ICO += structure_summary["ICO"]
            total_OTHER += structure_summary["OTHER"]

            # Append summary
            structure_type = "FCC"  # Placeholder for more advanced structure detection
            push!(summary, (basename(file), structure_type))

            println(f, "File: $(basename(file)) - FCC: $(structure_summary["FCC"]), HCP: $(structure_summary["HCP"]), BCC: $(structure_summary["BCC"]), ICO: $(structure_summary["ICO"]), OTHER: $(structure_summary["OTHER"])")
        end

        # Print final totals
        println(f, "\nTotal FCC: $total_FCC")
        println(f, "Total HCP: $total_HCP")
        println(f, "Total BCC: $total_BCC")
        println(f, "Total ICO: $total_ICO")
        println(f, "Total OTHER: $total_OTHER")
    end

    # Print out the summary to the console and save it
    println(summary)
    CSV.write(joinpath(output_dir, "structure_summary.csv"), summary)
end

end  # End of module ClassifyPBC

# Example usage
input_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized"
output_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/minimized/minimized_results"
rCut = (1+sqrt(2))/2 * 3.7782
EBL = 3.7782  # Equilibrim bond length

# Run the directory processing function
ClassifyPBC.process_directory(input_dir, output_dir, rCut, EBL)