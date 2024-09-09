module ClassifyPBC

using Graphs, LinearAlgebra, StaticArrays, Printf

export findSymmetries, distance2_rhombic, distance2_cubic, classify_structure, generate_fcc_structure, vestaFile, RhombicBC, CubicBC

# Define structures to hold boundary conditions
struct RhombicBC
    box_length::Float64
    box_height::Float64
end

struct CubicBC
    box_length::Float64
end

# Distance calculation considering rhombic boundary conditions
function distance2_rhombic(a::SVector{3,Float64}, b::SVector{3,Float64}, bc::RhombicBC)
    b_y = b[2] + (sqrt(3)/2 * bc.box_length) * round((a[2] - b[2]) / (sqrt(3)/2 * bc.box_length))
    b_x = b[1] - b[2]/sqrt(3) + bc.box_length * round(((a[1] - b[1]) - 1/sqrt(3) * (a[2] - b[2])) / bc.box_length) + 1/sqrt(3) * b_y
    b_z = b[3] + bc.box_height * round((a[3] - b[3]) / bc.box_height)
    return norm(a - SVector(b_x, b_y, b_z))
end

# Distance calculation considering cubic boundary conditions
function distance2_cubic(a::SVector{3,Float64}, b::SVector{3,Float64}, bc::CubicBC)
    delta = a .- b
    for i in 1:3
        if abs(delta[i]) > bc.box_length / 2
            delta[i] -= sign(delta[i]) * bc.box_length
        end
    end
    return norm(delta)
end

# Function to generate an FCC structure
function generate_fcc_structure(n_atoms::Int)
    r_start = 3.7782  # minimum radius between atoms
    L_start = 2 * (r_start^2 / 2)^0.5  # Distance between adjacent atoms along the axes

    Cell_Repeats = round(cbrt(n_atoms / 4))  # Number of times the unit cell is repeated

    if Cell_Repeats^3 * 4 != n_atoms
        error("Number of atoms not correct for FCC")
    end

    pos = Vector{SVector{3, Float64}}()

    # Generate simple cubic atoms
    for i in 0:(Cell_Repeats - 1)
        x = i * L_start
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start
                push!(pos, SVector(x, y, z))
            end
        end
    end

    # Generate face-centered atoms
    for i in 0:(Cell_Repeats - 1)
        x = i * L_start + L_start / 2
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start + L_start / 2
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start
                push!(pos, SVector(x, y, z))
            end
        end
    end

    for i in 0:(Cell_Repeats - 1)
        x = i * L_start
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start + L_start / 2
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start + L_start / 2
                push!(pos, SVector(x, y, z))
            end
        end
    end

    for i in 0:(Cell_Repeats - 1)
        x = i * L_start + L_start / 2
        for j in 0:(Cell_Repeats - 1)
            y = j * L_start
            for k in 0:(Cell_Repeats - 1)
                z = k * L_start + L_start / 2
                push!(pos, SVector(x, y, z))
            end
        end
    end

    # Center the configuration at the origin
    center = SVector(Cell_Repeats * L_start / 2, Cell_Repeats * L_start / 2, Cell_Repeats * L_start / 2)
    pos = [p - center for p in pos]

    # Convert to Bohr
    AtoBohr = 1.8897259886
    pos = [p * AtoBohr for p in pos]

    length(pos) == n_atoms || error("Number of atoms and positions not the same - check starting config")

    # Define boundary conditions
    box_length = Cell_Repeats * L_start * AtoBohr
    bc = CubicBC(box_length)

    return pos, bc
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
This function classifies the entire structure based on the number of atoms with FCC, HCP, BCC, ICO, and OTHER symmetries.

Inputs:
    classifications: Dict{Vector{Int64}, String} - Dictionary of atom indices mapped to their symmetries.

Outputs:
    structure_classification: Dict{String, Int} - Dictionary mapping each symmetry type to the number of atoms classified with that symmetry.
"""
function classify_structure(classifications::Dict{Vector{Int64}, String})
    structure_classification = Dict("FCC" => 0, "HCP" => 0, "BCC" => 0, "ICO" => 0, "OTHER" => 0)
    
    for symmetry in values(classifications)
        structure_classification[symmetry] += 1
    end
    
    return structure_classification
end

# Function to generate a VESTA file
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

function colourAtom(atomNum, classification)
    for key in keys(classification)
        if atomNum in key
            symmetry = classification[key]
            if symmetry == "FCC"
                return "128  255  0"  # Green
            elseif symmetry == "HCP"
                return "0  0  255"  # Blue
            elseif symmetry == "ICO"
                return "255  0  0"  # Red
            elseif symmetry == "BCC"
                return "255  128  0"  # Orange
            else
                return "127  0  255"  # Purple
            end
        end
    end
    return "76  76  76"  # Grey for unclassified atoms
end

# Main function to generate FCC structure, classify, and generate VESTA file
function main_fcc_vesta(n_atoms::Int, output_dir::String, rCut::Float64, EBL::Float64)
    # Generate FCC structure
    pos, bc = generate_fcc_structure(n_atoms)

    # Create dummy classifications (all FCC for now)
    classifications = Dict{Vector{Int64}, String}([i] => "FCC" for i in 1:n_atoms)

    # Prepare configurations
    configurations = [pos]

    # Generate VESTA file
    fileName = "fcc_structure"
    vestaFile(output_dir, fileName, configurations, [classifications], rCut, EBL, bc)
    println("VESTA file generated in directory: $output_dir")
end

end  # End of module ClassifyPBC

# Example usage
n_atoms = 108  # Must be a multiple of 4 for FCC structure
output_dir = "/Users/samuelcase/Downloads"  # Change this to your desired output directory
rCut = 1+sqrt(2)/2 * 3.7782  # Example value, adjust as needed
EBL = 3.7782  # Equilibrium bond length in Angstroms for Neon2, adjust as needed

# Run the main function
ClassifyPBC.main_fcc_vesta(n_atoms, output_dir, rCut, EBL)
