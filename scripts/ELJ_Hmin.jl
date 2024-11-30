using Glob
using Optim
using ParallelTemperingMonteCarlo  # Assuming this module provides ELJPotentialEven and dimer_energy

# Specify the directories for reading and writing files
input_dir = "/nesi/nobackup/uoa02731/sam/Ar/Rhombic/150/100/Configs"
output_dir_minimized = "/nesi/nobackup/uoa02731/sam/Ar/Rhombic/150/100/Configs/Emin/minimized"

# Structures for handling periodic boundary conditions
struct EMCubicBC
    box_length::Float64
end

struct EMRhombicBC
    box_width::Float64
    box_height::Float64
end

# Function to read multiple structures from a .dat file
function read_structures(file_path::String)
    lines = readlines(file_path)  # Read all lines from the file
    structures = []
    idx = 1
    total_lines = length(lines)
    while idx <= total_lines       
       # Look for the header line starting with "# Cycle:"
        if startswith(lines[idx], "# Cycle:")
            # Extract cycle number and box parameters
            header = lines[idx]
            idx += 1  # Move to the next line after the header

            # Parse the header to extract cycle number, box parameters, and temperature
            # Example header: "# Cycle: 10000, Box Width: 20.794930958234172, Box Height: 16.97911180140579, Temperature: 150.0"
            m_cubic = match(r"# Cycle: (\d+), Box Length: ([\d\.]+), Temperature: ([\d\.]+)", header)
            m_rhombic = match(r"# Cycle: (\d+), Box Width: ([\d\.]+), Box Height: ([\d\.]+), Temperature: ([\d\.]+)", header)
            if m_cubic !== nothing
                cycle_index = parse(Int, m_cubic.captures[1])
                box_length = parse(Float64, m_cubic.captures[2])
                temp = parse(Float64, m_cubic.captures[3])
                bc = EMCubicBC(box_length)
            elseif m_rhombic !== nothing
                cycle_index = parse(Int, m_rhombic.captures[1])
                box_width = parse(Float64, m_rhombic.captures[2])
                box_height = parse(Float64, m_rhombic.captures[3])
                temp = parse(Float64, m_rhombic.captures[4])
                bc = EMRhombicBC(box_width, box_height)
            else
                error("Header line format not recognized: $header")
            end

            # Read atom lines until the next header or end of file
            atoms = []
            while idx <= total_lines && !startswith(lines[idx], "#")
                line = lines[idx]
                atom_info = split(line)
                if length(atom_info) == 4
                    atom, x, y, z = atom_info
                    push!(atoms, (atom, parse(Float64, x), parse(Float64, y), parse(Float64, z)))
                elseif isempty(line)
                    # Skip empty lines
                else
                    error("Incorrect atom format in file: $file_path at line $idx")
                end
                idx += 1
            end

            # Append the structure to the list
            push!(structures, (cycle_index, atoms, bc))
        else
            idx += 1  # Skip lines that are not headers
        end
    end
    return structures
end

# Function to wrap positions within the simulation box (for cubic boxes)
function wrap_positions!(pos::Vector{Float64}, bc::EMCubicBC)
    for i in 1:3
        pos[i] -= bc.box_length * floor(pos[i] / bc.box_length)
    end
end

# Function to wrap positions within the simulation box (for rhombic boxes)
function wrap_positions!(pos::Vector{Float64}, bc::EMRhombicBC)
    # Implement wrapping for rhombic boxes if necessary
    # For simplicity, we'll assume positions are already within the box
    return
end

# Flatten and rebuild atom coordinates for optimization
flatten_atoms(atoms) = [coord for atom in atoms for coord in atom[2:end]]

function rebuild_atoms(flat_coords, atoms, bc)
    new_atoms = []
    for i in 1:length(atoms)
        pos = [flat_coords[3i-2], flat_coords[3i-1], flat_coords[3i]]
        wrap_positions!(pos, bc)
        push!(new_atoms, (atoms[i][1], pos...))
    end
    return new_atoms
end

# Calculate the squared distance between two atoms under periodic boundary conditions
function distance2(a::Vector{Float64}, b::Vector{Float64}, bc::EMCubicBC)
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]
    dx -= bc.box_length * round(dx / bc.box_length)
    dy -= bc.box_length * round(dy / bc.box_length)
    dz -= bc.box_length * round(dz / bc.box_length)
    return dx^2 + dy^2 + dz^2
end

# Distance function for rhombic boxes
function distance2(a::Vector{Float64}, b::Vector{Float64}, bc::EMRhombicBC)
    sqrt3 = sqrt(3.0)
    delta_y = a[2] - b[2]
    delta_z = a[3] - b[3]
    b_y = b[2] + (sqrt3 / 2 * bc.box_width) * round(delta_y / (sqrt3 / 2 * bc.box_width))
    delta_x = a[1] - b[1]
    delta_xy = delta_x - (1 / sqrt3) * delta_y
    b_x = b[1] + bc.box_width * round(delta_xy / bc.box_width) - (1 / sqrt3) * (a[2] - b_y)
    b_z = b[3] + bc.box_height * round(delta_z / bc.box_height)
    adjusted_b = [b_x, b_y, b_z]
    dx = a[1] - adjusted_b[1]
    dy = a[2] - adjusted_b[2]
    dz = a[3] - adjusted_b[3]
    return dx^2 + dy^2 + dz^2
end

# Write minimized structures to .xyz files
function write_xyz(atoms, file_path, comment="")
    open(file_path, "w") do f
        println(f, length(atoms))  # Write the number of atoms
        println(f, comment)  # Write a comment line
        for (atom, x, y, z) in atoms
            println(f, "$atom $x $y $z")
        end
    end
end

# Write minimized structures to .dat files (including box parameters)
function write_dat(atoms, bc, file_path)
    open(file_path, "w") do f
        println(f, "# Minimized structure")  # Add a header line
        if bc isa EMCubicBC
            println(f, "Box Length: $(bc.box_length)")  # Write the box length
        elseif bc isa EMRhombicBC
            println(f, "Box Width: $(bc.box_width), Box Height: $(bc.box_height)")
        else
            error("Unknown boundary condition type")
        end
        for (atom, x, y, z) in atoms
            println(f, "$atom $x $y $z")
        end
    end
end

# Define the ELJ potential (assuming ELJPotentialEven and dimer_energy are defined in ParallelTemperingMonteCarlo)
function lj_elj(x, N, pot, bc, pressure)
    E = 0.0
    for i = 1:N-1
        for j = i+1:N
            a = [x[3i-2], x[3i-1], x[3i]]
            b = [x[3j-2], x[3j-1], x[3j]]
            d2 = distance2(a, b, bc)
            E += dimer_energy(pot, d2)
        end
    end
    # Calculate volume based on the cell type
    if bc isa EMCubicBC
        volume = bc.box_length^3
    elseif bc isa EMRhombicBC
        # Volume = (sqrt(3)/2) * box_width^2 * box_height
        volume = (sqrt(3)/2) * bc.box_width^2 * bc.box_height
    else
        error("Unknown boundary condition type")
    end
    # Return enthalpy: E + P * V
    return E + pressure * volume
end

# Perform optimization using the ELJ potential and show energy at each iteration
function optimize_structure(atoms, coeff, bc, pressure)
    N = length(atoms)
    pot = ELJPotentialEven(coeff)  # Using the ELJ potential
    x = flatten_atoms(atoms)
    # Define the wrapper function for optimization
    lj_wrapper(x) = lj_elj(x, N, pot, bc, pressure)

    # Set options to show trace and set the gradient tolerance
    options = Optim.Options(g_tol=5e-4, show_trace=true)
    
    # First iteration of optimization
    println("Starting first optimization iteration...")
    opt_result1 = optimize(lj_wrapper, x, ConjugateGradient(), options)
    println("First optimization complete. Enthalpy: ", opt_result1.minimum)

    # Rebuild atoms from the first optimization
    minimized_atoms1 = rebuild_atoms(opt_result1.minimizer, atoms, bc)
    
    # Second iteration of optimization
    println("Starting second optimization iteration...")
    x2 = flatten_atoms(minimized_atoms1)
    opt_result2 = optimize(lj_wrapper, x2, ConjugateGradient(), options)
    println("Second optimization complete. Enthalpy: ", opt_result2.minimum)

    # Rebuild atoms from the second optimization
    minimized_atoms2 = rebuild_atoms(opt_result2.minimizer, atoms, bc)
    return minimized_atoms2
end

# Main loop for processing .dat files and writing minimized structures
coeff = [-123.635101619510, 21262.8963716972, -3239750.64086661, 189367623.844691, -4304257347.72069, 35315085074.3605]

# Define the pressure value (you need to specify this)
pressure = 1.0  # Replace with the desired pressure value

# Ensure the output directory exists
if !isdir(output_dir_minimized)
    mkpath(output_dir_minimized)
end

for file_path in glob("*.dat", input_dir)
    structures = read_structures(file_path)
    base_name = replace(basename(file_path), ".dat" => "")  # Get the base name without extension
    for (cycle_index, atoms, bc) in structures
        minimized_atoms = optimize_structure(atoms, coeff, bc, pressure)

        # Generate filenames using the base name and cycle index
        minimized_xyz_path = joinpath(output_dir_minimized, "minimized_$(base_name)_cycle$(cycle_index).xyz")
        minimized_dat_path = joinpath(output_dir_minimized, "minimized_$(base_name)_cycle$(cycle_index).dat")

        # Write the minimized structure to both .xyz and .dat files
        write_xyz(minimized_atoms, minimized_xyz_path, "Minimized structure")
        write_dat(minimized_atoms, bc, minimized_dat_path)

        # Print the box information based on the boundary condition type
        if bc isa EMCubicBC
            println("Processed file: $(basename(file_path)), cycle: $cycle_index with Box Length: ", bc.box_length)
        elseif bc isa EMRhombicBC
            println("Processed file: $(basename(file_path)), cycle: $cycle_index with Box Width: ", bc.box_width, ", Box Height: ", bc.box_height)
        end
    end
end
