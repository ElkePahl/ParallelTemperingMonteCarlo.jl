using DelimitedFiles
using Plots
using Glob

# Define boundary condition types
struct RhombicBC
    box_length::Float64
    box_height::Float64
end

# struct CubicBC
#     box_length::Float64
# end

# Function to compute the squared distance between two points with rhombic PBCs
function distance2_rhombic(a::Vector{Float64}, b::Vector{Float64}, bc::RhombicBC)
    sqrt3 = sqrt(3.0)
    delta_y = a[2] - b[2]
    delta_z = a[3] - b[3]
    b_y = b[2] + (sqrt3 / 2 * bc.box_length) * round(delta_y / (sqrt3 / 2 * bc.box_length))
    delta_x = a[1] - b[1]
    delta_xy = delta_x - (1 / sqrt3) * delta_y
    b_x = b[1] + bc.box_length * round(delta_xy / bc.box_length) - (1 / sqrt3) * (a[2] - b_y)
    b_z = b[3] + bc.box_height * round(delta_z / bc.box_height)
    adjusted_b = [b_x, b_y, b_z]
    return sum((a - adjusted_b).^2)
end

# function distance2(a::SVector{3, Float64}, b::SVector{3, Float64}, bc::RhombicBC)
#     sqrt3 = sqrt(3)
#     # Adjust b to account for periodic boundary conditions
#     b_y = b[2] + (sqrt3/2 * bc.box_width) * round((a[2] - b[2]) / (sqrt3/2 * bc.box_width))
#     b_x = b[1] - b[2]/sqrt3 + bc.box_width * round(((a[1] - b[1]) - (1/sqrt3)*(a[2] - b[2])) / bc.box_width) + (1/sqrt3) * b_y
#     b_z = b[3] + bc.box_height * round((a[3] - b[3]) / bc.box_height)
#     b_adjusted = SVector{3, Float64}(b_x, b_y, b_z)
#     # Return the squared distance between a and the adjusted b
#     return distance2(a, b_adjusted)
# end

# Function to compute the squared distance between two points with cubic PBCs
function distance2_cubic(a::Vector{Float64}, b::Vector{Float64}, bc::CubicBC)
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]
    dx -= bc.box_length * round(dx / bc.box_length)
    dy -= bc.box_length * round(dy / bc.box_length)
    dz -= bc.box_length * round(dz / bc.box_length)
    return dx^2 + dy^2 + dz^2
end

# Function to read configurations from a .dat file with multiple configurations
function read_configurations(filename::String)
    configurations = Vector{Dict{String, Any}}()
    current_positions = Vector{Vector{Float64}}()
    cell = Dict{String, Any}()
    config_count = 0  # Counter for configurations

    open(filename, "r") do file
        while !eof(file)
            line = readline(file)
            line = strip(line)

            if isempty(line)
                continue
            elseif startswith(line, "# Cycle:")
                # If we have a previous configuration, store it
                if !isempty(current_positions)
                    config = Dict("positions" => deepcopy(current_positions), "cell" => deepcopy(cell))
                    push!(configurations, config)
                    current_positions = Vector{Vector{Float64}}()
                    config_count += 1
                end

                # Parse cell parameters from the line
                cell = Dict{String, Any}()

                # Extract cycle number
                m_cycle = match(r"# Cycle: (\d+)", line)
                if m_cycle !== nothing
                    cycle_number = parse(Int, m_cycle.captures[1])
                    cell["cycle_number"] = cycle_number
                else
                    cell["cycle_number"] = 0  # Default if not found
                end

                m_cubic = match(r"Box Length: (\d+\.?\d*)", line)
                if m_cubic !== nothing
                    box_length = parse(Float64, m_cubic.captures[1])
                    cell["box_length"] = box_length
                    cell["cell_type"] = "cubic"
                else
                    # Try to match rhombic box
                    m_rhombic = match(r"Box Width: (\d+\.?\d*), Box Height: (\d+\.?\d*)", line)
                    if m_rhombic !== nothing
                        box_length = parse(Float64, m_rhombic.captures[1])
                        box_height = parse(Float64, m_rhombic.captures[2])
                        cell["box_length"] = box_length
                        cell["box_height"] = box_height
                        cell["cell_type"] = "rhombic"
                    else
                        error("Box information not found or invalid in file: $filename")
                    end
                end

                # Extract temperature if present
                m_temp = match(r"Temperature: (\d+\.?\d*\.?\d*)", line)
                if m_temp !== nothing
                    temperature = parse(Float64, m_temp.captures[1])
                    cell["temperature"] = temperature
                end
            elseif startswith(line, "Ar") || startswith(line, "ar") || startswith(line, "AR")
                data = split(line)
                if length(data) >= 4
                    x = parse(Float64, data[2])
                    y = parse(Float64, data[3])
                    z = parse(Float64, data[4])
                    push!(current_positions, [x, y, z])
                else
                    println("Invalid atom line in file: $filename")
                end
            else
                # Ignore other lines
                continue
            end
        end

        # After the loop, store the last configuration if any
        if !isempty(current_positions)
            config = Dict("positions" => deepcopy(current_positions), "cell" => deepcopy(cell))
            push!(configurations, config)
            config_count += 1
        end
    end

    println("Total configurations read from $filename: $config_count")
    return configurations
end

# Function to compute the RDF from positions and cell parameters
function compute_rdf_from_positions(positions::Vector{Vector{Float64}}, cell::Dict{String, Any}, Δr::Float64, r_max::Float64)
    n_bins = Int(floor(r_max / Δr))
    rdf_counts = zeros(n_bins)
    N = length(positions)

    # Check the cell type and get parameters
    if haskey(cell, "cell_type")
        cell_type = cell["cell_type"]
    else
        error("Cell type not specified in cell parameters.")
    end

    if cell_type == "rhombic"
        # Ensure cell parameters are available
        if !haskey(cell, "box_length") || !haskey(cell, "box_height")
            error("Cell parameters 'box_length' and 'box_height' are required for rhombic cells.")
        end
        box_length = cell["box_length"]
        box_height = cell["box_height"]
        bc = RhombicBC(box_length, box_height)
        for i in 1:(N-1)
            for j in (i+1):N
                a = positions[i]
                b = positions[j]
                r2 = distance2_rhombic(a, b, bc)
                r = sqrt(r2)
                if r < r_max
                    bin_index = Int(floor(r / Δr)) + 1
                    if bin_index <= n_bins
                        rdf_counts[bin_index] += 2  # Each pair contributes twice
                    end
                end
            end
        end
    elseif cell_type == "cubic"
        # Ensure cell parameter is available
        if !haskey(cell, "box_length")
            error("Cell parameter 'box_length' is required for cubic cells.")
        end
        box_length = cell["box_length"]
        bc = CubicBC(box_length)
        for i in 1:(N-1)
            for j in (i+1):N
                a = positions[i]
                b = positions[j]
                r2 = distance2_cubic(a, b, bc)
                r = sqrt(r2)
                if r < r_max
                    bin_index = Int(floor(r / Δr)) + 1
                    if bin_index <= n_bins
                        rdf_counts[bin_index] += 2  # Each pair contributes twice
                    end
                end
            end
        end
    else
        error("Unknown cell type: $cell_type")
    end

    return rdf_counts
end

# Function to compute the average RDF from configurations
function compute_average_rdf(configurations::Vector{Dict{String, Any}}, Δr::Float64, r_max::Float64)
    n_bins = Int(floor(r_max / Δr))
    rdf_total = zeros(n_bins)
    num_configs = length(configurations)

    for config in configurations
        positions = config["positions"]
        cell = config["cell"]
        rdf_counts = compute_rdf_from_positions(positions, cell, Δr, r_max)
        rdf_total .+= rdf_counts
    end

    # Average RDF over configurations
    rdf_average = rdf_total / num_configs
    return rdf_average
end

# Main function to process all configuration files and plot RDFs with temperature colormap
function process_and_plot_all_rdfs_with_colormap(
    config_dir::String,
    Δr::Float64,
    r_max::Float64;
    trajectory_indices::Union{Nothing, AbstractVector{Int}}=nothing,
    temperature_range::Union{Nothing, Tuple{Float64, Float64}}=nothing
)
    # Get list of all configuration files
    config_files = glob("*.dat", config_dir)

    if isempty(config_files)
        println("No configuration files found in directory: $config_dir")
        return
    end

    # Sort config files
    config_files = sort(config_files)

    bins = collect(Δr:Δr:r_max)
    rdf_list = Vector{Vector{Float64}}()  # Specify the type explicitly
    temperatures = Float64[]  # Store temperatures
    labels = String[]  # Labels for legend
    selected_indices = Int[]

    # Process each configuration file
    for (index, file_path) in enumerate(config_files)
        filename = basename(file_path)
        # Read configurations from the file
        configurations = read_configurations(file_path)
        # Compute the average RDF over all configurations in the file
        rdf_average = compute_average_rdf(configurations, Δr, r_max)
        # Extract temperature from cell parameters (assuming all configurations have the same temperature)
        temp = NaN
        if !isempty(configurations)
            cell = configurations[1]["cell"]
            if haskey(cell, "temperature")
                temp = cell["temperature"]
            else
                temp = NaN
            end
        end

        # Determine if this trajectory should be included based on indices or temperature range
        include_trajectory = true

        if trajectory_indices !== nothing
            include_trajectory = index in trajectory_indices
        end

        if temperature_range !== nothing
            tmin, tmax = temperature_range
            if isnan(temp) || temp < tmin || temp > tmax
                include_trajectory = false
            end
        end

        if include_trajectory
            # Store the RDF and temperature
            push!(rdf_list, rdf_average)
            push!(temperatures, temp)
            push!(labels, !isnan(temp) ? "$(temp) K" : filename)
            push!(selected_indices, index)
        end
    end

    if isempty(rdf_list)
        println("No trajectories selected for plotting.")
        return
    end

    # Sort RDFs and temperatures in order of increasing temperature
    sorted_indices = sortperm(temperatures)
    rdf_list = rdf_list[sorted_indices]
    temperatures = temperatures[sorted_indices]
    labels = labels[sorted_indices]

    # Normalize temperatures to [0, 1] for color mapping
    min_temp = minimum(temperatures)
    max_temp = maximum(temperatures)
    temp_norm = (temperatures .- min_temp) ./ (max_temp - min_temp)

    # Define a color gradient from blue to red
    color_gradient = cgrad([:blue, :red])

    # Get colors for each temperature using the 'get' function
    colors = get.(Ref(color_gradient), temp_norm)

    # Plot all RDFs together with colormap
    p = plot()

    for i in 1:length(temperatures)
        r = bins[1:length(rdf_list[i])]
        g_r = rdf_list[i]
        color = colors[i]
        plot!(p, r, g_r, label=labels[i], lw=2, color=color)
    end

    # Set labels, title, and grid
    plot!(p,
          xlabel="Distance r (units)",
          ylabel="g(r)",
          title="Radial Distribution Functions",
          grid=true,
          primary=false,
          legend=false)

    # Display the plot
    display(p)
end

# Replace with your actual directory containing the configuration files
config_dir = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/216"

# Parameters for RDF calculation
Δr = 0.15  # Bin width
r_max = 15.0  # Maximum distance to consider

# Specify trajectory indices (e.g., trajectories 5 to 10)
trajectory_indices = 1:25

# Alternatively, specify a temperature range (e.g., temperatures between 100 K and 200 K)
#temperature_range = (1, 5)

# Call the function with the desired parameters
process_and_plot_all_rdfs_with_colormap(
    config_dir,
    Δr,
    r_max;
    trajectory_indices=trajectory_indices,
    # Uncomment the next line to use temperature range instead
    #temperature_range=temperature_range
)
