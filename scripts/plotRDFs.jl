using DelimitedFiles
using Plots
using Colors

# Function to read and plot RDFs from the concatenated file
function plot_rdfs_with_colormap_no_temps(filepath::String; ranges::Union{Nothing, Vector{UnitRange{Int}}}=nothing)
    # Read the file content
    lines = readlines(filepath)
    rdf_data_list = []
    i = 1
    while i <= length(lines)
        line = lines[i]
        if strip(line) == "RDF"
            i += 1  # Move to the line with RDF data
            if i <= length(lines)
                rdf_line = lines[i]
                rdf_values = parse.(Float64, split(rdf_line, ','))
                push!(rdf_data_list, rdf_values)
            end
        end
        i += 1  # Move to the next line
    end

    # Determine the number of RDFs
    n_rdfs = length(rdf_data_list)

    # Validate and combine ranges
    selected_indices = []
    if ranges === nothing
        # If no ranges provided, select all RDFs
        selected_indices = 1:n_rdfs
    else
        # Flatten all ranges into a single list of indices
        for r in ranges
            if r.start < 1 || r.stop > n_rdfs || r.start > r.stop
                error("Invalid range specified. Ensure ranges are within 1 to $(n_rdfs).")
            end
            append!(selected_indices, r)
        end
    end

    # Ensure unique and sorted indices
    selected_indices = sort(unique(selected_indices))

    # Create a color gradient from blue to red
    colors_list = [RGB(0, 0, 1), RGB(1, 0, 0)]  # Blue to Red
    colormap = cgrad(colors_list, length(selected_indices))

    # Plot the selected RDFs with the color map
    plt = plot()
    for (plot_idx, rdf_idx) in enumerate(selected_indices)
        rdf_data = rdf_data_list[rdf_idx]
        bin_indices = 1:length(rdf_data)
        plot!(
            bin_indices,
            rdf_data,
            label="RDF $(rdf_idx)",
            color=colormap[plot_idx],
            linewidth=2,
            legend=false
        )
    end
    xlabel!("Bin Index")
    ylabel!("g(r)")
    title!("Radial Distribution Functions")
    display(plt)
end

# Usage example:
save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data"
filename = "all_rdfs.csv"
filepath = joinpath(save_directory, filename)

# Plot all RDFs
plot_rdfs_with_colormap_no_temps(filepath)

# # Plot specific ranges (e.g., 1:3 and 6:9)
# plot_rdfs_with_colormap_no_temps(filepath; ranges=[1:1, 20:32])
