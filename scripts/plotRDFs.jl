using DelimitedFiles
using Plots
using Colors

# Function to read and plot RDFs from the concatenated file
function plot_rdfs_with_colormap_no_temps(filepath::String)
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

    # Create a color gradient from blue to red
    colors_list = [RGB(0, 0, 1), RGB(1, 0, 0)]  # Blue to Red
    colormap = cgrad(colors_list, n_rdfs)

    # Plot all RDFs over each other with the color map
    plt = plot()
    for idx in 1:n_rdfs
        rdf_data = rdf_data_list[idx]
        bin_indices = 1:length(rdf_data)
        plot!(
            bin_indices,
            rdf_data,
            label="RDF $(idx)",
            color=colormap[idx],
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
save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"
filename = "all_rdfs.csv"
filepath = joinpath(save_directory, filename)
plot_rdfs_with_colormap_no_temps(filepath)
