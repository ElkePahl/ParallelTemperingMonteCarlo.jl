using DelimitedFiles
using Plots
using Colors

# Function to read multihistograms from the CSV file
function read_multihistograms(filepath::String)
    lines = readlines(filepath)
    histograms = Vector{Matrix{Float64}}()
    i = 1
    while i <= length(lines)
        line = strip(lines[i])
        if startswith(line, "Histogram")
            # Start of a new histogram
            i += 1
            data_lines = []
            while i <= length(lines) && strip(lines[i]) != ""
                push!(data_lines, lines[i])
                i += 1
            end
            # Convert data_lines to a matrix
            data_matrix = [parse.(Float64, split(line, ',')) for line in data_lines]
            # Convert data_matrix to a Matrix{Float64}
            histogram_matrix = hcat(data_matrix...)'  # hcat and transpose
            push!(histograms, histogram_matrix)
        else
            i += 1
        end
    end
    return histograms
end

# Function to plot each histogram as a line with color mapping
function plot_histograms_with_colormap(
    histograms::Vector{Matrix{Float64}},
    temperatures::Union{Nothing, AbstractVector{Float64}}=nothing;
    sum_over::Symbol = :rows  # :rows or :columns
)
    n_histograms = length(histograms)
    plt = plot()
    
    # Create a color gradient from blue to red
    colors_list = [RGB(0, 0, 1), RGB(1, 0, 0)]  # Blue to Red
    colormap = cgrad(colors_list, n_histograms)
    
    # Set x_label based on sum_over
    if sum_over == :rows
        x_label = "Bin Index (Columns)"
    elseif sum_over == :columns
        x_label = "Bin Index (Rows)"
    else
        error("Invalid value for sum_over. Use :rows or :columns.")
    end
    
    for idx in 1:n_histograms
        histogram = histograms[idx]
        # Sum over the specified dimension
        if sum_over == :rows
            summed = sum(histogram, dims=2)  # Sum over rows
        elseif sum_over == :columns
            summed = sum(histogram, dims=1)  # Sum over columns
        end
        # Flatten the summed array
        summed = vec(summed)
        # x-axis: bin indices
        bin_indices = 1:length(summed)
        # Label based on temperature if available
        if temperatures !== nothing
            temp = temperatures[idx]
            label = "T=$(round(temp, digits=1)) K"
        else
            label = "Trajectory $(idx)"
        end
        plot!(
            bin_indices,
            summed,
            label=label,
            color=colormap[idx],
            linewidth=2,
            legend=false
        )
    end
    xlabel!(x_label)
    ylabel!("Counts")
    title!("Multihistograms (Summed over $(sum_over))")
    display(plt)
end

# Usage example:
save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar"
filename = "all_histograms.csv"
filepath = joinpath(save_directory, filename)

# Read the histograms from the file
histograms = read_multihistograms(filepath)

# Convert temperatures to Vector if it's an SVector
temperatures = Vector(temp.t_grid)  # Replace with your actual temperatures vector

# Plot the histograms, summing over rows
println("Plotting histograms summed over rows:")
plot_histograms_with_colormap(histograms, temperatures; sum_over=:rows)

# Plot the histograms, summing over columns
println("Plotting histograms summed over columns:")
plot_histograms_with_colormap(histograms, temperatures; sum_over=:columns)
