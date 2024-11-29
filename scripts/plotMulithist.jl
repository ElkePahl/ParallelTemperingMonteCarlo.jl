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
save_directory = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data"
filename = "all_histograms.csv"
filepath = joinpath(save_directory, filename)

# Read the histograms from the file
histograms = read_multihistograms(filepath)

# Convert temperatures to Vector if it's an SVector
temperatures = [80.0, 81.26278617353782, 82.5455052085766, 83.84847174189014, 85.17200537673932, 86.51643076126726, 87.88207766813188, 89.26928107539554, 90.67838124869142, 92.1097238246869, 93.56365989586439, 95.04054609664054, 96.54074469084456, 98.06462366057758, 99.61255679647466, 101.18492378939139, 102.78211032353789, 104.40450817108291, 106.05251528825116, 107.72653591293754, 109.42698066386231, 111.15426664129124, 112.90881752934573, 114.69106369992772, 116.5014423182854, 118.3403974502447, 120.20838017113391, 122.10584867642748, 124.03326839413614, 125.99111209897137, 127.9798600283118, 130.0]

#[80.0, 81.63483962166853, 83.3030880006943, 85.00542786158023, 86.74255588064271, 88.51518297112362, 90.32403457412903, 92.16985095551334, 94.0533875088303, 95.97541506447506, 97.93672020514364, 99.93810558773907, 101.9803902718557, 104.06441005497632, 106.1910178145193, 108.36108385687537, 110.57549627357744, 112.83516130474878, 115.14100370997832, 117.49396714677528, 119.89501455675739, 122.34512855973131, 124.84531185582605, 127.39658763584443, 130.0] #Vector(temp.t_grid)  # Replace with your actual temperatures vector

# Plot the histograms, summing over rows
println("Plotting histograms summed over rows:")
plot_histograms_with_colormap(histograms, temperatures; sum_over=:rows)

# Plot the histograms, summing over columns
println("Plotting histograms summed over columns:")
plot_histograms_with_colormap(histograms, temperatures; sum_over=:columns)
