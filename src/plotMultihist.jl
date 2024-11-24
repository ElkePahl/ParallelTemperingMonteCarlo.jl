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
temperatures = [600.0, 613.566861360761, 627.4404889334922, 641.6278191426404, 656.1359452551665, 670.9721209269811, 686.143763829568, 701.6584593586122, 717.5239644264847, 733.7482113404809, 750.339311768752, 767.3055607959113, 784.6554410703432, 802.3976270452902, 820.5409893158353, 839.0945990539515, 858.067732543832, 877.4698758197733, 897.3107294089251, 917.6002131812822, 938.348471309341, 959.5658773399009, 981.2630393805467, 1003.4508054034047, 1026.1402686688248, 1049.3427732716982, 1073.0699198131877, 1097.333571200702, 1122.1458585790165, 1147.519187395506, 1173.4662436025192, 1200.0] #Vector(temp.t_grid)  # Replace with your actual temperatures vector

# Plot the histograms, summing over rows
println("Plotting histograms summed over rows:")
plot_histograms_with_colormap(histograms, temperatures; sum_over=:rows)

# Plot the histograms, summing over columns
println("Plotting histograms summed over columns:")
plot_histograms_with_colormap(histograms, temperatures; sum_over=:columns)
