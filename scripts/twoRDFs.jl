using DelimitedFiles
using Plots

# Function to read and extract a single RDF from a file
function read_single_rdf(filepath::String, index::Int)
    # Read all lines from the file
    lines = readlines(filepath)
    # Extract the RDF line based on the index
    rdf_data = []
    current_index = 0
    for line in lines
        if startswith(line, "RDF")
            current_index += 1
        end
        if current_index == index && !startswith(line, "RDF")
            rdf_data = parse.(Float64, split(line, ','))
            break
        end
    end
    if isempty(rdf_data)
        error("RDF data for index $index not found in file $filepath.")
    end
    return rdf_data
end

# Function to plot RDFs from two sources
function plot_rdfs_two_sources(file1::String, file2::String, index::Int)
    # Read RDF data
    rdf1 = read_single_rdf(file1, index)
    rdf2 = read_single_rdf(file2, index)

    # Generate x-axis based on the RDF data length
    x1 = 1:length(rdf1)
    x2 = 1:length(rdf2)

    # Plot both RDFs
    plt = plot(x1, rdf1, label="FCC", linewidth=2, color=:blue)
    plot!(plt, x2, rdf2, label="HCP", linewidth=2, color=:red)

    # Add labels and title
    xlabel!("Bin Index")
    ylabel!("g(r)")
    #title!("Radial Distribution Functions (FCC vs. HCP)")
    display(plt)
end

# File paths
file1 = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar/all_rdfs.csv"
file2 = "/Users/samuelcase/Dropbox/PTMC_Lit&Coding/Sam_Results/Data/Ar_HCP/all_rdfs.csv"

# Plot the RDFs for index 1
plot_rdfs_two_sources(file1, file2, 1)
