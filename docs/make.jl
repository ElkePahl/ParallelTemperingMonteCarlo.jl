"""
Running `julia --project make.jl` while in the `docs` directory will generate the documentation pages for all submodules defined in the lists below.
"""

using Documenter, Literate
using ParallelTemperingMonteCarlo

EXAMPLES_INPUT = joinpath(@__DIR__, "../examples")
EXAMPLES_OUTPUT = joinpath(@__DIR__, "src/")
EXAMPLES_FILES = filter(endswith(".jl"), readdir(EXAMPLES_INPUT))
EXAMPLES_PAIRS = Pair{String,String}[]
EXAMPLES_NUMS = Int[]

#Top layer of submodules
surface = sort([
    "BoundaryConditions",
    "Configurations",
    "Ensembles",
    "MachineLearningPotential",
    "EnergyEvaluation",
    "MCStates",
    "InputParams",
    "MCMoves",
    "Exchange",
    "MCSampling",
    "ReadSave",
    "Initialization",
    "MCRun",
    "Multihistogram",
    "Multihistogram_NPT",
    "Multihistogram_NVT"
])

#Submodules of a submodule in the ParallelTemperingMonteCarlo module
submodules = [
    "MachineLearningPotential" => [
        "MachineLearningPotential",
        "Cutoff",
        "DeltaMatrix",
        "ForwardPass",
        "SymmetryFunctions"
    ]
]
main_module = "ParallelTemperingMonteCarlo"
"""
Generates the documentation markdown files for a (sub)module given the path of its parent module and its name. These will be automatically included in the documentation pages.
"""
function write_md(parent_module::String, module_name::String)
    fpath = joinpath(dirname(@__FILE__), "src", module_name * ".md")
    open(fpath, "w") do io
        write(io, "# $module_name\n\n```@autodocs\nModules = [$parent_module.$module_name]\n```")
    end
end

#Create home page
open(joinpath(dirname(@__FILE__), "src", "index.md"), "w") do io
    write(io, "# $main_module\n\n```@contents\n```")
end

#Dynamically create the documentation pages argument for makedocs.
#Surface modules will be independent in the sidebar, but submodules will be grouped under a heading.
pages = Vector{Any}()
push!(pages, "Home" => "index.md")

for module_name in surface
    write_md(main_module, module_name) #Generate the markdown file for the module
    push!(pages, module_name => module_name * ".md") #Associate a page name(currently the name of the module) with the markdown file.
end

#Grouping submodules under a heading, and creating submodule pages.
for (parent, children) in submodules
    subpages = Vector{Any}()
    for child in children
        write_md("$main_module.$parent", child)
        push!(subpages, child => "$child.md")
    end
    push!(pages, parent => subpages) #Currently the name of the parent module is the name of the navbar group
end

THRESHOLD_IGNORE = String[]
#Adding examples
for fn in EXAMPLES_FILES
    fnmd_full = Literate.markdown(
        joinpath(EXAMPLES_INPUT, fn), EXAMPLES_OUTPUT;
        documenter = true, execute = true
        )
    filepath = chop(fn; head = 0, tail = 2) * "md"
    push!(EXAMPLES_PAIRS, fn => filepath)
    push!(THRESHOLD_IGNORE, filepath)
end
push!(pages, "Examples" => EXAMPLES_PAIRS)

Documenter.HTMLWriter.HTML(
    size_threshold_ignore = THRESHOLD_IGNORE,
)

makedocs(sitename="$main_module",
    pages = pages,
)

deploydocs(
    repo = "github.com/ElkePahl/ParallelTemperingMonteCarlo.jl.git"
)