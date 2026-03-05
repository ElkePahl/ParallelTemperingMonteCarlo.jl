pro_file = Dict(
    "test" => "ne13.jl",
    "release" => "ne55.jl",
)
mode = "test"
call = :(include(joinpath(@__DIR__,"..","scripts",pro_file[$mode])))
eval(call)

using StatProfilerHTML, ProfileCanvas
if mode == "release"
    Profile.init(n = 500_000_000)
end
@profilehtml eval(call)
@profview_allocs eval(call)