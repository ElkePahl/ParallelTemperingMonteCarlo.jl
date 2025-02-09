using Test

call = :(include(joinpath(@__DIR__,"..","scripts","ne55.jl")))
eval(call)


benchmark_data = [
    1133.804270116, #elapsed_time
    3435518568, #bytes_allocated
]
profile = @timed eval(call)
threshold = 0.2
@testset "time_profile" begin
    @test profile[2] < benchmark_data[1] * (1+threshold)
end
@testset "memory_profile" begin
    @test profile[3] < benchmark_data[2] * (1+threshold)
end
