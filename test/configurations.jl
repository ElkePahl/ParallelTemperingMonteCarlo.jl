using Test
using ParallelTemperingMonteCarlo
using StaticArrays

@testset "Configurations" begin
    points_vec_int = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    points_tuple = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    points_f32 = [
        SVector{3,Float32}(1, 0, 0),
        SVector{3,Float32}(0, 1, 0),
        SVector{3,Float32}(0, 0, 1),
    ]
    points_svec_int = [SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1)]

    boundary_condition = SphericalBC(; radius=5.0)

    config1 = Config(points_vec_int, boundary_condition)
    config2 = Config(points_tuple, boundary_condition)
    config3 = Config(points_f32, boundary_condition)
    config4 = Config(points_svec_int, boundary_condition)

    @test config1 == config2 == config3 == config4
    @test eltype(config1) == eltype(config2) == eltype(config4) == SVector{3,Float64}
    @test eltype(config3) == SVector{3,Float32}
    @test config1 == points_svec_int

    config1[1] = (-1, -1, -1)
    config2[1] = [-1.0, -1.0, -1.0]
    config3[1] = (-1, -1, -1)

    @test config1[1] ≡ SVector(-1.0, -1.0, -1.0)
    @test config1 == config2 == config3

    @test get_centre(config4) ≈ [1 / 3, 1 / 3, 1 / 3]
    @test get_centre(recentre!(config4)) ≈ [0, 0, 0] atol = √eps(Float64)

    @test summary(config1) == "3-element Config with SphericalBC{Float64}(25.0)"
end
