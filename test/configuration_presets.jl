using ParallelTemperingMonteCarlo
using ParallelTemperingMonteCarlo: min_distance
using Test

@testset "face_centred_cubic" begin
    config1 = face_centred_cubic(1)
    @test min_distance(config1) ≈ 1
    @test length(config1.positions) == 32
    @test config1.boundary_condition isa CubicBC

    config2 = face_centred_cubic(2; r_min=2.0)
    @test min_distance(config2) ≈ 2
    @test length(config2.positions) == 108

    config3 = face_centred_cubic(3; boundary_condition=RectangularBC)
    @test min_distance(config3) ≈ 1
    @test length(config3.positions) == 256
    @test config3.boundary_condition isa RectangularBC

    @test_throws ArgumentError face_centred_cubic(4; boundary_condition=SphericalBC)
end

@testset "body-centred cubic" begin
    config1 = body_centred_cubic(1)
    @test min_distance(config1) ≈ 1
    @test length(config1.positions) == 16
    @test config1.boundary_condition isa CubicBC

    config2 = body_centred_cubic(2; r_min=0.5)
    @test min_distance(config2) ≈ 0.5
    @test length(config2.positions) == 54
    @test config2.boundary_condition isa CubicBC

    config3 = body_centred_cubic(3; boundary_condition=RectangularBC)
    @test min_distance(config3) ≈ 1
    @test length(config3.positions) == 128
    @test config3.boundary_condition isa RectangularBC

    @test_throws ArgumentError body_centred_cubic(4; boundary_condition=SphericalBC)
end

@testset "magic_cluster" begin
    for (idx, n_atoms) in enumerate([13, 55, 147, 309, 561, 923])
        if n_atoms ≠ 309
            r_min = rand([0.2, 1.0, 1.5])
            config = magic_cluster(idx; r_min)
            @test length(config.positions) == n_atoms
            @test min_distance(config) ≈ r_min
            @test config.boundary_condition isa SphericalBC
        end
    end
    # TODO: find a file for index 4
    @test_throws ErrorException magic_cluster(4)
    @test_throws ArgumentError magic_cluster(0)
    @test_throws ArgumentError magic_cluster(7)
    @test_throws ArgumentError magic_cluster(100)
end
