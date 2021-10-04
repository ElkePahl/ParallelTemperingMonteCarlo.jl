using Test
using ParallelTemperingMonteCarlo
using StaticArrays

# @testset "Point" begin
#     p1 = Point(1.,1.,1.)
#     @test p1.x == p1.y == p1.z == 1.

#     pointarray = [Point(rand(),rand(),rand()) for _ in 1:10]
#     bc = SphericalBC(1.0)    
#     config = @inferred Config{10}(pointarray,bc)
#     @test length(config) == 10
#     @test_throws ErrorException @inferred Config(pointarray,bc)
#     p2 = Point(-1.,-1.,-1.)
#     @test dist2(p1,p2) == 12.0
# end

@testset "Config" begin
    bc = SphericalBC(1.0)
    v1 = SVector(1., 2., 3.)
    conf = Config{3}([v1,v1,v1],bc)

    @test conf.bc.radius2 == 1.0
    @test conf.pos[1] == v1

    posarray = [SVector(rand(),rand(),rand()) for _ in 1:10]
    config = @inferred Config{10}(posarray,bc)
    @test length(config.pos) == 10
    @test_throws ErrorException @inferred Config(posarray,bc)

    delta = SVector(0.,1.,2.)
    conf1 = move_atom!(conf, 1, delta)
    @test conf.pos[1] == SVector(1.,3.,5.)
    @test conf1.pos[1] == SVector(1.,3.,5.)
end