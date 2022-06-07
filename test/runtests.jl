using Test
using ParallelTemperingMonteCarlo
using StaticArrays, LinearAlgebra

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
    bc = SphericalBC(radius=2.0)
    v1 = SVector(1., 2., 3.)
    conf = Config{3}([v1,v1,v1],bc)

    @test conf.bc.radius2 == 4.0
    @test conf.pos[1] == v1

    posarray = [SVector(rand(),rand(),rand()) for _ in 1:10]
    config = @inferred Config{10}(posarray,bc)
    @test length(config.pos) == 10
    @test_throws ErrorException @inferred Config(posarray,bc)

    posarray = [[rand(),rand(),rand()] for _ in 1:10]
    config = Config(posarray,bc)
    @test length(config) == 10

    v2 = SVector(2.,4.,6.)
    @test distance2(v1,v2) == 14.0

    v3 = SVector(0., 1., 0.)
    conf2 = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf2)
    @test d2mat[1,3] == 11.0
    @test d2mat[2,1] == d2mat[1,2]

    delta = SVector(0.,1.,2.)
    pos = move_atom!(conf.pos[1],delta,bc)
    @test pos == SVector(1.,3.,5.)
    @test pos == SVector(1.,3.,5.)

    
end

@testset "BoundaryConditions" begin
    bc = SphericalBC(radius=1.0)
    @test bc.radius2 == 1.

    @test check_boundary(bc,SVector(0,0.5,1.))
    @test check_boundary(bc,SVector(0,0.5,0.5)) == false
end