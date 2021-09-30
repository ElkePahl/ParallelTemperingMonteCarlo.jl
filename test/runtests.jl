using Test
using ParallelTemperingMonteCarlo

@testset "Point" begin
    p1 = Point(1.,1.,1.)
    @test p1.x == p1.y == p1.z == 1.

    pointarray = [Point(rand(),rand(),rand()) for _ in 1:10]
    bc = SphericalBC(1.0)    
    config = @inferred Config{10}(pointarray,bc)
    @test length(config) == 10
    @test_throws ErrorException @inferred Config(pointarray,bc)
    p2 = Point(-1.,-1.,-1.)
    @test dist2(p1,p2) == 12.0
end
