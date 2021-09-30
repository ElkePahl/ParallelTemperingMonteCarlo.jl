using Test
using ParallelTemperingMonteCarlo

@testset "Point" begin
    p1 = Point(1.,1.,1.)
    @test p1.x == p1.y == p1.z == 1.
end