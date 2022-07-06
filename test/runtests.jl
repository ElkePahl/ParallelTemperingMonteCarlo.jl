using Test
using ParallelTemperingMonteCarlo
using StaticArrays, LinearAlgebra

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

    displ = 0.1
    @test_throws ErrorException atom_displacement(v1,displ,bc)
    trial_pos = atom_displacement(v3,displ,bc)
    @test norm(trial_pos-v3) < displ

    move_atom=AtomMove(10, displ)
    @test move_atom.frequency == 10
end

@testset "BoundaryConditions" begin
    bc = SphericalBC(radius=1.0)
    @test bc.radius2 == 1.
    @test check_boundary(bc,SVector(0,0.5,1.))
    @test check_boundary(bc,SVector(0,0.5,0.5)) == false
end

@testset "Potentials" begin
    c = [-2.,0.,1.]
    pot =  ELJPotential{3}(c)
    @test dimer_energy(pot,1.) == -1.0
    c1 = [-2.,1.]
    pot1 = ELJPotentialEven{2}(c1)
    @test dimer_energy(pot1,1.) == -1.0
    @test dimer_energy(pot1,2.) == dimer_energy(pot,2.)
    c=[-1.,2.,3.,-4.,5.]
    pot =  ELJPotential{5}(c)
    @test dimer_energy(pot,1.) == sum(c)
    pot1 = ELJPotential(c)
    @test dimer_energy(pot1,1.) == sum(c)
    @test dimer_energy(pot,2.) == dimer_energy(pot1,2.)
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = SphericalBC(radius=2.0)
    conf2 = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf2)
    @test dimer_energy_atom(2,d2mat[2,:],pot) < 0
    en_vec,en_tot = dimer_energy_config(d2mat,3,pot)
    @test en_vec[2] == dimer_energy_atom(2,d2mat[2,:],pot)
    #@test en_vec
end

@testset "EnergyEvaluation"
    pos_ne13 = [[2.825384495892464, 0.928562467914040, 0.505520149314310],
                [2.023342172678102,	-2.136126268595355, 0.666071287554958],
                [2.033761811732818,	-0.643989413759464, -2.133000349161121],
                [0.979777205108572,	2.312002562803556, -1.671909307631893],
                [0.962914279874254,	-0.102326586625353, 2.857083360096907],
                [0.317957619634043,	2.646768968413408, 1.412132053672896],
                [-2.825388342924982, -0.928563755928189, -0.505520471387560],
                [-0.317955944853142, -2.646769840660271, -1.412131825293682],
                [-0.979776174195320, -2.312003751825495, 1.671909138648006],
                [-0.962916072888105, 0.102326392265998,	-2.857083272537599],
                [-2.023340541398004, 2.136128558801072,	-0.666071089291685],
                [-2.033762834001679, 0.643989905095452, 2.132999911364582],
                [0.000002325340981,	0.000000762100600, 0.000000414930733]]
    AtoBohr = 1.88973
    pos_ne13 = pos_ne13 * AtoBohr
    bc_ne13 = SphericalBC(radius=5.32*AtoBohr)
    config_ne13 = Config(pos_ne13, bc_ne13)
    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
    pot = ELJPotentialEven{6}(c)
    dist2_mat = get_distance2_mat(config_ne13)
    @test dist2_mat[1,1] == 0.0
    @test dist2_mat[1,2] > 35.92 && dist2_mat[1,2] < 35.93
    @test length(config_ne13.pos) == 13
    en_atom_vec, en_tot = dimer_energy_config(dist2_mat, 13, pot)
    @test en_atom_vec[1] < -0.00081 && en_atom_vec[1] > -0.00082
    @test en_tot < -0.00563 && en_tot > -0.005634686479115
end

#@testset "MCLoop"

#end

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