using Test
using SafeTestsets
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

@testset "TemperatureGrid" begin
    n_traj = 32
    temp = TempGrid{n_traj}(2, 16)
    kB = 3.16681196E-6
    @test temp.t_grid[1] ≈ 2.0
    @test length(temp.t_grid) == n_traj
    @test length(temp.beta_grid) == n_traj
    @test 1. /(temp.t_grid[1]*temp.beta_grid[1]) ≈ kB
    temp1 = TempGrid{n_traj}(2, 16; tdistr = :equally_spaced)
    @test (temp1.t_grid[2] - temp1.t_grid[1]) ≈ (temp1.t_grid[n_traj] - temp1.t_grid[n_traj-1])
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

@safetestset "RuNNer" begin
    include("test_runner_forward.jl")
end

@safetestset "script testing" begin
    function read_save_data(filename)
        readfile = open(filename, "r+")
        filecontents = readdlm(readfile)
        step, configdata = read_input(filecontents)
        close(readfile)
        return step, configdata
    end
    mycompare(a, b) = a == b
    mycompare(a::Number, b::Number) = a ≈ b

    println("starting script testing. Hang on tight ...")
    @testset "Cu55" begin
        include("../scripts/test_Cu55.jl")
        # 46.922331 seconds (765.86 M allocations: 57.507 GiB, 10.54% gc time, 0.01% compilation time)

        step, configdata = read_save_data("save.data")
        # reference data has been produced on a single thread
        step_ref, configdata_ref = read_save_data("testing_data/save.data")

        @test step == step_ref # the script successfully finished

        # The matrix `configdata` has strings and numbers
        @test all(mycompare.(configdata, configdata_ref)) # identical configurations

        # clean up
        # rm("save.data")
        # rm("params.data")
    end
end
