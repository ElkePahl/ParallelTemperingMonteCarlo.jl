using Test
using SafeTestsets
using ParallelTemperingMonteCarlo
using StaticArrays, LinearAlgebra




@testset "Ensembles" begin
    x = MoveStrategy(NVT(10))    
    @test length(x.movestrat) == length(x)

    bc = SphericalBC(radius=2.0)
    v1 = SVector(1., 2., 3.)
    conf = Config{3}([v1,v1,v1],bc)

    envars_nvt = set_ensemble_variables(conf,NVT(1))
    @test typeof(envars_nvt) == NVTVariables{Float64}
    @test typeof(envars_nvt.index) == Int64 
    @test length(envars_nvt.trial_move) == 3

    y = MoveStrategy(NPT(5,101325))
    @test length(y.movestrat) == length(y)
    conf2 = Config{3}([v1,v1,v1] , PeriodicBC(8.7674))
    envars_npt = set_ensemble_variables(conf2,NPT(3,101325))

    @test envars_npt.r_cut == conf2.bc.box_length^2/4
    @test size(envars_npt.new_dist2_mat) ==  (3,3)


end

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

end

@testset "Config_cubic" begin
    bc = PeriodicBC(10.0)
    v1 = SVector(1., 2., 3.)
    conf = Config{3}([v1,v1,v1],bc)

    @test conf.bc.box_length == 10.0
    @test conf.pos[1] == v1

    v2 = SVector(2.,4.,6.)
    @test distance2(v1,v2,bc) == 14.0

    v3 = SVector(3., 6., 9.)
    @test distance2(v1,v3) == 56.0
    @test distance2(v1,v3,bc) == 36.0
    
    conf2 = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf2)
    @test d2mat[1,3] == 36.0
    @test d2mat[2,1] == d2mat[1,2]

    max_v = 0.1
    trial_config = volume_change(conf2,max_v,50)
    @test trial_config.bc.box_length/bc.box_length <= exp(0.5*max_v)^(1/3)
    @test trial_config.bc.box_length/bc.box_length >= exp(-0.5*max_v)^(1/3)
    @test abs(trial_config.bc.box_length/bc.box_length - trial_config.pos[1][1]/v1[1]) <= 10^(-15)

    displ = 0.1
    trial_pos = atom_displacement(v1,displ,bc)
    @test norm(trial_pos-v1) < displ


end

@testset "Tangent" begin
    bc = SphericalBC(radius=10.0)
    v1 = SVector(5., 0., 0.)
    v2 = SVector(-3.,0.,4.)
    v3 = SVector(-2.,0., -3.)
    conf = Config{3}([v1,v2,v3],bc)
    mat = get_tantheta_mat(conf,bc)

    @test mat[1,2]==-2.0
    @test mat[1,3]==7/3
    @test mat[2,3]==1/7
    
    bc = PeriodicBC(10.0)
    conf = Config{3}([v1,v2,v3],bc)
    mat = get_tantheta_mat(conf,bc)

    @test mat[1,2]==-1/2
    @test mat[1,3]==1.0
    @test mat[2,3]==-1/3
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
    include("test_potentials.jl")
end


@safetestset "RuNNer" begin
    include("test_runner_forward.jl")
end

# @safetestset "script testing" begin
#     function read_save_data(filename)
#         readfile = open(filename, "r+")
#         filecontents = readdlm(readfile)
#         step, configdata = read_input(filecontents)
#         close(readfile)
#         return step, configdata
#     end
#     mycompare(a, b) = a == b
#     mycompare(a::Number, b::Number) = a ≈ b

#     println("starting script testing. Hang on tight ...")
#     @testset "Cu55" begin
#         include("test_Cu55.jl")
#         # 46.922331 seconds (765.86 M allocations: 57.507 GiB, 10.54% gc time, 0.01% compilation time)

#         step, configdata = read_save_data("save.data")
#         # reference data has been produced on a single thread
#         step_ref, configdata_ref = read_save_data("testing_data/save.data")

#         @test step == step_ref # the script successfully finished

#         # The matrix `configdata` has strings and numbers
#         @test all(mycompare.(configdata, configdata_ref)) # identical configurations

#         # clean up
#         rm("save.data")
#         rm("params.data")
#     end
# end

@safetestset "multihist" begin
    include("multihist_test.jl")
end
