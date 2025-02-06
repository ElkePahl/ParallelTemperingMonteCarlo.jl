using ParallelTemperingMonteCarlo.MachineLearningPotential.ForwardPass: lib_path

@testset "ELJPotentials" begin 
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

    @test dimer_energy(pot1,1.) == sum(c1)

    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = SphericalBC(radius=2.0)
    conf1 = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf1)

    vars = set_variables(conf1,d2mat,pot)
    evars = set_ensemble_variables(conf1,NVT(3))
    @test dimer_energy_atom(2,d2mat[2,:],pot) < 0
    en_vec,en_tot = dimer_energy_config(d2mat,3,vars,pot)
    @test en_vec[2] == dimer_energy_atom(2,d2mat[2,:],pot)
    en,vars = initialise_energy(conf1,d2mat,vars,evars,pot)
    @test en ≈ en_tot

    bc2 = CubicBC(4.) 
    conf2 = Config{3}([v1,v2,v3],bc2)


    evars1 = set_ensemble_variables(conf2,NPT(3,10))
    en_vec_pbc,en_tot_pbc = dimer_energy_config(d2mat,3,vars,4.,bc2,pot1)
    @test en_vec_pbc[2] == dimer_energy_atom(2,d2mat[2,:],4.,pot)
    en_pbc,vars_pbc = initialise_energy(conf2,d2mat,vars,evars1,pot1)
    @test en_pbc == en_tot_pbc
end
@testset "EmbeddedAtomTest" begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = SphericalBC(radius=2.0)
    conf = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf)
    pot1 = EmbeddedAtomPotential(1.,1.,1.,1.,1.)
    @test pot1.ean == 0.5 
    vars = set_variables(conf,d2mat,pot1)
    evars = set_ensemble_variables(conf,NVT(3))
    @test typeof(vars.component_vector) == Matrix{Float64}
    @test vars.component_vector[:,1] == vars.component_vector[:,2]
    E,vars = initialise_energy(conf,d2mat,vars,evars,pot1)
    @test E ≈ -1.3495549581716526
end

@testset "RuNNerPotentialTest" begin
    include("potentialfile.jl")
    @test isa(runnerpotential,AbstractMachineLearningPotential)
    v1 = SVector(2.36, 2.36, 0.0)
    v2 = SVector(6.99, 2.33, 0.0)
    v3 = SVector(2.33, 6.99, 0.0)
    v4 = SVector(-2.36, 2.36, 0.0)
    v5 = SVector(-6.99, 2.33, 0.0)
    v6 = SVector(-2.33, 6.99, 0.0)
    bc = SphericalBC(radius=7.0)
    conf = Config{6}([v1,v2,v3,v4,v5,v6],bc)
    d2mat = get_distance2_mat(conf)
    vars = set_variables(conf,d2mat,runnerpotential)
    evars = set_ensemble_variables(conf,NNVT([4,2]))

    @test vars.f_matrix[1,2] == cutoff_function(sqrt(d2mat[1,2]),runnerpotential.r_cut)
    @test vars.f_matrix[2,2] == 1.
    @test isa(vars.g_matrix,MMatrix)
    E,vars = initialise_energy(conf,d2mat,vars,evars,runnerpotential)
    @test E ≈-0.261652899
end