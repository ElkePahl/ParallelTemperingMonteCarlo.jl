using Random 

@testset "States"  begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = SphericalBC(radius=5.0)
    conf1 = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf1)
    c1 = [-2.,1.]
    pot1 = ELJPotentialEven{2}(c1)

    ensemble = NVT(3)

    temp = TempGrid{2}(10,15)

    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf1,ensemble,pot1)
    state2 = MCState(temp.t_grid[2],temp.beta_grid[2],conf1,ensemble,pot1)
    @test typeof(state.ensemble_variables) == NVTVariables{Float64}
    @test typeof(state.potential_variables) == DimerPotentialVariables{Float64}

    @test state.en_tot == dimer_energy_config(state.dist2_mat,3,state.potential_variables,pot1)[2]

    state.ensemble_variables.index = 1

    Random.seed!(1234)

    trialpos = atom_displacement(state.config.pos[1] , state.max_displ[1] , state.config.bc)

    Random.seed!(1234)
    generate_move!(state,"atommove")

    @test trialpos == state.ensemble_variables.trial_move
    state = get_energy!(state,pot1,"atommove")
    @test state.new_en - state.en_tot ≈ -4.99895252855e-5
    @test metropolis_condition("atommove",state,ensemble) == 1.0

    MCRun.swap_config!(state,"atommove")
    @test state.ensemble_variables.trial_move == state.config.pos[1]
    @test state.new_dist2_vec == state.dist2_mat[1,:]

    @test state2.en_tot - state.en_tot  ≈ 4.99895252855e-5

    update_max_stepsize!(state,3,ensemble,0.4,0.6)
    @test state.max_displ[1] < 0.1
    exc_trajectories!(state,state2)
    @test state2.en_tot - state.en_tot  ≈ -4.99895252855e-5
    @test state.config.pos == conf1.pos


end 
@testset "States V"  begin 

    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = CubicBC(10.0)
    conf1 = Config{3}([v1,v2,v3],bc)

    d2mat = get_distance2_mat(conf1)
    c1 = [-2.,1.]
    pot1 = ELJPotentialEven{2}(c1)
    ensemble = NPT(3,101325*3.398928944382626e-14)

    temp = TempGrid{2}(10,15)

    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf1,ensemble,pot1)
    state2 = MCState(temp.t_grid[2],temp.beta_grid[2],conf1,ensemble,pot1)


    @test typeof(state.ensemble_variables) == NPTVariables{Float64}
    @test typeof(state.potential_variables) == DimerPotentialVariables{Float64}

    @test state.en_tot == dimer_energy_config(state.dist2_mat,3,state.potential_variables,state.ensemble_variables.r_cut,state.config.bc,pot1)[2]

    state.ensemble_variables.index = 1
    Random.seed!(1234)

    trialpos = atom_displacement(state.config.pos[1] , state.max_displ[1] , state.config.bc)

    Random.seed!(1234)
    generate_move!(state,"atommove")

    @test trialpos == state.ensemble_variables.trial_move
    state = get_energy!(state,pot1,"atommove")
    @test state.new_en - state.en_tot ≈ -4.99895252855e-5
    @test metropolis_condition("atommove",state,ensemble) == 1.0

    MCRun.swap_config!(state,"atommove")
    @test state.ensemble_variables.trial_move == state.config.pos[1]
    @test state.new_dist2_vec == state.dist2_mat[1,:]
    @test state2.en_tot - state.en_tot  ≈ 4.99895252855e-5

    update_max_stepsize!(state,3,ensemble,0.4,0.6)
    @test state.max_displ[1] < 0.1
    exc_trajectories!(state,state2)
    @test state2.en_tot - state.en_tot  ≈ -4.99895252855e-5
    @test state.config.pos == conf1.pos
end


@testset "States NNVT" begin
    v1 = SVector(2.36, 2.36, 0.0)
    v2 = SVector(6.99, 2.33, 0.0)
    v3 = SVector(2.33, 6.99, 0.0)
    v4 = SVector(-2.36, 2.36, 0.0)
    v5 = SVector(-6.99, 2.33, 0.0)
    v6 = SVector(-2.33, 6.99, 0.0)
    bc = SphericalBC(radius=8.0)
    conf = Config{6}([v1,v2,v3,v4,v5,v6],bc)
    d2mat = get_distance2_mat(conf)
    vars = set_variables(conf,d2mat,runnerpotential)
    nnvtens = NNVT([4,2])
    temp = TempGrid{2}(500,650)
    include("potentialfile.jl")

 
    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf,nnvtens,runnerpotential)
    state2 = MCState(temp.t_grid[2],temp.beta_grid[2],conf,nnvtens,runnerpotential)
    refstate = MCState(temp.t_grid[1],temp.beta_grid[1],conf,nnvtens,runnerpotential)

    @test isa(state.ensemble_variables,NNVTVariables)
    @test isa(state.potential_variables,NNPVariables2a)
    testgmat = MMatrix{88,6}(total_symm_calc(state.config.pos,d2mat,state.potential_variables.f_matrix,runnerpotential.radsymfunctions,runnerpotential.angsymfunctions,5,26,4,2))
    @test testgmat == state.potential_variables.g_matrix

    #test moves
    Random.seed!(123)
    generate_move!(state,"atommove")

    Random.seed!(123)
    trialpos = atom_displacement(state.config.pos[1] , state.max_displ[1] , state.config.bc)

    @test trialpos == state.ensemble_variables.trial_move
    state = get_energy!(state,runnerpotential,"atommove")
    delen = state.new_en - state.en_tot
    @test delen ≈ -7.5971516e-5
    #test exc after atom move
    @test  metropolis_condition("atommove",state,ensemble) == 1.
    MCRun.swap_config!(state,"atommove")
    @test state.ensemble_variables.trial_move == state.config.pos[1]
    @test state.en_tot - state2.en_tot  ≈ delen 
    # test parallel_tempering_exchange
    exc_trajectories!(state,state2)
    @test state2.en_tot - state.en_tot  ≈ delen
    @test state.config.pos == conf.pos
    #test atomswap
    Random.seed!(123)
    MCMoves.swap_atoms(state)
    @test state.ensemble_variables.swap_indices[2] > 4
    state = get_energy!(state,runnerpotential,"atomswap")
    delen2 = state.new_en - state.en_tot
    @test delen2 ≈ -0.0017084901948

    MCRun.acc_test!(state,nnvtens,"atomswap")
    
    refmat = copy(refstate.dist2_mat[6,:] )
    refmat[6] = refstate.dist2_mat[6,3]
    refmat[3] = 0.

    @test refmat == state.dist2_mat[3,:]
    @test state.en_tot - refstate.en_tot == delen2
    @test state.config.pos[3] == refstate.config.pos[6]
    @test state.config.pos[6] == refstate.config.pos[3]

end
