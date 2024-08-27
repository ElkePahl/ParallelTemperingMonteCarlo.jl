using Random 


@testset "separated_scale"  begin 
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    pos = [v1,v2,v3]
    pos_xy = scale_xy(pos,2.0)
    @test pos_xy[1] == [2.0, 4.0, 3.0]

    pos_z = scale_z(pos,2.0)
    @test pos_z[1] == [1.0, 2.0, 6.0]
end

@testset "separated_v_change" begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = RhombicBC(10.0,10.0)
    conf = Config{3}([v1,v2,v3],bc)

    trial_config_xy,scale1 = volume_change_xy(conf,bc,0.1,12.0,bc.box_length/bc.box_height)
    @test trial_config_xy.pos[1][1] == conf.pos[1][1]*scale1
    @test trial_config_xy.pos[1][3] == conf.pos[1][3]

    trial_config_z,scale2 = volume_change_z(conf,bc,0.1,12.0,bc.box_length/bc.box_height)
    @test trial_config_z.pos[1][1] == conf.pos[1][1]
    @test abs(trial_config_z.pos[1][3] - conf.pos[1][3]/scale2) < 10^(-12)
end

@testset "rhombic_v_changes_elj" begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = RhombicBC(10.0,10.0)
    conf = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf)
    c1 = [-2.,1.]
    pot1 = ELJPotentialEven{2}(c1)

    ensemble = NPT(3,101325*3.398928944382626e-14,true)

    temp = TempGrid{2}(10,15)

    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf,ensemble,pot1)

    state_new = volume_change(state)

    @test abs(state_new.ensemble_variables.trial_config.pos[1][1]/state.config.pos[1][1] - state_new.ensemble_variables.trial_config.bc.box_length/state.config.bc.box_length) < 10^(-12)
    @test abs(state_new.ensemble_variables.trial_config.pos[1][3]/state.config.pos[1][3] - state_new.ensemble_variables.trial_config.bc.box_height/state.config.bc.box_height) < 10^(-12)

    state_new_xyz = volume_change_xyz(state)

    @test abs(state_new_xyz.ensemble_variables.trial_config.pos[1][1]/state.config.pos[1][1] - state_new_xyz.ensemble_variables.trial_config.bc.box_length/state.config.bc.box_length) < 10^(-12)
    @test abs(state_new_xyz.ensemble_variables.trial_config.pos[1][3]/state.config.pos[1][3] - state_new_xyz.ensemble_variables.trial_config.bc.box_height/state.config.bc.box_height) < 10^(-12)
    @test state_new_xyz.ensemble_variables.trial_config.pos[1][1]/state.config.pos[1][1] ==1.0 || state_new_xyz.ensemble_variables.trial_config.pos[1][3]/state.config.pos[1][3] == 1.0
    
    for i=1:5
        state_new_decide = volume_change(state,ensemble.separated_volume)
        @test state_new_decide.ensemble_variables.trial_config.pos[1][1]/state.config.pos[1][1] ==1.0 || state_new_decide.ensemble_variables.trial_config.pos[1][3]/state.config.pos[1][3] == 1.0
    end


end


@testset "rhombic_v_changes_potB" begin
    v1 = SVector(1., 2., 3.)
    v2 = SVector(2.,4.,6.)
    v3 = SVector(0., 1., 0.)
    bc = RhombicBC(10.0,10.0)
    conf = Config{3}([v1,v2,v3],bc)
    d2mat = get_distance2_mat(conf)

    a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
    b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
    c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
    potB = ELJPotentialB{6}(a,b,c1)

    ensemble = NPT(3,101325*3.398928944382626e-14,true)

    temp = TempGrid{2}(10,15)

    state = MCState(temp.t_grid[1],temp.beta_grid[1],conf,ensemble,potB)

    @test abs(state.potential_variables.tan_mat[1,2]+0.7453559924999299) <= 10^(-12)
    @test abs(state.potential_variables.tan_mat[1,3]-0.47140452079103173) <= 10^(-12)

    @test state.en_tot == -0.00016263185592172208

end



