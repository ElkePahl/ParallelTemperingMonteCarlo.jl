using Random

@testset "separated_scale" begin
    v1 = SVector(1.0, 2.0, 3.0)
    v2 = SVector(2.0, 4.0, 6.0)
    v3 = SVector(0.0, 1.0, 0.0)
    pos = [v1, v2, v3]
    pos_xy = scale_xy(pos, 2.0)
    @test pos_xy[1] == [2.0, 4.0, 3.0]

    pos_z = scale_z(pos, 2.0)
    @test pos_z[1] == [1.0, 2.0, 6.0]
end

@testset "separated_v_change" begin
    v1 = SVector(1.0, 2.0, 3.0)
    v2 = SVector(2.0, 4.0, 6.0)
    v3 = SVector(0.0, 1.0, 0.0)
    bc = RhombicBC(10.0, 10.0)
    conf = Config([v1, v2, v3], bc)

    trial_config_xy, scale1 = volume_change_xy(
        conf, 0.1, 12.0, bc.box_length / bc.box_height
    )
    @test trial_config_xy[1][1] == conf[1][1] * scale1
    @test trial_config_xy[1][3] == conf[1][3]

    trial_config_z, scale2 = volume_change_z(conf, 0.1, 12.0, bc.box_length / bc.box_height)
    @test trial_config_z[1][1] == conf[1][1]
    @test abs(trial_config_z[1][3] - conf[1][3] / scale2) < 1e-12
end

@testset "rhombic_v_changes_elj" begin
    v1 = SVector(1.0, 2.0, 3.0)
    v2 = SVector(2.0, 4.0, 6.0)
    v3 = SVector(0.0, 1.0, 0.0)
    bc = RhombicBC(10.0, 10.0)
    conf = Config([v1, v2, v3], bc)
    d2mat = get_distance2_mat(conf)
    c1 = [-2.0, 1.0]
    pot1 = ELJPotentialEven{2}(c1)

    ensemble = NPT(3, 101325 * 3.398928944382626e-14, true)

    temp = TempGrid{2}(10, 15)

    state = MCState(temp.t_grid[1], temp.beta_grid[1], conf, ensemble, pot1)

    state_new = volume_change(state)

    # Position scaled as much as boundary condition
    config = state.config
    trial_config = state_new.ensemble_variables.trial_config

    @test trial_config[1][1] / config[1][1] ≈
        trial_config.boundary_condition.box_length / config.boundary_condition.box_length
    @test trial_config[1][3] / config[1][3] ≈
        trial_config.boundary_condition.box_height / config.boundary_condition.box_height
    @test trial_config[1][3] / config[1][3] ≈
        trial_config[1][1] / config[1][1]

    state_new_xyz = volume_change(state, true)
    trial_config_xyz = state_new_xyz.ensemble_variables.trial_config

    @test trial_config_xyz[1][1] / config[1][1] ≈
        trial_config_xyz.boundary_condition.box_length / config.boundary_condition.box_length
    @test trial_config_xyz[1][3] / config[1][3] ≈
        trial_config_xyz.boundary_condition.box_height / config.boundary_condition.box_height

    for i in 1:100
        state_new = volume_change(state, true)
        ensemble_variables = state_new.ensemble_variables
        trial_config = ensemble_variables.trial_config
        if ensemble_variables.xy_or_z == 2
            @test trial_config[2][3] ≠ config[2][3]
            @test trial_config[2][1] == config[2][1]
            @test trial_config[2][2] == config[2][2]
        elseif ensemble_variables.xy_or_z == 1
            @test trial_config[2][3] == config[2][3]
            @test trial_config[2][1] ≠ config[2][1]
            @test trial_config[2][2] ≠ config[2][2]
        elseif ensemble_variables.xy_or_z == 0
            @test trial_config[2][3] ≠ config[2][3]
            @test trial_config[2][1] ≠ config[2][1]
            @test trial_config[2][2] ≠ config[2][2]
        end
    end
end

@testset "rhombic_v_changes_potB" begin
    v1 = SVector(1.0, 2.0, 3.0)
    v2 = SVector(2.0, 4.0, 6.0)
    v3 = SVector(0.0, 1.0, 0.0)
    bc = RhombicBC(10.0, 10.0)
    conf = Config([v1, v2, v3], bc)
    d2mat = get_distance2_mat(conf)

    a = [0.0005742, -0.4032, -0.2101, -0.0595, 0.0606, 0.1608]
    b = [-0.01336, -0.02005, -0.1051, -0.1268, -0.1405, -0.1751]
    c1 = [-0.1132, -1.5012, 35.6955, -268.7494, 729.7605, -583.4203]
    potB = ELJPotentialB{6}(a, b, c1)

    ensemble = NPT(3, 101325 * 3.398928944382626e-14, true)

    temp = TempGrid{2}(10, 15)

    state = MCState(temp.t_grid[1], temp.beta_grid[1], conf, ensemble, potB)

    @test state.potential_variables.tan_mat[1, 2] ≈ 0.7453559924999299 #TODO: sign difference
    @test state.potential_variables.tan_mat[1, 3] ≈ 0.47140452079103173

    @test state.en_tot == -0.00016263185592172208

    state_new = volume_change(state, ensemble.separated_volume)

    @test metropolis_condition("volumemove", state_new, ensemble) ≈ metropolis_condition(
        ensemble,
        state_new.new_en - state.en_tot,
        get_volume(state_new.ensemble_variables.trial_config.boundary_condition),
        get_volume(state.config.boundary_condition),
        state.beta,
    )
end
