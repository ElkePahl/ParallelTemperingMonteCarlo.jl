using Test
using SafeTestsets
using ParallelTemperingMonteCarlo
using StaticArrays, LinearAlgebra
using Random
"Lenient comparison operator for `struct`, both mutable and immutable (type with \\eqsim)."
function ≂(x, y)
    if x == y
        return true
    end
    try
        if x ≈ y
            return true
        end
    catch
    end
    # If x and y are of different types, they are not equal
    if typeof(x) != typeof(y)
        return false
    end
            
    # If x and y are arrays, compare their elements recursively
    if x isa AbstractArray && y isa AbstractArray
        if length(x) != length(y)
            return false
        end
        for i in eachindex(x)
            if !(x[i] ≂ y[i])
                return false
            end
        end
        return true
    end
    
    # If x and y are structs, compare their fields recursively
    for field in fieldnames(typeof(x))
        if !(getfield(x, field) ≂ getfield(y, field))
            return false
        end
    end
    
    return true
end

#Ensemble
begin
    @testset "get_r_cut" begin
        cubic_bc = CubicBC(10.0)
        @test get_r_cut(cubic_bc) == 25.0
        
        rhombic_bc = RhombicBC(4.0, 5.0)
        @test get_r_cut(rhombic_bc) == 3.0
    end
    
    
    @testset "NVT" begin
        nvt = NVT(10)
        @test nvt == NVT(10, 10, 0)
    end
    
    @testset "MoveStrategy" begin
        test_ensemble = NVT(0, 2, 3)
        x = MoveStrategy(test_ensemble)    
        @test x ≂ MoveStrategy{5, NVT}(NVT(0,2,3),
            [
                "atommove",
                "atommove",
                "atomswap",
                "atomswap",
                "atomswap"
            ]
        )
    end
    
    @testset "NVTVariables" begin
        bc = SphericalBC(radius=2.0)
        v1 = SVector(1., 2., 3.)
        conf = Config{3}([v1,v1,v1],bc)
    
        envars_nvt = set_ensemble_variables(conf,NVT(1))
        @test envars_nvt ≂ NVTVariables{Float64}(
            1,
            SVector{3, Float64}(0.0, 0.0, 0.0)
        )
    end
    
    @testset "NPT" begin
        npt = NPT(10, 101325)
        @test npt ≂ NPT(10, 10, 1, 0, 101325)
    
        y = MoveStrategy(npt)
        @test y ≂ MoveStrategy{11, NPT}(
            NPT(10, 10, 1, 0, 101325),
            [
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "atommove",
                "volumemove"
            ]
        )
    end
    
    @testset "NPTVariables" begin
        v1 = SVector(1., 2., 3.)
        conf2 = Config{3}([v1,v1,v1] , CubicBC(8.7674))
        envars_npt = set_ensemble_variables(conf2,NPT(3,101325))
    
        @test envars_npt ≂ NPTVariables{Float64}(1, [0.0, 0.0, 0.0], Config{3, CubicBC{Float64}, Float64}(SVector{3, Float64}[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], CubicBC{Float64}(8.7674)), [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 19.21682569, 0.0)
        conf3 = Config{3}([v1,v1,v1] , RhombicBC(10.0,10.0))
        envars_npt = set_ensemble_variables(conf3,NPT(3,101325))
    
        @test envars_npt ≂ NPTVariables{Float64}(1, [0.0, 0.0, 0.0], Config{3, RhombicBC{Float64}, Float64}(SVector{3, Float64}[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], RhombicBC{Float64}(10.0, 10.0)), [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 18.75, 0.0)
    
        conf4 = Config{3}([v1,v1,v1] , RhombicBC(10.0,5.0))
        envars_npt = set_ensemble_variables(conf4,NPT(3,101325))
        @test envars_npt ≂ NPTVariables{Float64}(1, [0.0, 0.0, 0.0], Config{3, RhombicBC{Float64}, Float64}(SVector{3, Float64}[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], RhombicBC{Float64}(10.0, 5.0)), [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 6.25, 0.0)

    end
end

#Configurations
begin
    @testset "Config constructor" begin
        #Testing Config with SphericalBC and the `inferred` macro 
        bc = SphericalBC(radius=4.0)
        Random.seed!(1234)
        posarray = [SVector(rand(),rand(),rand()) for _ in 0:9]
        config = @inferred Config{10}(posarray,bc)
        @test config ≂ Config{10, SphericalBC{Float64}, Float64}(SVector{3, Float64}[[0.32597672886359486, 0.5490511363155669, 0.21858665481883066], [0.8942454282009883, 0.35311164439921205, 0.39425536741585077], [0.9531246272848422, 0.7955469475347194, 0.4942498668904206], [0.7484150218874741, 0.5782319465613976, 0.7279350012266056], [0.007448006132865004, 0.19937661409915552, 0.4392431254532684], [0.6825326622623844, 0.9567409540049077, 0.6478553157718558], [0.996665291437684, 0.7491940599574348, 0.11008426115113379], [0.4913831957970459, 0.5651453592612876, 0.2538117862083361], [0.626793910352374, 0.23410455326227375, 0.1247919570769006], [0.609874865666702, 0.6727928883390367, 0.7619157626781667]], SphericalBC(radius=4.0))
        @test_throws ErrorException @inferred Config(posarray,bc)

        #Testing Config with SphericalBC with random positions
        Random.seed!(4321)
        posarray = [[rand(),rand(),rand()] for _ in 0:9]
        config = Config(posarray,bc)
        @test config ≂ Config{10, SphericalBC{Float64}, Float64}(SVector{3, Float64}[[0.549646602186891, 0.1295805209303995, 0.8647701448945764], [0.048563077743689065, 0.10271005862089821, 0.4011373724024988], [0.7440394139194013, 0.8323080594185164, 0.13083238137887598], [0.36406839362517673, 0.006071525002571687, 0.526382759960887], [0.6666411661708421, 0.0036467654234990654, 0.6273931086515868], [0.48479966793423024, 0.7034479876260851, 0.6011627120836686], [0.5561188879723905, 0.5407795689288655, 0.47159602863934924], [0.23198889927997512, 0.8284954591439163, 0.2983174292260927], [0.7214616029365842, 0.6318691091778196, 0.6320480377407123], [0.8417026388331795, 0.9363914241340269, 0.4044825339606498]], SphericalBC(radius=4.0))

        # Testing Config with CubicBC
        bc = CubicBC(10.0)
        v1 = SVector(1., 2., 3.)
        conf = Config{3}([v1,v1,v1],bc)
        @test conf ≂ Config{3, CubicBC{Float64}, Float64}(SVector{3, Float64}[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], CubicBC{Float64}(10.0))

        # Testing Config with RhombicBC
        bc = RhombicBC(10.0,10.0)
        v1 = SVector(1., 2., 3.)
        conf = Config{3}([v1,v1,v1],bc)
        @test conf ≂ Config{3, RhombicBC{Float64}, Float64}(SVector{3, Float64}[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], RhombicBC{Float64}(10.0, 10.0))
    end

    @testset "distance2" begin
        v1 = SVector(1., 2., 3.)
        v2 = SVector(2.,4.,6.)
        v3 = SVector(3., 6., 9.)

        # Basic distance2 with 2 arguments
        @test distance2(v1,v2) == 14.0
        @test distance2(v1,v3) == 56.0

        # SphericalBC(same function as distance2(a,b))
        bc = SphericalBC(radius=4.0)
        @test distance2(v1,v2,bc) == 14.0

        # CubicBC 
        bc = CubicBC(10.0)
        @test distance2(v1,v2,bc) == 14.0
        @test distance2(v1,v3,bc) == 36.0

        # RhombicBC
        bc = RhombicBC(10.0,10.0)
        @test distance2(v1,v2,bc) == 14.0
        @test distance2(v1,v3,bc) == 36.0
    end

    @testset "get_distance2_mat" begin
        v1 = SVector(1., 2., 3.)
        v2 = SVector(2.,4.,6.)
        v3 = SVector(3., 6., 9.)

        # SphericalBC
        bc = SphericalBC(radius=4.0)
        conf2 = Config{3}([v1,v2,v3],bc)
        d2mat = get_distance2_mat(conf2)
        @test d2mat ≂ [0.0 14.0 11.0; 14.0 0.0 49.0; 11.0 49.0 0.0]

        # CubicBC
        bc = CubicBC(10.0)
        conf2 = Config{3}([v1,v2,v3],bc)
        d2mat = get_distance2_mat(conf2)
        @test d2mat ≂ [0.0 14.0 36.0; 14.0 0.0 14.0; 36.0 14.0 0.0]

        # RhombicBC
        bc = RhombicBC(10.0,10.0)
        conf2 = Config{3}([v1,v2,v3],bc)
        d2mat = get_distance2_mat(conf2)
        @test d2mat ≂ [0.0 14.0 36.0; 14.0 0.0 14.0; 36.0 14.0 0.0]
    end

    @testset "get_volume" begin
        # CubicBC
        bc = CubicBC(10.0)
        v = get_volume(bc)
        @test v==1000.0

        # RhombicBC
        bc = RhombicBC(10.0,10.0)
        v = get_volume(bc)
        @test v==3^0.5/2*1000.0
    end

    @testset "get_tantheta_mat" begin
        v1 = SVector(5., 0., 0.)
        v2 = SVector(-3.,0.,4.)
        v3 = SVector(-2.,0., -3.)

        # SphericalBC
        bc = SphericalBC(radius=10.0)
        conf = Config{3}([v1,v2,v3],bc)
        mat = get_tantheta_mat(conf,bc)
        #Here we do not test the whole matrix due to floating point errors.
        @test mat[1,2]==-2.0
        @test mat[1,3]==7/3
        @test mat[2,3]==1/7

        # CubicBC
        bc = CubicBC(10.0)
        conf = Config{3}([v1,v2,v3],bc)
        mat = get_tantheta_mat(conf,bc)
        @test mat ≂ [0.0 -0.5 1.0; -0.5 0.0 -1/3; 1.0 -1/3 0.0]

        # RhombicBC
        bc = RhombicBC(5.0,5.0)
        conf = Config{3}([v1,v2,v3],bc)
        mat = get_tantheta_mat(conf,bc)
        @test mat ≂ [0.0 2.0 -1.0; 2.0 0.0 0.5; -1.0 0.5 0.0]
    end

    @testset "get_tan" begin
        #Testing get_tan with two arguments
        Random.seed!(1234)
        v1 = SVector(rand(),rand(),rand()) .* 100.0
        v2 = SVector(rand(),rand(),rand()) .* 100.0
        result = get_tan(v1,v2)
        @test result ≈ -3.421783622152092

        #Testing get_tan with SphericalBC
        bc = SphericalBC(radius=4.0)
        @test get_tan(v1,v2,bc) ≈ result

        #Testing get_tan with CubicBC
        bc = CubicBC(10.0)
        @test get_tan(v1,v2,bc) ≈ 1.3147700496470467

        #Testing get_tan with RhombicBC
        bc = RhombicBC(10.0,10.0)
        @test get_tan(v1,v2,bc) ≈ 1.6043117438283387
    end

    @testset "get_centre" begin
        Random.seed!(123)
        posarray = [SVector(rand(),rand(),rand()) for _ in 0:9]
        centre = get_centre(posarray,length(posarray))
        @test centre ≈ [0.35696456049919784, 0.5529939892376119, 0.4947065292905407]
    end

    @testset "recentre!" begin
        Random.seed!(12345)
        posarray = [SVector(rand(),rand(),rand()) for _ in 0:9]
        conf = Config{10}(posarray,SphericalBC(radius=4.0))
        recentre!(conf)
        @test conf.pos ≈ [[0.33274081229118546, 0.4209766403293317, -0.11327404143498188], [-0.47593323469348237, -0.12341715941321452, -0.20033667722153614], [0.04911880568421301, 0.28271819272111476, -0.14472477651440618], [-0.044242857008182535, -0.3533067945298938, 0.4152680044610666], [-0.34449953781765097, -0.3274297148530201, 0.3576092195706506], [0.2412225238635951, 0.5496964754639848, 0.36711961990878605], [0.22241332842251083, 0.2887754754988734, -0.06927266526967824], [0.29055949320826313, -0.11511933009752517, -0.32126060011204605], [-0.4385268254254845, -0.30709079237074044, 0.12957429101832707], [0.16714749147503305, -0.3158029927489111, -0.42070237440618186]]
    end
end

#BoundaryConditions
begin
    @testset "SphericalBC" begin
        bc = SphericalBC(radius=1.0)
        @test bc.radius2 == 1.
    end

    @testset "CubicBC" begin
        bc = CubicBC(10.0)
        @test bc.box_length == 10.0
    end

    @testset "RhombicBC" begin
        bc = RhombicBC(15.0,10.0)
        @test bc.box_length == 15.0
        @test bc.box_height == 10.0
    end

    @testset "check_boundary" begin
        bc = SphericalBC(radius=1.0)
        @test check_boundary(bc,SVector(0,0.5,1.))
        @test !check_boundary(bc,SVector(0,0.5,0.5))
    end
end

#InputParams
begin
    @testset "MCParams" begin
        mc_params = MCParams(100, 32, 10; eq_percentage = 0.5)
        @test mc_params ≂ MCParams(100, 50, 1, 32, 10, 100, 100, 0.4, 0.6)
    end
    @testset "TempGrid" begin
        n_traj = 32
        temp = TempGrid{n_traj}(2, 16)
        @test temp ≂ TempGrid{32, Float64}([2.0, 2.1387593971421346, 2.2871458794318933, 2.4458273711349365, 2.615518136901144, 2.796981996846505, 2.991035764696417, 3.198552924466336, 3.420467562229416, 3.6577785706690062, 3.911554145341731, 4.1829365928899485, 4.473147472846541, 4.7834930961765645, 5.115370405306075, 5.470273262105569, 5.849799172131822, 6.255656475395606, 6.689672036022699, 7.153799465421251, 7.650127915970039, 8.180891484810147, 8.748479270068886, 9.355446124781496, 10.004524156916695, 10.698635027270536, 11.440903100584428, 12.234669509083764, 13.08350719174062, 13.991236976955975, 14.961944781053553, 16.0], [157887.4926315486, 147643.99664826435, 138065.08281908667, 129107.63408317252, 120731.33074780596, 112898.46899948658, 105573.79118973778, 98724.32713170841, 92319.24569320552, 86329.71601814106, 80728.7777517163, 75491.21968519525, 70593.46627401712, 66013.47151843816, 61730.619729032696, 57725.63273037514, 53980.48308520315, 50478.31294846281, 47203.35818597755, 44140.87741618053, 41277.08565550918, 38599.09226878412, 36094.84294527119, 33753.06543924675, 31563.218830831054, 29515.446078700246, 27600.52965110479, 25809.85003547882, 24135.346939881703, 22569.483011629993, 21105.20990980838, 19735.936578943576])
        temp1 = TempGrid{n_traj}(2, 16; tdistr = :equally_spaced)
        @test temp1 ≂ TempGrid{32, Float64}([2.0, 2.4516129032258065, 2.903225806451613, 3.354838709677419, 3.806451612903226, 4.258064516129032, 4.709677419354838, 5.161290322580645, 5.612903225806452, 6.064516129032258, 6.516129032258064, 6.967741935483871, 7.419354838709677, 7.870967741935484, 8.32258064516129, 8.774193548387096, 9.225806451612904, 9.677419354838708, 10.129032258064516, 10.580645161290322, 11.032258064516128, 11.483870967741936, 11.935483870967742, 12.387096774193548, 12.838709677419354, 13.29032258064516, 13.741935483870968, 14.193548387096774, 14.64516129032258, 15.096774193548386, 15.548387096774194, 16.0], [157887.4926315486, 128802.95451521069, 108766.93936840016, 94125.23599188475, 82957.83511149163, 74159.27684209101, 67048.11330928777, 61181.40339472509, 56258.761742275936, 52069.27948487241, 48460.517540376306, 45319.55807016673, 42560.97627459136, 40118.95304572137, 37941.95559362796, 35989.06082042652, 34227.35854250354, 32630.08181052005, 31175.237398586032, 29844.587021817115, 28622.87878115794, 27497.25995268543, 26456.823089610847, 25492.251414468785, 24595.539053155815, 23759.768308631104, 22978.930852478905, 22247.783052627303, 21561.72806862558, 20916.719109307724, 20309.17955011621, 19735.936578943576])
    end 
    #Output functions are just constructors, with very little logic to test
end

#EnergyEvaluation
begin
    Random.seed!(1234)
    n_atoms = 10
    c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
    eljpot_even = ELJPotentialEven{6}(c)
    a=[0.0005742,-0.4032,-0.2101,-0.0595,0.0606,0.1608]
    b=[-0.01336,-0.02005,-0.1051,-0.1268,-0.1405,-0.1751]
    c1=[-0.1132,-1.5012,35.6955,-268.7494,729.7605,-583.4203]
    eljpot_b = ELJPotentialB{6}(a,b,c1)
    index = 5
    pos = [SVector(rand()*7,rand()*7,rand()*7) for _ in 1:n_atoms]
    sphericalbc = SphericalBC(radius=4.0)
    rhombicbc = RhombicBC(5.0,10.0)
    cubicbc = CubicBC(10.0)
    spherical_config = Config(pos,sphericalbc)
    cubic_config = Config(pos,cubicbc)
    rhombic_config = Config(pos,rhombicbc)
    d2mat_spherical = get_distance2_mat(spherical_config)
    d2mat_cubic = get_distance2_mat(cubic_config)
    d2mat_rhombic = get_distance2_mat(rhombic_config)
    d2vec_spherical = d2mat_spherical[index,:]
    d2vec_cubic = d2mat_cubic[index,:]
    d2vec_rhombic = d2mat_rhombic[index,:]
    thetamat_spherical = get_tantheta_mat(spherical_config,sphericalbc)
    thetamat_cubic = get_tantheta_mat(cubic_config,cubicbc)
    thetamat_rhombic = get_tantheta_mat(rhombic_config,rhombicbc)
    r_cut = 100
    @testset "dimer_energy_atom" begin
        r_cut = 50
        @test dimer_energy_atom(index,d2vec_spherical,eljpot_even) ≈ 0.023911356198916628
        @test dimer_energy_atom(index,d2vec_spherical,r_cut,eljpot_even) ≈ 0.02402606005477939
        @test dimer_energy_atom(index,d2vec_spherical,thetamat_spherical,eljpot_b) ≈ -8.13586254832503e-5
        @test dimer_energy_atom(index,d2vec_spherical,thetamat_spherical,r_cut,eljpot_b) ≈ -7.97412096809367e-5
    end

    dimer_potential_variables = DimerPotentialVariables([rand() for _ in 1:n_atoms])
    eljb_potential_variables = ELJPotentialBVariables([rand() for _ in 1:n_atoms],thetamat_spherical,[rand() for _ in 1:n_atoms])
    @testset "dimer_energy_config" begin
        @test dimer_energy_config(d2mat_spherical,n_atoms,dimer_potential_variables,eljpot_even) ≂ ([1.3557145257410774e7, 4.490459175475829, 56.12295154741882, 1.1988466992169222e7, 0.023911356198916628, 365.20355684232544, 5.276074090915916, 1.3557153443327183e7, 12.126980651298927, 1.1988772333137097e7], 2.554599063498897e7)
        @test dimer_energy_config(d2mat_rhombic,n_atoms,dimer_potential_variables,r_cut,rhombicbc,eljpot_even) ≂ ([1.3630035267213766e7, 57046.0084073771, 3.669527911994271e8, 1.1988462632017003e7, 3.670081538109875e8, 1507.4729882659524, 3.409020761089101e10, 1.3557570527187362e7, 3.4090135626498367e10, 1.1988773086656507e7], 3.448276378869706e10)
        @test dimer_energy_config(d2mat_spherical,n_atoms,eljb_potential_variables,eljpot_b) ≂ ([0.09974246784456195, -0.0004949984150655538, 0.0004464986278423629, 0.09974315464916497, -8.991276776044012e-5, 0.10029787911494172, 1.4848011272009263e-5, 0.0996475159492499, -0.0002882452740609024, 0.19977959309929333], 0.2993994004197197)
        @test dimer_energy_config(d2mat_cubic,n_atoms,eljb_potential_variables,r_cut,cubicbc,eljpot_b) ≂ ([0.09974246784456195, -0.000527207943895608, 0.0004422256538654586, 0.09974237893004974, -0.00013164977594247492, 0.100297349592459, 1.0711892535640772e-5, 0.0996475159492499, -0.0002884321290201486, 0.19977959309929333], 0.2993539333415685)
    end
    variance = 1e2
    pos2 = map(x -> x * ((rand() + variance-0.5)*variance^-1),pos)
    config2 = Config(pos2, sphericalbc)
    new_d2_mat_spherical = get_distance2_mat(config2)
    new_d2_spherical_vec = new_d2_mat_spherical[index,:]
    new_tanvec_spherical = get_tantheta_mat(config2, sphericalbc)
    @testset "dimer_energy_update" begin
        r_cut = 50
        @test dimer_energy_update!(index,d2mat_spherical,new_d2_spherical_vec,0.0,eljpot_even) ≈ 0.00045929693400654364
        @test dimer_energy_update!(index,d2mat_spherical,new_d2_spherical_vec,0.0,r_cut,eljpot_even) ≈ 0.0004603835196235889
        @test dimer_energy_update!(index,d2mat_spherical,thetamat_spherical,new_d2_spherical_vec,new_tanvec_spherical,0.0,eljpot_b) ≈ 7.644500010850971e-6
        @test dimer_energy_update!(index,d2mat_spherical,thetamat_spherical,new_d2_spherical_vec,new_tanvec_spherical,0.0,r_cut,eljpot_b) ≈ 7.666736146333403e-6
    end
    @testset "energy_update!" begin
        r_cut = 50
        @test energy_update!(pos2, index, spherical_config, dimer_potential_variables, d2mat_spherical, new_d2_spherical_vec, 0.0, eljpot_even) ≂ (DimerPotentialVariables{Float64}([0.5888720595243433, 0.36585394350375, 0.13102565622085904, 0.9464532262313834, 0.5743234852783174, 0.6776499075995779, 0.5715863775229529, 0.07271614209221577, 0.7011158355140754, 0.09521752297832098]), 0.00045929693400654364)
        @test energy_update!(pos2, index, spherical_config, dimer_potential_variables, d2mat_spherical, new_d2_spherical_vec, 0.0, r_cut, eljpot_even) ≂ (DimerPotentialVariables{Float64}([0.5888720595243433, 0.36585394350375, 0.13102565622085904, 0.9464532262313834, 0.5743234852783174, 0.6776499075995779, 0.5715863775229529, 0.07271614209221577, 0.7011158355140754, 0.09521752297832098]), 0.0004603835196235889)
    end
end


@testset "Potentials" begin 
    include("test_potentials.jl")
end

@testset "States" begin
    include("test_states.jl")
end

@testset "Checkpoints" begin
    include("checkpoint_test.jl")
end

@safetestset "RuNNer" begin
    #include("test_runner_forward.jl")
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

@testset "Config_cubic" begin
    
    displ = 0.1
    trial_pos = atom_displacement(v1,displ,bc)
    @test norm(trial_pos-v1) < displ

    displ = 0.1
    # @test_throws ErrorException atom_displacement(v1,displ,bc)
    Random.seed!(3214)
    trial_pos = atom_displacement(v3,displ,bc)
    @test trial_pos ≂ [-0.00954294101413884, 1.0414628423920707, 0.03223299447843586]
    @test norm(trial_pos-v3) < displ
    
    max_v = 0.1
    trial_config, scale = volume_change(conf2,bc,max_v,50)
    @test trial_config.bc.box_length/bc.box_length <= exp(0.5*max_v)^(1/3)
    @test trial_config.bc.box_length/bc.box_length >= exp(-0.5*max_v)^(1/3)
    @test abs(trial_config.bc.box_length/bc.box_length - trial_config.pos[1][1]/v1[1]) <= 10^(-15)

end

@testset "Config_rhombic" begin
    

    max_v = 0.1
    trial_config, scale = volume_change(conf2,bc,max_v,50)
    @test trial_config.bc.box_length/bc.box_length <= exp(0.5*max_v)^(1/3)
    @test trial_config.bc.box_length/bc.box_length >= exp(-0.5*max_v)^(1/3)
    @test abs(trial_config.bc.box_length/bc.box_length - trial_config.pos[1][1]/v1[1]) <= 10^(-15)
    @test abs(trial_config.bc.box_length/bc.box_length - trial_config.bc.box_height/bc.box_height) <= 10^(-15)

    v5 = SVector(7.5, 4.330127018922193, 5.0)
    displ = 0.1
    trial_pos = atom_displacement(v5,displ,bc)
    @test norm(trial_pos-v5) < displ


end