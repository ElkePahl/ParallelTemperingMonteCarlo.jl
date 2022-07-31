using ParallelTemperingMonteCarlo
using BenchmarkTools, StaticArrays

ico_13  = [[-0.0000000049,       -0.0000000044,       -0.0000000033],
[-0.0000007312,       -0.0000000014,        0.6554619119],
 [0.1811648930,      -0.5575692094,        0.2931316798],
[-0.4742970242,       -0.3445967289,        0.2931309525],
[-0.4742970303,        0.3445967144,        0.2931309494],
 [0.1811648830,        0.5575692066,        0.2931316748],
 [0.5862626299,        0.0000000022,        0.2931321262],
[-0.1811648928,       -0.5575692153,       -0.2931316813],
[-0.5862626397,       -0.0000000109,       -0.2931321327],
[-0.1811649028,        0.5575692007,       -0.2931316863],
 [0.4742970144,        0.3445967202,       -0.2931309590],
 [0.4742970205,       -0.3445967231,       -0.2931309559],
 [0.0000007214,       -0.0000000073,       -0.6554619185]]

 
ico_55 = [[0.0000006584,       -0.0000019175,        0.0000000505],
[-0.0000005810,       -0.0000004871,        0.6678432175],
[0.1845874248,       -0.5681026047,        0.2986701538],
[-0.4832557457,       -0.3511072166,        0.2986684497],
[-0.4832557570,        0.3511046452,        0.2986669456],
[0.1845874064,        0.5681000550,        0.2986677202],
[0.5973371920,       -0.0000012681,        0.2986697030],
[-0.1845860897,       -0.5681038901,       -0.2986676192],
[-0.5973358752,       -0.0000025669,       -0.2986696020],
[-0.1845861081,        0.5680987696,       -0.2986700528],
[0.4832570624,        0.3511033815,       -0.2986683486],
[0.4832570738,       -0.3511084803,       -0.2986668445],
[0.0000018978,       -0.0000033480,       -0.6678431165],
[-0.0000017969,        0.0000009162,        1.3230014650],
[0.1871182835,       -0.5758942175,        0.9797717078],
[-0.4898861924,       -0.3559221410,       0.9797699802],
[-0.4898862039,        0.3559224872,        0.9797684555],
[0.1871182648,        0.5758945856,        0.9797692407],
[0.6055300485,        0.0000001908,        0.9797712507],
[0.7926501864,       -0.5758950093,        0.6055339635],
[0.3656681761,       -1.1254128670,        0.5916673591],
[-0.3027660545,       -0.9318173412,        0.6055326929],
[-0.9573332453,       -0.6955436707,        0.5916639831],
[-0.9797705418,       -0.0000006364,        0.6055294407],
[-0.9573332679,        0.6955423392,        0.5916610035],
[-0.3027660847,        0.9318160902,        0.6055287012],
[0.3656681396,        1.1254115783,        0.5916625380],
[0.7926501677,        0.5758937939,        0.6055314964],
[1.1833279992,       -0.0000006311,        0.5916664660],
[0.6770051458,       -0.9318186223,        0.0000033028],
[0.0000006771,       -1.1517907207,        0.0000025175],
[-0.6770037988,       -0.9318186442,        0.0000007900],
[-1.0954155825,       -0.3559242494,       -0.0000012200],
[-1.0954155940,        0.3559203788,       -0.0000027447],
[-0.6770038290,        0.9318147872,       -0.0000032017],
[0.0000006397,        1.1517868856,       -0.0000024165],
[0.6770051155,        0.9318148091,       -0.0000006889],
[1.0954168993,        0.3559204143,        0.0000013211],
[1.0954169108,       -0.3559242139,        0.0000028458],
[0.3027674014,       -0.9318199253,       -0.6055286002],
[-0.3656668229,       -1.1254154134,       -0.5916624370],
[-0.7926488510,       -0.5758976290,       -0.6055313954],
[-1.1833266824,       -0.0000032040,       -0.5916663649],
[-0.7926488697,        0.5758911742,       -0.6055338624],
[-0.3656668594,        1.1254090319,       -0.5916672580],
[0.3027673712,        0.9318135061,       -0.6055325919],
[0.9573345621,        0.6955398357,       -0.5916638820],
[0.9797718586,       -0.0000031986,       -0.6055293396],
[0.9573345846,       -0.6955461743,       -0.5916609025],
[-0.1871169480,       -0.5758984207,       -0.9797691397],
[-0.6055287318,       -0.0000040259,       -0.9797711497],
[-0.1871169667,        0.5758903824,       -0.9797716067],
[0.4898875091,        0.3559183059,       -0.9797698792],
[0.4898875207,       -0.3559263223,       -0.9797683545],
[0.0000031136,       -0.0000047513,       -1.3230013639]]
 n_traj = 10

 index = 5
 vec_shift = SVector(1,1,1)
 runnerdir = "/home/grayseff/Code/Brass_potential/"
 atomtype="Cu"
 pot = AbstractMLPotential(runnerdir,atomtype)


nmtobohr = 18.8973
copperconstant = 0.36258*nmtobohr
pos_cu13 = copperconstant*ico_13
pos_cu55 = copperconstant*ico_55
AtoBohr = 1.8897259886


 bc_cu13 = SphericalBC(radius=8*AtoBohr)
 bc_cu55 = SphericalBC(radius=14*AtoBohr)
#starting configuration
start_config13 = Config(pos_cu13, bc_cu13)
start_config55 = Config(pos_cu55, bc_cu55)
states13 = []
states55 = []
states55_a = []

for i = 1:n_traj
    push!(states13,start_config13)
    push!(states55,start_config55)
end

for i = 1:5*n_traj
    push!(states55_a,start_config55)
end

##
 function trialrun_singular_13a_10traj()
    for i in 1:n_traj
        E = RuNNer.getenergy(runnerdir,start_config13,atomtype,index,vec_shift)
    end
 end

 function trialrun_singular_55a_10traj()
    for i in 1:n_traj
        E = RuNNer.getenergy(runnerdir,start_config55,atomtype,index,vec_shift)
    end
 end
 function trialrun_singular_55a_50traj()
    for i in 1:5*n_traj
        E = RuNNer.getenergy(runnerdir,start_config55,atomtype,index,vec_shift)
    end
 end
function trialrun_multiple_13a_10traj()
    testfile = writeinit(runnerdir)

    for config in states13
        writeconfig(testfile,config,index,vec_shift,atomtype)
    end
    close(testfile)
    energyvec = getRuNNerenergy(runnerdir,n_traj)
end
function trialrun_multiple_55a_10traj()
    testfile = writeinit(runnerdir)

    for config in states55
        writeconfig(testfile,config,index,vec_shift,atomtype)
    end
    close(testfile)
    energyvec = getRuNNerenergy(runnerdir,n_traj)
end
function trialrun_multiple_55a_50traj()
    testfile = writeinit(runnerdir)

    for config in states55_a
        writeconfig(testfile,config,index,vec_shift,atomtype)
    end
    close(testfile)
    energyvec = getRuNNerenergy(runnerdir,n_traj)
end

##
thirteen_ten_single = @benchmark trialrun_singular_13a_10traj()
thirteen_ten_multiple = @benchmark trialrun_multiple_13a_10traj()
fiftyfive_ten_single = @benchmark trialrun_singular_55a_10traj()
fiftyfive_ten_multiple = @benchmark trialrun_multiple_55a_10traj()
fiftyfive_fifty_single = @benchmark trialrun_singular_55a_50traj()
fiftyfive_fifty_multiple = @benchmark trialrun_multiple_55a_50traj()
##