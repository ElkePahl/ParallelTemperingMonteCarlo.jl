module ReadSave


export restart


using StaticArrays
using DelimitedFiles
using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCRun


function readinput(savedata)
    step = savedata[1,5]

    paramdata = savedata[2:7 ,:] #when writing this function properly, remember there is a steps line at the top.
    configdata = savedata[8:end,:]

    return step,paramdata,configdata
end


function initialiseparams(paramdata)

    MC_param = MCParams(paramdata[2,2],paramdata[4,2],paramdata[5,2],mc_sample = paramdata[3,2], n_adjust = paramdata[6,2])

    return MC_param
end

function readconfig(oneconfigvec,n_atoms, potential)
    positions = []
    coord_atom = zeros(3)
    for j=1:n_atoms
        coord_atom = SVector(oneconfigvec[8+j,1] ,oneconfigvec[8+j,2] ,oneconfigvec[8+j,3] )
        push!(positions,coord_atom)
    end
    if oneconfigvec[7,2] == "SphericalBC{Float64}"
        boundarycondition = SphericalBC(radius=oneconfigvec[7,3])
    end
    counta = [oneconfigvec[5,2], oneconfigvec[5,3]]
    countv = [oneconfigvec[5,4], oneconfigvec[5,5]]
    countr = [oneconfigvec[5,6], oneconfigvec[5,7]]
    countx = [oneconfigvec[5,8], oneconfigvec[5,9]]
    
    mcstate = MCState(oneconfigvec[2,2], oneconfigvec[2,3],Config(positions,boundarycondition), potential ; max_displ=[oneconfigvec[4,2], oneconfigvec[4,3], oneconfigvec[4,4] ], count_atom=counta,count_vol=countv,count_rot=countr,count_exc=countx)

    return mcstate
end

function readconfigs(configvecs,n_atoms,n_traj,potential)
    states = []
    lines = Int64(9+n_atoms)
    for idx=1:n_traj
        oneconfig = configvecs( (idx - 1)*lines:(idx*lines), : )
        onestate = readconfig(oneconfig,n_atoms,potential)

        push!(states,onestate)

    end

    return states
end

function restart(potential ;directory = pwd())
    readfile = open("$(directory)/save.data ","r+")

    filecontents=readdlm(readingfile)

    step,paramdata,configdata = readinput(filecontents)
    mc_params = initialiseparams(paramdata)
    mc_states = readconfigs(configdata,mc_params.n_atoms,mc_params.n_traj,potential)

    return mc_params,mc_states,step

end