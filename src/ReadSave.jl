module ReadSave


export save_params,save_state,save_results,save_states
export restart_ptmc,read_results
export read_multihist
using StaticArrays
using DelimitedFiles
using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..EnergyEvaluation
using ..MCStates
using ..MCMoves



"""
    save_params(savefile::IOStream, mc_params::MCParams)
    (savefile::IOStream, mc_params::MCParams,move_strat,ensemble)
writes the MCParam struct to a savefile. Second method also writes the move strategy and ensemble.
"""
 function save_params(savefile::IOStream, mc_params::MCParams)
     write(savefile,"MC_Params \n")
     write(savefile,"total_cycles: $(mc_params.mc_cycles)\n")
     write(savefile,"mc_samples: $(mc_params.mc_sample)\n")
     write(savefile,"n_traj: $(mc_params.n_traj)\n")
     write(savefile, "n_atoms: $(mc_params.n_atoms)\n")
     write(savefile,"n_adjust: $(mc_params.n_adjust)\n")
     
    #  close(savefile)
 end
 function save_params(savefile::IOStream, mc_params::MCParams,move_strat,ensemble)
    write(savefile,"MC_Params \n")
    write(savefile,"total_cycles: $(mc_params.mc_cycles)\n")
    write(savefile,"mc_samples: $(mc_params.mc_sample)\n")
    write(savefile,"n_traj: $(mc_params.n_traj)\n")
    write(savefile, "n_atoms: $(mc_params.n_atoms)\n")
    write(savefile,"n_adjust: $(mc_params.n_adjust)\n")
    write(savefile,"ensemble: $ensemble\n" )
    write(savefile,"avr: $(atom_move_frequency(move_strat)) $(vol_move_frequency(move_strat)) $(rot_move_frequency(move_strat)) \n")
   #  close(savefile)
end
"""
    save_state(savefile::IOStream,mc_state::MCState)
saves a single mc_state struct to a savefile
"""
function save_state(savefile::IOStream,mc_state::MCState)
    write(savefile,"temp_beta: $(mc_state.temp) $(mc_state.beta) \n")
    write(savefile,"total_energy: $(mc_state.en_tot)\n")
    write(savefile,"max_displacement: $(mc_state.max_displ[1]) $(mc_state.max_displ[2]) $(mc_state.max_displ[3])\n")
    write(savefile, "counts_a/v/r/ex:  $(mc_state.count_atom[1])   $(mc_state.count_atom[2]) $(mc_state.count_vol[1]) $(mc_state.count_vol[2]) $(mc_state.count_rot[1]) $(mc_state.count_rot[2]) $(mc_state.count_exc[1]) $(mc_state.count_exc[2]) \n")

    if length(mc_state.ham) > 2
        ham1 = sum(mc_state.ham)
        ham2 = sum( mc_state.ham .* mc_state.ham)
    elseif length(mc_state.ham) == 2
        ham1 = mc_state.ham[1]
        ham2 = mc_state.ham[2]
    else
        ham1 = 0
        ham2 = 0
    end
    write(savefile, "E,E2: $ham1 $ham2 \n")
    if typeof(mc_state.config.bc) == SphericalBC{Float64}
        write(savefile, "Boundary: $(typeof(mc_state.config.bc))  $(sqrt(mc_state.config.bc.radius2)) \n")
    elseif typeof(mc_state.config.bc) == PeriodicBC{Float64}
        write(savefile, "Boundary: $(typeof(mc_state.config.bc))$(mc_state.config.bc.box_length) \n" )
    end
    write(savefile,"configuration \n")
    for row in mc_state.config.pos
        write(savefile,"$(row[1]) $(row[2]) $(row[3]) \n")
    end

end
"""
    save_results(results::Output; directory = pwd())
Saves the on the fly results and histogram information for re-reading.
"""

function save_results(results::Output, directory)

    resultsfile =  open("$(directory)/results.data","w+")
    rdf_file = open("$directory/RDF.data","w+")
    write(resultsfile,"emin,emax,nbins= $(results.en_min) $(results.en_max) $(results.n_bin) \n")
    write(resultsfile, "Histograms \n")
    writedlm(resultsfile,results.en_histogram)
    close(resultsfile)
    writedlm(rdf_file,results.rdf)
    close(rdf_file)
    
end
"""
    save_states(mc_params,mc_states,trial_index; directory = pwd())
opens a savefile, writes the mc params and states and the trial at which it was run. 
"""
function save_states(mc_params,mc_states,trial_index, directory; filename="save.data")
    dummy_index = 0 
    savefile = open("$(directory)/$(filename)","w+")

    if isfile("$directory/params.data") == false
        paramsfile = open("$directory/params.data","w+")
        save_params(paramsfile,mc_params)
        close(paramsfile)
    end

    write(savefile,"Save_made_at_step $trial_index \n") #
    for state in mc_states
        dummy_index += 1
        write(savefile, "config $dummy_index \n")
        save_state(savefile,state)
        write(savefile,"end \n")
    end
    close(savefile)
end
function save_states(mc_params,mc_states,trial_index, directory,move_strat,ensemble; filename="save.data")
    dummy_index = 0 
    savefile = open("$(directory)/$(filename)","w+")

    if isfile("$directory/params.data") == false
        paramsfile = open("$directory/params.data","w+")
        save_params(paramsfile,mc_params,move_strat,ensemble)
        close(paramsfile)
    end

    write(savefile,"Save made at step $trial_index \n") #
    for state in mc_states
        dummy_index += 1
        write(savefile, "config $dummy_index \n")
        save_state(savefile,state)
        write(savefile,"end \n")
    end
    close(savefile)
end
"""
    readinput(savedata)
takes the delimited contents of a savefile and splits it into paramdata to reinitialise MC_param, configuration data to reinitialise n_traj mc_states, and the step at which the save was made.
"""

function read_input(savedata)

    step = savedata[1,2]
    configdata = savedata[2:end,:]

    return step,configdata
end

"""
    initialiseparams(paramdata)
accepts an array of the delimited paramdata and returns an MCParam struct based on saved data
"""
function initialise_params(paramdata)

    MC_param = MCParams(paramdata[2,2],paramdata[4,2],paramdata[5,2],mc_sample = paramdata[3,2], n_adjust = paramdata[6,2])
    
    ensemble = eval(Meta.parse(paramdata[7,2]))
    a,v,r = paramdata[8,2],paramdata[8,3],paramdata[8,4]
    move_strat = MoveStrategy(atom_moves=a,vol_moves=v,rot_moves=r)
    
    return ensemble,move_strat,MC_param
    
end
function read_config(config_info)
    positions =  []
    if config_info[1,2] == "SphericalBC{Float64}"
        boundarycondition = SphericalBC(radius = (config_info[1,3]))
    end

    for row in eachrow(config_info[3:end,:])
        coord_atom = SVector(row[1] ,row[2] ,row[3] )
        push!(positions,coord_atom)
    end

    config = Config(positions,boundarycondition)
    
    return config
end
"""
    read_state(onestatevec,n_atoms, potential)
reads a single trajectory based on the savefile format. The potential must be manually added, though there is the possibility of including this in the savefile if required. Output is a single MCState struct.
"""
function read_state(onestatevec,n_atoms, potential)
    config = read_config(onestatevec[7:end-1,:])
    # for j=1:n_atoms
    #     coord_atom = SVector(onestatevec[8+j,1] ,onestatevec[8+j,2] ,onestatevec[8+j,3] )
    #     push!(positions,coord_atom)
    # end
    # if onestatevec[7,2] == "SphericalBC{Float64}"
    #     boundarycondition = SphericalBC(radius = (onestatevec[7,3]))
    # end
    counta = [onestatevec[5,2], onestatevec[5,3]]
    countv = [onestatevec[5,4], onestatevec[5,5]]
    countr = [onestatevec[5,6], onestatevec[5,7]]
    countx = [onestatevec[5,8], onestatevec[5,9]]
    
    mcstate = MCState(onestatevec[2,2], onestatevec[2,3],config, potential ; max_displ=[onestatevec[4,2], onestatevec[4,3], onestatevec[4,4] ], count_atom=counta,count_vol=countv,count_rot=countr,count_exc=countx) #initialise the mcstate

    mcstate.en_tot = onestatevec[3,2] #incl current energy

    push!(mcstate.ham,onestatevec[6,2])
    push!(mcstate.ham,onestatevec[6,3]) #incl the hamiltonian and hamiltonia squared vectors


    

    return mcstate
end

"""
    read_states(trajvecs,n_atoms,n_traj,potential)
takes the entirety of the trajectory information, splits it into n_traj configs and outputs them as a new mc_states vector.
"""
function read_states(trajvecs,n_atoms,n_traj,potential)
    states = []
    lines = Int64(9+n_atoms)
    for idx=1:n_traj
        onetraj = trajvecs[1+ (idx - 1)*lines:(idx*lines), :]
        onestate = read_state(onetraj,n_atoms,potential)

        push!(states,onestate)

    end

    return states
end

function read_results(;directory=pwd())
    resfile = open("$(directory)/results.data")
    rdffile = open("$directory/RDF.data")
    histinfo=readdlm(resfile)
    rdfdata = readdlm(rdffile)
    close(rdffile)
    close(resfile)
    
    emin,emax,nbins = histinfo[1,2:4]
    histograms = []
    rdf = []
    for i in 3:size(histinfo,1)
        hist = histinfo[i,:]
        push!(histograms,hist)
    end
    for row in eachrow(rdfdata)
        push!(rdf,row)
    end

    results = Output{Float64}(nbins; en_min = emin)
    results.en_min = emin
    results.en_max = emax
    results.en_histogram = histograms
    results.rdf = rdf

    return results

end

"""
    function restart(potential ;directory = pwd())
function takes a potential struct and optionally the directory of the savefile, this returns the params, states and the step at which data was saved.
"""
function restart_ptmc(potential ;directory = pwd())

    readfile = open("$(directory)/save.data","r+")

    filecontents=readdlm(readfile)

    step,configdata = read_input(filecontents)


    close(readfile)
    paramfile =  open("$(directory)/params.data")
    paramdata = readdlm(paramfile)

    close(paramfile)

    ensemble,move_strat,mc_params = initialise_params(paramdata)
    mc_states = read_states(configdata,mc_params.n_atoms,mc_params.n_traj,potential)
    results  = read_results(directory = directory)


    return results,ensemble,move_strat,mc_params,mc_states,step

end

"""
    read_multihist(;directory=pwd())
function designed to open and parse multihistogram information. Accepts a single keyword argument pointing to the location of the multihist file analysis.NVT
    returns Temp, heat capacity, its derivative and entropy. All vectors ready to be plotted. 

Not included: plotting functionality as this requires Plots to be added to the manifest and project, slowing down compile time.
"""
function read_multihist(;directory=pwd())
    multihist_file = open("$(directory)/analysis.NVT")
    multihist_info = readdlm(multihist_file)
    close(multihist_file)

    labels,M = multihist_info[1,:],multihist_info[2:end,:]
    T,Cv,dCv,S = M[:,1],M[:,3],M[:,4],M[:,5]

    return T,Cv,dCv,S
end

end