"""
    Checkpoint
Module designed to save relevant parameters and configurations throughout the simulation to allow restarting.
"""

module Checkpoint 

using StaticArrays,DelimitedFiles

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..Ensembles
using ..EnergyEvaluation
using ..MCStates


export save_init,save_histparams,checkpoint
export read_init,setresults,rebuild_states,read_config,build_states
#---------------------------------------------------------------#
#-----------------------Static Parameters-----------------------#
#---------------------------------------------------------------#
"""
    writeparams(savefile,params,temp)
Function to write the `mc_params` and `temp` data into a `savefile`. These are static parameters that define how the simulation is to proceed such as the number of cycles, trajectories and the temperatures to be covered.
"""
function writeparams(savefile,params,temp)
    headersvec = ["cycles:" "sample_rate:" "n_traj:" "n_atoms:" "min_acc:" "max_acc" "t_i:" "t_f"]
    paramsvec = [params.mc_cycles params.mc_sample params.n_traj params.n_atoms params.min_acc params.max_acc first(temp.t_grid) last(temp.t_grid)]
    writedlm(savefile, [headersvec, paramsvec], ' ' )
end
"""
    writeensemble(savefile,ensemble::NVT)
    writeensemble(savefile,ensemble::NPT)
Function to write the `ensemble` data into a savefile including the move types. First method is for the NVT ensemble which does not include volume changes, second method is NPT ensemble and does inclue volume moves.
"""
function writeensemble(savefile,ensemble::NVT)

    headersvec = ["ensemble" "n_atom_moves" "n_atom_swaps" ]
    valuesvec = ["NVT" ensemble.n_atoms ensemble.n_atom_moves ensemble.n_atom_swaps]
    writedlm(savefile, [headersvec, valuesvec], ' ' )
end
function writeensemble(savefile,ensemble::NPT)

    headersvec = ["ensemble" "n_atom_moves" "n_volume_moves" "n_atom_swaps" "pressure"]
    valuesvec = ["NPT" ensemble.n_atoms ensemble.n_atom_moves ensembles.n_volume_moves emsemble.n_atom_swaps ensemble.pressure]
    writedlm(savefile, [headersvec, valuesvec], ' ' )
end
"""
    writepotential(savefile,potential::Ptype)
    Ptype = AbstractDimerPotential,AbstractDimerPotentialB,EmbeddedAtomPotential,AbstractMachineLearningPotential
Function to write potential surface information into `savefile`. implemented methods are the Embedded Atom Model, Extended Lennard-Jones and ELJ in Magnetic Field. This does not work for machine learning potentials.
"""
function writepotential(savefile,potential::Ptype) where Ptype <: AbstractDimerPotential
    coeff_vec = transpose([potential.coeff[i] for i in eachindex(potential.coeff)])

    write(savefile,"$(typeof(potential)) " )
    writedlm(savefile, coeff_vec, ' ')
end
function writepotential(savefile,potential::Ptype) where Ptype <: AbstractDimerPotentialB

    coeff_vec_a = transpose([potential.coeff_a[i] for i in eachindex(potential.coeff_a)])
    coeff_vec_b = transpose([potential.coeff_b[i] for i in eachindex(potential.coeff_b)])
    coeff_vec_c = transpose([potential.coeff_c[i] for i in eachindex(potential.coeff_c)])
    write(savefile,"ELJB \n")
    writedlm(savefile, [coeff_vec_a, coeff_vec_b, coeff_vec_c])
end
function writepotential(savefile,potential::Ptype) where Ptype <: EmbeddedAtomPotential
    write(savefile,"EAM: $(potential.n) $(potential.m) $(potential.ean) $(potential.eCam) \n")
end 
function writepotential(savefile,potential::Ptype) where Ptype <: AbstractMachineLearningPotential
    write(savefile,"Define the potential elsewhere, it's too complicated for simple I/O \n")
end
"""
    save_init(potential,ensemble,params,temp)
Function to write all static parameters into a single parameters file. If a params file does not exist, it is created in ./checkpoint as params.data. This contains the `mc_params` `ensemble` and `potential` data.
"""
function save_init(potential,ensemble,params,temp)
    if ispath("./checkpoint/params.data") == true
    else
        mkpath("./checkpoint/")
        paramsfile = open("./checkpoint/params.data","w+")
        
        writeparams(paramsfile,params,temp)
        writeensemble(paramsfile,ensemble)
        writepotential(paramsfile,potential)
        
        close(paramsfile)
    end
end
#-----------------------------------------------------------------#
#-----------------------Post-Equilibration------------------------#
#-----------------------------------------------------------------#
"""
    save_histparams(results)
Initialises and populates a data file containing the information necessary to interpret histogram data
"""
function save_histparams(results)
    if ispath("./checkpoint/hist_info.data")==true
    else
        resfile = open("./checkpoint/hist_info.data","w+")
        headersvec = ["e_min" "e_max" "v_min" "v_max" "Δ_E" "Δ_V" "Δ_r2"]
        infovec = [results.en_min results.en_max results.v_min results.v_max results.delta_en_hist results.delta_v_hist results.delta_r2]
        writedlm(resfile,[headersvec , infovec], ' ')
        close(resfile)
    end
end
#-----------------------------------------------------------------#
#-----------------------Save Configurations-----------------------#
#-----------------------------------------------------------------#
"""
    checkpoint_config(savefile , config::Config{N,BC,T}, max_displ) 
        BC=SphericalBC,CubicBC,RhombicBC
Function writes a single config in the standard xyz format. `N` atoms, the comment line contains the boundary condition information (implemented for Spherical BC and bothtypes of Periodic BC) as well as `max_displ` information determining the stepsize used at the current step of the monte carlo simulation. The comment row is followed by 1 as a placeholder for the atom type to be implemented in future and the positions x,y,z in order.
"""
function checkpoint_config(savefile , state::MCState{T,N,BC,Ptype,Etype}) where {T,N,BC<:SphericalBC,Ptype,Etype}
    writedlm(savefile,[N,"$BC r2: $(state.config.bc.radius2) $(state.max_displ[1]) $(state.max_displ[2]) $(state.max_displ[3]) $(state.count_atom[1]) $(state.count_vol[1])"])
    
    for row in state.config.pos
        write(savefile, "1  $(row[1]) $(row[2]) $(row[3]) \n")
    end
end
function checkpoint_config(savefile ,state::MCState{T,N,BC,Ptype,Etype}) where {T,N,BC<:CubicBC,Ptype,Etype}
    writedlm(savefile,[N,"$BC box_length: $(state.config.bc.radius2) $(state.max_displ[1]) $(state.max_displ[2]) $(state.max_displ[3]) $(state.count_atom[1]) $(state.count_vol[1])"])
    
    for row in state.config.pos
        write(savefile, "1  $(row[1]) $(row[2]) $(row[3]) \n")
    end
end
function checkpoint_config(savefile , state::MCState{T,N,BC,Ptype,Etype}) where {T,N,BC<:RhombicBC,Ptype,Etype}
    writedlm(savefile,[N,"$BC box_dims: $(state.config.bc.radius2) $(state.max_displ[1]) $(state.max_displ[2]) $(state.max_displ[3]) $(state.count_atom[1]) $(state.count_vol[1])"])
    
    for row in state.config.pos
        write(savefile, "1  $(row[1]) $(row[2]) $(row[3]) \n")
    end
end
"""
    save_checkpoint(mc_states::Vector{stype}) where stype <: MCState
Function to save the configuration of each state in a vector of `mc_states`. writes each configuration according to [`writeconfig`](@ref) into a file config.i where i indicates the order of the states. 
"""
function save_configs(mc_states::Vector{stype}) where stype <: MCState
    for saveindex in eachindex(mc_states)
        checkpoint_file = open("./checkpoint/config.$saveindex","w")
        checkpoint_config(checkpoint_file,mc_states[saveindex])
        # checkpoint_config(checkpoint_file,mc_states[saveindex].config, mc_states[saveindex].max_displ, mc_states[saveindex].count_atom[1],mc_states[saveindex].count_volume[1])
        close(checkpoint_file)
    end
end
"""
    checkpoint(index,mc_states,results,ensemble;rdfsave=false)
Function to save relevant information about the current state of the system at step `index`. saves the configurations in each `mc_state` [`save_configs`](@ref) as well as the histograms stored in `results`. Optionally stores the volume histograms if using the NPT ensemble and the radial distribution functions if desired. 
"""
function checkpoint(index,mc_states,results,ensemble::NVT,rdfsave)
    
    indexfile = open("./checkpoint/index.txt","w+")
    writedlm(indexfile,index)
    close(indexfile)
    save_configs(mc_states)
    histfile = open("./checkpoint/histograms.data","w+")
    writedlm(histfile,results.en_histogram)
    close(histfile)
    if rdfsave == true 
        rdffile = open("./checkpoint/rdf.data","w+")
        writedlm(rdffile,results.rdf)
        close(rdffile)
    else
    end
    
end
function checkpoint(index,mc_states,results,ensemble::NPT, rdfsave)
    indexfile = open("./checkpoint/index.txt","w+")
    writedlm(indexfile,index)
    close(indexfile)
    save_configs(mc_states)
    histfile = open("./checkpoint/histograms.data","w+")
    writedlm(histfile,results.en_histogram)
    close(histfile)
    v_file = open("./checkpoint/volume_hist","w+")
    writedlm(v_file,ev_histogram)
    close(v_file)
    if rdfsave == true 
        rdffile = open("./checkpoint/rdf.data","w+")
        writedlm(rdffile,results.rdf)
        close(rdffile)
    else
    end    
end
#--------------------------------------------------------------------#
#-----------------------------Read Files-----------------------------#
#--------------------------------------------------------------------#
"""
    readensemble(ensemblevec)  
Function to convert delimited file contents `ensemblevec` and convert them into an ensemble.

"""
function readensemble(ensemblevec)
    if contains(ensemblevec[1],"NVT")
        return NVT(ensemblevec[2],ensemblevec[3],ensemblevec[4])
    elseif contains(ensemblevec[1],"NPT")
        return NPT(ensemblevec[2],ensemblevec[3],ensemblevec[4],ensemblevec[5],ensemblevec[6])
    end
end
"""
    readpotential(potinfovec)
Function to convert delimited file contents `potinfovec` into a potential. Implemented for:
    ELJEven,ELJ,ELJB and EmbeddedAtomPotential
"""
function readpotential(potinfovec)

    if contains(potinfovec[1,1],"ELJPotentialEven")
        coeffs= Vector(potinfovec[1,3:end])
        return ELJPotentialEven(coeffs)
    elseif contains(potinfovec[1,1],"ELJPotential{")
        coeffs = Vector(potinfovec[1,3:end])
        return ELJPotential(coeffs)
    elseif potinfovec[1,1] == "EAM:"
        return EmbeddedAtomPotential(potinfovec[1,2],potinfovec[1,3],potinfovec[1,4],potinfovec[1,5])
    elseif potinfovec[1,1] == "ELJB"
        a,b,c = potinfovec[2,:],potinfovec[3,:],potinfovec[4,:]
        return ELJPotentialB(a,b,c)
    end
end
"""
    read_params(paramsvec)
    read_params(paramsvec,restart)
Function to turn a delimited `paramsvec` into an MCParams and TempGrid struct. Second method includes a bool `restart` to determine whether or not to set eq_cycles=0 or 0.2*cycles if not restarting
"""
function read_params(paramsvec)
    parameters = MCParams(paramsvec[1],0,paramsvec[2],paramsvec[3],paramsvec[4],100,100,paramsvec[5],paramsvec[6])
    temps = TempGrid{Int(paramsvec[3])}(paramsvec[7],paramsvec[8])
    return parameters,temps
end

function read_params(paramsvec,restart)
    if restart == true
        parameters = MCParams(paramsvec[1],0,paramsvec[2],paramsvec[3],paramsvec[4],100,100,paramsvec[5],paramsvec[6])
    else
        parameters = MCParams(paramsvec[1],Int(floor(0.2*paramsvec[1])),paramsvec[2],paramsvec[3],paramsvec[4],100,100,paramsvec[5],paramsvec[6])
    end
    temps = TempGrid{Int(paramsvec[3])}(paramsvec[7],paramsvec[8])
    return parameters,temps
end
"""
    read_init()
Function to reinitialise the fixed parameters of the MC simulation as saved by the [`save_init`](@ref) function. returns 
"""
function read_init(restart::Bool)
    readfile=open("./checkpoint/params.data","r+")
    data=readdlm(readfile)
    close(readfile)

    paramsvec=data[2,:]
    ensemblevec=data[4,:]
    potinfovec=data[5:end,:]

    mc_params,temp = read_params(paramsvec,restart)
    ensemble = readensemble(ensemblevec)
    potential=readpotential(potinfovec)

    return mc_params,temp,ensemble,potential
end

"""
    read_checkpoint_config(xyzdata)
Function designed to take a single xyz-style checkpoint file and return the configuration and max displacement data associated with a saved mc_state. This is used to reconstruct an mc simulation from checkpoints
"""
function read_checkpoint_config(xyzdata)
    N=xyzdata[1,1]
    if xyzdata[2,1] == "SphericalBC{Float64}"
        bc = SphericalBC(;radius=sqrt(xyzdata[2,3]))
        
        max_displ = [xyzdata[2,4:6]]
        count_atom = xyzdata[2,7]
        count_vol=xyzdata[2,8]
    elseif xyzdata[2,1] == "CubicBC{Float64}"
        bc = CubicBC(xyzdata[2,3])

        max_displ = [xyzdata[2,4:6]]
        count_atom = xyzdata[2,7]
        count_vol=xyzdata[2,8]
    elseif xyzdata[2,1] == "RhombicBC"
        bc = RhombicBC(xyzdata[2,3],xyzdata[2,4])
        max_displ = [xyzdata[2,5:7]]
        count_atom = xyzdata[2,8]
        count_vol=xyzdata[2,9]
    end
    configvectors = [xyzdata[2+i,2:4] for i in 1:N]

    config=Config(configvectors,bc)

    return config , max_displ, count_atom, count_vol
end
"""
    read_config(xyzdata)
designed to read in one xyz-style file with one configuration and return this for starting a simulaiton from files without restarting.
"""
function read_config(xyzdata)
    N=xyzdata[1,1]
    if xyzdata[2,1] == "SphericalBC{Float64}"
        bc = SphericalBC(;radius=sqrt(xyzdata[2,3]))

    elseif xyzdata[2,1] == "CubicBC{Float64}"
        bc = CubicBC(xyzdata[2,3])

    elseif xyzdata[2,1] == "RhombicBC"
        bc = RhombicBC(xyzdata[2,3],xyzdata[2,4])
    end
    configvectors = [xyzdata[2+i,2:4] for i in 1:N]

    config=Config(configvectors,bc)

    return config
end
"""
    setresults(histparams,histdata,histv_data,r2data)
Function to re-initialise the results struct on restarting a simulation.
"""
function setresults(histparams,histdata,histv_data,r2data)
    results=Output{Float64}(100)
    results.en_min,results.en_max=histparams[1],histparams[2]
    results.v_min,results.v_max = histparams[3],histparams[4]
    results.delta_en_hist,results.delta_v_hist,results.delta_r2=histparams[5],histparams[6],histparams[7]

    for row in eachrow(histdata)
        push!(results.en_histogram,Vector(row))
    end
    if typeof(histv_data) == Matrix{Float64}
        for row in eachrow(histv_data)
            push!(results.ev_histogram,Vector(row))
        end
    end
    if typeof(r2data) == Matrix{Float64}
        for row in eachrow(r2data)
            push!(results.rdf,Vector(row))
        end
    end
    return results
end
"""
    rebuild_states(n_traj,ensemble,temps,potential)
Function to rebuild the MCStates vector and results struct from checkpoint information. The `ensemble` `temps` and `potential` along with `n_traj` are reconstructed elsewhere, but required to accurately recreate the states. 
"""
function rebuild_states(n_traj,ensemble,temps,potential)
    histinfofile=open("./checkpoint/hist_info.data","r+")
    histsfile=open("./checkpoint/histograms.data","r+")
    dataread = readdlm(histinfofile)
    histiread=readdlm(histsfile)
    close(histinfofile)
    close(histsfile)
    if isfile("./checkpoint/rdf.data") == true
        rdffile = open("./checkpoint/rdf.data","r+")
        rdfinfo= readdlm(rdffile)
        close(rdffile)
    else
        rdfinfo = false
    end
    if typeof(ensemble) == NPT
        ev_file = open("./checkpoint/volume_hist.data","r+")
        ev_info = readdlm(ev_file)
        close(ev_file)
    else
        ev_info = false
    end
    results = setresults(dataread[2,:],histiread,ev_info,rdfinfo)
    mcstates = []
    for index in 1:n_traj
        configfile = open("./checkpoint/config.$index","r+")
        configinfo=readdlm(configfile)
        close(configfile)
        conf,maxdisp,countatom,countvol = read_checkpoint_config(configinfo)

        state = MCState(temps.t_grid[index],temps.beta_grid[index],conf,ensemble,potential;max_displ=maxdisp[1],count_atom = [countatom,0],count_vol=[countvol,0])
        push!(mcstates,state)
    end

    return [state for state in mcstates] , results
end
"""
    build_states(mc_params,ensemble,temp,potential)
For use initialising states and outputs when NOT restarting, but beginning from files. Builds empty Output struct named `results` and a vector of `mc_states` using either: one configuration stored in config.data OR a series of configurations stored in config.i. NB if config.i doesn't exist the default will be config.1 . In this way states can be initialised with different starting configurations.  
"""
function build_states(mc_params,ensemble,temp,potential)
    if ispath("./checkpoint/config.1")
    confvec=[]
    for i in 1:mc_params.n_traj 
        if ispath("./checkpoint/config.$i")
            confinfo=readdlm("./checkpoint/config.$i")
        else
            confinfo=readdlm("./checkpoint/config.1")
        end
        conf=read_config(confinfo)
        push!(confvec,conf)
    end

    mc_states = [MCState(temp.t_grid[i],temp.beta_grid[i],confvec[i],ensemble,potential) for i in 1:mc_params.n_traj]
    results = Output{Float64}(mc_params.n_bin;en_min=mc_states[1].en_tot)

    elseif ispath("./checkpoint/config.data")

        confinfo = readdlm("./checkpoint/config.data")
        start_config = read_config(confinfo)
        mc_states = [MCState(temp.t_grid[i],temp.beta_grid[i],start_config,ensemble,potential) for i in 1:mc_params.n_traj]
        results = Output{Float64}(mc_params.n_bin;en_min=mc_states[1].en_tot)
    end
    return mc_states,results

end

end