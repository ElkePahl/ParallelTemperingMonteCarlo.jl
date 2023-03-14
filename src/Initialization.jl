"""
    module Initialization
In light of the increasing complexity of the ReadSave module, I aim to move the functions relating to actually initialising simulations in one place, as well as some dependencies relating to equilibration cycles.
"""

module Initialization

export init_sim,restart_ptmc, initialisation

using StaticArrays,DelimitedFiles,Random
using ..MCStates
using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..Exchange
using ..ReadSave


"""
    init_sim(pot ;dir=pwd())
Function designed to take an input file named by kwarg `file` and open and initalise the states and parameters required for the simulation. 
    
    ---This is not exactly the same as the restart as the params etc are distributed in separate files by the checkpoint function to prevent duplication and slowdown throughout the simulation. 
"""
function init_sim(pot ,file,eq_percentage)
    file=open(file,"r+")
    init = readdlm(file)
    close(file)
    paramsdata,simdata,config_data = init[1:8,:],init[9:11,:],init[12:end,:]

    ensemble,move_strat,mc_params = initialise_params(paramsdata,eq_percentage)

    temps=TempGrid{mc_params.n_traj}(simdata[1,2],simdata[1,3])

    #max_displ_atom=[0.1*sqrt(simdata[2,2]*t) for t in temps.t_grid]

    start_config = read_config(config_data)

    length(start_config.pos) == mc_params.n_atoms || error("number of atoms and positions not the same - check starting config")

    
    mc_states = [MCState(temps.t_grid[i], temps.beta_grid[i], start_config, pot) for i in 1:mc_params.n_traj]

    results = Output{Float64}(simdata[3,2]; en_min = mc_states[1].en_tot)

    return mc_states,move_strat,mc_params,pot,ensemble,results

end


"""
    function restart_ptmc(potential, directory)
function takes a potential struct and optionally the directory of the savefile, this returns the params, states and the step at which data was saved.
"""
function restart_ptmc(potential,directory)

    readfile = open("$(directory)/save.data","r+")

    filecontents=readdlm(readfile)

    step,configdata = read_input(filecontents)


    close(readfile)
    paramfile =  open("$(directory)/params.data")
    paramdata = readdlm(paramfile)

    close(paramfile)

    ensemble,move_strat,mc_params = initialise_params(paramdata,0.0)
    mc_states = read_states(configdata,mc_params.n_atoms,mc_params.n_traj,potential)
    results  = read_results(directory = directory)

    return mc_states,move_strat,mc_params,potential,ensemble,results,step

end
"""
    initialisation( pot, save_dir )
Function to restart parallel simulations through the restart_ptmc function. 
"""
function initialisation(restart::Bool, pot, save_dir;eq_percentage = 0.2,startfile="input.data")

    if restart == true
        mc_states,move_strat,mc_params,pot,ensemble,results,start_counter = restart_ptmc(pot,save_dir)
    else
        mc_states,move_strat,mc_params,pot,ensemble,results = init_sim(pot,"$(save_dir)/$(startfile)",eq_percentage)
        start_counter = 1
    end

    a,v,r = atom_move_frequency(move_strat),vol_move_frequency(move_strat),rot_move_frequency(move_strat)
    n_steps = a + v + r

    println("Total number of moves per MC cycle: ", n_steps)
    println()


    return mc_states,mc_params,move_strat,pot,ensemble,results,start_counter,n_steps,a,v,r

end
function initialisation(restart,mc_states, move_strat, mc_params, pot, ensemble, results;save=true,save_dir=pwd(),startfile="input.data")

    a,v,r = atom_move_frequency(move_strat),vol_move_frequency(move_strat),rot_move_frequency(move_strat)
    n_steps = a + v + r

    start_counter = 1

    return mc_states,mc_params,move_strat,pot,ensemble,results,start_counter,n_steps,a,v,r
end

end