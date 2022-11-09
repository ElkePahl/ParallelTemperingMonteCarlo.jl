module MCRun

export MCState
export metropolis_condition, mc_step!, mc_cycle!,ptmc_cycle!, ptmc_run!,save_states,save_params,save_results
export atom_move!,update_max_stepsize!
export exc_acceptance, exc_trajectories!

using StaticArrays,DelimitedFiles

using ..BoundaryConditions
using ..Configurations
using ..InputParams
using ..MCMoves
using ..EnergyEvaluation
using ..RuNNer

"""
    MCState(temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_vec, en_tot; 
        max_displ = [0.1,0.1,1.], count_atom = [0,0], count_vol = [0,0], count_rot = [0,0], count_exc = [0,0])
    MCState(temp, beta, config::Config, pot; kwargs...) 
Creates an MC state vector at a given temperature `temp` containing temperature-dependent information

Fieldnames:
- `temp`: temperature
- `beta`: inverse temperature
- `config`: actual configuration in Markov chain [`Config`](@ref)  
- `dist_2mat`: matrix of squared distances d_ij between atoms i and j; generated automatically when potential `pot` given
- `en_atom_vec`: vector of energy contributions per atom i; generated automatically when `pot` given
- `en_tot`: total energy of `config`; generated automatically when `pot` given
- `ham`: vector containing sampled energies - generated in MC run
- `max_displ`: max_diplacements for atom, volume and rotational moves; key-word argument
- `count_atom`: number of accepted atom moves - total and between adjustment of step sizes; key-word argument
- `count_vol`: number of accepted volume moves - total and between adjustment of step sizes; key-word argument
- `count_rot`: number of accepted rotational moves - total and between adjustment of step sizes; key-word argument
- `count_exc`: number of attempted (10%) and accepted exchanges with neighbouring trajectories; key-word argument
"""
mutable struct MCState{T,N,BC}
    temp::T
    beta::T
    config::Config{N,BC,T}
    dist2_mat::Matrix{T}
    en_atom_vec::Vector{T}
    en_tot::T
    ham::Vector{T}
    max_displ::Vector{T}
    count_atom::Vector{Int}
    count_vol::Vector{Int}
    count_rot::Vector{Int}
    count_exc::Vector{Int}
end    

function MCState(
    temp, beta, config::Config{N,BC,T}, dist2_mat, en_atom_vec, en_tot; 
    max_displ = [0.1,0.1,1.], count_atom = [0,0], count_vol = [0,0], count_rot = [0,0], count_exc = [0,0]
) where {T,N,BC}
    ham = T[]
    MCState{T,N,BC}(
        temp, beta, deepcopy(config), copy(dist2_mat), copy(en_atom_vec), en_tot, 
        ham, copy(max_displ), copy(count_atom), copy(count_vol), copy(count_rot), copy(count_exc)
        )
end

function MCState(temp, beta, config::Config, pot::AbstractDimerPotential; kwargs...) 
   dist2_mat = get_distance2_mat(config)
   n_atoms = length(config.pos)
   en_atom_vec, en_tot = dimer_energy_config(dist2_mat, n_atoms, pot)
   MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end

function MCState(temp,beta, config::Config, pot::AbstractMLPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    en_tot = RuNNer.getenergy(pot.dir, config,pot.atomtype)

    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)

end
function MCState(temp,beta, config::Config, pot::DFTPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    en_tot = getenergy_DFT(config.pos, pot)

    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end
function MCState(temp,beta, config::Config, pot::ParallelMLPotential;kwargs...)
    dist2_mat = get_distance2_mat(config)
    n_atoms = length(config.pos)
    en_atom_vec = zeros(n_atoms)
    
    en_tot = RuNNer.getenergy(pot.dir, config,pot.atomtype,pot.index)


    MCState(temp, beta, config, dist2_mat, en_atom_vec, en_tot; kwargs...)
end


"""
    metropolis_condition(ensemble, delta_en, beta)
Returns probability to accept a MC move at inverse temperature `beta` 
for energy difference `delta_en` between new and old configuration 
for given ensemble; implemented: 
    - `NVT`: canonical ensemble
    - `NPT`: NPT ensemble
Asymmetric Metropolis criterium, p = 1.0 if new configuration more stable, 
Boltzmann probability otherwise
"""
function metropolis_condition(::NVT, delta_energy, beta)
    prob_val = exp(-delta_energy*beta)
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

function metropolis_condition(::NPT, N, d_en, volume_changed, volume_unchanged, pressure, beta)
    delta_h = d_en + pressure*(volume_changed-volume_unchanged)*JtoEh*Bohr3tom3
    prob_val = exp(-delta_h*beta + NAtoms*log(volume_changed/volume_unchanged))
    T = typeof(prob_val)
    return ifelse(prob_val > 1, T(1), prob_val)
end

#function metropolis_condition(energy_unmoved, energy_moved, beta)
#    prob_val = exp(-(energy_moved-energy_unmoved)*beta)
#    T = typeof(prob_val)
#    return ifelse(prob_val > 1, T(1), prob_val)
#end


"""
    exc_acceptance(beta_1, beta_2, en_1, en_2)
Returns probability to exchange configurations of two trajectories with energies `en_1` and `en_2` 
at inverse temperatures `beta_1` and `beta_2`. 
"""
function exc_acceptance(beta_1, beta_2, en_1, en_2)
    d_en_acc = en_1 - en_2
    delta_beta = beta_1 - beta_2
    exc_acc = min(1.0,exp(delta_beta * d_en_acc))
    return exc_acc
end

"""
    exc_trajectories!(state_1::MCState, state_2::MCState)
Exchanges configurations and distance and energy information between two trajectories;
information contained in `state_1` and `state_2`, see [`MCState`](@ref)   
"""
function exc_trajectories!(state_1::MCState, state_2::MCState)
    state_1.config, state_2.config = state_2.config, state_1.config
    state_1.dist2_mat, state_2.dist2_mat = state_2.dist2_mat, state_1.dist2_mat
    state_1.en_atom_vec, state_2.en_atom_vec = state_2.en_atom_vec, state_1.en_atom_vec
    state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot
    return state_1, state_2
end 


"""
    update_max_stepsize!(mc_state::MCState, n_update, a, v, r)
Increases/decreases the max. displacement of atom, volume, and rotation moves to 110%/90% of old values
if acceptance rate is >60%/<40%. Acceptance rate is calculated after `n_update` MC cycles; 
each cycle consists of `a` atom, `v` volume and `r` rotation moves.
Information on actual max. displacement and accepted moves between updates is contained in `mc_state`, see [`MCState`](@ref).  
"""
function update_max_stepsize!(mc_state::MCState, n_update, a, v, r; min_acc = 0.4, max_acc = 0.6)
    #atom moves
    acc_rate = mc_state.count_atom[2] / (n_update * a)
    if acc_rate < min_acc
        mc_state.max_displ[1] *= 0.9
    elseif acc_rate > max_acc
        mc_state.max_displ[1] *= 1.1
    end
    mc_state.count_atom[2] = 0
    #volume moves
    if v > 0
        acc_rate = mc_state.count_vol[2] / (n_update * v)
        if acc_rate < min_acc
            mc_state.max_displ[2] *= 0.9
        elseif acc_rate > max_acc
            mc_state.max_displ[2] *= 1.1
        end
        mc_state.count_vol[2] = 0
    end
    #rotation moves
    if r > 0
        acc_rate = mc_state.count_rot[2] / (n_update * r)
        if acc_rate < min_acc
            mc_state.max_displ[3] *= 0.9
        elseif acc_rate > mac_acc
            mc_state.max_displ[3] *= 1.1
        end
        mc_state.count_rot[2] = 0
    end
    return mc_state
end
    
#    for i in 1:length(count_acc)
#       acc_rate =  count_accept[i] / (displ.update_step * n_atom)
#        if acc_rate < 0.4
#            displ.max_displacement[i] *= 0.9
#        elseif acc_rate > 0.6
#            displ.max_displacement[i] *= 1.1
#        end
#        count_accept[i] = 0
#    end
#    return displ, count_accept
#end

#function mc_step_atom!(config, beta, dist2_mat, en_tot, i_atom, max_displacement, count_acc, count_acc_adjust, pot)
    #move randomly selected atom (obeying the boundary conditions)
    #trial_pos = atom_displacement(config.pos[i_atom], max_displacement, config.bc)
    #find new distances of moved atom - might not be always needed?
    #dist2_new = [distance2(trial_pos,b) for b in config.pos]
    #en_moved = energy_update(i_atom, dist2_new, pot)
    #recalculate old 
    #en_unmoved = energy_update(i_atom, dist2_mat[i_atom,:], pot)
    #one might want to store dimer energies per atom in vector?
    #decide acceptance
    #if metropolis_condition(en_unmoved, en_moved, beta) >= rand()
        #new config accepted
    #    config.pos[i_atom] = copy(trial_pos)
    #    dist2_mat[i_atom,:] = copy(dist2_new)
    #    dist2_mat[:,i_atom] = copy(dist2_new)
    #    en_tot = en_tot - en_unmoved + en_moved
    #    count_acc += 1
    #    count_acc_adjust += 1
    #end 
    #return config, entot, dist2mat, count_acc, count_acc_adjust
#end

"""
    atom_move!(mc_state::MCState, i_atom, pot, ensemble)
Moves selected `i_atom`'s atom randomly (max. displacement as in `mc_state`).
Calculates energy update (depending on potential `pot`).
Accepts move according to Metropolis criterium (asymmetric choice, depending on `ensemble`)
Updates `mc_state` if move accepted.
"""
function atom_move!(mc_state::MCState, i_atom, pot, ensemble)
    #move randomly selected atom (obeying the boundary conditions)
    trial_pos = atom_displacement(mc_state.config.pos[i_atom], mc_state.max_displ[1], mc_state.config.bc)
    #find new distances of moved atom 

    d_en_move, dist2_new = energy_update(trial_pos, i_atom, mc_state.config, mc_state.dist2_mat, pot)

    #decide acceptance
    if metropolis_condition(ensemble, d_en_move, mc_state.beta) >= rand()
        #new config accepted
        mc_state.config.pos[i_atom] = trial_pos #copy(trial_pos)
        mc_state.dist2_mat[i_atom,:] = dist2_new #copy(dist2_new)
        mc_state.dist2_mat[:,i_atom] = dist2_new
        mc_state.en_tot += d_en_move
        mc_state.count_atom[1] += 1
        mc_state.count_atom[2] += 1
    end
    return mc_state #config, entot, dist2mat, count_acc, count_acc_adjust
end

"""
    mc_step!(mc_state::MCState, pot, ensemble, a, v, r)
Performs an individual MC step.
Chooses type of move randomly according to frequency of moves `a`,`v` and `r` 
for atom, volume and rotation moves.
Performs the selected move.   
"""
function mc_step!(mc_state::MCState, pot, ensemble, a, v, r)
    ran_atom = rand(1:(a+v+r)) #choose move randomly
    if ran_atom <= a
        mc_state = atom_move!(mc_state, ran_atom, pot, ensemble)
    #else if ran <= v
    #    vol_move!(mc_state, pot, ensemble)
    #else if ran <= r
    #    rot_move!(mc_state, pot, ensemble)
    end
    return mc_state
end 

"""
    mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
Performs a MC cycle consisting of `n_steps` individual MC steps 
(frequencies of moves given by `a`,`v` and `r`).
Attempts parallel tempering step for 10% of cycles.
Exchanges trajectories if exchange accepted.
"""
function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
    #perform one MC cycle
    for i_traj = 1:mc_params.n_traj
        for i_step = 1:n_steps
            #mc_states[i_traj] = mc_step!(type_moves[ran][2], type_moves[ran][1], mc_states[i_traj], ran, pot, ensemble)
            @inbounds mc_states[i_traj] = mc_step!(mc_states[i_traj], pot, ensemble, a, v, r)
        end
        #push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
    end
    #parallel tempering
    if rand() < 0.1 #attempt to exchange trajectories
        n_exc = rand(1:mc_params.n_traj-1)
        mc_states[n_exc].count_exc[1] += 1
        mc_states[n_exc+1].count_exc[1] += 1

        exc_acc = exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot)

        if exc_acc > rand()
            mc_states[n_exc].count_exc[2] += 1
            mc_states[n_exc+1].count_exc[2] += 1

            mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
        end
    end

    return mc_states
end
"""
    mc_cycle!(mc_states, move_strat, mc_params, pot::AbstractMLPotential,ensemble,n_steps,a,v,r)
Method for the MC cycle when using a machine learning potential. While functionally we can use the energyupdate! method for these potentials this is inefficient when using an external program, as such this is the parallelised energy version.

    We perturb one atom per trajectory, write them all out (see RuNNer.writeconfig) run the program and then read the energies (see RuNNer.getRuNNerenergy). We then batch-determine whether any configuration will be saved and update the relevant mc_state parameters.
"""
function mc_cycle!(mc_states, move_strat, mc_params, pot::AbstractMLPotential, ensemble, n_steps, a, v, r)
    file = RuNNer.writeinit(pot.dir)
    #this for loop creates n_traj perturbed atoms
    indices = []
    trials = []
    #we require parallelisation here, but will need to avoid a race condition
    for mc_state in mc_states
        #for i_step = 1:n_steps
            ran = rand(1:(a+v+r))
            trial_pos = atom_displacement(mc_state.config.pos[ran], mc_state.max_displ[1], mc_state.config.bc)
            writeconfig(file,mc_state.config,ran,trial_pos, pot.atomtype)
            push!(indices,ran)
            push!(trials,trial_pos)
        #end
    end
    #after which we require energy evaluations of the n_traj new configurations
    close(file)    
    energyvec = getRuNNerenergy(pot.dir,mc_params.n_traj)    
    #this replaces the atom_move! function
    #parallelisation here is fine

    Threads.@threads for indx in 1:mc_params.n_traj
        if metropolis_condition(ensemble, (energyvec[indx] - mc_states[indx].en_tot), mc_states[indx].beta ) >=rand()
            mc_states[indx].config.pos[indices[indx]] = trials[indx]
            mc_states[indx].en_tot = energyvec[indx]
            mc_states[indx].count_atom[1] +=1
            mc_states[indx].count_atom[2] += 1
        end
    end


    if rand() < 0.1 #attempt to exchange trajectories
        n_exc = rand(1:mc_params.n_traj-1)
        mc_states[n_exc].count_exc[1] += 1
        mc_states[n_exc+1].count_exc[1] += 1
        exc_acc = exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot)
        if exc_acc > rand()
            mc_states[n_exc].count_exc[2] += 1
            mc_states[n_exc+1].count_exc[2] += 1
            mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
        end
    end


    return mc_states
end
"""
    mc_cycle!(mc_states, move_strat, mc_params, pot::AbstractMLPotential,ensemble,n_steps,a,v,r)
Method for the MC cycle when using a machine learning potential. While functionally we can use the energyupdate! method for these potentials this is inefficient when using an external program, as such this is the parallelised energy version.

    We perturb one atom per trajectory, write them all out (see RuNNer.writeconfig) run the program and then read the energies (see RuNNer.getRuNNerenergy). We then batch-determine whether any configuration will be saved and update the relevant mc_state parameters.
"""
function mc_cycle!(mc_states, move_strat, mc_params, pot::AbstractMLPotential, ensemble, n_steps, a, v, r)
    file = RuNNer.writeinit(pot.dir)
    #this for loop creates n_traj perturbed atoms
    indices = []
    trials = []
    #we require parallelisation here, but will need to avoid a race condition
    for mc_state in mc_states
        #for i_step = 1:n_steps
            ran = rand(1:(a+v+r))
            trial_pos = atom_displacement(mc_state.config.pos[ran], mc_state.max_displ[1], mc_state.config.bc)
            writeconfig(file,mc_state.config,ran,trial_pos, pot.atomtype)
            push!(indices,ran)
            push!(trials,trial_pos)
        #end
    end
    #after which we require energy evaluations of the n_traj new configurations
    close(file)    
    energyvec = getRuNNerenergy(pot.dir,mc_params.n_traj)    
    #this replaces the atom_move! function
    #parallelisation here is fine
    Threads.@threads for i in 1:mc_params.n_traj
        if metropolis_condition(ensemble, (mc_states[i].en_tot - energyvec[i]), mc_states[i].beta ) >=rand()
            mc_states[i].config.pos[indices[i]] = trials[i]
            mc_states[i].en_tot = energyvec[i]
            mc_states[i].count_atom[1] +=1
            mc_states[i].count_atom[2] += 1
        end
    end


    if rand() < 0.1 #attempt to exchange trajectories
        n_exc = rand(1:mc_params.n_traj-1)
        mc_states[n_exc].count_exc[1] += 1
        mc_states[n_exc+1].count_exc[1] += 1
        exc_acc = exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot)
        if exc_acc > rand()
            mc_states[n_exc].count_exc[2] += 1
            mc_states[n_exc+1].count_exc[2] += 1
            mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
        end
    end


    return mc_states
end

"""
    sampling_step(mc_params, mc_states, i, saveham::Bool)
A function to store the information at the end of an MC_Cycle, replacing the manual if statements previously in PTMC_run. 
"""
function sampling_step!(mc_params,mc_states,i, saveham::Bool)  
        if rem(i, mc_params.mc_sample) == 0
            for i_traj=1:mc_params.n_traj
                if saveham == true
                    push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
                else
                    mc_states[i_traj].ham[1] += mc_states[i_traj].en_tot
                    #add E,E**2 to the correct positions in the hamiltonian
                    mc_states[i_traj].ham[2] += (mc_states[i_traj].en_tot*mc_states[i_traj].en_tot)
                end
            end
        end 
end

"""
    function save_params(savefile::IOStream, mc_params::MCParams)
writes the MCParam struct to a savefile
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
"""
    function save_state(savefile::IOStream,mc_state::MCState)
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
        write(savefile, "Boundary: $(typeof(mc_state.config.bc))  $(mc_state.config.bc.radius2) \n")
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
function save_results(results::Output; directory = pwd())
    resultsfile =  open("$(directory)/results.data","w+")
    write(resultsfile,"emin,emax,nbins= $(results.en_min) $(results.en_max) $(results.n_bin) \n")
    write(resultsfile, "Histograms \n")
    writedlm(resultsfile,results.en_histogram)
    close(resultsfile)
    #requires: en_min,en_max,n_bin,en_hist
    #reading doesn't require the rest as that is handled as a post-process
end
"""
    function save_states(mc_params,mc_states,trial_index; directory = pwd())
opens a savefile, writes the mc params and states and the trial at which it was run. 
"""
function save_states(mc_params,mc_states,trial_index; directory = pwd())
    i = 0 
    savefile = open("$(directory)/save.data","w+")
    write(savefile,"Save made at step $trial_index \n") #at $(format(now(),"HH:MM") )\n")
    save_params(savefile,mc_params)
    for state in mc_states
        i += 1
        write(savefile, "config $i \n")
        save_state(savefile,state)
        write(savefile,"end \n")
    end
    close(savefile)
end
"""
    initialise_histograms!(mc_params,results,T)
functionalised the step in which we build the energy histograms  
"""
function initialise_histograms!(mc_params,mc_states,results; full_ham = true,e_bounds = [0,0])    
    T = typeof(mc_states[1].en_tot)
    en_min = T[]
    en_max = T[]
    if full_ham == true
        for i_traj in 1:mc_params.n_traj
            push!(en_min,minimum(mc_states[i_traj].ham))
            push!(en_max,maximum(mc_states[i_traj].ham))
        end
    
        global_en_min = minimum(en_min)
        global_en_max = maximum(en_max)
    else
        #we'll give ourselves a 10% leeway here
        global_en_min = e_bounds[1] - abs(0.05*e_bounds[1])
        global_en_max = e_bounds[2] + abs(0.05*e_bounds[2])
        
        for i_traj = 1:mc_params.n_traj
            histogram = zeros(results.n_bin + 2 ) #bin 1 is too small bin nbin+2 is too large
            push!(results.en_histogram, histogram)
        end
    end

    delta_en = (global_en_max - global_en_min) / (results.n_bin - 1)

    results.en_min = global_en_min
    results.en_max = global_en_max
    
   
    return  delta_en
    

end

function updatehistogram!(mc_params,mc_states,results,delta_en ; fullham=true)

    for i_traj in 1:mc_params.n_traj
        if fullham == true #this is done at the end of the cycle

            hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
            for en in mc_states[i_traj].ham
                index = floor(Int,(en - results.en_min) / delta_en) + 1
                hist[index] += 1
            end
        push!(results.en_histogram, hist)

        else #this is done throughout the simulation
            en = mc_states[i_traj].en_tot

            index = floor(Int,(en - results.en_min) / delta_en) + 1 

            if index < 1 #if energy too low
                results.en_histogram[i_traj][1] += 1 #add to place 1
            elseif index > results.n_bin #if energy too high
                results.en_histogram[i_traj][(results.n_bin +2)] += 1 #add to place n_bin +2
            else
                results.en_histogram[i_traj][(index+1)] += 1
            end
        end
    end

end
"""
    function ptmc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i ;delta_en=0. ) 
functionalised the main body of the ptmc_run! code. Runs a single mc_state, samples the results, updates the histogram and writes the savefile if necessary.
"""
function ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i ;delta_en=0. )

    @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot,  ensemble, n_steps, a, v, r) 
    #sampling step
    sampling_step!(mc_params,mc_states,i,save_ham)

    if save_ham == false
        updatehistogram!(mc_params,mc_states,results,delta_en,fullham=save_ham)
    end

    #step adjustment
    if rem(i, mc_params.n_adjust) == 0
        for i_traj = 1:mc_params.n_traj
            update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
        end 
    end

    if save == true
        if rem(i,1000) == 0
            save_states(mc_params,mc_states,i)
            if save_ham == false
                save_results(results)
            end
        end
    end

end


# function ptmc_cycle( pot::nested)
#    for i =1:pot.cycle
#       ptmc_cycle!( pot::LJ)
# end


"""

    sampling_step(mc_params, mc_states, i, saveham::Bool)
A function to store the information at the end of an MC_Cycle, replacing the manual if statements previously in PTMC_run. 
"""
function sampling_step!(mc_params,mc_states,save_index, saveham::Bool)  
        if rem(save_index, mc_params.mc_sample) == 0
            for indx_traj=1:mc_params.n_traj
                if saveham == true
                    push!(mc_states[indx_traj].ham, mc_states[indx_traj].en_tot) #to build up ham vector of sampled energies
                else
                    mc_states[indx_traj].ham[1] += mc_states[indx_traj].en_tot
                    #add E,E**2 to the correct positions in the hamiltonian
                    mc_states[indx_traj].ham[2] += (mc_states[indx_traj].en_tot*mc_states[indx_traj].en_tot)
                end
            end
        end 
end
"""
    function save_params(savefile::IOStream, mc_params::MCParams)
writes the MCParam struct to a savefile
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
"""
    function save_state(savefile::IOStream,mc_state::MCState)
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
        write(savefile, "Boundary: $(typeof(mc_state.config.bc))  $(mc_state.config.bc.radius2) \n")
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
function save_results(results::Output; directory = pwd())
    resultsfile =  open("$(directory)/results.data","w+")
    rdf_file = open("$directory/RDF.data","w+")
    write(resultsfile,"emin,emax,nbins= $(results.en_min) $(results.en_max) $(results.n_bin) \n")
    write(resultsfile, "Histograms \n")
    writedlm(resultsfile,results.en_histogram)
    close(resultsfile)
    writedlm(rdf_file,results.rdf)
    close(rdf_file)
    #requires: en_min,en_max,n_bin,en_hist
    #reading doesn't require the rest as that is handled as a post-process
end
"""
    function save_states(mc_params,mc_states,trial_index; directory = pwd())
opens a savefile, writes the mc params and states and the trial at which it was run. 
"""
function save_states(mc_params,mc_states,trial_index, directory; filename="save.data")
    dummy_index = 0 
    savefile = open("$(directory)/$(filename)","w+")

    if isfile("$directory/params.data") == false
        paramsfile = open("$directory/params.data")
        save_params(paramsfile,mc_params)
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

    initialise_histograms!(mc_params,results,T)
functionalised the step in which we build the energy histograms  
"""
function initialise_histograms!(mc_params,mc_states,results; full_ham = true,e_bounds = [0,0])    
    T = typeof(mc_states[1].en_tot)
    en_min = T[]
    en_max = T[]

    r_max = 4*mc_states[1].config.bc.radius2 #we will work in d^2
    delta_r = r_max/results.n_bin/5 #we want more bins for the RDFs

    if full_ham == true
        for i_traj in 1:mc_params.n_traj
            push!(en_min,minimum(mc_states[i_traj].ham))
            push!(en_max,maximum(mc_states[i_traj].ham))
        end
    
        global_en_min = minimum(en_min)
        global_en_max = maximum(en_max)
    else

        #we'll give ourselves a 6% leeway here
        global_en_min = e_bounds[1] - abs(0.03*e_bounds[1])
        global_en_max = e_bounds[2] + abs(0.03*e_bounds[2])
    end

    for i_traj = 1:mc_params.n_traj
        histogram = zeros(results.n_bin + 2)
        push!(results.en_histogram, histogram)
        RDF = zeros(results.n_bin*5)
        push!(results.rdf,RDF)
    end
    

    delta_en_hist = (global_en_max - global_en_min) / (results.n_bin - 1)


    results.en_min = global_en_min
    results.en_max = global_en_max

  
        return  delta_en_hist
end
"""
    updaterdf!(mc_states,results,delta_r2)
For each state in a vector of mc_states, we use the distance squared matrix to determine which bin (between zero and 2*r_bound) the distance falls into, we then update results.rdf[bin] to build the radial distribution function
"""
function updaterdf!(mc_states,results,delta_r2)
    for j_traj in eachindex(mc_states)
        for element in mc_states[j_traj].dist2_mat 
            rdf_index=floor(Int,(element/delta_r2))
            if rdf_index != 0
                results.rdf[j_traj][rdf_index] +=1
            end
        end
    end
end
"""
    updatehistogram!(mc_params,mc_states,results,delta_en_hist ; fullham=true)
Performed either at the end or during the mc run according to fullham=true/false (saved all datapoints or calculated on the fly). Uses the energy bounds and the previously defined delta_en_hist to calculate the bin in which te current energy value falls for each trajectory. This is used to build up the energy histograms for post-analysis.
"""
function updatehistogram!(mc_params,mc_states,results,delta_en_hist ; fullham=true)

    for update_traj_index in 1:mc_params.n_traj
        
        if fullham == true #this is done at the end of the cycle

            hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
            for en in mc_states[update_traj_index].ham
                hist_index = floor(Int,(en - results.en_min) / delta_en_hist) + 1
                hist[hist_index] += 1

            end
        push!(results.en_histogram, hist)

        else #this is done throughout the simulation

            en = mc_states[update_traj_index].en_tot

            hist_index = floor(Int,(en - results.en_min) / delta_en_hist) + 1 

            if hist_index < 1 #if energy too low
                results.en_histogram[update_traj_index][1] += 1 #add to place 1
            elseif hist_index > results.n_bin #if energy too high
                results.en_histogram[update_traj_index][(results.n_bin +2)] += 1 #add to place n_bin +2
            else
                results.en_histogram[update_traj_index][(hist_index+1)] += 1
            end

        end
    end

end
"""
    function ptmc_cycle!(mc_states,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i ;delta_en=0. ) 
functionalised the main body of the ptmc_run! code. Runs a single mc_state, samples the results, updates the histogram and writes the savefile if necessary.
"""
function ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir ;delta_en_hist=0.)


    mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot,  ensemble, n_steps, a, v, r) 
    #sampling step
    sampling_step!(mc_params,mc_states,i,save_ham)

    if save_ham == false
        updatehistogram!(mc_params,mc_states,results,delta_en_hist,fullham=save_ham)
        updaterdf!(mc_states,results,(4*mc_states[1].config.bc.radius2/(results.n_bin*5)))

    end

    #step adjustment
    if rem(i, mc_params.n_adjust) == 0
        for i_traj = 1:mc_params.n_traj
            update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
        end 
    end

    if save == true
        if rem(i,1000) == 0

            save_states(mc_params,mc_states,i,save_dir)
            if save_ham == false
                save_results(results)
            end
        end
    end

end

"""
    ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results)

Main function, controlling the parallel tempering MC run.
Calculates number of MC steps per cycle.
Performs equilibration and main MC loop.  
Energy is sampled after `mc_sample` MC cycles. 
Step size adjustment is done after `n_adjust` MC cycles.    
Evaluation: including calculation of inner energy, heat capacity, energy histograms;
saved in `results`.

The booleans control:
save_ham: whether or not to save every energy in a vector, or calculate averages on the fly.
save: whether or not to save the parameters and configurations every 1000 steps
restart: this controls whether to run an equilibration cycle, it additionally requires an integer restartindex which says from which cycle we have restarted the process.
"""

function ptmc_run!(mc_states, move_strat, mc_params, pot, ensemble, results; save_ham::Bool = true, save::Bool=true, restart::Bool=false,restartindex::Int64=0,save_dir = pwd())

    #restart isn't compatible with saving the hamiltonian at the moment

    if restart == true
        save_ham = false
    end
    
    if save_ham == false
        if restart == false
            #If we do not save the hamiltonian we still need to store the E, E**2 terms at each cycle
            for i_traj = 1:mc_params.n_traj
                push!(mc_states[i_traj].ham, 0)
                push!(mc_states[i_traj].ham, 0)
            end

        end
    end

    a = atom_move_frequency(move_strat)
    v = vol_move_frequency(move_strat)
    r = rot_move_frequency(move_strat)
    #number of MC steps per MC cycle
    n_steps = a + v + r

    println("Total number of moves per MC cycle: ", n_steps)
    println()

    
    #equilibration cycle
    if restart == false
        #this initialises the max and min energies for histograms
        if save_ham == false
            ebounds = [100. , -100.] #emin,emax
        end
        
        for i = 1:mc_params.eq_cycles
            @inbounds mc_states = mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
            #verbose way to save the highest and lowest energies

            if save_ham == false
                for i_traj = 1:mc_params.n_traj
                    if mc_states[i_traj].en_tot < ebounds[1]
                        ebounds[1] = mc_states[i_traj].en_tot
                    end

                    if mc_states[i_traj].en_tot > ebounds[2]
                        ebounds[2] = mc_states[i_traj].en_tot
                    end
                end
            end
            #update stepsizes

            if rem(i, mc_params.n_adjust) == 0
                for i_traj = 1:mc_params.n_traj
                    update_max_stepsize!(mc_states[i_traj], mc_params.n_adjust, a, v, r)
                end 
            end
        end
    #re-set counter variables to zero
        for i_traj = 1:mc_params.n_traj
            mc_states[i_traj].count_atom = [0, 0]
            mc_states[i_traj].count_vol = [0, 0]
            mc_states[i_traj].count_rot = [0, 0]
            mc_states[i_traj].count_exc = [0, 0]
        end
        #initialise histogram for non-saving hamiltonian 
        if save_ham == false

            delta_en_hist = initialise_histograms!(mc_params,mc_states,results, full_ham=false,e_bounds=ebounds)

        end

        println("equilibration done")


        if save == true
            save_states(mc_params,mc_states,0,save_dir)
        end
    end



    #main MC loop

    if restart == false

        for i = 1:mc_params.mc_cycles
            if save_ham == false

                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir;delta_en_hist=delta_en_hist)
            else
                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir)
            end

        end

    else #if restarting

        for i = restartindex:mc_params.mc_cycles
            if save_ham == false

                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir;delta_en_hist=delta_en_hist)
            else
                @inbounds ptmc_cycle!(mc_states,results,move_strat, mc_params, pot, ensemble ,n_steps ,a ,v ,r, save_ham, save, i,save_dir)
            end
            

        end 
    end
    println("MC loop done.")
    #Evaluation
    #average energy
    n_sample = mc_params.mc_cycles / mc_params.mc_sample

    if save_ham == true
        en_avg = [sum(mc_states[i_traj].ham) / n_sample for i_traj in 1:mc_params.n_traj] #floor(mc_cycles/mc_sample)
        en2_avg = [sum(mc_states[i_traj].ham .* mc_states[i_traj].ham) / n_sample for i_traj in 1:mc_params.n_traj]
    else
        en_avg = [mc_states[i_traj].ham[1] / n_sample  for i_traj in 1:mc_params.n_traj]
        en2_avg = [mc_states[i_traj].ham[2] / n_sample  for i_traj in 1:mc_params.n_traj]
    end


    results.en_avg = en_avg

    #heat capacity
    results.heat_cap = [(en2_avg[i]-en_avg[i]^2) * mc_states[i].beta for i in 1:mc_params.n_traj]

    #acceptance statistics
    results.count_stat_atom = [mc_states[i_traj].count_atom[1] / (mc_params.n_atoms * mc_params.mc_cycles) for i_traj in 1:mc_params.n_traj]
    results.count_stat_exc = [mc_states[i_traj].count_exc[2] / mc_states[i_traj].count_exc[1] for i_traj in 1:mc_params.n_traj]

    println(results.heat_cap)

    #energy histograms
    if save_ham == true
        # T = typeof(mc_states[1].ham[1])

        delta_en_hist= initialise_histograms!(mc_params,mc_states,results)
        updatehistogram!(mc_params,mc_states,results,delta_en_hist)

    
    end
    #     for i_traj in 1:mc_params.n_traj
    #         hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
    #         for en in mc_states[i_traj].ham
    #             index = floor(Int,(en - global_en_min) / delta_en) + 1
    #             hist[index] += 1
    #         end
    #         push!(results.en_histogram, hist)
    #     end
    # end

    #TO DO
    # volume (NPT ensemble),rot moves ...
    # move boundary condition from config to mc_params?
    # rdfs

    println("done")
    return 
end

end
