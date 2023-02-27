
# """
#     exc_trajectories!(state_1::MCState, state_2::MCState)
# Exchanges configurations and distance and energy information between two trajectories;
# information contained in `state_1` and `state_2`, see [`MCState`](@ref)   
# """
# function exc_trajectories!(state_1::MCState, state_2::MCState)
#     state_1.config, state_2.config = state_2.config, state_1.config
#     state_1.dist2_mat, state_2.dist2_mat = state_2.dist2_mat, state_1.dist2_mat
#     state_1.en_atom_vec, state_2.en_atom_vec = state_2.en_atom_vec, state_1.en_atom_vec
#     state_1.en_tot, state_2.en_tot = state_2.en_tot, state_1.en_tot
#     return state_1, state_2
# end 

# """
#     atom_move!(mc_state::MCState, i_atom, pot, ensemble)
# Moves selected `i_atom`'s atom randomly (max. displacement as in `mc_state`).
# Calculates energy update (depending on potential `pot`).
# Accepts move according to Metropolis criterium (asymmetric choice, depending on `ensemble`)
# Updates `mc_state` if move accepted.
# """
# function atom_move!(mc_state::MCState, i_atom, pot, ensemble)
#     #move randomly selected atom (obeying the boundary conditions)

#     # trial_pos = atom_displacement(mc_state.config.pos[i_atom], mc_state.max_displ[1], mc_state.config.bc)
    
#     trial_pos = atom_displacement(mc_state,i_atom)
#     #find new distances of moved atom 

#     delta_en_move, dist2_new = energy_update(trial_pos, i_atom, mc_state.config, mc_state.dist2_mat, pot)

#     #decide acceptance
#     if metropolis_condition(ensemble, delta_en_move, mc_state.beta) >= rand()
#         #new config accepted
#         mc_state.config.pos[i_atom] = trial_pos #copy(trial_pos)
#         mc_state.dist2_mat[i_atom,:] = dist2_new #copy(dist2_new)
#         mc_state.dist2_mat[:,i_atom] = dist2_new
#         mc_state.en_tot += delta_en_move
#         mc_state.count_atom[1] += 1
#         mc_state.count_atom[2] += 1
#     end
#     return mc_state #config, entot, dist2mat, count_acc, count_acc_adjust
# end

# """
#     mc_step!(mc_state::MCState, pot, ensemble, a, v, r)
# Performs an individual MC step.
# Chooses type of move randomly according to frequency of moves `a`,`v` and `r` 
# for atom, volume and rotation moves.
# Performs the selected move.   
# """
# function mc_step!(mc_state::MCState, pot, ensemble, a, v, r)
#     ran_atom = rand(1:(a+v+r)) #choose move randomly
#     if ran_atom <= a
#         mc_state = atom_move!(mc_state, ran_atom, pot, ensemble)
#     #else if ran <= v
#     #    vol_move!(mc_state, pot, ensemble)
#     #else if ran <= r
#     #    rot_move!(mc_state, pot, ensemble)
#     end
#     return mc_state
# end 

# """
#     mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
# Performs a MC cycle consisting of `n_steps` individual MC steps 
# (frequencies of moves given by `a`,`v` and `r`).
# Attempts parallel tempering step for 10% of cycles.
# Exchanges trajectories if exchange accepted.
# """
# function mc_cycle!(mc_states, move_strat, mc_params, pot, ensemble, n_steps, a, v, r)
#     #perform one MC cycle
#     for i_traj = 1:mc_params.n_traj
#         for i_step = 1:n_steps
#             #mc_states[i_traj] = mc_step!(type_moves[ran][2], type_moves[ran][1], mc_states[i_traj], ran, pot, ensemble)
#             @inbounds mc_states[i_traj] = mc_step!(mc_states[i_traj], pot, ensemble, a, v, r)
#         end
#         #push!(mc_states[i_traj].ham, mc_states[i_traj].en_tot) #to build up ham vector of sampled energies
#     end
#     #parallel tempering
#     if rand() < 0.1 #attempt to exchange trajectories
#         n_exc = rand(1:mc_params.n_traj-1)
#         mc_states[n_exc].count_exc[1] += 1
#         mc_states[n_exc+1].count_exc[1] += 1

#         exc_acc = exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot)

#         if exc_acc > rand()
#             mc_states[n_exc].count_exc[2] += 1
#             mc_states[n_exc+1].count_exc[2] += 1

#             mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
#         end
#     end

#     return mc_states
# end
# """
#     mc_cycle!(mc_states, move_strat, mc_params, pot::AbstractMachineLearningPotential,ensemble,n_steps,a,v,r)
# Method for the MC cycle when using a machine learning potential. While functionally we can use the energyupdate! method for these potentials this is inefficient when using an external program, as such this is the parallelised energy version.

#     We perturb one atom per trajectory, write them all out (see RuNNer.writeconfig) run the program and then read the energies (see RuNNer.getRuNNerenergy). We then batch-determine whether any configuration will be saved and update the relevant mc_state parameters.
# """
# function mc_cycle!(mc_states, move_strat, mc_params, pot::AbstractMachineLearningPotential, ensemble, n_steps, a, v, r)
#     file = RuNNer.writeinit(pot.dir)
#     #this for loop creates n_traj perturbed atoms
#     indices = []
#     trials = []
#     #we require parallelisation here, but will need to avoid a race condition
#     for mc_state in mc_states
#         #for i_step = 1:n_steps
#             ran = rand(1:(a+v+r))
#             trial_pos = atom_displacement(mc_state.config.pos[ran], mc_state.max_displ[1], mc_state.config.bc)
#             writeconfig(file,mc_state.config,ran,trial_pos, pot.atomtype)
#             push!(indices,ran)
#             push!(trials,trial_pos)
#         #end
#     end
#     #after which we require energy evaluations of the n_traj new configurations
#     close(file)    
#     energyvec = getRuNNerenergy(pot.dir,mc_params.n_traj)    
#     #this replaces the atom_move! function
#     #parallelisation here is fine

#     Threads.@threads for indx in 1:mc_params.n_traj
#         if metropolis_condition(ensemble, (energyvec[indx] - mc_states[indx].en_tot), mc_states[indx].beta ) >=rand()
#             mc_states[indx].config.pos[indices[indx]] = trials[indx]
#             mc_states[indx].en_tot = energyvec[indx]
#             mc_states[indx].count_atom[1] +=1
#             mc_states[indx].count_atom[2] += 1
#         end
#     end


#     if rand() < 0.1 #attempt to exchange trajectories
#         n_exc = rand(1:mc_params.n_traj-1)
#         mc_states[n_exc].count_exc[1] += 1
#         mc_states[n_exc+1].count_exc[1] += 1
#         exc_acc = exc_acceptance(mc_states[n_exc].beta, mc_states[n_exc+1].beta, mc_states[n_exc].en_tot,  mc_states[n_exc+1].en_tot)
#         if exc_acc > rand()
#             mc_states[n_exc].count_exc[2] += 1
#             mc_states[n_exc+1].count_exc[2] += 1
#             mc_states[n_exc], mc_states[n_exc+1] = exc_trajectories!(mc_states[n_exc], mc_states[n_exc+1])
#         end
#     end


#     return mc_states
# end


# """

#     sampling_step(mc_params, mc_states, save_index, saveham::Bool)
# A function to store the information at the end of an MC_Cycle, replacing the manual if statements previously in PTMC_run. 
# """
# function sampling_step!(mc_params,mc_states,save_index, saveham::Bool)  
#         if rem(save_index, mc_params.mc_sample) == 0
#             for indx_traj=1:mc_params.n_traj
#                 if saveham == true
#                     push!(mc_states[indx_traj].ham, mc_states[indx_traj].en_tot) #to build up ham vector of sampled energies
#                 else
#                     mc_states[indx_traj].ham[1] += mc_states[indx_traj].en_tot
#                     #add E,E**2 to the correct positions in the hamiltonian
#                     mc_states[indx_traj].ham[2] += (mc_states[indx_traj].en_tot*mc_states[indx_traj].en_tot)
#                 end
#             end
#         end 
# end
# """

#     initialise_histograms!(mc_params,results,T)
# functionalised the step in which we build the energy histograms  
# """
# function initialise_histograms!(mc_params,mc_states,results; full_ham = true,e_bounds = [0,0])    
#     T = typeof(mc_states[1].en_tot)
#     en_min = T[]
#     en_max = T[]

#     r_max = 4*mc_states[1].config.bc.radius2 #we will work in d^2
#     delta_r = r_max/results.n_bin/5 #we want more bins for the RDFs

#     if full_ham == true
#         for i_traj in 1:mc_params.n_traj
#             push!(en_min,minimum(mc_states[i_traj].ham))
#             push!(en_max,maximum(mc_states[i_traj].ham))
#         end
    
#         global_en_min = minimum(en_min)
#         global_en_max = maximum(en_max)
#     else

#         #we'll give ourselves a 6% leeway here
#         global_en_min = e_bounds[1] - abs(0.03*e_bounds[1])
#         global_en_max = e_bounds[2] + abs(0.03*e_bounds[2])
#     end

#     for i_traj = 1:mc_params.n_traj
#         histogram = zeros(results.n_bin + 2)
#         push!(results.en_histogram, histogram)
#         RDF = zeros(results.n_bin*5)
#         push!(results.rdf,RDF)
#     end
    

#     delta_en_hist = (global_en_max - global_en_min) / (results.n_bin - 1)


#     results.en_min = global_en_min
#     results.en_max = global_en_max

  
#         return  delta_en_hist
# end
# """
#     updaterdf!(mc_states,results,delta_r2)
# For each state in a vector of mc_states, we use the distance squared matrix to determine which bin (between zero and 2*r_bound) the distance falls into, we then update results.rdf[bin] to build the radial distribution function
# """
# function updaterdf!(mc_states,results,delta_r2)
#     for j_traj in eachindex(mc_states)
#         for element in mc_states[j_traj].dist2_mat 
#             rdf_index=floor(Int,(element/delta_r2))
#             if rdf_index != 0
#                 results.rdf[j_traj][rdf_index] +=1
#             end
#         end
#     end
# end
# """
#     updatehistogram!(mc_params,mc_states,results,delta_en_hist ; fullham=true)
# Performed either at the end or during the mc run according to fullham=true/false (saved all datapoints or calculated on the fly). Uses the energy bounds and the previously defined delta_en_hist to calculate the bin in which te current energy value falls for each trajectory. This is used to build up the energy histograms for post-analysis.
# """
# function updatehistogram!(mc_params,mc_states,results,delta_en_hist ; fullham=true)

#     for update_traj_index in 1:mc_params.n_traj
        
#         if fullham == true #this is done at the end of the cycle

#             hist = zeros(results.n_bin)#EnHist(results.n_bin, global_en_min, global_en_max)
#             for en in mc_states[update_traj_index].ham
#                 hist_index = floor(Int,(en - results.en_min) / delta_en_hist) + 1
#                 hist[hist_index] += 1

#             end
#         push!(results.en_histogram, hist)

#         else #this is done throughout the simulation

#             en = mc_states[update_traj_index].en_tot

#             hist_index = floor(Int,(en - results.en_min) / delta_en_hist) + 1 

#             if hist_index < 1 #if energy too low
#                 results.en_histogram[update_traj_index][1] += 1 #add to place 1
#             elseif hist_index > results.n_bin #if energy too high
#                 results.en_histogram[update_traj_index][(results.n_bin +2)] += 1 #add to place n_bin +2
#             else
#                 results.en_histogram[update_traj_index][(hist_index+1)] += 1
#             end

#         end
#     end

# end