module MCSampling

export sampling_step!

export sampling_step!, initialise_histograms!,finalise_results


using StaticArrays,LinearAlgebra
using ..MCStates
using ..Configurations
using ..InputParams
using ..BoundaryConditions

"""
    update_energy_tot(mc_state)

function to update the current energy and energy squared values for coarse analysis of averages at the end. 
"""
# function update_energy_tot(mc_state)
#     mc_state.ham[1] +=mc_state.en_tot
#     mc_state.ham[2] +=(mc_state.en_tot*mc_state.en_tot)

#     return mc_state
# end
function update_energy_tot(mc_states)
    for indx_traj in eachindex(mc_states)      
        mc_states[indx_traj].ham[1] += mc_states[indx_traj].en_tot
            #add E,E**2 to the correct positions in the hamiltonian
        mc_states[indx_traj].ham[2] += (mc_states[indx_traj].en_tot*mc_states[indx_traj].en_tot)            
    end
end
"""
    find_hist_index(mc_state,results,delta_en_hist)
returns the histogram index of a single mc_state energy and returns this value. 
"""
function find_hist_index(mc_state,results,delta_en_hist)

    hist_index = floor(Int,(mc_state.en_tot - results.en_min)/delta_en_hist ) +1

    if hist_index < 1
        return 1
    elseif hist_index > results.n_bin
        return results.n_bin+2
    else
        return hist_index +1
    end
end
"""
    initialise_histograms!(mc_params,results,e_bounds,bc)
Function to create the energy and radial histograms at the end of equilibration. The min/max energy values are extracted from e_bounds and (with 2% either side additionally) used to determine the energy grating for the histogram (delta_en_hist). For spherical boundary conditions the radius squared is used to define a diameter squared since the greatest possible atomic distance is 2*r2 and distance**2 is used throughout the simulation. Histogram contains overflow bins, rdf has 5 times the number of bins as en_histogram

Returns delta_en_hist,delta_r2
"""
function initialise_histograms!(mc_params,results,e_bounds,bc::SphericalBC)

    # incl 6% leeway

    results.en_min = e_bounds[1] #- abs(0.02*e_bounds[1])
    results.en_max = e_bounds[2] #+ abs(0.02*e_bounds[2])

    delta_en_hist = (results.en_max - results.en_min) / (results.n_bin - 1)
    delta_r2 = 4*bc.radius2/results.n_bin/5 

    for i_traj in 1:mc_params.n_traj       

        push!(results.en_histogram,zeros(results.n_bin + 2))
        push!(results.rdf,zeros(results.n_bin*5))

    end
    return delta_en_hist,delta_r2
end
function initialise_histograms!(mc_params,results,e_bounds,bc::AdjacencyBC)

    # incl 6% leeway

    results.en_min = e_bounds[1] #- abs(0.02*e_bounds[1])
    results.en_max = e_bounds[2] #+ abs(0.02*e_bounds[2])

    delta_en_hist = (results.en_max - results.en_min) / (results.n_bin - 1)
    delta_r2 = (4*15*15)/results.n_bin/5                       #testing purposes ONLY

    for i_traj in 1:mc_params.n_traj       

        push!(results.en_histogram,zeros(results.n_bin + 2))
        push!(results.rdf,zeros(results.n_bin*5))

    end
    return delta_en_hist,delta_r2
end
"""
    update_histograms!(mc_states,results,delta_en_hist)
Self explanatory name, updates the energy histograms in results using the current mc_states.en_tot

"""
function update_histograms!(mc_states,results,delta_en_hist)
     for i_traj in eachindex(mc_states)
        @inbounds histindex = find_hist_index(mc_states[i_traj],results,delta_en_hist)
        results.en_histogram[i_traj][histindex] +=1
    end

end

rdf_index(r2val,delta_r2) = floor(Int,(r2val/delta_r2))
"""
    update_rdf!(mc_states,results,delta_r2)
Self explanatory name, iterates over mc_states and adds to the appropriate results.rdf histogram. Type stable by the initialise function specifying a vector of integers.  

"""
function update_rdf!(mc_states,results,delta_r2)
    for j_traj in eachindex(mc_states)
        for element in mc_states[j_traj].dist2_mat 
            idx=rdf_index(element,delta_r2)
             if 0 < idx < (results.n_bins*5)
                 results.rdf[j_traj][idx] +=1
            elseif idx >= (results.nbins*5)
                results.rdf[j_traj][results.nbins*5] +=1
            end
        end
    end
    #if results.rdf[j_traj][100] >1000000
    #     println("Warning: Bin size too small")
    # end
end
"""
    sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist,delta_r2)
    sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist)

Function performed at the end of an mc_cycle! after equilibration. Updates the E,E**2 totals for each mc_state, updates the energy and radial histograms and then returns the modified mc_states and results.
Second method does not perform the rdf calculation. This is designed to improve the speed of sampling where the rdf is not required.


TO IMPLEMENT:
This function benchmarked at 7.84μs, the update RDF step takes 7.545μs of this. Removing the rdf information should become a toggle-able option in case faster results with less information are wanted. 
"""
function sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist,delta_r2)
    if rem(save_index, mc_params.mc_sample) == 0

        update_energy_tot(mc_states)
        
        update_histograms!(mc_states,results,delta_en_hist)
        update_rdf!(mc_states,results,delta_r2)
    end 
end
function sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist)
    if rem(save_index, mc_params.mc_sample) == 0

        update_energy_tot(mc_states)
        
        update_histograms!(mc_states,results,delta_en_hist)

    end   
end
"""
    finalise_results(mc_states,mc_params,results)
Function designed to take a complete mc simulation and calculate the averages. 
"""
function finalise_results(mc_states,mc_params,results)

    #Energy average
    n_sample = mc_params.mc_cycles / mc_params.mc_sample
    en_avg = [mc_states[i_traj].ham[1] / n_sample  for i_traj in 1:mc_params.n_traj]
    en2_avg = [mc_states[i_traj].ham[2] / n_sample  for i_traj in 1:mc_params.n_traj]
    results.en_avg = en_avg
    #heat capacity
    results.heat_cap = [(en2_avg[i]-en_avg[i]^2) * mc_states[i].beta for i in 1:mc_params.n_traj]
    #count stats 
    results.count_stat_atom = [mc_states[i_traj].count_atom[1] / (mc_params.n_atoms * mc_params.mc_cycles) for i_traj in 1:mc_params.n_traj]
    results.count_stat_exc = [mc_states[i_traj].count_exc[2] / mc_states[i_traj].count_exc[1] for i_traj in 1:mc_params.n_traj]

    println(results.heat_cap)

    return results

end

end #module