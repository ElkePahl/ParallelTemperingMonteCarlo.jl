module MCSampling

#export sampling_step!


using StaticArrays,LinearAlgebra
using ..MCStates
using ..Configurations
using ..InputParams
using ..BoundaryConditions

"""
    update_energy_tot(mc_state)
function to update the current energy and energy squared values for coarse analysis of averages at the end. 
"""
function update_energy_tot(mc_state)
    mc_state.ham[1] +=mc_state.en_tot
    mc_state.ham[2] +=(mc_state.en_tot*mc_state.en_tot)

    return mc_state
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

    # incl 4% leeway
    results.en_min = e_bounds[1] - abs(0.02*e_bounds[1])
    results.en_max = e_bounds[2] + abs(0.02*e_bounds[2])
    delta_en_hist = (results.en_max - results.en_min) / (results.n_bin - 1)
    delta_r2 = 4*bc.radius2/results.n_bin/5 

    for i_traj in 1:mc_params.n_traj       
        push!(results.en_histogram,zeros(Int,results.n_bin + 2))
        push!(results.rdf,zeros(Int,results.n_bin*5))
    end
    return delta_en_hist,delta_r2
end

function update_histograms!(mc_states,results,delta_en_hist)
     for i in eachindex(mc_states)
        @inbounds index = find_hist_index(mc_states[i],results,delta_en_hist)
        results.en_histogram[i][index] +=1
    end
    return results
end

rdf_index(r2val,delta_r2) = floor(Int,(r2val/delta_r2))
function updaterdf!(mc_states,results,delta_r2)
    for j_traj in eachindex(mc_states)
        for element in mc_states[j_traj].dist2_mat 
            idx=rdf_index(element,delta_r2)
            if rdf_index != 0
                results.rdf[j_traj][idx] +=1
            end
        end
    end
end
"""
    sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist,delta_r2)
Function performed at the end of an mc_cycle! after equilibration. Updates the E,E**2 totals for each mc_state, updates the energy and radial histograms and then returns the modified mc_states and results.

"""
function sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist,delta_r2)
    if rem(save_index, mc_params.mc_sample) == 0
        for state in mc_states
            state = update_energy_tot(state)
        end
        results = update_histograms!(mc_states,results,delta_en_hist)
        results = update_rdf!(mc_states,results,delta_r2)
    end
    return mc_states,results
end



end