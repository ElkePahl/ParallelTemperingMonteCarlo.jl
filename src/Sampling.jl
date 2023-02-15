module MCSampling


using StaticArrays
using ..MCStates
using ..Configurations
using ..InputParams
#using ..BoundaryConditions

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
    update_one_histval!(histogram,index)
To be used in conjunction with the find index function above to correctly update the overall results. Mostly a wrapper function to allow correct broascasting, this then can be used on en_histogram vector of vectors. See the update_histogram! function defined below.
"""
function update_one_histval!(histogram,index)
    histogram[index] += 1
end
function update_histograms!(mc_states,results,delta_en_hist)
    indices = find_hist_index.(mc_states,Ref(results),Ref(delta_en_hist))
    broadcast(update_one_histval!,results.en_histogram,indices)

    return results
end

rdf_index(r2val,delta_r2) = floor(Int,(r2val/delta_r2))
"""
    find_rdf_index(mc_state,delta_r2)
for each element in the dist2_matrix we calculate an rdf index using the anonymous rdf_index function above.
"""
function find_rdf_index(mc_state,delta_r2)
    rdf_indices = rdf_index.(mc_state.dist2_mat,Ref(delta_r2))   
    return rdf_indices
end
"""
    update_one_rdf!(mc_state,histogram,delta_r2)
Function accepts one mc_state and the rdf histogram corresponding to it. It calculates the indices and broadcasts the update hist function defined for the energy histograms. This is a wrapper, becaues while results.rdf is a vector, results is not.
"""
function update_one_rdf!(mc_state,histogram,delta_r2)
    rdf_indices = find_rdf_index(mc_state,delta_r2)
    broadcast(update_one_histval!,Ref(histogram),rdf_indices)
end

function sampling_step!(mc_params,mc_states,save_index,results,delta_en_hist,delta_r2)
    if rem(save_index, mc_params.mc_sample) == 0
        mc_states =  update_energy_tot.(mc_states)#update energies

        results = update_histograms!(mc_states,results,delta_en_hist) #update histograms

        broadcast(update_one_rdf!,mc_states,results.rdf,Ref(delta_r2)) #update rdf's
    end
end



end