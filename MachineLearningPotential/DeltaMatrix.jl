"""
    module DeltaMatrix 
A module designed to update an existing matrix of symmetry function values based on a small perturbation in the positions. 
"""
module DeltaMatrix

using ..Cutoff
using ..SymmetryFunctions 

using StaticArrays,LinearAlgebra

export total_symm!,symmetry_calculation!,total_thr_symm!

#-------------------------------------------------------------------#
#-------------------------Adjust Functions--------------------------#
#-------------------------------------------------------------------#
"""
    adjust_symm_val!(g_value,r_sum,f_prod,η,g_norm)
Designed to update the radial symmetry function value `g_value`. Accepts the hyperparameter `η` as well as `r_sum`, `f_prod` and `g_norm` and adds the individual contribution of g_{ij}. 
"""
function adjust_symm_val!(g_value,r_sum,f_prod,η,g_norm)
    #adjusts radial type 2 symmetry function
    g_value += exponential_part(η,r_sum,f_prod*g_norm)
    return g_value
end
"""
    adjust_radial_symm_val!(g_value1,g_value2,rnew_ij,r2_ij,fnew_ij,f2_ij,η,g_norm)
Taking the i and j symmetry function values `g_value#` we use the hyperparameters `η,g_norm` as well as the old and new distances and cutoff functions `rnew_ij,fnew_ij` `r2_ij,f2_ij` adds the new symmetry value and subtracts the old one based on g_{ij}
"""
function adjust_radial_symm_val!(g_value1,g_value2,rnew_ij,r2_ij,fnew_ij,f2_ij,η,g_norm)
 
    g_value1,g_value2 = adjust_symm_val!(g_value1,rnew_ij,fnew_ij,η,g_norm),adjust_symm_val!(g_value2,rnew_ij,fnew_ij,η,g_norm)
    g_value1,g_value2 = adjust_symm_val!(g_value1,r2_ij,-f2_ij,η,g_norm),adjust_symm_val!(g_value2,r2_ij,-f2_ij,η,g_norm)

    return g_value1,g_value2
end
"""
    adjust_angular_symm_val!(g_value,θ_new,θ_old,exp_new,exp_old,tpz)
    adjust_angular_symm_val!(g_value,exp_old,exp_new,θ_old,θ_new,λ,ζ,tpz)

Functions for adjusting angular symmetry function value from `g_value` by calculating the exponential component `exp_old,exp_new`, theta components `θ_val_old,θ_val_new` from the angles `θ_old,θ_new` and the normalisaiton factor `tpz` These are used to subtract the old g value and add the new one. 
"""

function adjust_angular_symm_val!(g_value,θ_new,θ_old,exp_new,exp_old,tpz)

    g_value += exp_new*θ_new*tpz
    g_value -= exp_old*θ_old*tpz

    return g_value
end

function adjust_angular_symm_val!(g_value,exp_old,exp_new,θ_old,θ_new,λ,ζ,tpz)
    θ_val_old,θ_val_new = theta_part(θ_old,λ,ζ),theta_part(θ_new,λ,ζ)
    return adjust_angular_symm_val!(g_value,θ_val_new,θ_val_old,exp_new,exp_old,tpz)
end
#---------------------------------------------------------------------------#
#-----------------------------Functional Calls------------------------------#
#---------------------------------------------------------------------------#
"""
    calc_new_symmetry_value!(g_vector,indexi,indexj,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,g_norm)
Call for the radial symmetry value designed to curry the input from `g_vector` at positions `indexi,indexj` to the adjust_radial_symm_val! function. It unpacks the radial distances from `dist2_mat,new_dist2_vector` and the cutoff functions from `f_matrix,new_f_vec` as well as the hyperparameters `η,g_norm` and gives these values to the lower level functions. 
"""
function calc_new_symmetry_value!(g_vector,indexi,indexj,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,g_norm)
    g_vector[indexi],g_vector[indexj] = adjust_radial_symm_val!(g_vector[indexi],g_vector[indexj],new_dist2_vector[indexj],dist2_mat[indexi,indexj],new_f_vector[indexj],f_matrix[indexi,indexj],η,g_norm)
    
end
"""
    calc_new_symmetry_value!(g_vector,indices,newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ,tpz)
    calc_new_symmetry_value!(g_vector,indexi,indexj,indexk,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,λ,ζ,tpz)

Currying functions from higher-level data structures such as the radial distances in `dist2_mat,new_dist2_vector` and cutoff functions `f_matrix,new_f_vector` as well as positions in `newposition,position` and passes these to lower level functions along with hyperparameters `η,λ,ζ,tpz` to adjust the values in `g_vector` at positions `indexi,indexj,indexk` calculates the required exponential component and angles to pass down to adjust_angular_symm_val!
"""
function calc_new_symmetry_value!(g_vector,indices,newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ,tpz)

    θ_new_vec,θ_old_vec = all_angular_measure(newposition,position2,position3,rnew_ij,rnew_ik,r2_jk),all_angular_measure(position1,position2,position3,r2_ij,r2_ik,r2_jk)

    exp_new,exp_old = exponential_part(η,rnew_ij,rnew_ik,r2_jk,fnew_ij,fnew_ik,f_jk),exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk)

    for (θ_old,θ_new,index) in zip(θ_old_vec,θ_new_vec,indices)
        g_vector[index] = adjust_angular_symm_val!(g_vector[index],exp_old,exp_new,θ_old,θ_new,λ,ζ,tpz)
    end

    return g_vector
end
function calc_new_symmetry_value!(g_vector,indexi,indexj,indexk,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,λ,ζ,tpz)
    return calc_new_symmetry_value!(g_vector,[indexi,indexj,indexk],newposition,position[indexi],position[indexj],position[indexk],new_dist2_vector[indexj],new_dist2_vector[indexk],dist2_mat[indexi,indexj],dist2_mat[indexi,indexk],dist2_mat[indexj,indexk],new_f_vector[indexj],new_f_vector[indexk],f_matrix[indexi,indexj],f_matrix[indexi,indexk],f_matrix[indexj,indexk],η,λ,ζ,tpz)

end
#-----------------------------------------------------------------------#
#---------------------------Looping Functions---------------------------#
#-----------------------------------------------------------------------#
"""
    symmetry_calculation!(g_vector,atomindex,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,symmetry_function::RadialType2)
    symmetry_calculation!(g_vector,atomindex,newposition,position,dist2_mat,new_dis_vector,f_matrix,new_f_vector,symmetry_function::AngularType3)

Method one is designed for radial symmetry functions. Given an atom at `atomindex` along with high level data structures: `dist2_mat,new_dist2_vector,f_matrix,new_f_vector` containing the new and old positions, distances and cutoff functions. Given a single `symmetry_function` we iterate over all other atoms and pass their index to lower level currying functions.  The positions `newposition,position` are included for consistency with the higher-level function.

Method two is designed to do the same for the angular symmetry function using the same inputs. Double loop over all j all k and use calc_new_symmetry_value! over `g_vector`
"""
function radial_symmetry_calculation!(g_vector,atomindex,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,symmetry_function::RadialType2)
    if symmetry_function.type_vec == Int(11)

        η,g_norm = symmetry_function.eta,symmetry_function.G_norm
        for index2 in eachindex(g_vector)
            if index2 != atomindex
                 calc_new_symmetry_value!(g_vector,atomindex,index2,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,g_norm)
            end
        end
    end

    return g_vector
end
function angular_symmetry_calculation!(g_vector,atomindex,newposition,position,dist2_mat,new_dis_vector,f_matrix,new_f_vector,symmetry_function::AngularType3)

    N = length(g_vector)

    if symmetry_function.type_vec == Int(111)

        η,λ,ζ,tpz = symmetry_function.eta,symmetry_function.lambda,symmetry_function.zeta,symmetry_function.tpz

        for j_index in 1:N
            if j_index != atomindex
                for k_index in j_index+1:N
                    if k_index != atomindex
                        g_vector = calc_new_symmetry_value!(g_vector,atomindex,j_index,k_index,newposition,position,dist2_mat,new_dis_vector,f_matrix,new_f_vector,η,λ,ζ,tpz) 
                    end
                end
            end
        end
        
    end

    return g_vector
end
"""
    total_symm!(g_matrix,position,new_position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,atomindex,total_symmetry_vector)
Top level function to calculate the total change to the matrix of symmetry function values `g_matrix`. Given `position,dist2_matrix,f_matrix` containing the original state of the system, and `new_position,new_dist_vector,new_f_vector` the change to this state based on the motion of `atomindex`, we iterate over the `total_symmetry_vector` using the defined symmetry_calculation function. 
"""
# function total_symm!(g_matrix,position,new_position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,atomindex,total_symmetry_vector)
#     for g_index in eachindex(total_symmetry_vector)
#         g_matrix[g_index,:] = symmetry_calculation!(g_matrix[g_index,:],atomindex,new_position,position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,total_symmetry_vector[g_index])
#     end

#     return g_matrix
# end


function total_symm!(g_matrix,position,new_position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,atomindex,radsymmfunctions,angsymmfunctions,Nrad,Nang)
    for g_index in 1:Nrad
#@views
      g_matrix[g_index,:] = radial_symmetry_calculation!(g_matrix[g_index,:],atomindex,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,radsymmfunctions[g_index])
    end
    for g_index in 1:Nang
#@views
    truindex = g_index+Nrad
        g_matrix[truindex ,:] = angular_symmetry_calculation!(g_matrix[truindex,:],atomindex,new_position,position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,angsymmfunctions[g_index])
    end

    return g_matrix 
end






function total_thr_symm!(g_matrix,position,new_position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,atomindex,total_symmetry_vector)
    Threads.@threads for g_index in eachindex(total_symmetry_vector)
        g_matrix[g_index,:] = symmetry_calculation!(g_matrix[g_index,:],atomindex,new_position,position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,total_symmetry_vector[g_index])
    end

    return g_matrix
end


# #------------------------------------------------------------------------#
# #new version with fewer allocs?#
# #------------------------------------------------------------------------#
# function calc_new_symmetry_value!(g_vector,θ_new_vec,θ_old_vec ,indices,newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ,tpz)

#     θ_new_vec = all_angular_measure(θ_new_vec, newposition,position2,position3,rnew_ij,rnew_ik,r2_jk)

#     exp_new,exp_old = exponential_part(η,rnew_ij,rnew_ik,r2_jk,fnew_ij,fnew_ik,f_jk),exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk)

#     for (θ_old,θ_new,index) in zip(θ_old_vec,θ_new_vec,indices)
#         g_vector[index] = adjust_angular_symm_val!(g_vector[index],exp_old,exp_new,θ_old,θ_new,λ,ζ,tpz)
#     end

#     return g_vector
# end
# function calc_new_symmetry_value!(g_vector,old_theta,new_theta,indexi,indexj,indexk,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,λ,ζ,tpz)
#     return calc_new_symmetry_value!(g_vector,old_theta,new_theta,[indexi,indexj,indexk],newposition,position[indexi],position[indexj],position[indexk],new_dist2_vector[indexj],new_dist2_vector[indexk],dist2_mat[indexi,indexj],dist2_mat[indexi,indexk],dist2_mat[indexj,indexk],new_f_vector[indexj],new_f_vector[indexk],f_matrix[indexi,indexj],f_matrix[indexi,indexk],f_matrix[indexj,indexk],η,λ,ζ,tpz)

# end
# function symmetry_calculation!(g_vector,atomindex,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,old_theta,new_theta,symmetry_function::RadialType2)
#     if symmetry_function.type_vec == [1.,1.]

#         η,g_norm = symmetry_function.eta,symmetry_function.G_norm
#         for index2 in eachindex(g_vector)
#             if index2 != atomindex
#                 g_vector = calc_new_symmetry_value!(g_vector,atomindex,index2,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,g_norm)
#             end
#         end
#     end

#     return g_vector
# end
# function symmetry_calculation!(g_vector,atomindex,newposition,position,dist2_mat,new_dis_vector,f_matrix,new_f_vector,old_theta,new_theta,symmetry_function::AngularType3)

#     N = length(g_vector)

#     if symmetry_function.type_vec == [1.,1.,1.]

#         η,λ,ζ,tpz = symmetry_function.eta,symmetry_function.lambda,symmetry_function.zeta,symmetry_function.tpz

#         for j_index in 1:N
#             if j_index != atomindex
#                 for k_index in j_index+1:N
#                     if k_index != atomindex
#                         g_vector = calc_new_symmetry_value!(g_vector,old_theta,new_theta,atomindex,j_index,k_index,newposition,position,dist2_mat,new_dis_vector,f_matrix,new_f_vector,η,λ,ζ,tpz) 
#                     end
#                 end
#             end
#         end
        
#     end

#     return g_vector
# end

# function total_symm!(g_matrix,position,new_position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,atomindex,old_theta,new_theta,total_symmetry_vector)
#     for g_index in eachindex(total_symmetry_vector)
#         g_matrix[g_index,:] = symmetry_calculation!(g_matrix[g_index,:],atomindex,new_position,position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,old_theta,new_theta,total_symmetry_vector[g_index])
#     end

#     return g_matrix
# end

end