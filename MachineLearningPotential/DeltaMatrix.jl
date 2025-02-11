"""
    module DeltaMatrix 
A module designed to update an existing matrix of symmetry function values based on a small perturbation in the positions. 
"""
module DeltaMatrix

using ..Cutoff
using ..SymmetryFunctions 

using StaticArrays,LinearAlgebra

export total_symm!,total_thr_symm!
export calc_delta_matrix,calc_swap_matrix


#----------------------------------------------------------------------#
#------------------------ New Format Functions ------------------------#
#----------------------------------------------------------------------#
"""
    new_radial_symm_val!(rnew_ij,r2_ij,fnew_ij,f2_ij,η)
Function to calculate the updated value of a radial symmetry function: That is, how much does the radial symmetry value calculated using the old distance r2_ij change when using the new distance rnew_ij using the old and new cutoff values f2_ij,fnew_ij and the parameter η. 

"""
function new_radial_symm_val!(rnew_ij,r2_ij,fnew_ij,f2_ij,η)
    return exponential_part(η,rnew_ij,fnew_ij) - exponential_part(η,r2_ij,f2_ij)
end
"""
    new_angular_symm_vals(newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ)
Function to calculate the updated value of an angular symmetry function: Using the old and new interatomic distances and positions along with cutoff values and parameters, we calculate the old and new symmetry value and return a static vector of the three new angular symmetry value, one for each atom i,j,k being the centre of the calculation.

"""
function new_angular_symm_vals(newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ)

    θ_new,θ_old = all_angular_measure(newposition,position2,position3,rnew_ij,rnew_ik,r2_jk),all_angular_measure(position1,position2,position3,r2_ij,r2_ik,r2_jk)

    θ_val_old,θ_val_new = theta_part.(θ_old,Ref(λ),Ref(ζ)),theta_part.(θ_new,Ref(λ),Ref(ζ))

    exp_new,exp_old = exponential_part(η,rnew_ij,rnew_ik,r2_jk,fnew_ij,fnew_ik,f_jk),exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk)

    return SVector{3}(θ_val_new.*exp_new .- θ_val_old.*exp_old)
end
"""
    calc_delta_symm_val!(g_vector,atomindex,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,n1,n2,η,g_norm)
    calc_delta_symm_val!(g_vector,positions,newposition,atomindex,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,n1,n2,η,λ,ζ,tpz)

Generic function to calculate the total update to the vector of symmetry values having moved a single atom defined by atomindex. The first method calculates the changes to a vector of radial symmetry values, the second calculates the changes to a vector of angular symmetry values. 
"""
function calc_delta_symm_val!(g_vector,atomindex,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,n1,n2,η,g_norm)
    N = n1+n2

    
    ind = ifelse(atomindex <= n1 , 1 , 2 )
    #calculate i-Cu
    for index2 in 1:n1
        if index2 != atomindex
            g_new = new_radial_symm_val!(new_dist2_vector[index2],dist2_mat[atomindex,index2],new_f_vector[index2],f_matrix[atomindex,index2],η)

            g_vector[1,atomindex] += g_new
            g_vector[ind,index2] = g_new
        end
    end
    #calculate i-Zn
    for index2 in n1+1:N
        if index2 != atomindex
            g_new = new_radial_symm_val!(new_dist2_vector[index2],dist2_mat[atomindex,index2],new_f_vector[index2],f_matrix[atomindex,index2],η)

            g_vector[2,atomindex] += g_new
            g_vector[ind,index2] += g_new
        end
    end
    
    g_vector[1,1:n1] = g_vector[1,1:n1] .* g_norm[1]
    g_vector[2,1:n1] = g_vector[2,1:n1] .* g_norm[2]

    g_vector[1,1+n1:N] = g_vector[1,1+n1:N] .* g_norm[3]
    g_vector[2,1+n1:N] = g_vector[2,1+n1:N] .* g_norm[4]
    
    return g_vector

end

function calc_delta_symm_val!(g_vector,positions,newposition,atomindex,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,n1,n2,η,λ,ζ,tpz)

    N = n1+n2


    i_val = MVector{4,Int}(1,2,2,1)
    if atomindex > n1
        i_val .+= 1
    end
    #calculate i-CuCu i->row 1; j,k -> row 1/2
    for j_index in 1:n1
        if j_index != atomindex
            for k_index in j_index+1:n1
                if k_index != atomindex

                    g_vals = new_angular_symm_vals(newposition,positions[atomindex],positions[j_index],positions[k_index],new_dist2_vector[j_index],new_dist2_vector[k_index],dist2_mat[atomindex,j_index],dist2_mat[atomindex,k_index],dist2_mat[j_index,k_index],new_f_vector[j_index],new_f_vector[k_index],f_matrix[atomindex,j_index],f_matrix[atomindex,k_index],f_matrix[j_index,k_index],η,λ,ζ)

                    g_vector[ 1 , atomindex] += g_vals[1]
                    g_vector[ i_val[1] , j_index] += g_vals[2]
                    g_vector[ i_val[1] , k_index] += g_vals[3]
                end
            end
        end
    end
    # calculate i-ZnZn i->row 3; j,k -> row 2/3
    for j_index in n1+1:N
        if j_index != atomindex
            for k_index in j_index+1:N
                if k_index != atomindex
                    g_vals = new_angular_symm_vals(newposition,positions[atomindex],positions[j_index],positions[k_index],new_dist2_vector[j_index],new_dist2_vector[k_index],dist2_mat[atomindex,j_index],dist2_mat[atomindex,k_index],dist2_mat[j_index,k_index],new_f_vector[j_index],new_f_vector[k_index],f_matrix[atomindex,j_index],f_matrix[atomindex,k_index],f_matrix[j_index,k_index],η,λ,ζ)
                    
                    g_vector[ 3 , atomindex] += g_vals[1]
                    g_vector[ i_val[2] , j_index] += g_vals[2]
                    g_vector[ i_val[2] , k_index] += g_vals[3]
                    
                end
            end
        end
    end
    # calculate i-CuZn i->row 2; j -> 2/3; k -> 1/2
    for j_index in 1:n1
        if j_index !=atomindex
            for k_index in n1+1:N
                if k_index != atomindex

                        g_vals = new_angular_symm_vals(newposition,positions[atomindex],positions[j_index],positions[k_index],new_dist2_vector[j_index],new_dist2_vector[k_index],dist2_mat[atomindex,j_index],dist2_mat[atomindex,k_index],dist2_mat[j_index,k_index],new_f_vector[j_index],new_f_vector[k_index],f_matrix[atomindex,j_index],f_matrix[atomindex,k_index],f_matrix[j_index,k_index],η,λ,ζ)

                        g_vector[ 2 , atomindex] += g_vals[1]
                        g_vector[ i_val[3] , j_index] += g_vals[2]
                        g_vector[ i_val[4] , k_index] += g_vals[3]
                    
                end
            end
        end
    end
    
    g_vector[1,1:n1] = g_vector[1,1:n1].*tpz[1]
    g_vector[2,1:n1] = g_vector[2,1:n1].*tpz[2]
    g_vector[3,1:n1] = g_vector[3,1:n1].*tpz[3]
    
    g_vector[1,n1+1:N] = g_vector[1,n1+1:N].*tpz[4]
    g_vector[2,n1+1:N] = g_vector[2,n1+1:N].*tpz[5]
    g_vector[3,n1+1:N] = g_vector[3,n1+1:N].*tpz[6]
    return g_vector
end
"""
    calc_delta_matrix(g_mat,positions,newposition,atomindex,dist2_mat,new_dist2_vector,f_mat,new_f_vector,radsymmfunctions,angsymmfunctions,nrad,nang,n1,n2)
Having moved a single atom indexed by atomindex, we calcualte the changes to the total symmetry matrix g_mat using the calc_delta_symm_val functions
"""
function calc_delta_matrix(g_mat,positions,newposition,atomindex,dist2_mat,new_dist2_vector,f_mat,new_f_vector,radsymmfunctions,angsymmfunctions,nrad,nang,n1,n2)

    for g_index in 1:nrad
        idx=(g_index-1)*2+1

        g_mat[idx:idx+1,:] = calc_delta_symm_val!(g_mat[idx:idx+1,:],atomindex,dist2_mat,new_dist2_vector,f_mat,new_f_vector,n1,n2,radsymmfunctions[g_index].eta,radsymmfunctions[g_index].G_norm)   

    end

    for g_index in 1:nang 
        idx = nrad*2 + (g_index-1)*3 + 1 

       g_mat[idx:idx+2,:] = calc_delta_symm_val!(g_mat[idx:idx+2,:],positions,newposition,atomindex,dist2_mat,new_dist2_vector,f_mat,new_f_vector,n1,n2,angsymmfunctions[g_index].eta,angsymmfunctions[g_index].lambda,angsymmfunctions[g_index].zeta,angsymmfunctions[g_index].tpz)
    end

    return g_mat
end

#---------------------------------------------------------------------#
#---------------------------- Atom Swaps -----------------------------#
#---------------------------------------------------------------------#
"""
    calc_swap_symm_val(g_vector,atomindex1,atomindex2,dist2_mat,f_mat,n1,n2,η,g_norm)
    calc_swap_symm_val(g_vector,positions,atomindex1,atomindex2,dist2_mat,f_mat,n1,n2,η,λ,ζ,tpz)
function to calculate the changes to a symmetry vector g_vector assuming we have swapped the positions of atomindex1 and atomindex2. The first method is defined for a radial symmetry function with parameters η and g_norm. Works for a system with n1 atoms of type 1 and n2 atoms of type 2.

"""
function calc_swap_symm_val(g_vector,atomindex1,atomindex2,dist2_mat,f_mat,n1,n2,η,g_norm)

    N = n1+n2 
    
    for index in 1:n1
        if index != atomindex1
            gval1 = calc_one_symm_val(dist2_mat[atomindex1,index],f_mat[atomindex1,index],η) #the value of the new Zn atom which will live at index2

            gval2 = calc_one_symm_val(dist2_mat[atomindex2,index],f_mat[atomindex2,index],η) #the value of the current Zn which will live at index1


            g_vector[1,index] += gval2-gval1
            g_vector[2,index] += gval1-gval2

            g_vector[1,atomindex1] += gval2-gval1
            
            g_vector[1,atomindex2] += gval1-gval2

        end
    end

    for index in n1+1:N
        if index != atomindex2
            gval1 = calc_one_symm_val(dist2_mat[atomindex1,index],f_mat[atomindex1,index],η) #the value of the new Zn atom which will live at index2
            gval2 = calc_one_symm_val(dist2_mat[atomindex2,index],f_mat[atomindex2,index],η) #the value of the old Zn which will live at index1

            g_vector[1,index] += gval2-gval1
            g_vector[2,index] += gval1-gval2

            g_vector[2,atomindex1] += gval2-gval1

            g_vector[2,atomindex2] += gval1-gval2

        end
    end


    g_vector[1,1:n1] = g_vector[1,1:n1] .* g_norm[1]
    g_vector[2,1:n1] = g_vector[2,1:n1] .* g_norm[2]

    g_vector[1,1+n1:N] = g_vector[1,1+n1:N] .* g_norm[3]
    g_vector[2,1+n1:N] = g_vector[2,1+n1:N] .* g_norm[4]

    return g_vector
end



function calc_swap_symm_val(g_vector,positions,atomindex1,atomindex2,dist2_mat,f_mat,n1,n2,η,λ,ζ,tpz)
    N = n1+n2
    for j_index in 1:n1
        if j_index != atomindex1
            for k_index in j_index+1:n1
                if k_index != atomindex1

                    g_vals1 = calc_one_symm_val(positions[atomindex1],positions[j_index],positions[k_index],dist2_mat[atomindex1,j_index],dist2_mat[atomindex1,k_index],dist2_mat[j_index,k_index],f_mat[atomindex1,j_index],f_mat[atomindex1,k_index],f_mat[j_index,k_index],η,λ,ζ)

                    g_vals2 = calc_one_symm_val(positions[atomindex2],positions[j_index],positions[k_index],dist2_mat[atomindex2,j_index],dist2_mat[atomindex2,k_index],dist2_mat[j_index,k_index],f_mat[atomindex2,j_index],f_mat[atomindex2,k_index],f_mat[j_index,k_index],η,λ,ζ)

                    g_vector[ 1 , atomindex1 ] += g_vals2[1] - g_vals1[1]
                    g_vector[ 1 , atomindex2 ] += g_vals1[1] - g_vals2[1]

                    g_vector[ 1 , j_index ] += g_vals2[2] - g_vals1[2]
                    g_vector[ 1 , k_index ] += g_vals2[3] - g_vals1[3]

                    g_vector[ 2 , j_index ] += g_vals1[2] - g_vals2[2]
                    g_vector[ 2 , k_index ] += g_vals1[3] - g_vals2[3]

                end
            end
        end
    end

    for j_index in n1+1:N
        if j_index != atomindex2
            for k_index in j_index+1:N
                if k_index != atomindex2

                    g_vals1 = calc_one_symm_val(positions[atomindex1],positions[j_index],positions[k_index],dist2_mat[atomindex1,j_index],dist2_mat[atomindex1,k_index],dist2_mat[j_index,k_index],f_mat[atomindex1,j_index],f_mat[atomindex1,k_index],f_mat[j_index,k_index],η,λ,ζ)

                    g_vals2 = calc_one_symm_val(positions[atomindex2],positions[j_index],positions[k_index],dist2_mat[atomindex2,j_index],dist2_mat[atomindex2,k_index],dist2_mat[j_index,k_index],f_mat[atomindex2,j_index],f_mat[atomindex2,k_index],f_mat[j_index,k_index],η,λ,ζ)

                    g_vector[ 3 , atomindex1 ] += g_vals2[1] - g_vals1[1]
                    g_vector[ 3 , atomindex2 ] += g_vals1[1] - g_vals2[1]

                    g_vector[ 2 , j_index ] += g_vals2[2] - g_vals1[2]
                    g_vector[ 2 , k_index ] += g_vals2[3] - g_vals1[3]

                    g_vector[ 3 , j_index ] += g_vals1[2] - g_vals2[2]
                    g_vector[ 3 , k_index ] += g_vals1[3] - g_vals2[3]

                end
            end
        end
    end

    for j_index in 1:n1
        if j_index != atomindex1
            for k_index in n1+1:N
                if k_index != atomindex2

                    g_vals1 = calc_one_symm_val(positions[atomindex1],positions[j_index],positions[k_index],dist2_mat[atomindex1,j_index],dist2_mat[atomindex1,k_index],dist2_mat[j_index,k_index],f_mat[atomindex1,j_index],f_mat[atomindex1,k_index],f_mat[j_index,k_index],η,λ,ζ)

                    g_vals2 = calc_one_symm_val(positions[atomindex2],positions[j_index],positions[k_index],dist2_mat[atomindex2,j_index],dist2_mat[atomindex2,k_index],dist2_mat[j_index,k_index],f_mat[atomindex2,j_index],f_mat[atomindex2,k_index],f_mat[j_index,k_index],η,λ,ζ)

                    g_vector[ 2 , atomindex1 ] += g_vals2[1] - g_vals1[1]
                    g_vector[ 2 , atomindex2 ] += g_vals1[1] - g_vals2[1]
                     

                    g_vector[ 2 , j_index ] += g_vals2[2] - g_vals1[2]
                    g_vector[ 1 , k_index ] += g_vals2[3] - g_vals1[3]

                    g_vector[ 3 , j_index ] += g_vals1[2] - g_vals2[2]
                    g_vector[ 2 , k_index ] += g_vals1[3] - g_vals2[3]

                end
            end
        end
    end

    for k_index in 1:n1
        if k_index != atomindex1

        g_vals = calc_one_symm_val(positions[atomindex1],positions[atomindex2],positions[k_index],dist2_mat[atomindex1,atomindex2],dist2_mat[atomindex1,k_index],dist2_mat[atomindex2,k_index],f_mat[atomindex1,atomindex2],f_mat[atomindex1,k_index],f_mat[atomindex2,k_index],η,λ,ζ)

        g_vector[ 2 , atomindex1 ] -= g_vals[1]
        g_vector[ 1 , atomindex2 ] += g_vals[1]

        g_vector[ 1 , atomindex2 ] -= g_vals[2]
        g_vector[ 2 , atomindex1 ] += g_vals[2]

        end
    end

    for k_index in n1+1:N 
        if k_index != atomindex2

        g_vals = calc_one_symm_val(positions[atomindex1],positions[atomindex2],positions[k_index],dist2_mat[atomindex1,atomindex2],dist2_mat[atomindex1,k_index],dist2_mat[atomindex2,k_index],f_mat[atomindex1,atomindex2],f_mat[atomindex1,k_index],f_mat[atomindex2,k_index],η,λ,ζ)

        g_vector[ 3 , atomindex1 ] -= g_vals[1]
        g_vector[ 2 , atomindex2 ] += g_vals[1]

        g_vector[ 2 , atomindex2 ] -= g_vals[2]
        g_vector[ 3 , atomindex1 ] += g_vals[2]

        end
    end

    g_vector[1,1:n1] = g_vector[1,1:n1].*tpz[1]
    g_vector[2,1:n1] = g_vector[2,1:n1].*tpz[2]
    g_vector[3,1:n1] = g_vector[3,1:n1].*tpz[3]
    
    g_vector[1,n1+1:N] = g_vector[1,n1+1:N].*tpz[4]
    g_vector[2,n1+1:N] = g_vector[2,n1+1:N].*tpz[5]
    g_vector[3,n1+1:N] = g_vector[3,n1+1:N].*tpz[6]
    
    return g_vector
end

"""
    calc_swap_matrix(g_mat,positions,atomindex1,atomindex2,dist2_mat,f_mat,radsymmfunctions,angsymmfunctions,nrad,nang,n1,n2)
having swapped atom at atomindex1 and atomindex2 in a system with n1 atoms of type 1 and n2 atoms of type 2, with nrad radial and nang angular symmetry functions, we calculate the changes to g_mat based on the swap. 
"""
function calc_swap_matrix(g_mat,positions,atomindex1,atomindex2,dist2_mat,f_mat,radsymmfunctions,angsymmfunctions,nrad,nang,n1,n2)
    for g_index in 1:nrad
        idx=(g_index-1)*2+1

        g_mat[idx:idx+1,:] = calc_swap_symm_val(g_mat[idx:idx+1,:],atomindex1,atomindex2,dist2_mat,f_mat,n1,n2,radsymmfunctions[g_index].eta,radsymmfunctions[g_index].G_norm)   

    end
    for g_index in 1:nang 
        idx = nrad*2 + (g_index-1)*3 + 1 

       g_mat[idx:idx+2,:] = calc_swap_symm_val(g_mat[idx:idx+2,:],positions,atomindex1,atomindex2,dist2_mat,f_mat,n1,n2,angsymmfunctions[g_index].eta,angsymmfunctions[g_index].lambda,angsymmfunctions[g_index].zeta,angsymmfunctions[g_index].tpz)
    end

    return g_mat
end
#---------------------------------------------------------------------------#
# These are the old/defunct functions, they will have to go, but getenergy has matching methods, so not yet#
"""
    adjust_symm_val!(g_value,r_sum,f_prod,η,g_norm)
Designed to update the radial symmetry function value `g_value`. Accepts the hyperparameter `η` as well as `r_sum`, `f_prod` and `g_norm` and adds the individual contribution of `g_{ij}`. 
"""
function adjust_symm_val!(g_value,r_sum,f_prod,η,g_norm)
    #adjusts radial type 2 symmetry function
    g_value += exponential_part(η,r_sum,f_prod*g_norm)
    return g_value
end
"""
    adjust_radial_symm_val!(g_value1,g_value2,rnew_ij,r2_ij,fnew_ij,f2_ij,η,g_norm)
Taking the `i` and `j` symmetry function values `g_value1,g_value2` we use the hyperparameters `η,g_norm` as well as the old and new distances and cutoff functions `rnew_ij,fnew_ij` `r2_ij,f2_ij` adds the new symmetry value and subtracts the old one based on `g_{ij}`
"""
function adjust_radial_symm_val!(g_value1,g_value2,rnew_ij,r2_ij,fnew_ij,f2_ij,η,g_norm)
 
    g_value1,g_value2 = adjust_symm_val!(g_value1,rnew_ij,fnew_ij,η,g_norm),adjust_symm_val!(g_value2,rnew_ij,fnew_ij,η,g_norm)
    g_value1,g_value2 = adjust_symm_val!(g_value1,r2_ij,-f2_ij,η,g_norm),adjust_symm_val!(g_value2,r2_ij,-f2_ij,η,g_norm)

    return g_value1,g_value2
end
"""
    adjust_angular_symm_val!(g_value,θ_new,θ_old,exp_new,exp_old,tpz)
    adjust_angular_symm_val!(g_value,exp_old,exp_new,θ_old,θ_new,λ,ζ,tpz)

Functions for adjusting angular symmetry function value from `g_value` by calculating the exponential component `exp_old,exp_new`, theta components `θ_val_old,θ_val_new` from the angles `θ_old,θ_new` and the normalisaiton factor `tpz` These are used to subtract the old `g` value and add the new one. 
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
Call for the radial symmetry value designed to curry the input from `g_vector` at positions `indexi,indexj` to the [`adjust_radial_symm_val!`](@ref) function. It unpacks the radial distances from `dist2_mat,new_dist2_vector` and the cutoff functions from `f_matrix,new_f_vec` as well as the hyperparameters `η,g_norm` and gives these values to the lower level functions. 
"""
function calc_new_symmetry_value!(g_vector,indexi,indexj,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,g_norm)
    g_vector[indexi],g_vector[indexj] = adjust_radial_symm_val!(g_vector[indexi],g_vector[indexj],new_dist2_vector[indexj],dist2_mat[indexi,indexj],new_f_vector[indexj],f_matrix[indexi,indexj],η,g_norm)
    
end
"""
    calc_new_symmetry_value!(g_vector,indices,newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ,tpz)
    calc_new_symmetry_value!(g_vector,indexi,indexj,indexk,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,λ,ζ,tpz)

Currying functions from higher-level data structures such as the radial distances in `dist2_mat,new_dist2_vector` and cutoff functions `f_matrix,new_f_vector` as well as positions in `newposition,position` and passes these to lower level functions along with hyperparameters `η,λ,ζ,tpz` to adjust the values in `g_vector` at positions `indexi,indexj,indexk` calculates the required exponential component and angles to pass down to [`adjust_angular_symm_val!`](@ref)
"""
function calc_new_symmetry_value!(g_vector,indexi,indexj,indexk,newposition,position1,position2,position3,rnew_ij,rnew_ik,r2_ij,r2_ik,r2_jk,fnew_ij,fnew_ik,f_ij,f_ik,f_jk,η,λ,ζ,tpz)

    θ_new,θ_old = all_angular_measure(newposition,position2,position3,rnew_ij,rnew_ik,r2_jk),all_angular_measure(position1,position2,position3,r2_ij,r2_ik,r2_jk)

    exp_new,exp_old = exponential_part(η,rnew_ij,rnew_ik,r2_jk,fnew_ij,fnew_ik,f_jk),exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk)
    
    g_vector[indexi] = adjust_angular_symm_val!(g_vector[indexi],exp_old,exp_new,θ_old[1],θ_new[1],λ,ζ,tpz)
    g_vector[indexj] = adjust_angular_symm_val!(g_vector[indexj],exp_old,exp_new,θ_old[2],θ_new[2],λ,ζ,tpz)
    g_vector[indexk] = adjust_angular_symm_val!(g_vector[indexk],exp_old,exp_new,θ_old[3],θ_new[3],λ,ζ,tpz)
    # for (θ_old,θ_new,index) in zip(θ_old_vec,θ_new_vec,indices)
    #     g_vector[index] = adjust_angular_symm_val!(g_vector[index],exp_old,exp_new,θ_old,θ_new,λ,ζ,tpz)
    # end
    return g_vector
end
function calc_new_symmetry_value!(g_vector,indexi,indexj,indexk,newposition,position,dist2_mat,new_dist2_vector,f_matrix,new_f_vector,η,λ,ζ,tpz)
    return calc_new_symmetry_value!(g_vector,indexi,indexj,indexk,newposition,position[indexi],position[indexj],position[indexk],new_dist2_vector[indexj],new_dist2_vector[indexk],dist2_mat[indexi,indexj],dist2_mat[indexi,indexk],dist2_mat[indexj,indexk],new_f_vector[indexj],new_f_vector[indexk],f_matrix[indexi,indexj],f_matrix[indexi,indexk],f_matrix[indexj,indexk],η,λ,ζ,tpz)

end
"""
    radial_symmetry_calculation!(g_vector, atomindex, dist2_mat, new_dist2_vector, f_matrix, new_f_vector, symmetry_function::RadialType2)
    symmetry_calculation!(g_vector,atomindex,newposition,position,dist2_mat,new_dis_vector,f_matrix,new_f_vector,symmetry_function::AngularType3)
This method is designed for radial symmetry functions. Given an atom at `atomindex` along with high level data structures: `dist2_mat,new_dist2_vector,f_matrix,new_f_vector` containing the new and old positions, distances and cutoff functions. We iterate over all other atoms and pass their index to lower level currying functions.  The positions `newposition,position` are included for consistency with the higher-level function.
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
"""
    angular_symmetry_calculation!(g_vector, atomindex, newposition, position, dist2_mat, new_dis_vector, f_matrix, new_f_vector, symmetry_function::AngularType3)
This method is designed to do the same as the radial symmetry function for the angular symmetry function using the same inputs. Double loop over all `j,k` and use [`calc_new_symmetry_value!`](@ref) over `g_vector`.
"""
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
Top level function to calculate the total change to the matrix of symmetry function values `g_matrix`. Given `position,dist2_matrix,f_matrix` containing the original state of the system, and `new_position,new_dist_vector,new_f_vector` the change to this state based on the motion of `atomindex`, we iterate over the `total_symmetry_vector` using the defined [`radial_symmetry_calculation!`](@ref) and [`angular_symmetry_calculation!`](@ref) functions. 
"""
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
function total_thr_symm!(g_matrix,position,new_position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,atomindex,radsymmfunctions,angsymmfunctions,Nrad,Nang)
    Threads.@threads for g_index in 1:Nrad
        #@views
        g_matrix[g_index,:] = radial_symmetry_calculation!(g_matrix[g_index,:],atomindex,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,radsymmfunctions[g_index])
    end
    Threads.@threads for g_index in 1:Nang
#@views
        truindex = g_index+Nrad
        g_matrix[truindex ,:] = angular_symmetry_calculation!(g_matrix[truindex,:],atomindex,new_position,position,dist2_matrix,new_dist_vector,f_matrix,new_f_vector,angsymmfunctions[g_index])
    end
    return g_matrix
end


end