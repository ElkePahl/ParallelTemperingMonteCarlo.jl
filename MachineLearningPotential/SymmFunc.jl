module SymmetryFunctions

using ..Cutoff
using StaticArrays

export AbstractSymmFunction,AngularSymmFunction,RadialSymmFunction 
export RadialType2,AngularType3,RadialType2a,AngularType3a
export exponential_part,theta_part
export calc_one_symm_val,calc_symm_vals!,update_g_vals!
export total_symm_calc,total_thr_symm_calc



#------------------------------------------------#
#----------------Type Definitions----------------#
#------------------------------------------------#

abstract type AbstractSymmFunction{T} end 
abstract type RadialSymmFunction{T} <: AbstractSymmFunction{T} end
abstract type AngularSymmFunction{T} <: AbstractSymmFunction{T} end

#------------------------------------------------#
#-------------Symm Func Definitions--------------#
#------------------------------------------------#

struct RadialType2{T} <: RadialSymmFunction{T}
    eta::T
    r_cut::T
    type_vec::Int
    G_offset::T
    G_norm::T
end
"""
    RadialType2{T}(eta,r_cut,type_vector) 
    RadialType2{T}(eta,r_cut,type_vector::Vector,G_vals::Vector) 

Various definitions of the RadialType2 struct to account for new normalisation factors required by the neural network to simplify the math. One only accepts the standard hyperparameters trained by the neural network and sets the offset and normalisation factors to zero and one respectively. The other accepts G_min,G_max and calculates the normalisation and offset manually
"""
function RadialType2{T}(eta,r_cut,type_vector) where {T}
    return RadialType2(eta,r_cut,type_vector,0.,1.)
end
function RadialType2{T}(eta,r_cut,type_vector,G_vals::Vector) where {T}
    G_norm = 1/(G_vals[1] - G_vals[2])
    G_offset = -G_vals[2]*G_norm
    return RadialType2(eta,r_cut,type_vector,G_offset,G_norm)
end
#------------------------------------------------------------------------------#
#-----------------------------2 atom radial struct-----------------------------#
#------------------------------------------------------------------------------#
struct RadialType2a{T} <: RadialSymmFunction{T}
    eta::T
    r_cut::T
    type_vec::Int
    G_offset::SVector{4,T}
    G_norm::SVector{4,T}
end
function RadialType2a{T}(eta,r_cut,type_vector) where {T}
    return RadialType2a(eta,r_cut,type_vector,SVector{2}(0.,0.),SVector{2}(1.,1.))
end
function RadialType2a{T}(eta,r_cut,type_vector,G_vals_a::Vector,G_vals_b) where {T}
    G_norm1 = 1/(G_vals_a[1][1] - G_vals_a[1][2])
    G_offset1 = -G_vals_a[1][2]*G_norm1

    G_norm2 = 1/(G_vals_a[2][1] - G_vals_a[2][2])
    G_offset2 = -G_vals_a[2][2]*G_norm2

    G_norm3 = 1/(G_vals_b[1][1] - G_vals_b[1][2])
    G_offset3 = -G_vals_b[1][2]*G_norm3

    G_norm4 = 1/(G_vals_b[2][1] - G_vals_b[2][2])
    G_offset4 = -G_vals_b[2][2]*G_norm4


    return RadialType2a(eta,r_cut,type_vector,SVector{4}(G_offset1,G_offset2,G_offset3,G_offset4),SVector{4}(G_norm1,G_norm2,G_norm3,G_norm4))

end
#---------------------------------------------------------------------------------------#
struct AngularType3{T} <:AngularSymmFunction{T}
    eta::T
    lambda::T
    zeta::T
    r_cut::T
    type_vec::Int
    tpz::T
    G_offset::T
    #G_norm::T
    
end
"""
    AngularType3{T}(eta,lambda,zeta,r_cut,type_vec::Vector) where {T}
    AngularType3{T}(eta,lambda,zeta,r_cut,type_vector::Vector,G_vals::Vector) where {T}
Functions to initialise the AngularType3 structs based on various different definitions. If we don't include the offset and normalisation factors the two power of (one minus) zeta factor inlcudes no normalisation, and the offset is zero. 
Second definition includes a vector containing G_max and G_min in a vector, it sets the offset and renormalises tpz to include G_norm. 
"""
function AngularType3{T}(eta,lambda,zeta,r_cut,type_vec) where {T}
    tpz = 2.0^(1-zeta)
    return AngularType3(eta,lambda,zeta,r_cut,type_vec,tpz,0.)
end
function AngularType3{T}(eta,lambda,zeta,r_cut,type_vector,G_vals::Vector) where {T}
    G_norm = 1/(G_vals[1] - G_vals[2])
    G_offset = -G_vals[2]*G_norm
    tpz = 2.0^(1-zeta)*G_norm
    return AngularType3(eta,lambda,zeta,r_cut,type_vector,tpz,G_offset)
end
#-----------------------------------------------------------------------------#
struct AngularType3a{T} <:AngularSymmFunction{T}
    eta::T
    lambda::T
    zeta::T
    r_cut::T
    type_vec::Int
    tpz::SVector{6,T}
    G_offset::SVector{6,T}
    #G_norm::T
end
function AngularType3a{T}(eta,lambda,zeta,r_cut,type_vec) where {T}
    tpz = 2.0^(1-zeta)

    return AngularType3a(eta,lambda,zeta,r_cut,type_vec,tpz,SVector{6}(0.,0.,0.,0.,0.,0.))
end
function AngularType3a{T}(eta,lambda,zeta,r_cut,type_vector,G_valsa::Vector,G_valsb::Vector) where {T}

    G_norm1 = 1/(G_valsa[1][1] - G_valsa[1][2])
    G_offset1 = -G_valsa[1][2]*G_norm1
    tpz1 = 2.0^(1-zeta)*G_norm1

    G_norm2 = 1/(G_valsa[2][1] - G_valsa[2][2])
    G_offset2 = -G_valsa[2][2]*G_norm2
    tpz2 = 2.0^(1-zeta)*G_norm2

    G_norm3 = 1/(G_valsa[3][1] - G_valsa[3][2])
    G_offset3 = -G_valsa[3][2]*G_norm3
    tpz3 = 2.0^(1-zeta)*G_norm3

    G_norm4 = 1/(G_valsb[1][1] - G_valsb[1][2])
    G_offset4 = -G_valsb[1][2]*G_norm4
    tpz4 = 2.0^(1-zeta)*G_norm4

    G_norm5 = 1/(G_valsb[2][1] - G_valsb[2][2])
    G_offset5 = -G_valsb[2][2]*G_norm5
    tpz5 = 2.0^(1-zeta)*G_norm5

    G_norm6 = 1/(G_valsb[3][1] - G_valsb[3][2])
    G_offset6 = -G_valsb[3][2]*G_norm6
    tpz6 = 2.0^(1-zeta)*G_norm6

    return AngularType3a(eta,lambda,zeta,r_cut,type_vector,SVector{6}(tpz1,tpz2,tpz3,tpz4,tpz5,tpz6),SVector{6}(G_offset1,G_offset2,G_offset3,G_offset4,G_offset5,G_offset6))
end
#------------------------------------------------------------------#
#-------------------Calculating One Symm Val-----------------------#
#------------------------------------------------------------------#
"""
    exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk)
    exponential_part(η,rsum,f_prod)
calculates the exponential portion of the symmetry function for the angular symmetry function. Preserves the values we can maintain throughout iterating over theta. Second method simply reduces the inputs to what is actually required. 
"""
exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk) = exp(-η*(r2_ij+r2_ik+r2_jk))* f_ij * f_ik * f_jk
exponential_part(η,rsum,f_prod) = exp(-η*(rsum))*f_prod
"""
    theta_part(θ,λ,ζ)
Calculates the angular portion of a single symmetry function, this requires iteration over each of the three angles.
"""
theta_part(θ,λ,ζ) = (1+λ*θ)^ζ
"""
    symmfunc_calc(θ_vec,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)
Calculates the three g_values corresponding to the three atoms iterated over, builds the foundation of the total symm function as calculated below.
"""
function symmfunc_calc(θ_vec,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)

    exp_part = exponential_part(η,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk)
    g_values = MVector{3}([exp_part* theta_part(θ,λ,ζ) for θ in θ_vec])
    
    return g_values
end
"""
    update_g_vals!(g_vec,g_vals,atomindex,index2,index3)
function to correctly update the symmvalues 'g_vals' at the indices in 'g_vec'
 """
function update_g_vals!(g_vec,g_vals,atomindex,index2,index3)

    g_vec[atomindex] += g_vals[1]
    g_vec[index2] += g_vals[2]
    g_vec[index3] += g_vals[3]
    
    return g_vec
end

"""
    calc_one_symm_val(r2_ij,fc_ij,η)
Accepts interatomic distance squared `r2_ij`, the cutoff function 'fc_ij' and a gaussian parameter `η`  it then calculates the radial symmetry function value for a single pair of atoms.
    calc_one_symm_val(θ,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)
    (position1,position2,position3,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)

returns a single symmetry function value from the double-sum. accepts `θ` the angle between ijk centred on i, and the squared distances `r2_ij`,`r2_ik`, `r2_jk`, the cutoff function values `f_ij,f_ik,f_jk` along with the symmetry funciton parameters `η`,`λ`,`ζ`, and the cutoff radius `r_cut`.

The version with `position_i` calculates the angle between positions before calculating the symmetry functions according to the previous method.  
"""
calc_one_symm_val(r2_ij,fc_ij,g_norm,η) = ifelse(fc_ij!=0. && fc_ij!=1., fc_ij*exp(-η*r2_ij)*g_norm, 0.)


calc_one_symm_val(r2_ij,fc_ij,η) = ifelse(fc_ij!=0. && fc_ij!=1., fc_ij*exp(-η*r2_ij), 0.)


function calc_one_symm_val(position1,position2,position3,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)
    θ_vec = all_angular_measure(position1,position2,position3,r2_ij,r2_ik,r2_jk)
    
    g_vals = symmfunc_calc(θ_vec,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)

    return g_vals
end
# function calc_one_symm_val(position1,position2,position3,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)
#     θ_vec = all_angular_measure(position1,position2,position3,r2_ij,r2_ik,r2_jk)
    
#     g_vals = symmfunc_calc(θ_vec,r2_ij,r2_ik,r2_jk,f_ij,f_ik,f_jk,η,λ,ζ)

#     return g_vals
# end
#----------------------------------------------------------------#
#------------------Total Symmetry Calculation--------------------#
#----------------------------------------------------------------#

""" 
    calc_symm_vals!(positions,dist2_mat,f_mat,g_vec,symm_func::RadialType2)
Accepts `positions` for consistency with angular calculation, `dist2_mat` and `f_mat` containing the distances and cutoff functions relevant to the symmetry values, lastly accepts the symmetry function over which to iterate. `g_vec` is an N_atom vector into which the total contributions of each atom are inputted. Returns the same vector. 
"""
function calc_symm_vals!(positions,dist2_mat,f_mat,g_vec,symm_func::RadialType2)
    N=length(g_vec)
    if symm_func.type_vec == Int(11)
        g_norm,η =  symm_func.G_norm,symm_func.eta
        for atomindex in eachindex(g_vec)
            for index2 in (atomindex+1):N
                g_val =  calc_one_symm_val(dist2_mat[atomindex,index2],f_mat[atomindex,index2],g_norm,η)
                g_vec[atomindex] += g_val 
                g_vec[index2] += g_val
            end
        end

        return g_vec .+ symm_func.G_offset
    else
        return zeros(N) 
    end    
end

function calc_symm_vals!(positions,dist2_mat,f_mat,g_vec,symm_func::AngularType3)
    N = length(g_vec)  
    if symm_func.type_vec == Int(111)
        η,λ,ζ = symm_func.eta,symm_func.lambda,symm_func.zeta
        for atomindex in eachindex(g_vec)
            for index2 in (atomindex+1):N
                for index3 in (index2+1):N

                    g_vals=calc_one_symm_val(positions[atomindex],positions[index2],positions[index3],dist2_mat[atomindex,index2],dist2_mat[atomindex,index3],dist2_mat[index2,index3],f_mat[atomindex,index2],f_mat[atomindex,index3],f_mat[index2,index3],η,λ,ζ)

                    g_vals .*= symm_func.tpz 
                    
                    g_vec = update_g_vals!(g_vec,g_vals,atomindex,index2,index3)
                end
            end
        end
        return g_vec .+ symm_func.G_offset
        
    else
       return zeros(N) 
    end

end
#-----------------------------------------------------------------------------------------#
function calc_symm_vals!(position,dist2_mat,f_mat,g_vec,n1,n2,η,g_norm,G_offset)
    N = n1+n2
    # CuCu populates row 1 for both i and j
    for atomindex in 1:n1
        for index2 in atomindex+1:n1

            g_val =  calc_one_symm_val(dist2_mat[atomindex,index2],f_mat[atomindex,index2],η)

            g_vec[1,atomindex] += g_val 
            g_vec[1,index2] += g_val
        end
    end
    #ZnZn populates row 2 for both i and j
    for atomindex in n1+1:N
        for index2 in atomindex+1:N

            g_val =  calc_one_symm_val(dist2_mat[atomindex,index2],f_mat[atomindex,index2],η)

            g_vec[2,atomindex] += g_val 
            g_vec[2,index2] += g_val
        end
    end
    #CuZn is row 2 for i row 1 for Zn
    for atomindex in 1:n1
        for index2 in n1+1:N

            g_val =  calc_one_symm_val(dist2_mat[atomindex,index2],f_mat[atomindex,index2],η)

            g_vec[2,atomindex] += g_val 
            g_vec[1,index2] += g_val
        end
    end
    
    g_vec[1,1:n1] = g_vec[1,1:n1] .* g_norm[1] .+ G_offset[1]
    g_vec[2,1:n1] = g_vec[2,1:n1] .* g_norm[2] .+ G_offset[2]

    g_vec[1,1+n1:N] = g_vec[1,1+n1:N] .* g_norm[3] .+ G_offset[3]
    g_vec[2,1+n1:N] = g_vec[2,1+n1:N] .* g_norm[4] .+ G_offset[4]

    return g_vec
end

function calc_symm_vals!(positions,dist2_mat,f_mat,g_vec,n1,n2,η,λ,ζ,tpz,G_offset)
    N = n1+n2
 #   CuCuCu row 1
    for atomindex in 1:n1
        for index2 in atomindex+1:n1
            for index3 in index2+1:n1
                g_vals=calc_one_symm_val(positions[atomindex],positions[index2],positions[index3],dist2_mat[atomindex,index2],dist2_mat[atomindex,index3],dist2_mat[index2,index3],f_mat[atomindex,index2],f_mat[atomindex,index3],f_mat[index2,index3],η,λ,ζ)

                g_vec[1,atomindex] += g_vals[1]
                g_vec[1,index2] += g_vals[2]
                g_vec[1,index3] += g_vals[3]
            end
        end
    end
#   ZnZnZn row 3
    for atomindex in n1+1:N
        for index2 in atomindex+1:N
            for index3 in index2+1:N
                g_vals=calc_one_symm_val(positions[atomindex],positions[index2],positions[index3],dist2_mat[atomindex,index2],dist2_mat[atomindex,index3],dist2_mat[index2,index3],f_mat[atomindex,index2],f_mat[atomindex,index3],f_mat[index2,index3],η,λ,ζ)

                g_vec[3,atomindex] += g_vals[1]
                g_vec[3,index2] += g_vals[2]
                g_vec[3,index3] += g_vals[3]
            end
        end
    end
#CuCuZn row 2 for Cu, row 1 for Zn
    for atomindex in 1:n1
        for index2 in atomindex+1:n1
            for index3 in n1+1:N
                
                g_vals=calc_one_symm_val(positions[atomindex],positions[index2],positions[index3],dist2_mat[atomindex,index2],dist2_mat[atomindex,index3],dist2_mat[index2,index3],f_mat[atomindex,index2],f_mat[atomindex,index3],f_mat[index2,index3],η,λ,ζ)
                

                g_vec[2,atomindex] += g_vals[1]
                g_vec[2,index2] += g_vals[2]
                g_vec[1,index3] += g_vals[3]
            end
        end
    end
# CuZnZn row 3 for Cu row 2 for Zn
    for atomindex in 1:n1
        for index2 in n1+1:N
            for index3 in index2+1:N
                g_vals=calc_one_symm_val(positions[atomindex],positions[index2],positions[index3],dist2_mat[atomindex,index2],dist2_mat[atomindex,index3],dist2_mat[index2,index3],f_mat[atomindex,index2],f_mat[atomindex,index3],f_mat[index2,index3],η,λ,ζ)

                g_vec[3,atomindex] += g_vals[1]
                g_vec[2,index2] += g_vals[2]
                g_vec[2,index3] += g_vals[3]
            end
        end
    end
    g_vec[1,1:n1] = g_vec[1,1:n1].*tpz[1] .+ G_offset[1]
    g_vec[2,1:n1] = g_vec[2,1:n1].*tpz[2] .+ G_offset[2]
    g_vec[3,1:n1] = g_vec[3,1:n1].*tpz[3] .+ G_offset[3]
    
    g_vec[1,n1+1:N] = g_vec[1,n1+1:N].*tpz[4] .+ G_offset[4]
    g_vec[2,n1+1:N] = g_vec[2,n1+1:N].*tpz[5] .+ G_offset[5]
    g_vec[3,n1+1:N] = g_vec[3,n1+1:N].*tpz[6] .+ G_offset[6]

    return g_vec
end
#-----------------------------------------------------------------#
#--------------------Calculate Symmetry Matrix--------------------#
#-----------------------------------------------------------------#
"""
    init_symm_vecs(dist2_mat,total_symm_vec)
Prepares the symmetry matrix `g_mat` by taking the dimensions of the `dist2_mat` containing the squared distance of each atom with its pair, and `total_symm_vec` with all of the symmetry functions. 
"""
function init_symm_vecs(dist2_mat,total_symm_vec)
    g_mat=zeros(length(total_symm_vec),size(dist2_mat)[1])
    return g_mat 
end
"""
    total_symm_calc(positions,dist2_mat,f_mat,total_symm_vec)
Function to run over a vector of symmetry functions `total_symm_vec` and determining the value for each symmetry function for each atom at position `positions` with distances `dist2_mat` and a matrix of cutoff functions `f_mat` between each atom pair.
"""
function total_symm_calc(positions,dist2_mat,f_mat,radsymmfunctions,angsymmfunctions,Nrad,Nang) 

    g_mat = zeros(Nrad+Nang,length(positions))

    for g_index in 1:Nrad
        g_mat[g_index,:] = calc_symm_vals!(positions,dist2_mat,f_mat,g_mat[g_index,:],radsymmfunctions[g_index])
    end
    for g_index in Nrad+1:Nang
        g_mat[g_index,:] = calc_symm_vals!(positions,dist2_mat,f_mat,g_mat[g_index,:],angsymmfunctions[g_index-Nrad])
    end
    return g_mat 
end

function total_symm_calc(positions,dist2_mat,f_mat,radsymmfunctions,angsymmfunctions,nrad,nang,n1,n2)
    g_mat = zeros(nrad*2 + nang*3 , length(positions))
    for g_index in 1:nrad
        idx=(g_index-1)*2+1
        g_mat[idx:idx+1,:] = calc_symm_vals!(positions,dist2_mat,f_mat,g_mat[idx:idx+1,:],n1,n2,radsymmfunctions[g_index].eta,radsymmfunctions[g_index].G_norm,radsymmfunctions[g_index].G_offset)
    end
    for g_index in 1:nang 
        idx = nrad*2 + (g_index-1)*3 + 1 

        g_mat[idx:idx+2,:] = calc_symm_vals!(positions,dist2_mat,f_mat,g_mat[idx:idx+2,:],n1,n2,angsymmfunctions[g_index].eta,angsymmfunctions[g_index].lambda,angsymmfunctions[g_index].zeta,angsymmfunctions[g_index].tpz,angsymmfunctions[g_index].G_offset)
    end
    return g_mat
end
"""
    total_thr_symm_calc(positions,dist2_mat,f_mat,total_symm_vec)
This operates as the total_symm_calc function only threaded over the symmetry functions. 
"""
function total_thr_symm_calc(positions,dist2_mat,f_mat,total_symm_vec)
    g_mat = init_symm_vecs(dist2_mat,total_symm_vec)

    Threads.@threads for g_index in eachindex(total_symm_vec)
        g_mat[g_index,:] = calc_symm_vals!(positions,dist2_mat,f_mat,g_mat[g_index,:],total_symm_vec[g_index])
    end
    
    return g_mat
end

end