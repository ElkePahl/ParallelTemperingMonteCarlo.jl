module Cutoff

using LinearAlgebra
using StaticArrays

export distance2,find_distance2_mat,angular_measure,all_angular_measure
export cutoff_function

#----------------------------------------------------------------#
#-------------------------Measurements---------------------------#
#----------------------------------------------------------------#

#Beginning with basic distance funtions required throughout MLP calculations
"""
    dist2(a,b)
squared distance of two vectors `a` `b` 
"""
dist2(a,b) = (a-b)⋅(a-b)
"""
    find_distance2_mat(pos)
given a vector called `pos` comprised of (ideally) static vectors we return a lengthXlength symmetric matrix of the squared distance
"""
find_distance2_mat(pos) = [dist2(a,b) for a in pos, b in pos]

"""
    thetacalc(xy,xz,disxy,disxz)
calculates cosine theta of two vectors `xy,xz` with their sqared-distances `disxy,disxz`. 
"""
thetacalc(xy,xz,disxy,disxz) = xy⋅xz/(disxy*disxz)

"""
    angular_measure(a,b,c,r2ij,r2ik)
    angular_measure(a,b,c)
angular measure accepts three vectors `a`,`b`,`c` and can either accept or calculate the squared distances between them `r2_ab`,`r2_bc`, centred on vector `a`. Returns cos(θ) labelled as θ: the angular measure.
"""
function angular_measure(a,b,c,r2ab,r2ac)
    θ = (a - b)⋅(a - c)/sqrt(r2ab*r2ac) 
    return θ
end
function angular_measure(a,b,c)
    r2_ab,r2_ac,r2_bc = dist2(a,b),dist2(a,c),dist2(b,c)
    θ = angular_measure(a,b,c,r2_ab,r2_ac)   
    return θ,r2_ab,r2_ac,r2_bc
end


function all_angular_measure(a,b,c,r2ab,r2ac,r2bc)
    ab,ac,bc = (a-b),(a-c),(b-c)
    dis2ab,dis2ac,dis2bc = sqrt(r2ab),sqrt(r2ac),sqrt(r2bc)
    thetavec = [thetacalc(ab,ac,dis2ab,dis2ac),thetacalc(-ab,bc,dis2ab,dis2bc),thetacalc(-ac,-bc,dis2ac,dis2bc) ] 

    return thetavec
end
#------------------------------------------------------------------#
#----------------------Type 2 cutoff function----------------------#
#------------------------------------------------------------------#
"""
    cutoff_function(r_scaled)

    cutoff_function(r_ij,r_cut)
    cutoff_function(dist_vec::T,r_cut) where {T<:Array}

Implementation of the type 2 cutoff function. Either accepts scaled radius `r_scaled` or the interatomiic distance `r_ij` and the cutoff radius `r_cut`. Calculation is described in the RuNNer documentation, given as 1/2 (cos(πx) + 1) where x is (r_ij - r_i,c)/(rc - r_i,c). As an inner cutoff is not used by the potentials we are interested in, we have not included a method. A third method is included for creating a matrix or vector to match the distances provided. 
"""
function cutoff_function(r_scaled)
    
    cutoff= 0.5*(cos(π*r_scaled) + 1)
    
    return cutoff
end
cutoff_function(r_ij::Float64,r_cut) = ifelse(r_ij<r_cut ,cutoff_function(r_ij/r_cut),0.)

function cutoff_function(dist_vec::T,r_cut) where {T<:Array} 
    return cutoff_function.(sqrt.(dist_vec),Ref(r_cut))
end

end