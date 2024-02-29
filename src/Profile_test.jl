using LinearAlgebra

abstract type AbstractEnsemble end

struct VT <: AbstractEnsemble
    n_atoms::Int
end

struct PT <: AbstractEnsemble
    n_atoms::Int
    pressure::Real
end

mutable struct result{T}
    vector1::Vector{T}
    vector2::Vector{T}
end

function result(vec1,vec2)
    T = typeof(vec1[1])
    return result(vec1,vec2)
end

function give_vec!(a,b,result,::VT)
    result.vector1=a
    return result1
end

function give_vec!(a,b,result1,::PT)
    result1.vector1=a
    result1.vector2=b
    return result1
end

function vec_square(result1)
    return result1.vector1⋅result1.vector1+result1.vector2⋅result1.vector2
end


#function vec_square(vec::Vector{Vector{Float64}})
    #return vec[1]⋅vec[1]+vec[2]⋅vec[2]
#end

function vec_sum(a,b,result1,ensemble)
    sum=0.
    for i=1:1000000
        #vec=give_vec(a,b,ensemble)
        sum+=vec_square(give_vec!(a,b,result1,ensemble))
    end
    return sum
end

ensemble = VT(1)
#ensemble = PT(1,1.)
a=[1.,0.,0.]
b=[0.,1.,0.]
result1=result(zeros(3),zeros(3))


println(vec_sum(a,b,result1,ensemble))





