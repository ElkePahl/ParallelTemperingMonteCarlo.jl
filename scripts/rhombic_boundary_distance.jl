using LinearAlgebra
using Plots
using BenchmarkTools

distance_2(a,b) = (a-b)⋅(a-b)


# directly calculate the coordinates after minimum image convention and transformation
function distance_rhombic(a,b,l,h)
    b_after_y=b[2]+(3^0.5/2*l)*round((a[2]-b[2])/(3^0.5/2*l))
    b_after_x=b[1]-b[2]/3^0.5 + l*round(((a[1]-b[1])-1/3^0.5*(a[2]-b[2]))/l) + 1/3^0.5*b_after_y
    b_after_z=b[3]+h*round((a[3]-b[3])/h)
    return distance_2(a,[b_after_x,b_after_y,b_after_z])
end

l=10
h=10
a=[0,0,0]
b=[5,4*3^0.5,0]
c=[4,1*3^0.5,0]

#println(distance_rhombic(a,b,l,h))
#println()
#println(distance_rhombic(a,c,l,h))
#println()


# transformation and rectangular mic functions
function rh_to_re(a)
    return [a[1]-1/3^0.5*a[2],a[2],a[3]]
end

function re_to_rh(a)
    return [a[1]+1/3^0.5*a[2],a[2],a[3]]
end

function mic_rec(a,b,l,h)
    return [b[1]+l*round(a[1]-b[1]/l),b[2]+(3^0.5/2*l)*round((a[2]-b[2])/(3^0.5/2*l)),b[3]+h*round((a[3]-b[3])/h)]
end

function distance_rhombic_2(a,b,l,h)
    #b_new=re_to_rh(mic_rec(a,rh_to_re(b),l,h))
    return distance_2(a,re_to_rh(mic_rec(a,rh_to_re(b),l,h)))
end

#println(distance_rhombic_2(a,b,l,h))
#println()
#println(distance_rhombic_2(a,c,l,h))

function distance_rec(a,b,l)
    b_after_y=b[2]+l*round((a[2]-b[2])/l)
    b_after_x=b[1]+l*round((a[1]-b[1])/l)
    b_after_z=b[3]+l*round((a[3]-b[3])/l)
    return distance2(a,[b_after_x,b_after_y,b_after_z])
    #return distance2(a,b+[round((a[1]-b[1])/l)*l, round((a[2]-b[2])/l)*l, round((a[3]-b[3])/l)*l])

end

#@benchmark distance_rhombic_2(a,b,l,h) setup=(a=[rand(10),rand(10),0],b=[rand(10),rand(10),0],l=10,h=10)

x=Array{Float32}(undef, 151)
y=Array{Float32}(undef, 151)
for i=0:150
    x[i+1]=i/10
    y[i+1]=i/10
end
d=Matrix{Float64}(undef,151,151)
for i=1:151
    for j=1:151
        d[j,i]=distance_rhombic([0,0,0],[x[i],y[j],0],l,h)^0.5
    end
end

#println(d)

#gr()
heatmap(d,xlabel="x values", ylabel="y values",title="Distance",aspect_ratio=:equal)

#heatmap(d,aspect_ratio=:equal)
#@btime distance_rec(a,b,l)
    