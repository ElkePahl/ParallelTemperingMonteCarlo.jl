
function saap(r,a0,a1,a2,a3,a4,a5)
    e_1=exp(a1*r)
    e_2=exp(a3*r)
    invr=1/r
    r2=invr^2
    r6=r2^3
    e=((a0*invr)*e_1+a2*e_2+a4)/(1+a5*r6)
    return e
end

function elj(r2,a0,a1,a2,a3,a4,a5)
    e=r2^3*(a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*a5)))))
    return e
end

function saap2(a::Vector,r)
    e_1=exp(a[2]*r)
    e_2=exp(a[4]*r)
    invr=1/r
    r2=invr^2
    r6=r2^3
    e=((a[1]*invr)*e_1 + a[3]*e_2 + a[5])/(1+a[6]*r6)
    return e
end

function elj2(pot::Vector,r2)
    r6inv = 1/(r2*r2*r2)

    r2inv=1/r2
    #r6inv=r2inv^3

    sum = 0.
    for i = 1:6
        sum += pot[i] * r6inv
        r6inv /= r2
        #r6inv *= r2inv
    end
    return sum
end

function elj3(pot::Vector,r2)
    e=r2^3*(pot[1]+r2*(pot[2]+r2*(pot[3]+r2*(pot[4]+r2*(pot[5]+r2*pot[6])))))
    return e
end