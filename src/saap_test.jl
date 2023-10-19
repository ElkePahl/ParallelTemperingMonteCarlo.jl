
function saap(r,a0,a1,a2,a3,a4,a5)
    e_1=exp(a1*r)
    e_2=exp(a3*r)
    r2=r^(-2)
    r6=r2^3
    e=((a0/r)*e_1+a2*e_2+a4)/(1+a5*r6)
    return e
end

function elj(r,a0,a1,a2,a3,a4,a5)
    r2=r^(-2)
    e=r2^3*(a0+r2*(a1+r2*(a2+r2*(a3+r2*(a4+r2*a5)))))
    return e
end