module multihist_NPT

p=ensemble.pressure
k=3.166811429/10^6
temp=temp.t_grid
beta=temp.beta_grid
tempnumber=length(beta)
Emin=results.en_min
Emax=results.en_max
Vmin=results.v_min
Vmax=results.v_max
Ebins=results.n_bin
Vbins=results.n_bin
dEhist=(Emax-Emin)/Ebins
dVhist=(Vmax-Vmin)/Vbins
EVhistogram=results.ev_histogram

free_energy=Array{Float64}(undef,tempnumber)
new_free_energy=Array{Float64}(undef,tempnumber)
normalconst=Array{Float64}(undef,tempnumber)
ncycles=Array{Float64}(undef,tempnumber)
conv_threshold=10^(-4)  #convergence threshold

for i=1:tempnumber   #initialisation
    free_energy[i]=0
    new_free_energy[i]=0
    #beta[i]=1/k/temp[i]
    normalconst[i]=0
    ncycles[i]=0
    for m=1:Ebins
        for n=1:Vbins
            ncycles[i]=ncycles[i]+EVhistogram[i][m+1,n+1]
        end
    end
end


function quasiprob(betat,m,n)
    energy_t=Emin+(m-0.5)*dEhist
    volume=Vmin+(n-0.5)*dVhist
    quasiprob=0
    denom=0
    offset=-10^9
    for i=1:tempnumber
        offset=max(offset,-beta[i]*(energy_t+p*volume*3.398928944382626e-14)-free_energy[i])
    end
    offset=offset+log(10^3)
    for i=1:tempnumber
        quasiprob=quasiprob+EVhistogram[i][m+1,n+1]
        denom=denom+ncycles[i]*exp(-beta[i]*(energy_t+p*volume*3.398928944382626e-14)-free_energy[i]-offset)
    end

    quasiprob=quasiprob/denom*exp(-betat*(energy_t+p*volume*3.398928944382626e-14)-offset)
end


for it=1:1000
    println("iteration=",it)
    for i=1:tempnumber
        local betat
        betat=beta[i]
        new_free_energy[i]=0
        for m=1:Ebins
            for n=1:Vbins
                new_free_energy[i]=new_free_energy[i]+quasiprob(betat,m,n)
            end
        end

        new_free_energy[i]=log(new_free_energy[i])
    end
    
    local delta
    delta=0
    for i=1:tempnumber
        delta=delta+abs(new_free_energy[i]-free_energy[i])^2
        free_energy[i]=new_free_energy[i]
    end
    println(delta)

    if delta<conv_threshold
        println("iteration finished")
        break             #if converged, exit the loop
    end
end

for i=1:tempnumber
    betat=beta[i]
    for m=1:Ebins
        for n=1:Vbins
            normalconst[i]=normalconst[i]+quasiprob(betat,m,n)
        end
    end
end

for i=1:tempnumber
    betat=beta[i]
    eenergy=0
    eenergy2=0
    evolume=0
    evolume2=0
    eenthalpy=0
    eenthalpy2=0
    for m=1:Ebins
        for n=1:Vbins
            energy_t=Emin+(m-0.5)*dEhist
            volume=Vmin+(n-0.5)*dVhist

            eenergy=eenergy+quasiprob(betat,m,n)/normalconst[i]*energy_t
            eenergy2=eenergy2+quasiprob(betat,m,n)/normalconst[i]*energy_t^2

            evolume=evolume+quasiprob(betat,m,n)/normalconst[i]*volume
            evolume2=evolume2+quasiprob(betat,m,n)/normalconst[i]*volume^2

            eenthalpy=eenthalpy+quasiprob(betat,m,n)/normalconst[i]*(energy_t+p*volume*3.398928944382626e-14)
            eenthalpy2=eenthalpy2+quasiprob(betat,m,n)/normalconst[i]*(energy_t+p*volume*3.398928944382626e-14)^2
        end
    end
    println("temperature: ",temp[i])
    println("energy: ",eenergy)
    println("volume: ", evolume)
    println("enthalpy: ", eenthalpy)
    println("heat capacity: ", (eenthalpy2-eenthalpy^2)/(k*temp[i]^2))
    println()
end
