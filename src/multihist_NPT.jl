module Multihistogram_NPT

using DelimitedFiles, LinearAlgebra, StaticArrays

using ..InputParams

export multihistgram_NPT

function temp_trajectories(temp)
    tempnumber = length(temp.t_grid)
    tempnumber_result = tempnumber * 3
    return tempnumber,tempnumber_result
end

function histogram_initialise(ensemble,temp,results)
    p=ensemble.pressure
    k=3.166811429/10^6
    temp_o=temp.t_grid
    beta=temp.beta_grid
    Emin=results.en_min
    Emax=results.en_max
    Vmin=results.v_min
    Vmax=results.v_max
    Ebins=results.n_bin
    Vbins=results.n_bin
    dEhist=(Emax-Emin)/Ebins
    dVhist=(Vmax-Vmin)/Vbins
    EVhistogram=results.ev_histogram
    return p,k,temp_o,beta,Emin,Vmin,Ebins,Vbins,dEhist,dVhist,EVhistogram
end

function Temp_grid_result(ti,tf,tempnumber_result)
    temp_grid_result = TempGrid{tempnumber_result}(ti,tf) 
    temp_result=temp_grid_result.t_grid
    beta_result=temp_grid_result.beta_grid
    return temp_result,beta_result
end

function free_energy_initialise(EVhistogram,Ebins,Vbins,tempnumber,tempnumber_result)
    free_energy=Array{Float64}(undef,tempnumber)
    new_free_energy=Array{Float64}(undef,tempnumber)
    normalconst=Array{Float64}(undef,tempnumber_result)
    ncycles=Array{Float64}(undef,tempnumber)

    for i=1:tempnumber
        free_energy[i]=0
        new_free_energy[i]=0
        ncycles[i]=0
        for m=1:Ebins
            for n=1:Vbins
                ncycles[i]=ncycles[i]+EVhistogram[i][m+1,n+1]
            end
        end
    end
    
    for i=1:tempnumber_result
        normalconst[i]=0
    end
    return free_energy, new_free_energy, normalconst, ncycles
end

function quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)
    energy_t=Emin+(m-0.5)*dEhist
    volume=Vmin+(n-0.5)*dVhist
    quasiprob=0
    denom=0
    offset=-10^6
    for i=1:tempnumber
        offset=max(offset,-beta[i]*(energy_t+p*volume)-free_energy[i])
    end
    offset=offset+log(10^3)
    for i=1:tempnumber
        quasiprob=quasiprob+EVhistogram[i][m+1,n+1]
        denom=denom+ncycles[i]*exp(-beta[i]*(energy_t+p*volume)-free_energy[i]-offset)
    end

    quasiprob=quasiprob/denom*exp(-betat*(energy_t+p*volume)-offset)
    return quasiprob
end

function multihistgram_NPT(ensemble, temp, results, conv_threshold, readfile)
    if readfile==false
        tempnumber,tempnumber_result = temp_trajectories(temp)
        p,k,temp_o,beta,Emin,Vmin,Ebins,Vbins,dEhist,dVhist,EVhistogram = histogram_initialise(ensemble,temp,results)
    end
    temp_result,beta_result = Temp_grid_result(temp_o[1],temp_o[tempnumber],tempnumber_result)

    free_energy, new_free_energy, normalconst, ncycles = free_energy_initialise(EVhistogram,Ebins,Vbins,tempnumber,tempnumber_result)
    
    for it=1:1000
        println("iteration=",it)
        for i=1:tempnumber
            local betat
            betat=beta[i]
            new_free_energy[i]=0
            for m=1:Ebins
                for n=1:Vbins
                    new_free_energy[i]=new_free_energy[i]+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)
                end
            end
            
            #println(new_free_energy[i])
            new_free_energy[i]=log(new_free_energy[i])
            #println(new_free_energy[i])
        end
        
        local delta
        delta=0
        for i=1:tempnumber
            delta=delta+abs(new_free_energy[i]-free_energy[i])^2
            free_energy[i]=new_free_energy[i]
        end
        println(delta)
        println()
    
        if delta<conv_threshold
            println("iteration finished")
            break             #if converged, exit the loop
        end
    end
    
    for i=1:tempnumber_result
        betat=beta_result[i]
        for m=1:Ebins
            for n=1:Vbins
                normalconst[i]=normalconst[i]+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)
            end
        end
    end
    
    cp=zeros(tempnumber_result)
    for i=1:tempnumber_result
        betat=beta_result[i]
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
    
                eenergy=eenergy+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)/normalconst[i]*energy_t
                eenergy2=eenergy2+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)/normalconst[i]*energy_t^2
    
                evolume=evolume+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)/normalconst[i]*volume
                evolume2=evolume2+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)/normalconst[i]*volume^2
    
                eenthalpy=eenthalpy+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)/normalconst[i]*(energy_t+p*volume)
                eenthalpy2=eenthalpy2+quasiprob(betat,m,n,ncycles,dEhist,dVhist,Emin,Vmin,tempnumber,EVhistogram,beta,p,free_energy)/normalconst[i]*(energy_t+p*volume)^2
            end
        end
        println("temperature: ",temp_result[i])
        println("energy: ",eenergy)
        println("volume: ", evolume)
        println("enthalpy: ", eenthalpy)
        println("heat capacity: ", (eenthalpy2-eenthalpy^2)/(k*temp_result[i]^2))
        println()
        cp[i]=(eenthalpy2-eenthalpy^2)/(k*temp_result[i]^2)
    end
    println("temperature array: ",temp_result)
    println("heat capacity array: ",cp)
end



end