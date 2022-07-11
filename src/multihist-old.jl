module Multihistogram-OLD

export multihistogram,initialise,systemsolver,bvector,amatrix,Entropycalc,analysis


using Statistics
using BenchmarkTools
using Plots
using DelimitedFiles
using LinearAlgebra


function initialise(xdir::String)
    # Read in hist.data to obtain constants
    f = open("$(xdir)histE.data", "r+")
    datafile=readdlm(f)
    kB = datafile[1]
    NTraj = datafile[2]
    T = copy(datafile[3,:])
    emin = datafile[4,1]
    emax = datafile[4,2]
    NBins = datafile[4,3]
    beta = 1 ./(kB.*T)
    de = (emax-emin)/(NBins-1)
    #Below we initialise the histogram array
    HistArray = Array{Float64}(undef,NTraj,NBins)
    
    for i in 1:NTraj
        c = open("$(xdir)histE.$i", "r")
        hist = readdlm(c)
        HistArray[i,:] = hist[1:NBins,2] 
    end

    println("Files Read")
    energyvector = [(j-1)*de + emin for j=1:NBins] #and the energies

    HistArray,energyvector,beta,nsum,NBins = processhist!(HistArray,energyvector,beta,NBins)

    return HistArray,energyvector,beta,nsum,NTraj,NBins,kB
end


function processhist!(HistArray,energyvector,beta,NBins)
    for i in 1:NTraj
        #NB in Florent's original code this factor of NBins*i normalised everything
        HistArray[i,:] = HistArray[i,:]./(NBins)#*i)
    end

    nsum = zeros(NBins)

    for j = 1:NBins
        nsum[j] = sum(HistArray[:,j])
    end
    
    #as it causes colossal headaches, we will now delete all rows
    #which have exactly 0 histogram counts. Trust me it's needed.
    k=1
    while k <= NBins

        if nsum[k] == 0
            deleteat!(nsum,k)
            deleteat!(energyvector,k)
            HistArray= HistArray[1:end,1:end .!=k]
            NBins -= 1
        else
            k=k+1
        end
    end
    return HistArray,energyvector,beta,nsum,NBins
end



function nancheck(X :: Vector)
    N = length(X)
    check = 1
    for i=1:N
        if isnan(X[i]) == true
            check = 0
        end
    end
    return check
end
function nancheck(X::Matrix)
    check = 1
    N1 = size(X)[1]
    N2 = size(X)[2]
    for i = 1:N1
        for j = 1:N2
            if isnan(X[i,j]) == true
                check = 0
            end
        end
    end
    return check
end


function bvector(HistArray::Matrix,energyvector::Vector,beta::Vector,nsum::Vector,NTraj,NBins)
    #Below we find the matrix of values n_{ij}*(ln(n_{ij} + beta_iE_j)
    #which appears frequently
    logmat = Array{Float64}(undef,NTraj,NBins)
    for i in 1:NTraj
        for j in 1:NBins
            logmat[i,j] = log(HistArray[i,j]) + beta[i]*energyvector[j]
        end
    end
    bmat = Array{Float64}(undef, NTraj,NBins)
    B = zeros(NTraj)
    rhvec = zeros(NBins)
    #The loop below guarantees we do not get values of NaN, as -inf*0.0 != -inf*false
    for i = 1:NTraj
        for j = 1:NBins
            if HistArray[i,j] == 0.0
                bmat[i,j] = 0.0
            else
                bmat[i,j] = HistArray[i,j]*logmat[i,j]
            end
        end
    end
    #the penultimate step is the vectors of length j to be summed in the rhterm
    for j=1:NBins
        rhvec[j] = sum(bmat[:,j])
    end
#Now we have our two matrices, the ith element is a sum over the energies for bmat-nij*rhvec
    for i in 1:NTraj 
        B[i] = sum( bmat[i,:] .- HistArray[i,:] .*rhvec./nsum )
    end
    println("B Vector Calculated")
    c = nancheck(B)
    if c == 0
        println("Problem with B")
    end

    return B,bmat
end
# We can now calculate the much simpler A Matrix, this has two terms
#One for all indices and one for diagonals
function amatrix(HistArray :: Matrix,nsum,NTraj,NBins)
    A = Array{Float64}(undef,NTraj,NTraj)

    for i = 1:NTraj
        for ip = 1:NTraj
            
            A[i,ip] = -sum(HistArray[i,:].*HistArray[ip,:]./nsum[:])
            
            if i == ip
                A[i,ip] += sum(HistArray[i,:])
            end
        end
    end
    println("A Matrix Calculated")
    c = nancheck(A) #this function identifies any NaNs now before they cause problems
    if c == 0
        println("Problem with A")
    end

    return A
end

#This function translates a solved vector alpha into Entropy
function Entropycalc(alpha::Vector, bmat:: Matrix, HistArray::Matrix,nsum,NBins,kB)
    S_E = []
    for j = 1:NBins 
        var = (sum(bmat[:,j] .- HistArray[:,j].*alpha))/nsum[j]
        push!(S_E,var)
    end

    return S_E
end

function analysis(energyvector:: Vector, S_E :: Vector, beta::Vector,kB::Float64; NPoints=600)
    
    NBins = length(energyvector)
    Tvec = 1 ./ (kB*beta)
    dT = (last(Tvec) - 0.2)/NPoints
    T = [(i-1)*dT + 0.2 for i = 1:NPoints]
   #Initialise all relevant vectors
   y = Array{Float64}(undef,NPoints,NBins)
   XP = Array{Float64}(undef,NPoints,NBins)
   nexp = 0
   for x = [:Z, :U, :U2, :Cv, :dCv, :r2, :r3]
       @eval $x = Array{Float64}(undef,NPoints)
   end
   #below we begin the calculation of thermodynamic quantities
   for i = 1:NPoints
       #y is a matrix of free energy
       y[i,:] = S_E[:] .-energyvector[:]./(T[i]*kB)
       #here we set the zero of free energy
       nexp = maximum(y)
       #below we calculate the partition function
       @label start
       XP[i,:] = exp.(y[i,:].-nexp)
       Z[i] = sum(XP[i,:] )
       #this loop exists to make sure the scale of our partition function is sensible
        if Z[i] < 1.
            nexp -=2
            @goto start
        elseif Z[i] > 100.
            nexp +=2
            @goto start
        end
       U[i] = sum(XP[i,:].*energyvector[:])/Z[i]
       U2[i] = sum(XP[i,:].*energyvector[:].*energyvector[:])/Z[i]
       r2[i] = sum(XP[i,:].*(energyvector[:].-U[i] ).*(energyvector[:].-U[i] ) )/Z[i]
       r3[i] = sum(XP[i,:].*(energyvector[:].-U[i] ).*(energyvector[:].-U[i] ).*(energyvector[:].-U[i] ) )/Z[i]
       Cv[i] = (U2[i] - U[i]*U[i])/kB/(T[i]^2)
       dCv[i] = r3[i]/kB^2/T[i]^4 - 2*r2[i]/kB/T[i]^3

   end

return Z,Cv,dCv,T
end

#Here we put it aaaaaalllll together
function systemsolver(HistArray,energyvector,beta,nsum,NTraj,NBins,kB)
    #initialise the values
    #HistArray,energyvector,beta,nsum,NTraj,NBins,kB = initialise()
    #solve the b vector
    b,bmat = bvector(HistArray,energyvector,beta,nsum,NTraj,NBins)
    #solve the A matrix
    A = amatrix(HistArray,nsum,NTraj)
    #Check NaN
    c1 = nancheck(A)
    c2 = nancheck(b)
    if c1 == 1 && c2 == 1
        #If there isn't NaN we solve the system and get entropy
        alpha = A \ b
        #alpha = alpha.- alpha[NTraj]
        println("system solved!")
        S = Entropycalc(alpha, bmat, HistArray,nsum,NBins,kB)
        println("Entropy Found")
        return alpha, S
    else #if NaN is present we return an error
        println("system cannot be solved")
    end

end

function histplot(HistArray,energyvector,NTraj)
    histplt = plot(energyvector,HistArray[1,:], legend = false)
    for i = 2:NTraj
        histi = HistArray[i,:]
        histplt = plot!(histplt,energyvector,histi)
    end

    return histplt
end


function multihistogram(xdir::String)
     HistArray,energyvector,beta,nsum,NTraj,NBins,kB = initialise(xdir)
     hist=histplot(HistArray,energyvector,NTraj)
     png(hist,"$(xdir)histo")
     alpha,S = systemsolver(HistArray,energyvector,beta,nsum,NTraj,NBins,kB)
     Z,C,dC,T = analysis(energyvector,S,beta,kB,NPoints)
     println("Quantities found")
     cvplot = plot(T,C,xlabel="Temperature (K)",ylabel="Heat Capacity")
     png(cvplot,"$(xdir)Cv")
     dcvplot = plot(T,dC,xlabel="Temperature(K)",ylabel="dCv")
     png(dcvplot,"$(xdir)dC")
     println("analysis complete")
     cvfile = open("$(xdir)analysis.NVT", "w")
     writedlm(cvfile, ["T" "Cv" "dCv"])
     writedlm(cvfile, [T C dC])
     close(cvfile)
        
end


end



