using ParallelTemperingMonteCarlo

using Random,DelimitedFiles,Plots

script_folder = @__DIR__ # folder where this script is located
data_path = joinpath(script_folder, "data") # path to data files, so "./data/"


currentdir = pwd()
if ispath("saves")
    rm("saves",recursive=true)
end
mkdir("saves")
mkdir("saves/10000")
global numtrials = 10000

Xdata = readdlm("checkpoint/params.data")
Xdata[2,1] = numtrials
writedlm("checkpoint/params.data",Xdata)

ptmc_run!(false;save=1000)

cp("checkpoint","saves/10000/checkpoint")

while numtrials < 200000
    global numtrials
    numtrials += 10000
    Xdata = readdlm("checkpoint/params.data")
    Xdata[2,1] = numtrials
    writedlm("checkpoint/params.data",Xdata)

    ptmc_run!(true;save=1000)
    mkpath("saves/$(numtrials)")
    cp("checkpoint/","saves/$(numtrials)/checkpoint/")

end
##

for i in 19:20
    cd("$currentdir/saves_b/$(i*10000)")
    X = postprocess()
    png(plot(X[1],X[2],legend=false),"hists")
    png(plot(X[3],X[5],legend=false),"cv")
end
cd(currentdir)