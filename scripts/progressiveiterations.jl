using ParallelTemperingMonteCarlo

using Random,DelimitedFiles

script_folder = @__DIR__ # folder where this script is located
data_path = joinpath(script_folder, "data") # path to data files, so "./data/"


currentdir = pwd()
if ispath("saves")
    rm("saves",recursive=true)
end
mkdir("saves")
mkdir("saves/1m")
global numtrials = 10000

Xdata = readdlm("checkpoint/params.data")
Xdata[2,1] = numtrials
writedlm("checkpoint/params.data",Xdata)

ptmc_run!(false;save=100)

cp("checkpoint","saves/1m/checkpoint")

while numtrials < 50000
    global numtrials
    numtrials += 2500
    Xdata = readdlm("checkpoint/params.data")
    Xdata[2,1] = numtrials
    writedlm("checkpoint/params.data",Xdata)

    ptmc_run!(true;save=100)
    mkpath("saves/$(numtrials)")
    cp("checkpoint/","saves/$(numtrials)/checkpoint/")

end
