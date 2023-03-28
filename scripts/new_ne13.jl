using ParallelTemperingMonteCarlo
using Random

#demonstration of the new verison of the code using an input file. 

Random.seed!(1234)

#the potential still needs to be set manually

c=[-10.5097942564988, 989.725135614556, -101383.865938807, 3918846.12841668, -56234083.4334278, 288738837.441765]
pot = ELJPotentialEven{6}(c)
#NB on this machine I opened ParallelTemperingMonteCarlo and not scripts, hence pwd()/scripts 
directory = "$(pwd())/scripts"
#we just point to the working directory and potential as the inputs. 

testinput = pot,directory
#I've named the input file, input.data is the default if you don't do this.

@time ptmc_run!(testinput;startfile="test_input.data",save_dir=directory)


# #----------------------------------------------------#
# #--------------uncomment this to restart-------------#
# #----------------------------------------------------#

# restart_input = pot,directory

# ptmc_run!(restart_input;restart=true,save=true)