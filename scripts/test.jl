for (index,value) in enumerate(mc_states[1].config.pos)
    println(distance2(trial_pos,value))
    println(index)
end

trial_pos= mc_states[1].config.pos[1]+SVector(1.,1.,1.)