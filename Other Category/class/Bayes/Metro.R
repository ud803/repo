
Metropolis = function(sd,N,success,total) {

theta = seq(0,1,by=0.001)
current_pos = 0.01 #sample(theta, 1)


traject = c(current_pos)

for (i in 2:N){
	proposedJump = rnorm(1,0,sd)
	proposedPos = current_pos + proposedJump
	pproposed = proposedPos^(success) * (1-proposedPos)^(total-success)
	pcur = current_pos^(success) * (1-current_pos)^(total-success)
	
	pmove = min(1, pproposed/pcur)
	
	if(proposedPos > 1 | proposedPos <0)
		pmove = 0
	
	if( runif(1) < pmove )
		current_pos = proposedPos
	else
		current_pos = current_pos
	traject[i] = current_pos
}

return(traject)
}












