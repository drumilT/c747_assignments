for algo in "epsilon-greedy"
do
	#echo $algo
	for i in instances/*
	do
		#echo $i
		for h in 100 400 1600 6400 25600 102400
		do
			#echo $h
			for s in {0..49}
			do
				python bandit.py --instance $i --algorithm $algo --randomSeed $s --epsilon 0.02 --horizon $h
			done
		done
	done
done
