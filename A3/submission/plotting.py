from matplotlib import pyplot
import os

def makePlot(values, title=-1):
	"""
	plot timestep values with respect to episodes
	"""

	titles = ["Task 1: Deterministic Model without King's moves", "Task 2: Deterministic with King's moves",\
				 "Task 3: Stochastic Model with King's moves","Task 4: Deterministic Model with 9 moves", "Task 5: Stochastic Model without King's moves"]
	filename = None

	if title == "all":
		
		algos =  ["Sarsa","Expected_Sarsa","Q_Learning"]
		for itr in range(len(values)):
			pyplot.plot(values[itr], range(len(values[itr])), label=algos[itr])

		pyplot.xlabel("Time Steps")
		pyplot.ylabel("Episodes")
		pyplot.title("All Algorithms Combined")
		pyplot.legend(loc='lower right')
		filename = "plots/all.png"

	else:

		filename = "plots/task-%d.png"%(title)

		pyplot.plot(values, range(len(values)))
		pyplot.xlabel("Time Steps")
		pyplot.ylabel("Episodes")
		pyplot.title(titles[title-1])
		# pyplot.show()

	if "plots" not in os.listdir():
		os.mkdir("plots")
	pyplot.savefig(filename, bbox_inches='tight')