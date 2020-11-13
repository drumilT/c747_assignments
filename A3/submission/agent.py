from WindyGridworld import *
from plotting import *
import numpy as np
import sys

class Agent:
	"""
	The class encapsulating the agent and reinforcement algorithms
	"""

	def __init__(self, env):
		"""
		env is the Gridworld Object
		"""

		# initialize the agent
		self.env = env

		# get the int -> action mapping
		self.actions = []
		for action in self.env.actions:
			self.actions.append(action)
		self.actions.sort()

		self.Q = None

	def Control(self, alpha, epsilon, num_episodes,algorithm="Sarsa"):
		"""
		alpha is the step size
		epsilon is the parameter for epsilon greedy
		num_episodes is the number of episodes for which to run algorithm
		"""

		assert( algorithm in ["Sarsa","Expected_Sarsa","Q_Learning"])

		# initialize the values to be plotted and the time to be 0
		values = [0]
		time = 0

		# initialize Q
		self.Q = numpy.zeros((self.env.rows, self.env.columns, len(self.actions)))
		for idx in range(len(self.actions)):
			self.Q[self.env.goal[0], self.env.goal[1], idx] = 0

		for episode in range(num_episodes):

			# set the current state to initial state
			self.curr_state = list(self.env.start)

			# get the action according to epsilon greedy
			action = self.epsilonGreedy(self.curr_state, epsilon)

			# repeat while terminal state isn't reached
			while self.curr_state != list(self.env.goal):

				# get the new state, reward and new action
				state_, reward = self.env.take_step(self.curr_state, self.actions[action])
				action_ = None
				if algorithm=="Sarsa":
					action_ = self.epsilonGreedy(state_, epsilon)
				else:
					action_  = np.argmax(self.Q[state_[0], state_[1], :])

				if algorithm!="Expected_Sarsa":
					target = self.Q[state_[0], state_[1], action_] 
				else:
					target = (1-epsilon)*self.Q[state_[0], state_[1], action_] 
					pi = epsilon/(len(self.actions))
					for t_action in range(len(self.actions)):
						target+= pi * self.Q[state_[0], state_[1], t_action] 
					#print(target)

				# update the Q values
				self.Q[self.curr_state[0], self.curr_state[1], action] += \
				alpha * (reward + target - self.Q[self.curr_state[0], self.curr_state[1], action])

				# update the current state and action
				self.curr_state = state_
				action = self.epsilonGreedy(state_, epsilon)

				time += 1

			values.append(time)

		return numpy.array(values)

	def epsilonGreedy(self, state, epsilon):
		"""
		Return the action to be taken at state using self.Q
		"""

		# sample a uniform random number between 0 and 1
		rnd = numpy.random.uniform(0,1)

		# return the required action
		if rnd < epsilon:
			return numpy.random.choice(len(self.actions), 1)[0]
		else:
			return np.argmax(self.Q[state[0], state[1], :])


def main():

	seeds = [0,10,20,30,40,50,60,70,80,90] # a (not-so) random list of seeds

	# parameters
	alpha = 0.5
	epsilon = 0.1
	num_episodes = 200

	task = int(sys.argv[1])

	# if a single plot is to be generated
	if task != 0:
		timePlot = numpy.zeros((num_episodes+1))

		for seed in seeds:
			# set the random seed
			numpy.random.seed(seed)

			env = getDefaultGridworld(task)
			agent = Agent(env)
			timePlot += agent.Control(alpha, epsilon, num_episodes)

		timePlot /= (len(seeds)*1.)
		makePlot(timePlot, task)

	else: # all plots on 1 graph
		values = []
		for algorithm in ["Sarsa","Expected_Sarsa","Q_Learning"]:
			task =1
			timePlot = numpy.zeros((num_episodes+1))

			for seed in seeds:
				# set the random seed
				numpy.random.seed(seed)

				env = getDefaultGridworld(task)
				agent = Agent(env)
				timePlot += agent.Control(alpha, epsilon, num_episodes,algorithm=algorithm)

			timePlot /= (len(seeds)*1.)
			values.append(timePlot)

		makePlot(values, "all")

main()