import argparse
import numpy as np
import random
from pulp import *
from pulp.apis import  PULP_CBC_CMD


class MDP:
    def __init__(self, **entries):
        self.Vs = None
        self.Ps = None
        self.S = None
        self.A = None
        self.TR = None
        self.gamma = None
        self.type = None
        #self.__dict__.update(entries)

    def build_mdp(self,all_lines):
        self.S= int(all_lines[0].strip().split()[1])
        self.A = int(all_lines[1].strip().split()[1])
        self.st = int(all_lines[2].strip().split()[1])
        self.end_sts = list(map(int,all_lines[3].strip().split()[1:]))
        self.TR = dict({})
        for s1 in range(self.S):
            self.TR[s1]=dict({})
            for a in range(self.A):
                self.TR[s1][a]= dict({})
                for s2 in range(self.S):
                    self.TR[s1][a][s2]=[0,-5]
        for i in range(4,len(all_lines)-2):
            s1,a,s2,r,p = list(map(float,all_lines[i].strip().split()[1:]))
            s1 = int(s1)
            a = int(a)
            s2 = int(s2)
            self.TR[s1][a][s2] = [p,r]
        self.gamma = float(all_lines[-1].strip().split()[1])
        self.type = all_lines[-2].strip().split()[1]
        #args ={'S':S ,'A':A, 'TR':TR, 'gamma':gamma, 'type':typem,'st':st ,'end_sts':end_sts}

    def randomInitPolicy(self):
        """
        initializes Ps as a random Policy
        """
        self.Ps = [random.randint(0,self.A-1) for _ in range(self.S)]

    def randomInitVs(self):
        """
        initializes Vs as a random Value function as a zero - one vector
        """
        self.Vs = [random.randint(0,2) for _ in range(self.S)]

    def getOptimalPolicy(self):
        """
        calculates the optimal policy given
        that Vs is already calculated
        """

        # initialize the Ps array
        self.Ps = [None]*self.S

        QValue = lambda s, a, s_ : mdp.TR[s][a][s_][0]*(mdp.TR[s][a][s_][1] + self.gamma*self.Vs[s_])

        # iterate over all the states
        for s in range(self.S):
            QValues = [sum([QValue(s,a,s_) for s_ in range(self.S)]) for a in range(self.A)]
            self.Ps[s] = np.argmax(QValues)

    def getValueFunction(self):
        """
		calculates the value function given
		that Ps is already calculated
		Note : Ps need not be optimal
		It will calculate VPi and set Vs = VPi
		# use Pulp library to get Vs
		"""

        # initialize Vs if not already done
        if self.Vs is None:
            self.Vs = [None]*self.S

        # declare the problem variable
        problem = LpProblem()

        # make an array of S decision variables
        decisionVariables = [LpVariable("V_%d"%s) for s in range(self.S)]

        # remove the end states if it is as episodic task
        if self.type == "episodic":
            for i in self.end_sts :
                decisionVariables[i] = 0.


        # add the set of S equations to the problem
        for s in range(self.S):
            a = self.Ps[s]
            problem += sum([mdp.TR[s][a][s_][0]*(mdp.TR[s][a][s_][1] + self.gamma*decisionVariables[s_]) for s_ in range(self.S)]) \
                == decisionVariables[s], "Bellman's Equation, State %d"%s

        # solve the problem
        problem.solve(PULP_CBC_CMD(msg=0))

        # set the values in self.Vs
        for s, var in enumerate(decisionVariables):
            self.Vs[s] = value(var)
        # states = range(self.S)
        # A = np.array([[-1*self.gamma*self.TR[s1][self.Ps[s1]][s2][1] for s2 in states] for s1 in states])
        # A[states,states] += 1
        # B = [sum(map(lambda x: x[0]*x[1], self.TR[s1][self.Ps[s1]].values())) for s1 in states]
        # V = np.linalg.solve(A,B)
        # print(V)
        # self.Vs= V

    def printAns(self):
        # iterate over all states
        for s in range(self.S):
            print("%06f\t%d"%(self.Vs[s], self.Ps[s]))
                    
def LinearProgramSolver(mdp):
    """
    Uses Linear Programming to find V*.
    Finds and puts the values into mdp.Vs
    Finds and puts the policy into mdp.Ps
    """
    #initialize
    problem = LpProblem("MDP", LpMinimize)

    # create a list of decision variables. (V values)
    decisionVariables = [LpVariable("V_%d"%s) for s in range(mdp.S)]

    # remove the last state if it is as episodic task
    if mdp.type == "episodic":
        for i in mdp.end_sts :
            decisionVariables[i] = 0.

    # add the objective function to the problem
    problem += sum(decisionVariables), "Sum of V values"

    # add all the constraints to the problem
    for s in range(mdp.S):
        for a in range(mdp.A):
	        # add the constraint for state s and action a
            problem += sum([mdp.TR[s][a][s_][0]*(mdp.TR[s][a][s_][1] + mdp.gamma*decisionVariables[s_]) for s_ in range(mdp.S)]) \
                    <= decisionVariables[s], "Constraint, state %d, action %d"%(s,a)

    # solve the formulated problem
    problem.solve(PULP_CBC_CMD(msg=0))

    # set the values of the MDP
    mdp.Vs = [None]*mdp.S
    for s, var in enumerate(decisionVariables):
        mdp.Vs[s] = value(var)

    # save the optimal policy from Vs into Ps
    mdp.getOptimalPolicy()

def PolicyIterationSolver(mdp):
    # initialize the policy randomly
    mdp.randomInitPolicy()
    # iterate while not converged
    while True:
        mdp.getValueFunction()
        currPi = [0]*mdp.S
        currPi[-1] = mdp.Ps[-1] 
        for s in range(mdp.S ):
            if mdp.type == "episodic":
                if s in mdp.end_sts:
                    continue
            currPi[s] = np.argmax([sum([mdp.TR[s][a][s_][0]*(mdp.TR[s][a][s_][1] + mdp.gamma*mdp.Vs[s_])\
                        for s_ in range(mdp.S)]) for a in range(mdp.A)])
        if currPi == mdp.Ps:
            break
        for s in range(mdp.S):
            mdp.Ps[s] = currPi[s]

def ValueIterationSolver(mdp):
    # randomly init Vs
    mdp.randomInitVs()

    while True:
        currVi = [0]* mdp.S        
        for s in range(mdp.S ):
            if mdp.type == "episodic":
                if s in mdp.end_sts:
                    continue
            currVi[s]= np.max([sum([mdp.TR[s][a][s_][0]*(mdp.TR[s][a][s_][1] + mdp.gamma*mdp.Vs[s_])\
                        for s_ in range(mdp.S)]) for a in range(mdp.A)])

        if currVi == mdp.Vs:
            break
        for s in range(mdp.S):
            mdp.Vs[s] = currVi[s]

    mdp.getOptimalPolicy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--mdp", required=True,
            help="Path to mdp file")
    ap.add_argument("--algorithm", required=True,type=str,
            help="Algo to use")

    args = vars(ap.parse_args())
    algorithm = args["algorithm"]
    assert(algorithm in ["vi","hpi","lp"])
    mdp = MDP()
    mdp_lines = open(args["mdp"],"r").read().splitlines()
    mdp.build_mdp(mdp_lines)
    if algorithm == "lp":
        LinearProgramSolver(mdp)
    elif algorithm == "hpi":
        PolicyIterationSolver(mdp)
    elif algorithm == "vi":
        ValueIterationSolver(mdp)
    # print the mdp ans
    mdp.printAns()



