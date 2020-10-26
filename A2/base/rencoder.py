import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--grid")
args = parser.parse_args()
data = np.loadtxt(args.grid, dtype=int)

w,h = data.shape
final_reward = w*h*10
discount = 1-1/(w*h)
state = 0
num_actions = 4
actions = range(num_actions)
start = None
end = []
transitions = []
pos_to_state = {}

for i in range(h):
	for j in range(w):
		c = data[i][j]
		if c == 2:
			start = state
		elif c == 3:
			end.append(state)
		if c != 1:
			pos_to_state[i,j] = state
			state += 1

num_states = state 

def make_transition(pos1, pos2, action):
		
	t1,t2 = data[pos1], data[pos2]
	if t1 in [0,2] and t2 in [0,2]:
		transitions.append((pos_to_state[pos1], action, pos_to_state[pos2], -1, 1))
	elif t1 in [0,2] and t2 == 3:
		transitions.append((pos_to_state[pos1], action, pos_to_state[pos2], final_reward, 1))

for i in range(h):
	for j in range(w):
		c, r, b, l, t = None, None, None, None, None
		c = i,j
		r = i,j+1
		b = i+1,j

		if r[1]<w:
			make_transition(c,r,0)
			make_transition(r,c,2)
		if b[0]<h:
			make_transition(c,b,1)
			make_transition(b,c,3)

# print(len(transitions))
print("numStates", num_states)
print("numActions", num_actions)
print("start", start)
print("end", " ".join(map(str,end)))
for t in transitions:
	print("transition", " ".join(map(str,t)))
print("mdptype episodic")
print("discount", discount)
