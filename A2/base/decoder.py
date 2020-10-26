import argparse
import numpy as np
import random


def process_grid(gridfile,policyfile):

    grid = np.loadtxt(gridfile,dtype=int)
    N = len(grid)
    M = len(grid[0])
    st = None
    state_list = []
    end = []
    for i in range(N):
        for j in range(M):
            if grid[i][j]!=1:
                state_list.append([i,j])
                if grid[i][j]==2:
                    st = len(state_list)-1
                if grid[i][j]==3:
                    end.append(len(state_list)-1)

    inv_st_dict =dict({})
    for itr,[i,j] in enumerate(state_list):
        inv_st_dict[(i,j)] = itr
    #print(inv_st_dict)
    acts = ["N","S","W","E"]
    policy = np.loadtxt(policyfile)
    policy = policy[:,1].astype(int)
    path = []
    curr_state = st
    while curr_state not in end:
        path.append(acts[policy[curr_state]])
        curr_pos = state_list[curr_state]
        #print(curr_pos)
        if(policy[curr_state] == 0):
            next_pos = (curr_pos[0]-1,curr_pos[1])
        if(policy[curr_state] == 1):
            next_pos = (curr_pos[0]+1,curr_pos[1])
        if(policy[curr_state] == 2):
            next_pos = (curr_pos[0],curr_pos[1]-1)
        if(policy[curr_state] == 3):
            next_pos = (curr_pos[0],curr_pos[1]+1)
        curr_state = inv_st_dict[next_pos]

    print(*path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--grid", required=True,
            help="Path to grid file")
    ap.add_argument("--value_policy",required=True,
            help="Path to policy file")

    args = vars(ap.parse_args())
    process_grid(args["grid"],args["value_policy"])
            