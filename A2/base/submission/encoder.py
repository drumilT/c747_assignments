import argparse
import numpy as np
import random


def process_grid(file):
    
    grid = np.loadtxt(file,dtype=int)
    N = len(grid)
    M = len(grid[0])
    end_reward = N*M 
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
    TR = []

    for itr,[i,j] in enumerate(state_list):
        if itr == end[0]:
            continue
        if i > 0 :
            if grid[i-1][j]!=1:
                pos = state_list.index([i-1,j])
                if pos==end[0]:
                    TR.append([itr,0,pos,end_reward,1])
                else:
                    TR.append([itr,0,pos,-1,1])
        if i < N-1 :
            if grid[i+1][j]!=1:
                pos = state_list.index([i+1,j])
                if end[0]==pos:
                    TR.append([itr,1,pos,end_reward,1])
                else:
                    TR.append([itr,1,pos,-1,1])
        if j > 0 :
            if grid[i][j-1]!=1:
                pos = state_list.index([i,j-1])
                if pos==end[0]:
                    TR.append([itr,2,pos,end_reward,1])
                else:
                    TR.append([itr,2,pos,-1,1])
        if j < M-1 :
            if grid[i][j+1]!=1:
                pos = state_list.index([i,j+1])
                if pos==end[0]:
                    TR.append([itr,3,pos,end_reward,1])
                else:
                    TR.append([itr,3,pos,-1,1])

    print("numStates {}".format(len(state_list)))
    print("numActions 4")
    print("start {}".format(st))
    print("end {}".format(" ".join(list(map(str,end)))))
    for i in TR:
        print("transition {} {} {} {} {}".format(i[0],i[1],i[2],i[3],i[4]))
    print("mdptype episodic")
    print("discount {}".format(1)   ) 


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--grid", required=True,
            help="Path to grid file")
    args = vars(ap.parse_args())
    process_grid(args["grid"])
            