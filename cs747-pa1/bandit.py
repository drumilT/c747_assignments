import argparse
import numpy as np

ap = argparse.ArgumentParser()


ap.add_argument("--instance", required=True,
           help="Path to instance file")
ap.add_argument("--algorithm", required=True,type=str,
           help="Algo to use")
ap.add_argument("--randomSeed", required=False,type=int,
            help="seed")
ap.add_argument("--epsilon", required=False,type=float,
            help="in range [0,1]")
ap.add_argument("--horizon", required=False,type=int,
            help="limit")
args = vars(ap.parse_args())

fin = args["instance"]
al = args["algorithm"]
assert( al in ["epsilon-greedy","ucb","kl-ucb","thompson-sampling","thompson-sampling-with-hint"])
seed = 0
if "randomSeed" in args.keys():
    seed = args["randomSeed"]
ep = 0
if "epsilon" in args.keys():
    ep = args["epsilon"]
    assert( ep>=0 and ep <=1)
T = -1
if "horizon" in args.keys():
    T = args["horizon"]

np.random.seed(seed)

h_range=[100,400,1600,6400,25600,102400]

def ep_greed(p_vals,horizon,eps):
    vals = np.random.binomial(1,p_vals,(max(1,int(eps*horizon//len(p_vals))),len(p_vals)))
    emperical_mean = np.mean(vals,axis=0)
    max_p = p_vals[np.argmax(emperical_mean)]
    further_pulls = np.random.binomial(1,max_p,horizon-vals.size)
    #print(vals.size + further_pulls.size)
    assert( vals.size + further_pulls.size == horizon)
    #print(vals.size + further_pulls.size)
    return np.sum(vals)+np.sum(further_pulls)

def ucb(p_vals,horizon):
    assert( horizon > len(p_vals))
    round_robin = np.random.binomial(1,p_vals,(1,len(p_vals)))
    mean_num_pull = np.array([[i,1] for i in np.mean(round_robin,axis=0)])
    for i in range(len(p_vals),horizon+1):
        ucb_t = mean_num_pull[:,0] + np.sqrt( 2 * np.log(i) * np.reciprocal(mean_num_pull[:,1]))
        max_pos = np.argmax(ucb_t)
        res = np.random.binomial(1,p_vals[max_pos],1)
        mean_num_pull[max_pos] = [ (mean_num_pull[max_pos][0]* mean_num_pull[max_pos][1] + res) / (mean_num_pull[max_pos][1]+1) , mean_num_pull[max_pos][1]+1]
    return np.sum([i[0]*i[1] for i in mean_num_pull])

def KL(p, q):
    #print(p,q)
    if not ( p<1 and q<=1 and q>0):
        q=0.999
    if p == 1:
        return p*np.log(p/q)
    elif p == 0:
        return (1-p)*np.log((1-p)/(1-q))
    else:
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def solve_q(rhs, p_a):
    if p_a >= 1:
        return 1
    q = np.arange(p_a, 1, 0.001)
    #diff_array = np.array([KL(p_a,el) -rhs for el in q if el <=1])
    min_index = -1
    #for i in range(len(q)):
    #    val = KL(p_a,q[i]) -rhs
    #    min_index = i if (min_index==-1 and val>0) else -1
    #    if min_index > 0 :
    #        break
    begin = 0
    end = len(q) -1
    while True:
        start = KL(p_a,q[begin]) -rhs
        stop = KL(p_a,q[end]) -rhs
        if start > 0 or begin==int((begin+end)/2) or begin==end:
            return q[begin]
        else:
            mid = KL(p_a,q[int((begin+end)/2)]) -rhs
            if mid > 0:
                end = int((begin+end)/2)
            else:
                begin = int((begin+end)/2)

    return q[min_index]
    #start = KL(p_a,begin)
    #stop = KL(p_a,end)
    #if (stop - start) < 1e-3:
    #    return start
    #else:
    #    mid = (start+stop)/2
    #    if KL(p_a,mid) > rhs :
    #        return solve_q(rhs, p_a, start, mid)
    #    else:
    #        return solve_q(rhs, p_a, mid, stop)

def kl_ucb(p_vals,horizon):
    assert( horizon > len(p_vals))
    round_robin = np.random.binomial(1,p_vals,(2,len(p_vals)))
    mean_num_pull = np.array([[i,2] for i in np.mean(round_robin,axis=0)])
    for i in range(2*len(p_vals),horizon+1):
        rhs = (np.log(i) + 3*np.log(np.log(i)))
        kl_ucb_t = np.array([ solve_q((rhs/j[1]),j[0]) for j in mean_num_pull])
        max_pos = np.argmax(kl_ucb_t)
        res = np.random.binomial(1,p_vals[max_pos],1)
        mean_num_pull[max_pos] = [ (mean_num_pull[max_pos][0]* mean_num_pull[max_pos][1] + res) / (mean_num_pull[max_pos][1]+1) , mean_num_pull[max_pos][1]+1]
    return np.sum([i[0]*i[1] for i in mean_num_pull])

def beta_gen(num_bandits, arm_successes, arm_failures):
    beta_arms = np.zeros(num_bandits, dtype=float)
    for i in range(0,num_bandits):
        beta_arms[i] = np.random.beta(arm_successes[i]+1, arm_failures[i]+1)
    return beta_arms

def thompson(p_vals, horizon):
    round_robin = np.random.binomial(1,p_vals,(2,len(p_vals)))
    succ_num_pull = np.array([[i,2] for i in np.sum(round_robin,axis=0)])
    for i in range(2*len(p_vals),horizon+1):
        beta_arms = beta_gen(len(p_vals), succ_num_pull[:,0],succ_num_pull[:,1]- succ_num_pull[:,0])
        max_pos = np.argmax(beta_arms)
        res = np.random.binomial(1,p_vals[max_pos],1)
        succ_num_pull[max_pos] = [ succ_num_pull[max_pos][0] + res , succ_num_pull[max_pos][1]+1]
    return np.sum( succ_num_pull[:,0])
    
def modified_thompson(p_vals, horizon):
    hint = np.sort(p_vals)
    #print(p_vals)
    prior = np.ones((len(hint),len(hint)))/len(hint)
    cum_reward = 0
    for i in range(horizon):
        max_pos = np.random.choice(np.argwhere(prior[:,-1] == np.max(prior[:,-1])).flatten())
        #print(max_pos)
        res = np.random.binomial(1,p_vals[max_pos],1)
        prior[max_pos] *= hint if res ==1 else (1-hint)
        prior[max_pos] /= np.sum(prior[max_pos])
        cum_reward += res
        #print(prior)
    return cum_reward[0]

def load_instances(fin):
    f = open(fin,"r").read().splitlines()
    arr =[]
    for i in f:
        arr.append(float(i))
    return np.array(arr)

prob = load_instances(fin)
if al =="epsilon-greedy":
    reward = ep_greed(prob,T,ep)
elif al =="ucb":
    reward = ucb(prob,T)
elif al=="kl-ucb":
    reward = kl_ucb(prob,T)
elif al=="thompson-sampling":
    reward = thompson(prob,T)
elif al=="thompson-sampling-with-hint":
    reward = modified_thompson(prob,T)

reg = np.max(prob) * T - reward
print("{}, {}, {}, {}, {}, {}".format(fin, al, seed, ep, T, reg))
