import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--scatter", default=False, action='store_true')
parser.add_argument("--size", type=int, nargs=2, default=[18, 5])

args = parser.parse_args()

with open(args.file) as fp:
	lines = [l.strip().split(", ") for l in fp]

lines = [[l[0], l[1]+' '+l[3], l[2], l[3], l[4], l[5]]  if l[1] == 'epsilon-greedy' else l for l in lines]

collate = list(zip(*lines))
inst_set = set(collate[0])
algo_set = set(collate[1])
seed_count = len(set(collate[2]))
horizon_list = sorted(float(i) for i in set(collate[4]))

data_dict = dict((i, dict((a, dict((h, 0.0) for h in horizon_list)) for a in algo_set)) for i in inst_set)

for i, a, _, _, h, r  in lines:
	h, r = float(h), float(r)
	data_dict[i][a][h] += r / seed_count

del lines
print(data_dict)
plt.subplots(1, len(inst_set), figsize=tuple(args.size))
i = 1
for instance in sorted(inst_set):
	plt.subplot(1, len(inst_set), i)
	plt.title(instance)
	for algo in data_dict[instance]:
		X, Y = tuple(zip(*list(data_dict[instance][algo].items())))
		if args.scatter:
			plt.scatter(X, Y, label=algo)
		else:
			plt.plot(X, Y, label=algo)
	plt.xscale('log')
	plt.xlabel('Horizon (log scale)')
	plt.ylabel('Cumulative Regret')
	plt.legend()
	i += 1

fname = os.path.basename(args.file)
fname = fname[:fname.find('.')] + '.jpg'
# plt.show()
plt.savefig(fname, bbox_inches='tight')