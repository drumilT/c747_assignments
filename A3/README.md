## Assignment 3

### File Structure

+ ``./WindyGridworld.py`` contains the class WindyGridworld which returns the reward and next state according to the task number.
+ ``./agent.py`` contains the class Agent which as takes input an environment Object and tries to estimate Q(s,a).
+ ``./plotting.py`` contains the plotting code.
+ ``./plots`` contain all the necessary plots.

### How to run
To run, simply use ```python3 agent.py task``` where ``task`` can be any one from 0,1,2,3,4,5.
Tasks 1 to 5 will generate plots corresponding to the respective tasks ( as described in the report) whereas 0 will generate a combined plot of all 3 control algortihms on the base task.
The plots will automatically be saved in ``./plots`` folder.
