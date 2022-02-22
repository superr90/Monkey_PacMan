# Hierarchical Decision-Making Analysis for Behavior Data


## Quickstart


For Pacman data analysis, ypou should first [compute utility functions](./Behavior_Analysis/HierarchicalModel/PreEstimation.py) 
and [convert zero values to -inf](./Behavior_Analysis/HierarchicalModel/UtilityConvert.py).Then [fit strategy weights](./Behavior_Analysis/Fitting/MergeFitting.py).

We also provide codes of experiment analysis in ```Analysis/Analysis/FigAllSave.py```. 
However, running this script requires all the data. So you are free to check the code logic but they might raise 
exceptions if you run that file.  We have provided precomputed analysis results in ```Data/Behavior_Plot_Data```, 
with which the figures are plotted. Figures are plotted with  ```Behavior_Analysis/Plotting/FiPlotting``` and 
will be saved in ```Behavior_Analysis/Fig```.

## Data Description

We provide an example data of 10 trials as ```Data/TestExample/10_trial_data_Omega.pkl``` for you to test codes. 
Its utility values and fitted strategy weights are saved in ```Data/TestExample/10_trial_data_Omega-with_Q-inf-merge_weight-dynamic-res.pkl```.
 
The main features of the dataset and their descriptions are listed as below: 


|  Column Name |     Data Type    |                             Description                             |
|:------------:|:----------------:|:-------------------------------------------------------------------:|
|     file     |        str       |                       ID of each round of game                      |
|   pacmanPos  |      2-tuple     |                        position of the PacMan                       |
|   ghost1Pos  |      2-tuple     |                     position of the ghost Blinky                    |
|   ghost2Pos  |      2-tuple     |                     position of the ghost Clyde                     |
|   ifscared1  |        int       |                           status of Blinky                          |
|   ifscared2  |        int       |                           status of Clyde                           |
|     beans    | list of 2-tuples |                 positions of all the existing beans                 |
|  energizers  | list of 2-tuples |               positions of all the existing energizers              |
|   fruitPos   |      2-tuple     |                        position of the fruit                        |
|    Reward    |        int       |                          type of the fruit                          |
| contribution |       list       | normalized weights of all the agents that are fitted by our model |
