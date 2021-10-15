# Monkey Plays Pac-Man with Compositional Strategies and Hierarchical Decision-making


## Quickstart

We provide a script to reproduce figures in the following paper: 

Yang, Q., Lin, Z., Zhang, W., Li, J., Chen, X., Zhang, J., & Yang, T. (2021). Monkey Plays Pac-Man with Compositional Strategies and Hierarchical Decision-making. *bioRxiv*.

Please run the file: 

```
python Analysis/Plotting/FigPltting.py
```
All the figures will be saved into the ```Fig``` directory. 

We also provide codes of behavioral data analysis in ```Analysis/Analysis/FigAllSave.py```. 
Running this script requires full data sets for two monkeys' game-play. We will release access to these data in the future. Nontheless, we provided precomputed analysis results in ```Data/plot_data``, with all figures in the paper draft plotted.

## Data Description

We provide sample data of Monkey O and P playing the Pac-Man game at 05-Sep-2019, along with fitted strategy weights, in the file ```Data/fitted_data/05-Sep-2019-example_data.pkl```. 
The main features of this dataset and their descriptions are listed as below: 


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

## Directory Layout

```
Pacman_Behavioral_Analysis
│   README.md  
└───Analysis
│   │   Analysis
│   │   Plotting
└───Utils
│   │   ComputationUtils.py
│   │   CostVal.py
│   │   FigUtils.py
│   │   FieUtils.py
└───Data
│   │   constant
│   │   fitted_data
│   │   plot_data
└───Fig
```