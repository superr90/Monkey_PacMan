'''
Description:
    Convert zero utility values to -inf.
'''

import numpy as np
import pandas as pd
import copy


def _infConvert(cur_pos, cur_Q, adjacent_data):
    new_Q = copy.deepcopy(cur_Q)
    adjacent_pos = adjacent_data[cur_pos]
    for dir_idx, dir in enumerate(["left", "right", "up", "down"]):
        if adjacent_pos[dir] is None or isinstance(adjacent_pos[dir], float):
            new_Q[dir_idx] = -np.inf
    return new_Q


def infUtilityConvert(filename, adjacent_data):
    data = pd.read_pickle(filename)
    # convert to inf
    for a in ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]:
        data["{}_Q".format(a)] = data[["pacmanPos", "{}_Q".format(a)]].apply(
            lambda x: _infConvert(x.pacmanPos, x["{}_Q".format(a)], adjacent_data), axis=1
        )
    return data


if __name__ == '__main__':
    import sys
    sys.path.append("../../Utils")
    from FileUtils import readAdjacentMap
    adjacent_data = readAdjacentMap("../../Data/Constants/adjacent_map.csv")
    inf_Q_data = infUtilityConvert("../../Data/TestExample/10_trial_data_Omega-with_Q.pkl", adjacent_data)
    inf_Q_data.to_pickle("../../Data/TestExample/10_trial_data_Omega-with_Q-inf.pkl")