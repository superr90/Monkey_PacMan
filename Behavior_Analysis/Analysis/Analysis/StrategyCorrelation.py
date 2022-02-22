'''
Description:
    Compute the correlation of individual strategies.
'''

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sbn

import sys
sys.path.append("../../../Utils")
from FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath

import matplotlib.pyplot as plt
params = {
    "legend.fontsize": 14,
    "legend.frameon": False,
    "ytick.labelsize": 14,
    "xtick.labelsize": 14,
    # "figure.dpi": 600,
    "axes.prop_cycle": plt.cycler("color", plt.cm.Accent(np.linspace(0, 1, 5))),
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    # "patch.force_edgecolor": False,
}
plt.rcParams.update(params)

# ==================================================

def randomAgent(pacmanPos, adjacent_data):
    '''
    用random agent预测方向，作为chance level。
    '''
    def _availableDir(adj_pos):
        available_dir = []
        for idx, dir in enumerate(["left", "right", "up", "down"]):
            if None != adj_pos[dir] and not isinstance(adj_pos[dir], float):
                available_dir.append(idx)
        return available_dir

    adjacent_pos = pacmanPos.apply(lambda x: adjacent_data[x])
    available_dirs = adjacent_pos.apply(lambda x: _availableDir(x))
    random_dir = [np.random.choice(each, 1).item() for each in available_dirs]
    return random_dir


def agentDirectionCorrelation(data_filename, map_filename, save_base):
    '''
    Strategy 方向之间的 Pearson correlation。
    '''
    data = pd.read_pickle(data_filename)
    if isinstance(data, list):
        data = pd.concat([each[1] for each in data]).reset_index(drop=True)
    adjacent_data = readAdjacentMap(map_filename)
    # ------------
    # agent_list = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting", "random"]
    agent_list = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer", "random"]
    agent_dir = []
    for a in agent_list:
        if a =="random":
            agent_dir.append(randomAgent(data.pacmanPos, adjacent_data))
        else:
            agent_dir.append(data["{}_Q_norm".format(a)].apply(lambda x: np.argmax(x)).values)
    agent_dir = np.asarray(agent_dir)
    # ------------------------------
    corr_mat = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            corr_mat[i, j] = pearsonr(agent_dir[i], agent_dir[j])[0]
    # ------------------------------
    plt.figure(figsize=(7, 7))
    agent_tick_names = ["global", "local", "evade(B)", "evade(C)", "approach", "energizer", "random"]
    plt.title("Pearson Correlation of Strategy Action Sequence", fontsize=20)
    sbn.heatmap(corr_mat, linewidths=0.5, square=True, cmap="binary", cbar=False,
                annot=True, annot_kws={"fontsize": 15}, fmt=".2f", # mask=mask_mat,
                xticklabels=agent_tick_names, yticklabels=agent_tick_names)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig("{}/strategy_basis_correlation.pdf".format(save_base))
    plt.show()
    # -------------------------------
    # 两个random agent之间的correlation
    random1 = randomAgent(data.pacmanPos, adjacent_data)
    random2 = randomAgent(data.pacmanPos, adjacent_data)
    print("Random agents correlation = {}".format(pearsonr(random1, random2)[0]))


def segmentSameRatio(data_filename, map_filename, save_base):
    '''
    在segment中比较strategy方向之间的重合度。
    '''
    # agent_list = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting", "random"]
    agent_list = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer", "random"]
    adjacent_data = readAdjacentMap(map_filename)
    res_data = pd.read_pickle(data_filename)
    data = [each[1] for each in res_data]
    bkpt_data = [each[-1] for each in res_data]
    # -----
    same_list = []
    for t in range(len(data)):
        if (t+1) % 10 == 0:
            print("Complete processing for {} trials.".format(t+1))
        trial_data = data[t]
        for seg in bkpt_data[t]:
            seg_data = trial_data.reset_index(drop=True).iloc[seg[0]:seg[1]]
            # -----
            agent_dir = []
            for a in agent_list:
                if a == "random":
                    agent_dir.append(randomAgent(seg_data.pacmanPos, adjacent_data))
                else:
                    agent_dir.append(seg_data["{}_Q_norm".format(a)].apply(lambda x: np.argmax(x)).values)
            agent_dir = np.asarray(agent_dir)
            # -----
            same_mat = np.zeros((7, 7))
            for i in range(7):
                for j in range(7):
                    if len(agent_dir[i]) >= 2 and len(agent_dir[j]) >= 2:
                        # 需要segment中两个strategy的方向完全一致
                        same_mat[i, j] = np.all(agent_dir[i] == agent_dir[j])
                    else:
                        same_mat[i, j] = np.nan
            same_list.append(same_mat)
    same_list = np.asarray(same_list)
    # -----
    plt.figure(figsize=(7, 7))
    plt.title("Consistency of Strategy Action Sequence", fontsize=20)
    agent_tick_names = ["global", "local", "evade(B)", "evade(C)", "approach", "energizer", "random"]
    sbn.heatmap(np.nanmean(same_list, axis=0), linewidths=0.5, square=True, cmap="binary", cbar=False,
                annot=True, annot_kws={"fontsize": 15}, fmt=".1%",
                xticklabels=agent_tick_names, yticklabels=agent_tick_names)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig("{}/strategy_basis_consistency.pdf".format(save_base))
    plt.show()


if __name__ == '__main__':
    save_base = "../../Fig"
    data_filename = "./1000_trial_data_Omega-clean_res.pkl"
    map_filename = "../../../Data/constant/adjacent_map.csv"
    agentDirectionCorrelation(data_filename, map_filename, save_base)
    segmentSameRatio(data_filename, map_filename, save_base)