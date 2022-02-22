'''
Description:
    Analyze fitted strategy weight from simulation data.
'''
import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import sys
sys.path.append("../../../Utils")
from FileUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath

import matplotlib.pyplot as plt
params = {
    "legend.fontsize": 14,
    "legend.frameon": False,
    "ytick.labelsize": 14,
    "xtick.labelsize": 14,
    "figure.dpi": 600,
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

label_name = {
    "local": "local",
    "pessimistic": "evade",
    "evade": "evade",
    "pessimistic_blinky": "evade(Blinky)",
    "pessimistic_clyde": "evade(Clyde)",
    "evade_blinky": "evade(Blinky)",
    "evade_clyde": "evade(Clyde)",
    "global": "global",
    "suicide": "approach",
    "approach": "approach",
    "planned_hunting": "energizer",
    "energizer": "energizer",
    "vague": "vague"
}

agents = [
        "global",
        "local",
        "evade_blinky",
        "evade_clyde",
        "approach",
        "energizer",
]

status_color_mapping = {
    "approach": "#505099",
    "energizer": "#4F84C4",
    "global": "#6EBAB3",
    "evade": "#C4515C",
    "evade_blinky": "#C4515C",
    "evade_clyde": "#FFD07D",
    "evade-blinky": "#C4515C",
    "evade-clyde": "#FFD07D",
    "local": "#F5905F",
    "vague": "#929292",
    # "Planned Attack":RdBu_7.mpl_colors[0],
    # "Accidental Attack":RdBu_7.mpl_colors[-1],
    # "Suicide":RdBu_7.mpl_colors[0],
    # "Normal Die":RdBu_7.mpl_colors[-1],
    "Others": "#929292",
}

adjacent_data = readAdjacentMap("../Data/constant/adjacent_map.csv")

# ==================================================

def _removeInvalidFittedData(corresponding_true_data, fitted_data):
    remove_index = []
    for i in range(len(corresponding_true_data)):
        # -----------------------------------
        #       THIS PART IS DEPRECATED
        # opposite_dirs = {"up": "down", "down": "up", "left": "right", "right": "left"}
        # # 第一步的方向就是反的
        # if fitted_data[i].next_pacman_dir_fill.values[0] in opposite_dirs:
        #     if opposite_dirs[fitted_data[i].next_pacman_dir_fill.values[0]] == corresponding_true_data[i].next_pacman_dir_fill.values[0]:
        #         remove_index.append(i)
        # # fitted data里面有来回走的数据
        # if (fitted_data[i].next_pacman_dir_fill.replace(opposite_dirs) == fitted_data[i].next_pacman_dir_fill.shift()).sum() > 0:
        #     remove_index.append(i)
        # # true data里面有来回走的数据
        # if (corresponding_true_data[i].next_pacman_dir_fill.replace(opposite_dirs) == corresponding_true_data[i].next_pacman_dir_fill.shift()).sum() > 0:
        #     remove_index.append(i)
        # -----------------------------------

        # true data里面有pacman重复的位置数据
        if (corresponding_true_data[i].pacmanPos == corresponding_true_data[i].pacmanPos.shift()).sum() > 0:
            remove_index.append(i)
        # true data和fitted data里面鬼的第二个位置不一样
        if (
            fitted_data[i].shape[0] > 1 and (
            fitted_data[i].ghost1Pos.values[1]
            != corresponding_true_data[i].ghost1Pos.values[1]
            or fitted_data[i].ghost2Pos.values[1]
            != corresponding_true_data[i].ghost2Pos.values[1])
        ):
            remove_index.append(i)
    return remove_index


def _estimationVagueLabeling2(contributions):
    '''
    根据拟合结果预测strategy
    '''
    if isinstance(contributions, float):
        return np.nan
    all_agent_name = ["global", "local", "evade(Blinky)", "evade(Clyde)", "approach", "energizer"]
    sorted_contributions = np.sort(contributions)[::-1]
    if sorted_contributions[0] - sorted_contributions[1] <= 0.1:
        return "vague"
    else:
        label = all_agent_name[np.argmax(contributions)]
        return label

#TODO: 数据文件路径
def simulationAllCheck():
    fitted_omega_data = pd.read_pickle("./more_filtered_simulation/segment_simulation_Q-fit_res.pkl")
    fitted_omega_data = [each[1] for each in fitted_omega_data]
    fitted_patamon_data = pd.read_pickle("./patamon_filtered_simulaiton/segment_simulation_Q-fit_res.pkl")
    fitted_patamon_data = [each[1] for each in fitted_patamon_data]
    fitted_vague_data = pd.read_pickle("./vague_filtered_simulation/segment_simulation_Q-fit_res.pkl")
    fitted_vague_data = [each[1] for each in fitted_vague_data]
    fitted_data = fitted_omega_data + fitted_patamon_data + fitted_vague_data
    print("The num of fitted data : {}".format(len(fitted_data)))
    # -----
    # Take the corresponding true data
    true_omega_data = pd.read_pickle("./8478_trial_data_Omega-with_Q-clean-res.pkl")
    true_patamon_data = pd.read_pickle("./7294_trial_data_Patamon-with_Q-clean-res.pkl")
    true_data = pd.concat([each[1] for each in true_omega_data] + [each[1] for each in true_patamon_data]).reset_index(drop=True)
    trial_id = [each.original_trial_id.values[0] for each in fitted_data]
    trial_name = [each.split("/")[0] for each in trial_id]
    trial_step = [each.split("/")[1] for each in trial_id]
    trial_length = [each.shape[0] for each in fitted_data]
    corresponding_true_data = []
    for i in range(len(trial_name)):
        start_idx = true_data[(true_data.file == trial_name[i]) & (true_data.origin_index == int(trial_step[i]))].index[0]
        end_idx = start_idx + trial_length[i]
        corresponding_true_data.append(true_data.iloc[start_idx:end_idx])
    print("Num of corresponding data : {}".format(len(corresponding_true_data)))
    # -----
    removed_idx = _removeInvalidFittedData(corresponding_true_data, fitted_data)
    print("Num of removed idx : {}".format(len(np.unique(removed_idx))))
    # -----
    agent_list = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]
    agent_Q_list = [each + "_Q_norm" for each in agent_list]
    true_pos = [each.pacmanPos for each in corresponding_true_data]
    simulated_pos = [each.pacmanPos for each in fitted_data]
    simulated_Q = [each[agent_Q_list] for each in fitted_data]
    true_Q = [each[[each + "_Q_norm" for each in ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]]] for each in corresponding_true_data]
    # -----
    diff_path = removed_idx
    # -----
    # selected_idx = reduce(np.intersect1d, (is_dominant_idx, no_same_idx))
    selected_idx = list(np.arange(len(fitted_data)))
    original_weight = [each.predefined_weight.apply(lambda x: eval(x)).values[0] for each in fitted_data]
    fitted_weight = [each.contribution.values[0] for each in fitted_data]
    # -----
    used_original_weight = [original_weight[each] for each in selected_idx if each not in diff_path]
    removed_original_weight = [original_weight[each] for each in selected_idx if each in diff_path]
    used_simulated_weight = [fitted_weight[each] for each in selected_idx if each not in diff_path]
    removed_simulated_weight = [fitted_weight[each] for each in selected_idx if each in diff_path]

    used_original_label = [_estimationVagueLabeling2(each) for each in used_original_weight]
    used_fitted_label = [_estimationVagueLabeling2(each) for each in used_simulated_weight]
    # -----
    weight_MSE = [np.linalg.norm(used_original_weight[i] - used_simulated_weight[i]) for i in
                  range(len(used_original_weight))]
    weight_MSE_arg_sort = np.argsort(weight_MSE)
    no_diff_weight_MSE_arg_sort = [each for each in weight_MSE_arg_sort if each not in diff_path] #TODO: 用于测试
    avg_MSE = np.nanmean(weight_MSE)
    std_MSE = np.nanstd(weight_MSE)
    print("Fitted weight MSE = {} (std={})".format(avg_MSE, std_MSE))
    print("Label consistency rate = {}".format(np.nanmean(np.asarray(used_original_label) == np.asarray(used_fitted_label))))
    # -----
    used_original_lables = [_estimationVagueLabeling2(each) for each in used_original_weight]
    used_original_lables = ["evade" if "evade" in each else each for each in used_original_lables]
    used_fitted_lables = [_estimationVagueLabeling2(each) for each in used_simulated_weight]
    used_fitted_lables = ["evade" if "evade" in each else each for each in used_fitted_lables]
    agent_list = ["global", "local", "evade", "approach", "energizer", "vague"]
    confusion_mat = np.zeros((6, 6))
    for i in range(len(used_original_lables)):
        row_idx = agent_list.index(used_fitted_lables[i])
        col_idx = agent_list.index(used_original_lables[i])
        confusion_mat[row_idx, col_idx] += 1
    confusion_ratio_mat = confusion_mat / np.sum(confusion_mat, axis=0).reshape((1, -1))
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    sbn.heatmap(confusion_ratio_mat, square=True, cmap="binary", cbar=False, linewidth=0.5,
                annot_kws={"fontsize": 15}, fmt=".1%", annot=True)
    plt.ylabel("Fitted Strategy Label", fontsize=20)
    plt.xlabel("Ground-truth Strategy Label", fontsize=20, labelpad=8)
    plt.xticks(np.asarray([0, 1, 2, 3, 4, 5]) + 0.5, agent_list, fontsize=15)
    plt.yticks(np.asarray([0, 1, 2, 3, 4, 5]) + 0.5, agent_list, fontsize=15)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig("./simulation_recovery_confusion_mat.pdf")
    plt.show()
    # -----
    all_true_weight = []
    all_fitted_weight = []
    plt.figure(figsize=(9, 7))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i, a in enumerate(agents):
        if i==2: #evade
            true_a = np.asarray([each[i] for each in used_original_weight] + [each[i+1] for each in used_original_weight])
            fitted_a = np.asarray([each[i] for each in used_simulated_weight] + [each[i+1] for each in used_simulated_weight])
            selected_idx = list(set(np.where(~np.isnan(true_a))[0]).intersection(set(np.where(~np.isnan(fitted_a))[0])))
            true_a = true_a[selected_idx]
            fitted_a = fitted_a[selected_idx]
            a_r2 = r2_score(true_a, fitted_a)
        elif i==3:
            continue
        else:
            true_a = np.asarray([each[i] for each in used_original_weight])
            fitted_a = np.asarray([each[i] for each in used_simulated_weight])
            selected_idx = list(set(np.where(~np.isnan(true_a))[0]).intersection(set(np.where(~np.isnan(fitted_a))[0])))
            true_a = true_a[selected_idx]
            fitted_a = fitted_a[selected_idx]
            a_r2 = r2_score(true_a, fitted_a)
        # -----
        all_true_weight.extend(true_a)
        all_fitted_weight.extend(fitted_a)
        # -----
        if i==2:
            plt.scatter(true_a, fitted_a, color=status_color_mapping["evade"],
                        label="{} ($R^2$={:.2f})".format(label_name["evade"], a_r2), s=80, alpha=0.8)
            # plt.scatter([each[i] for each in removed_original_weight] + [each[i+1] for each in removed_original_weight],
            #             [each[i] for each in removed_simulated_weight] + [each[i+1] for each in removed_simulated_weight],
            #             color="#a6a6a6", s=80, alpha=0.8)
        else:
            plt.scatter(true_a, fitted_a, color=status_color_mapping[a],
                        label="{} ($R^2$={:.2f})".format(label_name[a], a_r2), s=80, alpha=0.8)
            # plt.scatter([each[i] for each in removed_original_weight],
            #             [each[i] for each in removed_simulated_weight],
            #             color="#a6a6a6", s=80, alpha=0.8)
    # all_r2 = r2_score(all_true_weight, all_fitted_weight)
    model = LinearRegression()
    model.fit(
        np.asarray(all_true_weight).reshape((-1, 1)),
        np.asarray(all_fitted_weight).reshape((-1, 1))
    )
    model_coef = model.coef_.item()
    model_intercept = model.intercept_.item()
    all_r2 = model.score(np.asarray(all_true_weight).reshape((-1, 1)), np.asarray(all_fitted_weight).reshape((-1, 1)))
    # all_r2 = r2_score(all_true_weight, all_fitted_weight)
    plt.plot([0.0, 1.0], [0.0 * model_coef + model_intercept, 1.0 * model_coef + model_intercept], "-", color="k",
             lw=4.5, alpha=0.5)
    plt.text(1.08, 0.9, "$y$={:.2f}$x$+{:.2f}".format(model_coef, model_intercept), fontsize=15)
    plt.title("Strategy Weight Recovery ($R^2$={:.2f})".format(all_r2), fontsize=20)
    plt.xlabel("Ground-truth strategy weight", fontsize=20)
    plt.ylabel("Fitted strategy weight", fontsize=20)
    plt.legend(frameon=False, bbox_to_anchor=(1.0, 0.8), fontsize=15)
    plt.tight_layout()
    plt.savefig("./simulation_recovery_weights.pdf")
    plt.show()


if __name__ == '__main__':
    simulationAllCheck()


