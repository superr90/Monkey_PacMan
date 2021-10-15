'''
Description:
    Figure plotting.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import pickle
import copy
import scipy.stats
from matplotlib import lines


# Configuration
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
pd.set_option("display.float_format", "{:.5f}".format)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

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
    "Others":"#929292",
}

agent_name = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]
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

black = "#000000"
black_RGB = "#000000"
dark_grey = "#4d4d4d"
light_grey = "#a6a6a6"

print("Finished configuration.")
print("="*50)

file_base = "../../Data/plot_data/"
pic_base = "../../Fig"


# ===================================================
#               FIG 1: BASICS
# ===================================================

def plotFig1B(monkey):
    print("----------- Fig 1B -----------")
    with open("{}/{}_1.B.pkl".format(file_base, monkey), "rb") as file:
        result_df = pickle.load(file)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    color_list = ["#000000", "#5b5a5a", "#898989", "#bebebd"]
    sns.set_palette(color_list)
    for i, c in enumerate(["straight", "L-shape", "fork", "cross"]):
        gpd = result_df[result_df.category == c]
        ax.plot(
            gpd["local_4dirs_diff"],
            gpd["mean"],
            marker=None,
            lw=4,
            color=color_list[i]
        )
        plt.fill_between(
            gpd["local_4dirs_diff"],
            gpd["mean"] - gpd["std"] / np.sqrt(gpd["count"]),
            gpd["mean"] + gpd["std"] / np.sqrt(gpd["count"]),
            color=color_list[i],
            alpha=0.4,
            linewidth=0.0
        )
    plt.yticks(fontsize = 20)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([0, 1, 2, 3, ">=4"], fontsize = 20)
    ax.set_xlabel("R", fontsize = 20)
    ax.set_ylabel(" Probability of Moving \n towards Largest Reward", fontsize = 20)
    ax.set_ylim(0.0, 1.0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/1.B.pdf")
    plt.show()


def plotFig1C(monkey):
    print("----------- Fig 1C -----------")
    with open("{}/{}_1.C.pkl".format(file_base, monkey), "rb") as file:
        rs = pickle.load(file)
    dd = {
        "ghost(red) normal": {"color": "#C4515C", "linestyle": "-"},
        "ghost(red) scared": {"color": "#C4515C", "linestyle": "-.", },
        "ghost(yellow) normal": {"color": "#FFD07D", "linestyle": "-"},
        "ghost(yellow) scared": {"color": "#FFD07D", "linestyle": "-.", },
    }
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for idx, gpd in rs.groupby(["ifscared1"]):
        plt.errorbar(
            gpd["level_0"],
            gpd["mean"],
            marker=None,
            capsize=4,
            lw = 3,
            barsabove=True,
            **dd[idx],
            label = idx,
        )
        plt.fill_between(
            gpd["level_0"],
            gpd["mean"] - gpd["std"] / np.sqrt(gpd["count"]),
            gpd["mean"] + gpd["std"] / np.sqrt(gpd["count"]),
            color=dd[idx]["color"],
            alpha=0.2,
            linewidth=0.0,
        )
    plt.xlabel("Distance between Pacman and Ghosts", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.ylabel("Probability of Moving towards Ghosts", fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0, 1)
    plt.legend(frameon=False, fontsize=20)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/1.C.new.pdf")
    plt.show()
    plt.close()


# ===================================================
#               FIG 2: STRATEGY FITTING
# ===================================================

def plotFig2B(monkey):
    print("----------- Fig 2B -----------")
    # read data
    width = 0.2
    random_data = np.load("../../Data/plot_data/100trial-all_random_is_correct.npy", allow_pickle=True).item()
    all_mean = np.array([np.mean(random_data["early"]), np.mean(random_data["middle"]), np.mean(random_data["end"])])
    all_size = np.array([len(random_data["early"]), len(random_data["middle"]), len(random_data["end"])])
    avg_random_cr = {each: np.nanmean(random_data[each]) for each in random_data}
    avg_random_cr["all"] = np.sum(all_mean * all_size) / np.sum(all_size)
    avg_hybrid_cr, std_hybrid_cr, avg_moving_cr, std_moving_cr, avg_preceptron_cr, std_perceptron_cr, type_name = \
        np.load("../../Data/plot_data/{}_2B.npy".format(monkey), allow_pickle=True)

    type_name = {"all": "Overall", "early": "Early \nGame", "end": "Late \nGame", "close-scared": " Scared \nGhosts"}
    black = "#545454"
    dark_grey = "#898989"
    light_grey = "#FFFFFF"

    plt.figure(figsize=(8, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.bar(x=np.arange(0, 4), height=[avg_moving_cr[each] for each in type_name], width=width,
            label="Dynamic Strategy",
            color=black,
            linewidth=2,
            edgecolor=black,
            yerr=[std_moving_cr[each] for each in type_name], capsize=7,
            error_kw={"capthick": 3, "elinewidth": 3})
    plt.bar(x=np.arange(0, 4) - width, height=[avg_hybrid_cr[each] for each in type_name], width=width,
            label="Static Strategy",
            color=dark_grey,
            linewidth=2,
            edgecolor=black,
            yerr=[std_hybrid_cr[each] for each in type_name], capsize=7,
            error_kw={"capthick": 3, "elinewidth": 3})
    plt.bar(x=np.arange(0, 4) - 2*width, height=[avg_preceptron_cr[each] for each in type_name], width=width,
            label="Perceptron",
            color=light_grey,
            linewidth=2,
            edgecolor = black,
            yerr=[std_perceptron_cr[each] for each in type_name], capsize=7,
            error_kw={"capthick": 3, "elinewidth": 3})
    x_index = [[i - 5 * width / 2, i + width / 2] for i in range(5)]
    for index, i in enumerate(type_name.keys()):
        plt.plot(x_index[index], [avg_random_cr[i], avg_random_cr[i]], "--", lw=4, color="k")
    plt.xticks(np.arange(0, 4) - width, [type_name[each] for each in type_name], fontsize=20)
    plt.ylim(0.4, 1.15)
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
    plt.ylabel("Prediction Accuracy", fontsize=20)
    plt.legend(frameon=False, fontsize=16, ncol=1)
    plt.tight_layout()
    plt.savefig(pic_base + "/2B.{}.pdf".format(monkey))
    plt.show()


def plotFig2CDE(monkey):
    print("---------- Fig 2CDE ----------")
    # Extract data
    with open("{}/{}_2DE.detail_dist.pkl".format(file_base, monkey), "rb") as file:
        detail_data = pickle.load(file)
    weight_name = ["1st", "2nd", "3rd"]
    avg_list = []
    std_list = []
    for i in range(3):
        avg_val = np.nanmean(detail_data[i])
        sem_val = np.nanstd(detail_data[i])
        avg_list.append(avg_val)
        std_list.append(sem_val)
        print("| {} Largest Weight | AVG = {}, STD = {}".format(weight_name[i], avg_val, sem_val))
    first_weight = detail_data[0]
    second_weight = detail_data[1]
    third_weight = detail_data[2]
    with open("{}/{}_2DE.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    first_second_diff = data["series"]
    # Histogram; Probability
    bin = np.arange(0, 1.1, 0.1)
    first_weight_bin, bin_edges = np.histogram(first_weight, bin)
    first_weight_bin = np.random.uniform(np.repeat(bin_edges[:-1], first_weight_bin), np.repeat(bin_edges[1:], first_weight_bin))
    second_weight_bin, bin_edges = np.histogram(second_weight, bin)
    second_weight_bin = np.random.uniform(np.repeat(bin_edges[:-1], second_weight_bin), np.repeat(bin_edges[1:], second_weight_bin))
    third_weight_bin, bin_edges = np.histogram(third_weight, bin)
    third_weight_bin = np.random.uniform(np.repeat(bin_edges[:-1], third_weight_bin), np.repeat(bin_edges[1:], third_weight_bin))
    diff_weight_bin, bin_edges = np.histogram(first_second_diff, bin)
    diff_weight_bin = np.random.uniform(np.repeat(bin_edges[:-1], diff_weight_bin),np.repeat(bin_edges[1:], diff_weight_bin))
    # Configurations
    if monkey == "Omega" or monkey == "Patamon":
        fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    spec = fig.add_gridspec(1, 4)

    ax = fig.add_subplot(spec[0, :3])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    sns.violinplot(
        data=[first_weight_bin, second_weight_bin, third_weight_bin],
        palette=[dark_grey, dark_grey, dark_grey]
    )
    plt.xticks([0.0, 1, 2], ["1st", "2nd", "3rd"], fontsize=20)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Strategy Weight", fontsize=20)
    ax = fig.add_subplot(spec[0, 3])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    sns.violinplot(
        data=[diff_weight_bin],
        palette=[dark_grey]
    )
    plt.xticks([0.0], ["1st - 2nd"], fontsize=20)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Strategy Difference", fontsize=20)
    plt.savefig("{}/".format(pic_base) + monkey + "/2.DE.new.pdf")
    plt.show()


def plotFig2F(monkey):
    print("----------- Fig 2F -----------")
    with open("{}/{}_2F.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    type = ["all", "early", "end","close-scared"]
    type_name = {
        "middle":"Middle Stage",
        "close-normal":"Closed Normal Ghost",
        "all": "Overall", "early": "Early \nGame",
        "end": "Late \nGame", "close-scared": " Scared \nGhosts"
    }
    agent_name = ["vague", "energizer", "evade", "approach", "global", "local"][::-1]
    x_index = np.arange(len(type))
    plt.figure(figsize=(8, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    bottom = np.zeros((4,))
    for i in x_index:
        for j, a in enumerate(agent_name):
            if a != "evade":
                d = data[type[i]][agent_name[j]]
            else:
                d = data[type[i]]["evade_blinky"] + data[type[i]]["evade_clyde"]
            if i == 0:
                plt.bar(i, d,
                        color=status_color_mapping[agent_name[j]], width=0.45, label=label_name[agent_name[j]],
                        bottom=bottom[i])
            else:
                plt.bar(i, d,
                        color=status_color_mapping[agent_name[j]], width=0.45,
                        bottom=bottom[i])
            bottom[i] += d
    plt.xticks(np.arange(len(type)), [type_name[i] for i in type], fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel("Probability", fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="upper center", fontsize=15, ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/2.F.pdf")
    plt.show()


# ===================================================
#               FIG 3: STRATEGY PATTERNS
# ===================================================

def plotFig3AB(monkey):
    def _combine(data, bin_size, max_val):
        data = np.array(data, dtype=object)
        res = []
        for i in range(0, max_val, bin_size):
            res.append(np.concatenate(data[i:i+bin_size]))
        res.append(np.concatenate(data[max_val:]))
        return np.array(res, dtype=object)

    print("----------- Fig 3B -----------")
    # Read Data
    data = np.load(file_base + "{}-strategy_ratio-resample.npy".format(monkey), allow_pickle=True).item()
    game_PG_data = data["scared-PG"]
    game_beans_data = data["beans_in10"]
    # Plotting
    agents = ["global", "local", "evade(Blinky)", "evade(Clyde)", "approach", "energizer", "vague"]
    repeat_num = 10
    rand_num = 100
    # Pacman-Ghost Distance
    game_PG_data = _combine(game_PG_data, bin_size=2, max_val=20)
    all_PG = [[] for t in range(game_PG_data.shape[0])]
    for t in range(game_PG_data.shape[0]):
        for r in range(repeat_num):
            tmp = [0 for _ in range(7)]
            ind = np.random.choice(game_PG_data[t], rand_num, replace=True)
            for i in ind:
                tmp[int(i)] += 1
            tmp = np.array(tmp) / np.sum(tmp)
            all_PG[t].append(copy.deepcopy(tmp))
    all_PG = np.array(all_PG)

    plt.figure(figsize=(8, 4))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    avg_PG = np.array([
        np.nanmean(all_PG[:, :, 1], axis = 1),
        np.nanmean(all_PG[:, :, 4], axis = 1),
        np.nanmean(all_PG[:, :, [0,2,3,5,6]].reshape(all_PG.shape[0], -1), axis = 1)
    ]).T
    se_PG = np.array([
        np.nanstd(all_PG[:, :, 1], axis=1),
        np.nanstd(all_PG[:, :, 4], axis=1),
        np.nanstd(all_PG[:, :, [0, 2, 3, 5, 6]].reshape(all_PG.shape[0], -1), axis=1)
    ]).T
    plt.plot(avg_PG[:, 0], label="local", color=status_color_mapping["local"], lw=8)
    plt.plot(avg_PG[:, 1], label="approach", color=status_color_mapping["approach"], lw=8)
    plt.plot(avg_PG[:, 2], label="other", color="k", lw=8, alpha=0.6)
    plt.fill_between(
        np.arange(avg_PG.shape[0]),
        avg_PG[:, 0] - se_PG[:, 0],
        avg_PG[:, 0] + se_PG[:, 0],
        color=status_color_mapping["local"],
        alpha=0.1,
        linewidth=0.0
    )
    plt.fill_between(
        np.arange(avg_PG.shape[0]),
        avg_PG[:, 1] - se_PG[:, 1],
        avg_PG[:, 1] + se_PG[:, 1],
        color=status_color_mapping["approach"],
        alpha=0.1,
        linewidth=0.0
    )
    plt.fill_between(
        np.arange(avg_PG.shape[0]),
        avg_PG[:, 2] - se_PG[:, 2],
        avg_PG[:, 2] + se_PG[:, 2],
        color = "k",
        alpha=0.1,
        linewidth=0.0
    )
    plt.xticks(np.arange(11), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, ">=20"], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Distance between Pacman and Scared Ghost", fontsize=20)
    plt.ylabel("Strategy Probability", fontsize=20)
    plt.xlim(0, 10)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=20, ncol = 3)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/3B.pdf")
    plt.show()
    plt.close()
    # ====================================
    print("----------- Fig 3A -----------")
    # Beans within 10 steps
    game_beans_data = _combine(game_beans_data, bin_size=1, max_val=5)
    all_beans = [[] for _ in range(game_beans_data.shape[0])]
    for t in range(game_beans_data.shape[0]):
        for r in range(repeat_num):
            tmp = [0 for _ in range(7)]
            ind = np.random.choice(game_beans_data[t], rand_num, replace=True)
            for i in ind:
                tmp[i] += 1
            tmp = np.array(tmp) / np.sum(tmp)
            all_beans[t].append(copy.deepcopy(tmp))
    all_beans = np.array(all_beans)

    plt.figure(figsize=(8, 4))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    avg_beans = np.array([
        np.nanmean(all_beans[:, :, 0], axis=1),
        np.nanmean(all_beans[:, :, 1], axis=1),
        np.nanmean(all_beans[:, :, [2, 3, 4, 5, 6]].reshape(all_beans.shape[0], -1), axis=1)
    ]).T
    se_beans = np.array([
        np.nanstd(all_beans[:, :, 0], axis=1),
        np.nanstd(all_beans[:, :, 1], axis=1),
        np.nanstd(all_beans[:, :, [2, 3, 4, 5, 6]].reshape(all_beans.shape[0], -1), axis=1)
    ]).T
    plt.plot(avg_beans[:, 0], label="global", color=status_color_mapping["global"], lw=8)
    plt.plot(avg_beans[:, 1], label="local", color=status_color_mapping["local"], lw=8)
    plt.plot(avg_beans[:, 2], label="other", color="k", lw=8, alpha=0.6)
    plt.fill_between(
        np.arange(avg_beans.shape[0]),
        avg_beans[:, 0] - se_beans[:, 0],
        avg_beans[:, 0] + se_beans[:, 0],
        color=status_color_mapping["global"],
        alpha=0.1,
        linewidth=0.0
    )
    plt.fill_between(
        np.arange(avg_beans.shape[0]),
        avg_beans[:, 1] - se_beans[:, 1],
        avg_beans[:, 1] + se_beans[:, 1],
        color=status_color_mapping["local"],
        alpha=0.1,
        linewidth=0.0
    )
    plt.fill_between(
        np.arange(avg_beans.shape[0]),
        avg_beans[:, 2] - se_beans[:, 2],
        avg_beans[:, 2] + se_beans[:, 2],
        color="k",
        alpha=0.1,
        linewidth=0.0
    )
    plt.xticks(np.arange(6), [0, 1, 2, 3, 4, ">=5"], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Number of Pellets within 10 Steps", fontsize=20)
    plt.ylabel("Strategy Probability", fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.xlim(0, 5)
    plt.legend(fontsize=20, ncol = 3, loc = "upper center")
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/3A.pdf")
    plt.show()
    plt.close()


def plotFig3CD(monkey):
    print("----------- Fig 3C -----------")
    with open("{}/{}_3C.detail.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    actual_length = data["actual"]
    optimal_length = data["dis"]
    actual_length = np.array(actual_length, dtype=np.int)
    optimal_length = np.array(optimal_length, dtype=np.int)
    use_index = np.intersect1d(np.where(optimal_length<=30)[0], np.where(actual_length<=30)[0])
    actual_length = actual_length[use_index]
    optimal_length = optimal_length[use_index]
    cnt_bin, x_bin, y_bin = np.histogram2d(
        optimal_length, actual_length,
        [list(range(0, 30, 6)) + [30],
         list(range(0, 30, 6)) + [30]]
    )
    cnt_size = cnt_bin / np.sum(cnt_bin)
    cnt_bin[cnt_bin==0] = np.nan

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    sns.heatmap(cnt_bin, square=False, cmap="binary", cbar=False, linewidth=0.5, annot=cnt_size, annot_kws={"fontsize":15}, fmt=".1%")
    plt.plot([0, 30], [0, 30], "--", color="#1abc9b", lw=4, alpha=0.8)
    plt.ylabel("Shortest Trajectory Length", fontsize=20)
    plt.xlabel("Monkey's Trajectory Length", fontsize=20, labelpad = 8)
    plt.xticks([0, 1, 2, 3, 4, 5], [0, 6, 12, 18, 24, 30], fontsize=15)
    plt.yticks([0, 1, 2, 3, 4, 5], [0, 6, 12, 18, 24, 30], fontsize=15)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/3.C.new.pdf")
    plt.show()
    # -------------------------------------
    print("----------- Fig 3D -----------")
    with open("{}/{}_3D.detail.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    actual_turn = data["actual"]
    optimal_turn = data["fewest"]
    actual_turn = np.array(actual_turn, dtype=np.int)
    optimal_turn = np.array(optimal_turn, dtype=np.int)
    optimal_turn = np.min([actual_turn, optimal_turn], axis=0)
    use_index = np.intersect1d(np.where(optimal_turn <= 5)[0], np.where(actual_turn <= 5)[0])
    actual_turn = actual_turn[use_index]
    optimal_turn = optimal_turn[use_index]
    cnt_bin, x_bin, y_bin = np.histogram2d(
        optimal_turn, actual_turn,
        [list(range(0, 5, 1)) + [5],
         list(range(0, 5, 1)) + [5]]
    )
    cnt_size = cnt_bin / np.sum(cnt_bin)
    cnt_bin[cnt_bin == 0] = np.nan

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    sns.heatmap(cnt_bin, square=False, cmap="binary", cbar=False, linewidth=0.5, annot=cnt_size, annot_kws={"fontsize":15}, fmt=".1%")
    plt.plot([0, 5], [0, 5], "--", color="#1abc9b", lw=3, alpha=0.8)
    plt.xticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], fontsize=15)
    plt.yticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], fontsize=15)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.ylabel("Fewest Trajectory Turns", fontsize=20)
    plt.xlabel("Monkey's Trajectory Turns", fontsize=20, labelpad = 8)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/3.D.new.pdf")
    plt.show()


def plotFig3E(monkey):
    print("----------- Fig 3E -----------")
    with open("{}/{}_3E.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    identity_map = {"ghost":"Ghost", "beans_sacc":"Pellets", "energizer_sacc":"Energizer", "others":"Others"}
    identiy_list = ["beans_sacc", "ghost","energizer_sacc", "others"]
    agent_list = ["local", "energizer", "approach", "evade", "global"]
    pie_color = ["#000000", "#666666", "#949494", "#F2F2F2"]
    explodes = [0, 0, 0, 0.1]

    agent_fixation_ratio = {each:{} for each in agent_list}
    for a in agent_list:
        tmp = 0.0
        for id in data[a]:
            agent_fixation_ratio[a][id[0]] = id[3]
            tmp += id[3]
        agent_fixation_ratio[a]["others"] = 1-tmp

    plt.figure(figsize=(8, 5))
    for a_ind in range(5):
        if a_ind == 2:
            ax = plt.subplot(2, 3, 3)
            plt.bar(0, 0, color=pie_color[0], label="Pellets")
            plt.bar(0, 0, color=pie_color[1], label="Ghost")
            plt.bar(0, 0, color=pie_color[2], label="Energizer")
            plt.bar(0, 0, color=pie_color[3], label="Others")
            plt.xticks([])
            plt.yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.legend(fontsize=18, ncol=1)
        if a_ind >= 2:
            pos = a_ind + 2
        else:
            pos = a_ind + 1
        agent_data = agent_fixation_ratio[agent_list[a_ind]]
        agent_fixation_val = [agent_data[each] for each in identiy_list]
        ax = plt.subplot(2, 3, pos)
        plt.title(agent_list[a_ind], fontsize=20, color=status_color_mapping[agent_list[a_ind]])
        wedges, _ , _ = ax.pie(
            agent_fixation_val,
            colors=pie_color,
            pctdistance=1.1, autopct="%.1f%%", textprops={'fontsize': 15},
            explode=explodes
        )
        for w in wedges:
            w.set_edgecolor('k')
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/3E.pie.pdf")
    plt.show()


def _remove_nan(mat, remove_trial):
    need_index = []
    for i in range(mat.shape[0]):
        if len(mat[i,:][np.isnan(mat[i,:])]) == 0:
            need_index.append(i)
        else:
            continue
    if remove_trial:
        return mat[need_index]
    else:
        return mat


def plotFig3F(monkey):
    print("----------- Fig 3F -----------")
    with open("{}/{}_3F.pkl".format(file_base, monkey),"rb") as file:
        data = pickle.load(file)
    vague_trans = data["vague"]
    wo_vague_trans = data["transition"]
    colors = [black, black]
    trans_ = [("local", "global"), ("global", "local")]
    for index, t_type in enumerate(trans_):
        temp_vague = np.array(vague_trans[t_type])
        temp_wo_vague = np.array(wo_vague_trans[t_type])
        temp_vague = _remove_nan(temp_vague, remove_trial=False)
        temp_wo_vague = _remove_nan(temp_wo_vague, remove_trial=False)
        temp_vague = _combine(temp_vague)
        temp_wo_vague = _combine(temp_wo_vague)

        # two sample t-test
        test = []
        for j in range(temp_vague.shape[1]):
            v = temp_vague[:, j][~np.isnan(temp_vague[:, j])]
            wov = temp_wo_vague[:, j][~np.isnan(temp_wo_vague[:, j])]
            test.append(scipy.stats.ttest_ind(v, wov)[1])
        print("| {} Time Step T-Test | {}".format(t_type, test))

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.title("{} $\\rightarrow$ {}".format(t_type[0], t_type[1]), fontsize=20)
        plt.plot(
            np.arange(temp_wo_vague.shape[1]),
            np.nanmean(temp_wo_vague, axis=0),
            color=colors[1],
            lw=3,
        )
        plt.plot(
            np.arange(temp_vague.shape[1]),
            np.nanmean(temp_vague, axis=0),
            "--",
            color=colors[0],
            lw=3,
        )
        # around vague
        y_se = []
        for j in np.arange(temp_vague.shape[1]):
            y_se.append(np.nanstd(temp_vague[:,j]) / np.sqrt(len(temp_vague[:,j][~np.isnan(temp_vague[:,j])])))
        y_se = np.array(y_se)
        plt.fill_between(
            np.arange(temp_vague.shape[1]),
            np.nanmean(temp_vague, axis = 0) - y_se,
            np.nanmean(temp_vague, axis = 0) + y_se,
            color=colors[0],
            alpha=0.4,
            linewidth=0.0
        )
        # w/o vague
        y_se = []
        for j in np.arange(temp_wo_vague.shape[1]):
            y_se.append(
                np.nanstd(temp_wo_vague[:, j]) / np.sqrt(len(temp_wo_vague[:, j][~np.isnan(temp_wo_vague[:, j])])))
        y_se = np.array(y_se)
        plt.fill_between(
            np.arange(temp_wo_vague.shape[1]),
            np.nanmean(temp_wo_vague, axis = 0) - y_se,
            np.nanmean(temp_wo_vague, axis = 0) + y_se,
            color=colors[1],
            alpha=0.4,
            linewidth=0.0
        )
        plt.ylabel("Normalized Pupil Size", fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks([0, 2.25, 4.5, 6.75, 9], ["-10", "-5", "transition", "5", "10"], fontsize=15)
        plt.xlim(0, temp_vague.shape[1] - 1)
        plt.xlabel("Time (s)", fontsize = 20)
        plt.legend(["W/ Strategy Transition", "W/O Strategy Transition"], fontsize=16)
        x_index = np.arange(temp_wo_vague.shape[1])
        min_y = plt.gca().get_ylim()[0]

        interval = 0.25
        gap = 0.05
        plt.ylim(min_y - gap - 0.01, plt.gca().get_ylim()[1])
        for i in x_index:
            if test[i] < 0.01:
                plt.plot([], [])
                if i == 0:
                    ax.add_line(
                        lines.Line2D([x_index[i] + interval, x_index[i]], [min_y - gap, min_y - gap],
                                     lw=8., color='k'))
                elif i == len(test) - 1:
                    ax.add_line(
                        lines.Line2D([x_index[i], x_index[i] - interval], [min_y - gap, min_y - gap],
                                     lw=8., color='k'))
                else:
                    ax.add_line(lines.Line2D([x_index[i] + interval, x_index[i] - interval],
                                             [min_y - gap, min_y - gap], lw=8., color='k'))
        plt.tight_layout()
        plt.savefig("{}/".format(pic_base) + monkey + "/3F.{}_{}_{}.pdf".format(index + 1, t_type[0], t_type[1]))
        plt.show()

    # ==============================

    vague_trans = data["vague"]
    wo_vague_trans = data["transition"]

    vague_all = [vague_trans[each] for each in vague_trans]
    vague_all = np.vstack(vague_all)

    wo_vague_all = [wo_vague_trans[each] for each in wo_vague_trans]
    wo_vague_all = np.vstack(wo_vague_all)

    vague_all = _remove_nan(vague_all, remove_trial=False)
    wo_vague_all = _remove_nan(wo_vague_all, remove_trial=False)

    vague_all = _combine(vague_all)
    wo_vague_all = _combine(wo_vague_all)

    colors = [black, black]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    test = []
    for j in range(vague_all.shape[1]):
        v = vague_all[:, j][~np.isnan(vague_all[:, j])]
        wov = wo_vague_all[:, j][~np.isnan(wo_vague_all[:, j])]
        test.append(scipy.stats.ttest_ind(v, wov)[1])
    print("| All Data Time Step T-Test | {}", test)

    plt.plot(
        np.arange(wo_vague_all.shape[1]),
        np.nanmean(wo_vague_all, axis=0),
        color=colors[1],
        lw=3,
    )
    plt.plot(
        np.arange(vague_all.shape[1]),
        np.nanmean(vague_all, axis=0),
        "--",
        color=colors[0],
        lw=3,
    )
    plt.ylabel("Normalized Pupil Size", fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks([0, 2.5, 4.5, 6.5, 9], ["-10", "-5", "transition", "5", "10"], fontsize=15)
    plt.xlim(0, vague_all.shape[1] - 1)
    plt.xlabel("Time(s)", fontsize=20)
    plt.legend(["W/ Strategy Transition", "W/O Strategy Transition"], fontsize=16)
    mu = np.nanmean(vague_all, axis=0)
    sigma = (
            np.nanstd(vague_all, axis=0) /
            np.array([
                np.sqrt(len(vague_all[:, j][~np.isnan(vague_all[:, j])])) for j in range(vague_all.shape[1])
            ])
    )
    y_se = []
    for j in np.arange(vague_all.shape[1]):
        y_se.append(
            np.nanstd(vague_all[:, j]) / np.sqrt(len(vague_all[:, j][~np.isnan(vague_all[:, j])])))
    y_se = np.array(y_se)
    plt.fill_between(
        np.arange(vague_all.shape[1]),
        np.nanmean(vague_all, axis = 0) - y_se,
        np.nanmean(vague_all, axis = 0) + y_se,
        color=colors[0],
        alpha=0.4,
        linewidth=0.0
    )
    # w/o vague
    mu = np.nanmean(wo_vague_all, axis=0)
    sigma = (
            np.nanstd(wo_vague_all, axis=0) /
            np.array([
                np.sqrt(len(wo_vague_all[:, j][~np.isnan(wo_vague_all[:, j])])) for j in
                range(wo_vague_all.shape[1])
            ])
    )
    y_se = []
    for j in np.arange(wo_vague_all.shape[1]):
        y_se.append(
            np.nanstd(wo_vague_all[:, j]) / np.sqrt(len(wo_vague_all[:, j][~np.isnan(wo_vague_all[:, j])])))
    y_se = np.array(y_se)
    plt.fill_between(
        np.arange(wo_vague_all.shape[1]),
        np.nanmean(wo_vague_all, axis = 0) - y_se,
        np.nanmean(wo_vague_all, axis = 0) + y_se,
        color=colors[1],
        alpha=0.4,
        linewidth=0.0
    )
    x_index = np.arange(wo_vague_all.shape[1])
    min_y = plt.gca().get_ylim()[0]
    interval = 0.5
    gap = 0.05
    plt.ylim(min_y - gap - 0.01, plt.gca().get_ylim()[1])
    for i in range(len(test)):
        if test[i] < 0.01:
            plt.plot([], [])
            if i == 0:
                ax.add_line(
                    lines.Line2D([x_index[i] + interval, x_index[i]], [min_y - gap, min_y - gap],
                                 lw=8., color='k'))
            elif i == len(test) - 1:
                ax.add_line(
                    lines.Line2D([x_index[i], x_index[i] - interval], [min_y - gap, min_y - gap],
                                 lw=8., color='k'))
            else:
                ax.add_line(lines.Line2D([x_index[i] + interval, x_index[i] - interval],
                                         [min_y - gap, min_y - gap], lw=8., color='k'))
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/3F.all.pdf")
    plt.show()

# ===================================================
#                   FIG 4: PLAN
# ===================================================

def _combine(mat, bin=2):
    p = mat.shape[1]
    N = mat.shape[0]
    res = np.zeros((N*2, p//bin))
    res[res==0] = np.nan
    seq = np.arange(0, p+1, bin)
    for i in range(len(seq)-1):
        res[:,i] = mat[:, seq[i]:seq[i+1]].reshape(-1)
    return res


def plotFig4B(monkey):
    print("----------- Fig 4B -----------")
    plt.figure(figsize=(6, 5))
    ax1 = plt.subplot(1,1,1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.title("Planned Attack", fontsize = 20)
    with open("{}/{}_4B_planned_weight.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    mean_data = np.nanmean(data, axis = 0)
    sem_data = scipy.stats.sem(data, axis=0, nan_policy="omit")
    tmp_agent_name = copy.deepcopy(agent_name)
    tmp_agent_name[2] = "evade(Blinky)"
    tmp_agent_name[3] = "evade(Clyde)"
    plt.fill_between([10, 20], [0,0], [1,1], color="#efeeee", linewidth=0.0)
    for i in range(6):
        plt.plot(mean_data[i], "-", lw = 5, color = status_color_mapping[agent_name[i]], label = tmp_agent_name[i])
        plt.fill_between(
            np.arange(0, 20),
            mean_data[i] - sem_data[i],
            mean_data[i] + sem_data[i],
            color=status_color_mapping[agent_name[i]],
            alpha=0.3,
            linewidth=4
        )
    plt.xticks([5, 10, 15], ["-5", "0", "5"], fontsize=15)
    plt.xlabel("Time before/after Eating Energizer (s)", fontsize=20)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
    plt.ylim(0, 1.0)
    plt.ylabel("Strategy Weight", fontsize = 20)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/4B.planned.pdf")
    plt.show()

    plt.figure(figsize=(6, 5))
    ax2 = plt.subplot(1, 1, 1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.title("Accidental Consumption", fontsize = 20)
    with open("{}/{}_4B_accidental_weight.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    mean_data = np.nanmean(data, axis=0)
    sem_data = scipy.stats.sem(data, axis=0, nan_policy="omit")
    tmp_agent_name = copy.deepcopy(agent_name)
    tmp_agent_name[2] = "evade(Blinky)"
    tmp_agent_name[3] = "evade(Clyde)"
    plt.fill_between([10, 20], [0,0], [1,1], color="#efeeee", linewidth=0.0)
    for i in range(6):
        plt.plot(mean_data[i], "-", lw=5, color=status_color_mapping[agent_name[i]], label=tmp_agent_name[i])
        plt.fill_between(
            np.arange(0, 20),
            mean_data[i] - sem_data[i],
            mean_data[i] + sem_data[i],
            color=status_color_mapping[agent_name[i]],
            alpha=0.3,
            linewidth=4
        )
    plt.xticks([5, 10, 15], ["-5", "0", "5"], fontsize=20)
    plt.xlabel("Time before/after Eating Energizer (s)", fontsize=20)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Strategy Weight", fontsize=20)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/4B.accidental.pdf")
    plt.show()


def plotFig4DEHI(monkey):
    # ===========================================================================
    print("----------- Fig 4D -----------")
    with open("{}/{}_4D.ghost_all.pkl".format(file_base, monkey), "rb") as file:
        detail_data = pickle.load(file)
    planned_data = detail_data["Planned Attack"]
    accidental_data = detail_data["Accidental Attack"]
    planned_data = [each[-1] for each in planned_data]
    planned_data = [planned_data[1], planned_data[2], planned_data[0]]
    accidental_data = [each[-1] for each in accidental_data]
    accidental_data = [accidental_data[1], accidental_data[2], accidental_data[0]]
    test = [scipy.stats.ttest_ind(planned_data[i], accidental_data[i]) for i in range(len(planned_data))]
    print("| Planned vs. Accidental Saccade Ratio T Test | Ghost = {} / Energizer = {} / Pacman = {}".format(test[0][1], test[1][1], test[2][1]))
    colors = [dark_grey, light_grey]
    identity_map = {"ghost1Pos_sacc":"Ghost Blinky", "ghost2Pos_sacc":"Ghost Clyde", "pacman_sacc":"Pacman",
                    "ghost": "Ghosts", "ghost_all": "Ghosts", "energizer_sacc": "Energizers"}
    identiy_list = ["ghost_all", "energizer_sacc", "pacman_sacc"]
    agent_list = ["Planned Attack", "Accidental Consumption"]
    identity_sacc_mean = {
        "ghost_all" : [np.nanmean(planned_data[0]), np.nanmean(accidental_data[0])],
        "energizer_sacc": [np.nanmean(planned_data[1]), np.nanmean(accidental_data[1])],
        "pacman_sacc": [np.nanmean(planned_data[2]), np.nanmean(accidental_data[2])],

    }
    identity_sacc_std = {
        "ghost_all": [scipy.stats.sem(planned_data[0], nan_policy="omit"), scipy.stats.sem(accidental_data[0], nan_policy="omit")],
        "energizer_sacc": [scipy.stats.sem(planned_data[1], nan_policy="omit"),scipy.stats.sem(accidental_data[1], nan_policy="omit")],
        "pacman_sacc": [scipy.stats.sem(planned_data[2], nan_policy="omit"), scipy.stats.sem(accidental_data[2], nan_policy="omit")],
    }
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    x_index = np.arange(len(identiy_list))
    ks_x_index = x_index+0.15
    ks_y = []
    for i, a in enumerate(identiy_list):
        temp_mean = identity_sacc_mean[a]
        ks_y.append(np.max(temp_mean) + 0.007)
        temp_std = identity_sacc_std[a]
        for j, agent in enumerate(agent_list):
            plt.bar(x_index[i] + j * 0.3, height=temp_mean[j],
                    yerr=temp_std[j], width=0.3, color=colors[j])
    plt.xticks(x_index+0.15, [identity_map[k] for k in identiy_list], fontsize=20)
    plt.ylabel("Average Fixation Ratio", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, np.max(list(identity_sacc_mean.values()))+0.05)
    plt.legend(agent_list, fontsize=15, loc="upper left")
    for i in range(len(test)):
        tmp_ks = test[i]
        if tmp_ks[1] < 0.001:
            t = "***"
        elif tmp_ks[1] < 0.01:
            t = "**"
        elif tmp_ks[1] < 0.05:
            t = "*"
        else:
            t = "n.s."
        plt.text(ks_x_index[i], ks_y[i], t, fontsize=20, ha="center", va="center")
    plt.tight_layout()
    plt.savefig('{}/{}/4D_ratio.pdf'.format(pic_base, monkey))
    plt.show()
    plt.close()
    # ===========================================================================
    print("----------- Fig 4E -----------")
    with open("{}/{}_4E.detail.pkl".format(file_base, monkey), "rb") as file:
        detail_data = pickle.load(file)
    planned_data = detail_data["planned"].values
    accidental_data = detail_data["accidental"].values
    temp_planned_data = []
    temp_accidental_data = []
    for i in range(0, 20):
        temp_planned_data.append(planned_data[i])
        temp_accidental_data.append(accidental_data[i])
    planned_data = temp_planned_data
    accidental_data = temp_accidental_data
    planned_data[:len(planned_data)//2] = planned_data[:len(planned_data)//2][::-1]
    accidental_data[:len(accidental_data)//2] = accidental_data[:len(accidental_data)//2][::-1]
    test = [
        scipy.stats.ttest_ind(planned_data[i][~np.isnan(planned_data[i])], accidental_data[i][~np.isnan(accidental_data[i])])
        for i in range(len(planned_data))
    ]
    print("| Planned vs. Accidental Pupil Size T Test (time step) | ", [i[1] for i in test])
    with open("{}/{}_4E.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colors = [black, black]
    area_color = [dark_grey, light_grey]
    type = ["planned", "accidental"]
    type_map = {"planned": "Planned Attack", "accidental": "Accidental Consumption"}
    max_mean_data = -99999
    min_mean_data = 99999

    for index, each in enumerate(type):
        mean_data = data[each]["mean"].values
        std_data = data[each]["std"].values / np.sqrt(data[each]["count"].values)

        if np.max(mean_data) > max_mean_data:
            max_mean_data = np.max(mean_data)
        if np.min(mean_data) < min_mean_data:
            min_mean_data = np.min(mean_data)

        mean_data = np.array(mean_data)
        std_data = np.array(std_data)

        if each == "accidental":
            plt.plot(
                np.arange(-10, 10, 1),
                mean_data,
                color=colors[index],
                lw=3,
            )
        else:
            plt.plot(
                np.arange(-10, 10, 1),
                mean_data,
                color=colors[index],
                lw=3,
            )
        mu = mean_data
        sigma = std_data
        plt.fill_between(
            np.arange(-10, 10, 1),
            mean_data - std_data,
            mean_data + std_data,
            color=area_color[index],
            alpha=1.0,
            linewidth=0.0,
            label = type_map[each],
        )

    plt.fill_between([-1, 9], [min_mean_data - 0.1, min_mean_data - 0.1], [max_mean_data + 0.1, max_mean_data + 0.1], color=dark_grey, linewidth=0.0, alpha=0.1)
    plt.ylabel("Normalized Pupil Size", fontsize = 20)
    plt.xlabel("Time before/after Eating Ghosts (s)", fontsize=20)
    plt.legend(loc="upper center", fontsize=15, ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.15))
    plt.xticks([-10, -5.25, -0.5, 4.25, 9], [-10, -5, 0, 5, 10], fontsize = 20)
    plt.yticks(fontsize=20)
    plt.xlim(-10, 9)
    x_index = np.arange(-10, 10)
    for i in range(len(test)):
        if test[i][1] < 0.01:
            plt.plot([], [])
            if i == 0:
                ax.add_line(lines.Line2D([x_index[i]+1, x_index[i]], [min_mean_data - 0.1, min_mean_data - 0.1], lw=8., color='k'))
            elif i == len(test)-1:
                ax.add_line(lines.Line2D([x_index[i], x_index[i]-1], [min_mean_data - 0.1, min_mean_data - 0.1], lw=8., color='k'))
            else:
                ax.add_line(lines.Line2D([x_index[i]+1, x_index[i]-1], [min_mean_data - 0.1, min_mean_data - 0.1], lw=8., color='k'))
    plt.tight_layout()
    plt.savefig('{}/{}/4E_pupil.pdf'.format(pic_base, monkey))
    plt.show()
    plt.close()
    # ===========================================================================
    print("----------- Fig 4H -----------")
    with open("{}/{}_4H.ghost_all.pkl".format(file_base, monkey), "rb") as file:
        detail_data = pickle.load(file)
    suicide_data = detail_data["Suicide"]
    normal_data = detail_data["Normal Die"]
    suicide_data = [each[-1] for each in suicide_data]
    suicide_data = [suicide_data[1], suicide_data[2], suicide_data[0]]
    normal_data = [each[-1] for each in normal_data]
    normal_data = [normal_data[1], normal_data[2], normal_data[0]]

    test = [scipy.stats.ttest_ind(suicide_data[i], normal_data[i]) for i in range(len(suicide_data))]
    print("| Suicide vs. Failure Saccade Ratio T Test | Ghost = {} / Pellets = {} / Pacman = {}".format(test[0][1], test[1][1], test[2][1]))
    colors = [dark_grey, light_grey]
    identity_map = {"ghost1Pos_sacc": "Ghost Blinky", "ghost2Pos_sacc": "Ghost Clyde", "pacman_sacc": "Pacman",
                    "ghost": "Ghosts", "ghost_all": "Ghosts", "beans_sacc": "Pellets"}
    identiy_list = ["ghost_all", "beans_sacc", "pacman_sacc"]
    agent_list = ["Suicide", "Normal Die"]

    identity_sacc_mean = {
        "ghost_all": [np.nanmean(suicide_data[0]), np.nanmean(normal_data[0])],
        "beans_sacc": [np.nanmean(suicide_data[1]), np.nanmean(normal_data[1])],
        "pacman_sacc": [np.nanmean(suicide_data[2]), np.nanmean(normal_data[2])],
    }
    identity_sacc_std = {
        "ghost_all": [scipy.stats.sem(suicide_data[0], nan_policy="omit"),
                  scipy.stats.sem(normal_data[0], nan_policy="omit")],
        "beans_sacc": [scipy.stats.sem(suicide_data[1], nan_policy="omit"),
                       scipy.stats.sem(normal_data[1], nan_policy="omit")],
        "pacman_sacc": [scipy.stats.sem(suicide_data[2], nan_policy="omit"),
                        scipy.stats.sem(normal_data[2], nan_policy="omit")],
    }
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    x_index = np.arange(len(identiy_list))
    ks_x_index = x_index + 0.15
    ks_y = []
    for i, a in enumerate(identiy_list):
        temp_mean = identity_sacc_mean[a]
        ks_y.append(np.max(temp_mean) + 0.007)
        temp_std = identity_sacc_std[a]
        for j, agent in enumerate(agent_list):
            plt.bar(x_index[i] + j * 0.3, height=temp_mean[j],
                    yerr=temp_std[j], width=0.3, color=colors[j])
    plt.xticks(x_index + 0.15, [identity_map[k] for k in identiy_list], fontsize=20)
    plt.ylabel("Average Fixation Ratio", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, np.max(list(identity_sacc_mean.values()))+0.05)
    if monkey == "Patamon" or monkey == "Omega":
        plt.legend(["Suicide", "Failure"], fontsize=18, loc="upper left")
    else:
        plt.legend(["Suicide", "Failure"], fontsize=18)
    for i in range(len(test)):
        tmp_ks = test[i]
        if tmp_ks[1] < 0.001:
            t = "***"
        elif tmp_ks[1] < 0.01:
            t = "**"
        elif tmp_ks[1] < 0.05:
            t = "*"
        else:
            t = "n.s."
        plt.text(ks_x_index[i], ks_y[i], t, fontsize=20, ha="center", va="center")
    plt.tight_layout()
    plt.savefig('{}/{}/4H_ratio.pdf'.format(pic_base, monkey))
    plt.show()
    plt.close()
    # ===========================================================================
    print("----------- Fig 4I -----------")
    with open("{}/{}_4I.detail.pkl".format(file_base, monkey), "rb") as file:
        detail_data = pickle.load(file)
    suicide_data = detail_data["suicide"]
    normal_data = detail_data["normal"]
    temp_suicide_data = []
    temp_normal_data = []
    for i in range(0, 8, 2):
        temp_suicide_data.append(np.concatenate(suicide_data[i:i + 2]))
        temp_normal_data.append(np.concatenate(normal_data[i:i + 2]))
    temp_suicide_data.append(np.array(suicide_data[-1]))
    temp_normal_data.append(np.array(normal_data[-1]))
    suicide_data = temp_suicide_data
    normal_data = temp_normal_data

    test = [scipy.stats.ttest_ind(suicide_data[i][~np.isnan(suicide_data[i])], normal_data[i][~np.isnan(normal_data[i])]) for i in range(len(suicide_data))]
    print("| Suicide vs. Falure Pupil Size T Test (time step) | ", [i[1] for i in test])
    with open("{}/{}_4I.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colors = [black, black]
    type = ["suicide", "normal"]
    type_map = {"suicide": "Suicide", "normal": "Faiure"}
    max_mean_data = -99999
    min_mean_data = 99999
    for index, each in enumerate(type):
        # 合并数据
        mean_data = []
        std_data = []
        for i in range(0, 8, 2):
            tmp_mean = np.array([data[each]["mean"].values[i], data[each]["mean"].values[i + 1]])
            tmp_std = np.array([data[each]["std"].values[i], data[each]["std"].values[i + 1]])
            tmp_size = np.array([data[each]["count"].values[i], data[each]["count"].values[i + 1]])

            combined_mean = np.sum(tmp_mean * tmp_size) / np.sum(tmp_size)
            mean_data.append(combined_mean)

            combined_std = np.sqrt(
                np.sum(tmp_std ** 2 * (tmp_size - 1) + tmp_size * (tmp_mean - combined_mean) ** 2) / (
                        np.sum(tmp_size) - 1)
            ) / np.sqrt(np.sum(tmp_size))
            std_data.append(combined_std)
        mean_data.append(data[each]["mean"].values[-1])
        std_data.append(data[each]["std"].values[-1] / np.sqrt(np.sum(data[each]["count"].values[-1])))
        # ========================
        if np.max(mean_data) > max_mean_data:
            max_mean_data = np.max(mean_data)
        if np.min(mean_data) < min_mean_data:
            min_mean_data = np.min(mean_data)

        mean_data = np.array(mean_data)
        std_data = np.array(std_data)
        if each == "suicide":
            plt.plot(
                np.arange(len(mean_data)),
                mean_data,
                color=colors[index],
                lw=3,
            )
        else:
            plt.plot(
                np.arange(len(mean_data)),
                mean_data,
                color=colors[index],
                lw=3,
            )
        mu = mean_data
        sigma = std_data

        plt.fill_between(
            np.arange(len(mean_data)),
            mean_data - std_data,
            mean_data + std_data,
            color=area_color[index],
            alpha=1.0,
            linewidth=0.0,
            label = type_map[each],
        )
    plt.legend(fontsize=18, loc = "lower left" if monkey != "Patamon" else "lower right")
    plt.ylabel("Normalized Pupil Size", fontsize=20)
    plt.xlabel("Time Before PacMan Getting Caught (s)", fontsize=20)
    plt.xticks([0, 1, 2, 3, 4], [-4, -3, -2, -1, 0], fontsize=20)
    plt.yticks(fontsize=18)
    plt.xlim(0, 4)
    x_index = np.arange(len(test))
    min_mean_data = plt.gca().get_ylim()[0]
    for i in range(len(test)):
        if test[i][1] < 0.01:
            plt.plot([], [])
            if i == 0:
                ax.add_line(
                    lines.Line2D([x_index[i] + 0.5, x_index[i]], [min_mean_data - 0.1, min_mean_data - 0.1],
                                 lw=8., color='k'))
            elif i == len(test):
                ax.add_line(
                    lines.Line2D([x_index[i], x_index[i] - 0.5], [min_mean_data - 0.1, min_mean_data - 0.1],
                                 lw=8., color='k'))
            else:
                ax.add_line(lines.Line2D([x_index[i] + 0.5, x_index[i] - 0.5],
                                         [min_mean_data - 0.1, min_mean_data - 0.1], lw=8., color='k'))
    plt.tight_layout()
    plt.savefig('{}/{}/4I_pupil.pdf'.format(pic_base, monkey))
    plt.show()
    plt.close()


def plotFig4CFG(monkey):
    # ===========================================================================
    print("----------- Fig 4C -----------")
    with open("{}/{}_4C.all.pkl".format(file_base, monkey), "rb") as file:
        data = pickle.load(file)
    data = {"1": data["1"], "2": data["2"]}
    all_data = data["1"]
    all_data["distance2"] = data["2"].distance2
    all_data["EG2_dis"] = data["2"].EG2_dis
    planned_indices = np.where(all_data.cate == "Planned Hunting")[0]
    accidental_indices = np.where(all_data.cate == "Accidentally Hunting")[0]
    planned_PG_dis = []
    planned_EG_dis = []
    planned_PE_dis = []
    for each in planned_indices:
        temp_data = all_data.iloc[each]
        if temp_data.EG1_dis < temp_data.EG2_dis:
            planned_PG_dis.append(temp_data.distance1)
            planned_EG_dis.append(temp_data.EG1_dis)
        else:
            planned_PG_dis.append(temp_data.distance2)
            planned_EG_dis.append(temp_data.EG2_dis)
        planned_PE_dis.append(temp_data.PE_dis)
    accidental_PG_dis = []
    accidental_EG_dis = []
    accidental_PE_dis = []
    for each in accidental_indices:
        temp_data = all_data.iloc[each]
        if temp_data.EG1_dis < temp_data.EG2_dis:
            accidental_PG_dis.append(temp_data.distance1)
            accidental_EG_dis.append(temp_data.EG1_dis)
        else:
            accidental_PG_dis.append(temp_data.distance2)
            accidental_EG_dis.append(temp_data.EG2_dis)
        accidental_PE_dis.append(temp_data.PE_dis)
    planned_PG_dis = np.array(planned_PG_dis)
    planned_EG_dis = np.array(planned_EG_dis)
    planned_PE_dis = np.array(planned_PE_dis)
    accidental_PG_dis = np.array(accidental_PG_dis)
    accidental_EG_dis = np.array(accidental_EG_dis)
    accidental_PE_dis = np.array(accidental_PE_dis)


    planned_avg = (planned_PG_dis + planned_EG_dis + planned_PE_dis) / 3
    accidental_avg = (accidental_PG_dis + accidental_EG_dis + accidental_PE_dis) / 3
    planned_avg = planned_avg[~np.isnan(planned_avg)]
    accidental_avg = accidental_avg[~np.isnan(accidental_avg)]

    step = 2
    bins = np.arange(0, 75, step)
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    planned_data_bin = np.histogram(planned_avg, bins)[0]
    accidental_data_bin = np.histogram(accidental_avg, bins)[0]
    plt.bar(bins[:-1], planned_data_bin / np.sum(planned_data_bin), width=step - 0.15, color=dark_grey,
            alpha=1.0, align="edge", label="Planned Attack")
    plt.bar(bins[:-1], -accidental_data_bin / np.sum(accidental_data_bin), width=step - 0.15, color=light_grey,
            alpha=1.0, align="edge", label="Accidental Consumption")
    plt.plot([np.mean(planned_avg), np.mean(planned_avg)], [0, ax.get_ylim()[1]], "--", lw=4, color=black)
    plt.plot([np.mean(accidental_avg), np.mean(accidental_avg)], [0, ax.get_ylim()[0]], "--", lw=4, color=black)
    plt.ylabel("Probability", fontsize=20)
    plt.xlim(0, 40)
    plt.xticks(fontsize=15)
    plt.yticks([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15], [0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15], fontsize=15)
    plt.xlabel("Average Distance between PacMan, Energizer and Ghost", fontsize=20)
    plt.legend(fontsize=20)
    test = scipy.stats.ttest_ind(planned_avg, accidental_avg)
    print("| Planned vs. Accidental Avg Distance T Test | ", test)
    if test[1] < 0.001:
        t = "***"
    elif test[1] < 0.01:
        t = "**"
    elif test[1] < 0.05:
        t = "*"
    else:
        t = "n.s."
    ax.text(ax.get_xlim()[1], 0.0, t, fontsize=22, ha="center",
            va="center", rotation=90)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/4C.PA_vs_AA.avg.pdf")
    plt.show()
    # ===========================================================================
    print("----------- Fig 4F -----------")
    with open("{}/{}_4F.detail.pkl".format(file_base, monkey), "rb") as file:
        detail_data = pickle.load(file)
    df_com = detail_data["df_com"]
    df_temp = detail_data["df_temp"]
    df_temp_re = detail_data["df_temp_re"]
    suicide_suicide_dist = df_com.suicide_dis.values
    suicide_reset_dist = df_com.reset_dis.values
    suicide_suicide_dist = suicide_suicide_dist[~np.isnan(suicide_suicide_dist)]
    suicide_reset_dist = suicide_reset_dist[~np.isnan(suicide_reset_dist)]

    evade_suicide_dist = df_temp.values
    evade_reset_dist = df_temp_re.values
    evade_suicide_dist = evade_suicide_dist[~np.isnan(evade_suicide_dist)]
    evade_reset_dist = evade_reset_dist[~np.isnan(evade_reset_dist)]

    suicide_suicide_dist = np.array(suicide_suicide_dist)
    suicide_reset_dist = np.array(suicide_reset_dist)
    evade_suicide_dist = np.array(evade_suicide_dist)
    evade_reset_dist = np.array(evade_reset_dist)

    suicide_diff = suicide_suicide_dist-suicide_reset_dist
    suicide_diff = suicide_diff[~np.isnan(suicide_diff)]
    evade_diff = evade_suicide_dist-evade_reset_dist
    evade_diff = evade_diff[~np.isnan(evade_diff)]

    step = 3
    bins = np.arange(-40, 40, step)
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    suicide_data_bin = np.histogram(suicide_diff, bins)[0]
    evade_data_bin = np.histogram(evade_diff, bins)[0]
    plt.bar(bins[:-1], suicide_data_bin / np.sum(suicide_data_bin), width=step - 0.15, color=dark_grey,
            alpha=1.0, align="edge", label="Suicide")
    plt.bar(bins[:-1], -evade_data_bin / np.sum(evade_data_bin), width=step - 0.15, color=light_grey,
            alpha=1.0, align="edge", label="Failure")
    plt.plot([np.mean(suicide_diff), np.mean(suicide_diff)], [0, ax.get_ylim()[1]], "--", lw=4, color=black)
    plt.plot([np.mean(evade_diff), np.mean(evade_diff)], [0, ax.get_ylim()[0]], "--", lw=4, color=black)
    plt.ylabel("Probability", fontsize=20)
    plt.yticks([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15], [0.15, 0.1, 0.05, 0.0, 0.05, 0.1, 0.15], fontsize=15)
    plt.xlim(-40, 40)
    plt.xticks(fontsize=15)
    plt.xlabel("PacMan-Pellet Distance —— Reset-Pellet Distance", fontsize=20)
    plt.legend(fontsize=20, ncol=2)
    test = scipy.stats.ttest_ind(suicide_diff, evade_diff)
    print("| Suicide vs. Failure PP-RP Distance T Test | ", test)

    if test[1] < 0.001:
        t = "***"
    elif test[1] < 0.01:
        t = "**"
    elif test[1] < 0.05:
        t = "*"
    else:
        t = "n.s."
    ax.text(ax.get_xlim()[1], 0.0, t, fontsize=22, ha="center",
            va="center", rotation=90)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/4F.suicide_vs_evade.reset.pdf")
    plt.show()
    # ===========================================================================
    print("----------- Fig 4G -----------")
    with open("{}/{}_4G.detail.PG.pkl".format(file_base, monkey), "rb") as file:
        PG_data = pickle.load(file)
    evade_PG1 = PG_data["evade"].distance1
    evade_PG2 = PG_data["evade"].distance2
    suicide_PG1 = PG_data["suicide"].distance1
    suicide_PG2 = PG_data["suicide"].distance2

    suicide_PG1 = np.array(suicide_PG1)
    suicide_PG2 = np.array(suicide_PG2)
    evade_PG1 = np.array(evade_PG1)
    evade_PG2 = np.array(evade_PG2)

    suicide_avg = (suicide_PG1 + suicide_PG2) / 2
    suicide_avg = suicide_avg[~np.isnan(suicide_avg)]
    evade_avg = (evade_PG1 + evade_PG2) / 2
    evade_avg = evade_avg[~np.isnan(evade_avg)]

    step = 1
    bins = np.arange(0, 30, step)
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    suicide_data_bin = np.histogram(suicide_avg, bins)[0]
    evade_data_bin = np.histogram(evade_avg, bins)[0]
    plt.bar(bins[:-1], suicide_data_bin / np.sum(suicide_data_bin), width=step - 0.15, color=dark_grey,
            alpha=1.0, align="edge", label="Suicide")
    plt.bar(bins[:-1], -evade_data_bin / np.sum(evade_data_bin), width=step - 0.15, color=light_grey,
            alpha=1.0, align="edge", label="Failure")
    plt.plot([np.mean(suicide_avg), np.mean(suicide_avg)],
             [0, ax.get_ylim()[1]], "--", lw=4, color=black)
    plt.plot([np.mean(evade_avg), np.mean(evade_avg)],
             [0, ax.get_ylim()[0]], "--", lw=4, color=black)
    plt.ylabel("Probability", fontsize=20)
    plt.yticks([-0.2, -0.1, 0.0, 0.1, 0.2], [0.2, 0.1, 0.0, 0.1, 0.2], fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlim(0, 30)
    plt.xlabel("Average PacMan-Ghost Distance.", fontsize=20)
    plt.legend(fontsize=20, ncol=2)
    test = scipy.stats.ttest_ind(suicide_avg, evade_avg)
    print("| Suicide vs. Failure AVG(PG) T Test | ", test)

    if test[1] < 0.001:
        t = "***"
    elif test[1] < 0.01:
        t = "**"
    elif test[1] < 0.05:
        t = "*"
    else:
        t = "n.s."
    ax.text(ax.get_xlim()[1], 0.0, t, fontsize=22, ha="center",
            va="center", rotation=90)
    plt.tight_layout()
    plt.savefig("{}/".format(pic_base) + monkey + "/4G.suicide_vs_evade.PG.pdf")
    plt.show()


# ===================================================
#             SUPP FIG 2: BASIC STATISTICS
# ===================================================
def _splitErrorGames(record, error_name):
    round_num = record["round_num"]
    for n_pair in error_name:
        n, _, try_cnt = n_pair
        if n == "14-Omega-02-Jul-2019" :
            for k in [
                'round_num', 'round_tile', 'round_reward', 'round_energizer',
                'round_drop', 'round_eaten_ghosts'
            ]:
                record[k].pop("14-Omega-02-Jul-2019", None)
            record['game_per_day']["Omega-02-Jul-2019"].remove("14")
            for j in [1, 2, 3, 4, 5]:
                record['round_per_day']["Omega-02-Jul-2019"].remove("14-{}".format(j))
            continue

        if n == "6-Patamon-13-Jul-2019" :
            for k in [
                'round_num', 'round_tile', 'round_reward', 'round_energizer',
                'round_drop', 'round_eaten_ghosts'
            ]:
                record[k].pop("6-Patamon-13-Jul-2019", None)
            record['game_per_day']["Patamon-13-Jul-2019"].remove("6")
            for j in [1, 2, 3]:
                record['round_per_day']["Patamon-13-Jul-2019"].remove("6-{}".format(j))
            continue

        #
        tmp_round = round_num[n]
        one_num = len(np.unique(tmp_round))
        if one_num == len(tmp_round):
            print(n_pair)
            continue
        ref = np.arange(one_num) * 2
        another_ind = (ref + 1)[:(len(tmp_round) - one_num)]
        one_ind = ref[:len(another_ind)]
        if (len(another_ind) + len(one_ind)) == len(tmp_round):
            continue
        if try_cnt == "1":
            one_ind = list(one_ind) + list(range(another_ind[-1]+1, len(tmp_round)))
        else:
            another_ind = list(another_ind) + list(range(another_ind[-1]+1, len(tmp_round)))
        date_name = "-".join(n.split("-")[1:])
        # for each record
        for k in [
            'round_num', 'round_tile', 'round_reward', 'round_energizer',
            'round_drop', 'round_eaten_ghosts', 'game_per_day'
        ]:
            if k == "game_per_day":
                record[k][date_name].append("extra")
                continue
            tmp = copy.deepcopy(record[k][n])
            record[k][n] = list(np.array(tmp)[one_ind])
            record[k][n + "-extra"] = list(np.array(tmp)[another_ind])
    return record


def _removeDay(record, day):
    round_num = record["round_num"]
    game_name = list(round_num.keys())
    excluding_game_name = [each for each in game_name if day in each]
    excluding_round_num = [len(round_num[each]) for each in excluding_game_name]
    print("Excluding {} games and {} rounds. ".format(len(excluding_game_name), sum(excluding_round_num)))
    print()
    record['game_per_day'].pop(day, None)
    record['round_per_day'].pop(day, None)
    for g in excluding_game_name:
        for k in [
                    'round_num', 'round_tile', 'round_reward', 'round_energizer',
                    'round_drop', 'round_eaten_ghosts'
                ]:
            record[k].pop(g, None)
    return record


def plotBasicStatistics():
    print("---------- Supp Fig ----------")
    # Read records
    record = np.load(file_base + "basic_statistics.npy", allow_pickle=True).item()
    omega_record = record["Omega"]
    patamon_record = record["Patamon"]
    # Deal with wrongly marked games
    omega_error = pd.read_pickle(file_base + "omega_abnormal_round.pkl").file.unique()
    patamon_error = pd.read_pickle(file_base + "patamon_abnormal_round.pkl").file.unique()
    omega_error_game = {}
    patamon_error_game = {}
    for t in omega_error:
        tmp_split = t.split(".")[0].split("-")
        game_name = "-".join([tmp_split[0]]+tmp_split[2:-1])
        if game_name not in omega_error_game:
            omega_error_game[game_name] = (game_name, tmp_split[1], tmp_split[-1])
        else:
            if int(tmp_split[1]) > int(omega_error_game[game_name][1]):
                omega_error_game[game_name] = (game_name, tmp_split[1], tmp_split[-1])
    for t in patamon_error:
        tmp_split = t.split(".")[0].split("-")
        game_name = "-".join([tmp_split[0]]+tmp_split[2:-1])
        if game_name not in patamon_error_game:
            patamon_error_game[game_name] = (game_name, tmp_split[1], tmp_split[-1])
        else:
            if int(tmp_split[1]) > int(patamon_error_game[game_name][1]):
                patamon_error_game[game_name] = (game_name, tmp_split[1], tmp_split[-1])
    omega_error_game = list(omega_error_game.values())
    patamon_error_game = list(patamon_error_game.values())
    patamon_record = _splitErrorGames(patamon_record, patamon_error_game)
    patamon_record = _removeDay(patamon_record,"Patamon-30-May-2019")
    omega_record = _splitErrorGames(omega_record, omega_error_game)
    # ------------------------------------
    omega_num_rounds = [len(omega_record["round_num"][each]) for each in omega_record["round_num"]]
    patamon_num_rounds = [len(patamon_record["round_num"][each]) for each in patamon_record["round_num"]]
    # plot the number of rounds (Omega)
    omega_count = np.zeros((10,))
    patamon_count = np.zeros((10,))
    plt.figure(figsize=(7,5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    omega_num_rounds = [each if each < 10 else 10 for each in omega_num_rounds]
    for i in omega_num_rounds:
        omega_count[i-1] += 1
    plt.bar(np.arange(10), omega_count, align="center", color = light_grey, width = 0.99, linewidth = 0.45, edgecolor = "k")
    plt.axvline(np.mean(omega_num_rounds)-1, lw=3, linestyle="--", color="k")
    plt.text(x=np.mean(omega_num_rounds)-1, y=ax.get_ylim()[1] + 10,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(omega_num_rounds), np.std(omega_num_rounds)),
             fontsize=18, ha = "center", va = "center")
    plt.xlabel("Number of Rounds to Finish One Game", fontsize = 20)
    plt.xticks(np.arange(10), [1,2,3,4,5,6,7,8,9, "$\geqslant$10"], fontsize=20)
    plt.yticks(fontsize = 20)
    plt.ylabel("count", fontsize = 20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.A.Omega.pdf")
    plt.show()
    # plot the number of rounds (Patamon)
    plt.figure(figsize=(7,5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    patamon_num_rounds = [each if each < 10 else 10 for each in patamon_num_rounds]
    for i in patamon_num_rounds:
        patamon_count[i-1] += 1
    plt.bar(np.arange(10), patamon_count, align="center", color = light_grey, width = 0.99, linewidth = 0.45, edgecolor = "k")
    plt.axvline(np.mean(patamon_num_rounds)-1, lw=3, linestyle="--", color="k")
    plt.text(x=np.mean(patamon_num_rounds) - 1, y=ax.get_ylim()[1] + 10,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(patamon_num_rounds), np.std(patamon_num_rounds)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Number of Rounds to Finish One Game", fontsize = 20)
    plt.xticks(np.arange(10), [1,2,3,4,5,6,7,8,9, "$\geqslant$10"], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize = 20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.A.Patamon.pdf")
    plt.show()
    # plot the number of ghosts eaten in a round (Omega)
    omega_round_eaten_ghost_all = [np.sum(omega_record["round_eaten_ghosts"][each]) for each in omega_record["round_eaten_ghosts"]]
    patamon_round_eaten_ghost_all = [np.sum(patamon_record["round_eaten_ghosts"][each]) for each in patamon_record["round_eaten_ghosts"]]
    omega_count = np.zeros((9,))
    patamon_count = np.zeros((9, ))
    for i in omega_round_eaten_ghost_all:
        omega_count[int(i)] += 1
    for i in patamon_round_eaten_ghost_all:
        patamon_count[int(i)] += 1
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.bar(np.arange(9), omega_count, align="center", color = light_grey, width = 0.99, linewidth = 0.45, edgecolor = "k")
    plt.axvline(np.mean(omega_round_eaten_ghost_all), lw=3, linestyle="--", color="k")
    plt.text(x=np.mean(omega_round_eaten_ghost_all), y=ax.get_ylim()[1] + 15,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(omega_round_eaten_ghost_all), np.std(omega_round_eaten_ghost_all)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Times of Eating Ghosts in One Game", fontsize=20)
    plt.xticks(np.arange(9), np.arange(9), fontsize=20)
    plt.ylim(0, 500)
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize = 20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.F.Omega.pdf")
    plt.show()
    # plot the number of ghosts eaten in a round (Patamon)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.bar(np.arange(9), patamon_count, align="center", color=light_grey, width = 0.99, linewidth = 0.45, edgecolor = "k")
    plt.axvline(np.mean(patamon_round_eaten_ghost_all), lw=3, linestyle="--", color="k")
    plt.text(x=np.mean(patamon_round_eaten_ghost_all), y=ax.get_ylim()[1] + 15,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(patamon_round_eaten_ghost_all),
                                              np.std(patamon_round_eaten_ghost_all)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Times of Eating Ghosts in One Game", fontsize=20)
    plt.xticks(np.arange(9), np.arange(9), fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize = 20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.F.Patamon.pdf")
    plt.show()
    # plot duration of a game (Omega)
    omega_round_time = np.array([sum(each) for each in list(omega_record["round_tile"].values())]) * (25 / 60)
    patamon_round_time = np.array([sum(each) for each in list(patamon_record["round_tile"].values())]) * (25 / 60)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    bins = list(np.arange(100, 256, 5))
    bins_index = [str(int(i)) for i in bins]
    bins_index[-1] = r'$\infty$'
    sel_index = np.arange(0, len(bins_index), 5)
    sns.histplot(omega_round_time, bins= bins, color=light_grey, edgecolor="k", linewidth=0.5, alpha = 1)
    plt.axvline(np.mean(omega_round_time), lw=3, color="k", linestyle="--")
    plt.text(x=np.mean(omega_round_time), y=ax.get_ylim()[1] + 10,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(omega_round_time), np.std(omega_round_time)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Duration of a Game (seconds)", fontsize=20)
    plt.xticks(
        list(np.array(bins)[sel_index]) + [plt.gca().get_xlim()[1]],
        list(np.array(bins_index)[sel_index]) + [r'$\infty$'],
        fontsize=20
    )
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.E.Omega.pdf")
    plt.show()
    # plot duration of a game (Patamon)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    patamon_round_time = [each if each < 250 else 250 for each in patamon_round_time]
    sns.histplot(patamon_round_time, bins= bins, color=light_grey, edgecolor="k", linewidth=0.5, alpha = 1)
    plt.axvline(np.mean(patamon_round_time), lw=3, color="k", linestyle="--")
    plt.text(x=np.mean(patamon_round_time), y=ax.get_ylim()[1] + 10,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(patamon_round_time), np.std(patamon_round_time)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Duration of a Game (seconds)", fontsize=20)
    plt.xticks(
        list(np.array(bins)[sel_index]) + [plt.gca().get_xlim()[1]],
        list(np.array(bins_index)[sel_index]) + [r'$\infty$'],
        fontsize=20
    )
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.E.Patamon.pdf")
    plt.show()
    # plot number of games per day (Omega)
    omega_num_game_per_day = [len(omega_record["game_per_day"][each]) for each in omega_record["game_per_day"]]
    patamon_num_game_per_day = [len(patamon_record["game_per_day"][each]) for each in patamon_record["game_per_day"]]
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    bins = [0, 10, 20, 30, 40, 50, 60]
    omega_bin = np.histogram(omega_num_game_per_day, bins)
    patamon_bin = np.histogram(patamon_num_game_per_day, bins)
    plt.bar(np.arange(6), omega_bin[0], align = "edge", color=light_grey, width = 0.99, linewidth = 0.45, edgecolor = "k")
    plt.axvline(np.mean(omega_num_game_per_day) / 10, lw=3, color="k", linestyle="--")
    plt.text(x=np.mean(omega_num_game_per_day) / 10, y=ax.get_ylim()[1] + 1.5,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(omega_num_game_per_day), np.std(omega_num_game_per_day)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Number of Games per Day", fontsize=20)
    plt.xticks(np.arange(7), bins, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.B.Omega.pdf")
    plt.show()
    # plot number of games per day (Patamon)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.bar(np.arange(6), patamon_bin[0], align = "edge", color=light_grey, width = 0.99, linewidth = 0.45, edgecolor = "k")
    plt.axvline(np.mean(patamon_num_game_per_day) / 10, lw=3, color="k", linestyle="--")
    plt.text(x=np.mean(patamon_num_game_per_day) / 10, y=ax.get_ylim()[1] + 1.5,
             s="{:.1f} ($\pm$ {:.1f})".format(np.mean(patamon_num_game_per_day),
                                              np.std(patamon_num_game_per_day)),
             fontsize=18, ha="center", va="center")
    plt.xlabel("Number of Games per Day", fontsize=20)
    plt.xticks(np.arange(7), bins, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.B.Patamon.pdf")
    plt.show()
    # plot average round number per game on a certain day (Omega)
    omega_num_game_per_day = {each:len(omega_record["game_per_day"][each]) for each in omega_record["game_per_day"]}
    patamon_num_game_per_day = {each:len(patamon_record["game_per_day"][each]) for each in patamon_record["game_per_day"]}
    omega_day = list(omega_num_game_per_day.keys())
    patamon_day = list(patamon_num_game_per_day.keys())
    omega_round_per_game_per_day = {each:[] for each in omega_record["game_per_day"]}
    for each in omega_record["round_num"]:
        tmp = each.split("-")
        if "extra" in tmp:
            tmp.remove("extra")
        omega_round_per_game_per_day["-".join(tmp[1:])].append(len(omega_record["round_num"][each]))
    omega_round_std = {each: np.std(omega_round_per_game_per_day[each])/len(omega_round_per_game_per_day[each]) for each in omega_round_per_game_per_day}
    omega_round_per_game_per_day = {each: np.mean(omega_round_per_game_per_day[each]) for each in omega_round_per_game_per_day}

    patamon_round_per_game_per_day = {each: [] for each in patamon_record["game_per_day"]}
    for each in patamon_record["round_num"]:
        tmp = each.split("-")
        if "extra" in tmp:
            tmp.remove("extra")
        patamon_round_per_game_per_day["-".join(tmp[1:])].append(len(patamon_record["round_num"][each]))
    patamon_round_std = {each: np.std(patamon_round_per_game_per_day[each])/len(patamon_round_per_game_per_day[each]) for each in patamon_round_per_game_per_day}
    patamon_round_per_game_per_day = {each: np.mean(patamon_round_per_game_per_day[each]) for each in patamon_round_per_game_per_day}
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.errorbar(
        [omega_num_game_per_day[each] for each in omega_day],
        [omega_round_per_game_per_day[each] for each in omega_day],
        yerr = [omega_round_std[each] for each in omega_day],
        linestyle="", ms=10, elinewidth=2,
        marker="o",
        c = light_grey,
    )
    plt.xlabel("Game per Day", fontsize=20)
    plt.ylabel("Average Round Number \n per Game on that Day", fontsize = 20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.C.Omega.pdf")
    plt.show()
    # plot average round number per game on a certain day (Patamon)
    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.errorbar(
        [patamon_num_game_per_day[each] for each in patamon_day],
        [patamon_round_per_game_per_day[each] for each in patamon_day],
        yerr=[patamon_round_std[each] for each in patamon_day],
        linestyle="", ms=10, elinewidth=2,
        marker="o",
        c=light_grey,
    )
    plt.xlabel("Game per Day", fontsize=20)
    plt.ylabel("Average Round Number \n per Game on that Day", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.C.Patamon.pdf")
    plt.show()
    # plot average round number on a certain day
    def dateStr(d):
        d = d.split("-")
        d = d[1:]
        d[1] = str(datetime.datetime.strptime(d[1], "%b").month)
        return datetime.datetime.strptime("-".join(d), '%d-%m-%Y')

    omega_num_game_per_day = {each:len(omega_record["game_per_day"][each]) for each in omega_record["game_per_day"]}
    patamon_num_game_per_day = {each:len(patamon_record["game_per_day"][each]) for each in patamon_record["game_per_day"]}
    omega_day = sorted(list(omega_num_game_per_day.keys()), key = lambda x : dateStr(x))
    patamon_day = sorted(list(patamon_num_game_per_day.keys()), key = lambda x : dateStr(x))

    omega_round_per_game_per_day = {each:[] for each in omega_record["game_per_day"]}
    for each in omega_record["round_num"]:
        tmp = each.split("-")
        if "extra" in tmp:
            tmp.remove("extra")
        omega_round_per_game_per_day["-".join(tmp[1:])].append(len(omega_record["round_num"][each]))
    omega_round_std = {each: np.std(omega_round_per_game_per_day[each])/len(omega_round_per_game_per_day[each]) for each in omega_round_per_game_per_day}
    omega_round_per_game_per_day = {each: np.mean(omega_round_per_game_per_day[each]) for each in omega_round_per_game_per_day}
    patamon_round_per_game_per_day = {each: [] for each in patamon_record["game_per_day"]}
    for each in patamon_record["round_num"]:
        tmp = each.split("-")
        if "extra" in tmp:
            tmp.remove("extra")
        patamon_round_per_game_per_day["-".join(tmp[1:])].append(len(patamon_record["round_num"][each]))
    patamon_round_std = {each: np.std(patamon_round_per_game_per_day[each])/len(patamon_round_per_game_per_day[each]) for each in patamon_round_per_game_per_day}
    patamon_round_per_game_per_day = {each: np.mean(patamon_round_per_game_per_day[each]) for each in patamon_round_per_game_per_day}

    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(
        np.arange(len(omega_day)),
        [omega_round_per_game_per_day[each] for each in omega_day],
        color=light_grey,
        linewidth=2.5,
    )
    plt.fill_between(
        np.arange(len(omega_day)),
        np.array([omega_round_per_game_per_day[each] for each in omega_day]) - np.array([omega_round_std[each] for each in omega_day]),
        np.array([omega_round_per_game_per_day[each] for each in omega_day]) + np.array([omega_round_std[each] for each in omega_day]),
        color=light_grey,
        alpha=0.2,
        linewidth=1.0
    )
    plt.ylabel("Average Round Number on that Day", fontsize = 20)
    tmp_omega_day = ["-".join(each.split("-")[1:]) for each in omega_day]
    temp_omega_idx = [1, 18, 33, 50]
    plt.xticks(np.arange(len(tmp_omega_day))[temp_omega_idx], np.array(tmp_omega_day)[temp_omega_idx], fontsize=15)
    plt.ylim(0, 8)
    plt.yticks(np.arange(9), np.arange(9), fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.D.Omega.pdf")
    plt.show()

    plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(
        np.arange(len(patamon_day)),
        [patamon_round_per_game_per_day[each] for each in patamon_day],
        color=light_grey,
        linewidth=2.5,
    )
    plt.fill_between(
        np.arange(len(patamon_day)),
        np.array([patamon_round_per_game_per_day[each] for each in patamon_day]) - np.array(
            [patamon_round_std[each] for each in patamon_day]),
        np.array([patamon_round_per_game_per_day[each] for each in patamon_day]) + np.array(
            [patamon_round_std[each] for each in patamon_day]),
        color=light_grey,
        alpha=0.2,
        linewidth=1.0
    )
    plt.ylabel("Average Round Number on that Day", fontsize=20)
    temp_patamon_day = ["-".join(each.split("-")[1:]) for each in patamon_day]
    temp_patamon_idx = [2, 15, 28, 38]
    plt.xticks(np.arange(len(temp_patamon_day))[temp_patamon_idx], np.array(temp_patamon_day)[temp_patamon_idx], fontsize=15)
    plt.ylim(0, 8)
    plt.yticks(np.arange(9), np.arange(9), fontsize=20)
    plt.tight_layout()
    plt.savefig(pic_base + "/Supp2.D.Patamon.pdf")
    plt.show()



if __name__ == '__main__':
    monkey = "all"

    # Fig 1: Basics
    plotFig1B(monkey)
    plotFig1C(monkey)

    # Fig 2: Strategy Fitting
    plotFig2B(monkey)
    plotFig2CDE(monkey)
    plotFig2F(monkey)

    # Fig 3: Strategy Patterns
    plotFig3AB(monkey)
    plotFig3CD(monkey)
    plotFig3E(monkey)
    plotFig3F(monkey)

    # Fig 4: Plan (Planned Attack vs. Accidental Assumption & Suicide vs. Failure)
    plotFig4B(monkey)
    plotFig4DEHI(monkey)
    plotFig4CFG(monkey)

    # Supp Fig. 2: Basic Statistics
    plotBasicStatistics()

