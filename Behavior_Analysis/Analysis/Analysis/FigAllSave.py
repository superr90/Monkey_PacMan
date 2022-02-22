'''
Description:
    Data analysis. Analyzing results are saved in files and used for the figure plotting.
'''
import numpy as np
import pandas as pd
import pickle
from more_itertools import consecutive_groups as cg
import copy
import scipy.stats

import sys
sys.path.append("../../../")
from Utils.ComputationUtils import _estimationVagueLabeling2, _closestScaredDist
from Utils.FileUtils import readLocDistance, readAdjacentMap
from Utils.FigUtils import add_states, consecutive_groups, vague_consecutive_groups, \
    evade_consecutive_groups, how_many_turns, how_many_turns_poss, \
    add_dis, generate_planned_accidental, add_PEG_dis, generate_suicide_normal_next, \
    z_score_wo_outlier2, extend_df_overlap, go_to_most_beans, toward_ghost_table, add_possible_dirs, \
    pupil_consecutive_groups, seq_center
from Utils.FigUtils import TURNING_POS, LOCS_DF


# =======================================================
# Directory for saving analyzing results
save_base = "../../../Data/Behavior_Plot_Data/"
# Load map data
LOCS_DF_4_CONTEXT = readLocDistance("../../../Data/Constants/dij_distance_map.csv")
# Load model fitted data for analyses
print("="*50)
print("Start reading data...")
df = pd.read_pickle(
    "/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/all_trials-merge_weight-new_sacc-dynamic-res.pkl"
    # "/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/all_trials-merge_weight-dynamic-res.pkl"
)
if isinstance(df, list):
    df = pd.concat([each[1] for each in df]).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)
df = df.drop(columns=["labels"], errors="ignore")
df = df.sort_values(by=["file", "index"])
if "level_0" in df.columns.values:
    df = df.drop(columns=["level_0"])
# Fitted labels
df_reset_comb = add_states(
    df.reset_index().drop("level_0", 1)
)
# Adjust "rt" positions
new_index = (
    df_reset_comb.loc[df_reset_comb[~df_reset_comb.rt.isnull()].index + 1, "pacmanPos"]
    .isin(TURNING_POS)
    .where(lambda x: x == True)
    .dropna()
    .index
)
df_reset_comb.loc[
    new_index, ["rt", "rt_std", "first_pos", "turns_previous", "previous_steps"]
] = df_reset_comb.loc[
    new_index - 1, ["rt", "rt_std", "first_pos", "turns_previous", "previous_steps"]
].values
df_reset_comb.loc[
    new_index - 1, ["rt", "rt_std", "first_pos", "turns_previous", "previous_steps"]
] = [np.nan] * 5
print("Finished reading data with the shape of {}.".format(df_reset_comb.shape))
print("="*50)

# =======================================================

def _readData(monkey):
    '''
    Extract data for a certain monkey.
    :param monkey: (str) Th monkey name ("Omega", "Patamon").
    :return: (pandas.DataFrame) A table of monkey data.
    '''
    if monkey in ["Omega", "Patamon"]:
        df_total = copy.deepcopy(df_reset_comb[df_reset_comb.file.str.contains(monkey)]).reset_index(drop = True)
    elif monkey =="all":
        df_total = copy.deepcopy(df_reset_comb)
    else:
        raise ValueError("Unknow monkey name {}!".format(monkey))
    print("Data shape : {}.".format(df_total.shape))
    return df_total

# =======================================================

def rewardOrientation(monkey):
    '''
    Analyze the probability of Pacman moving towards rewards.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("=" * 50)
    print("Reward Orientation for {} Data".format(monkey))
    df_total = _readData(monkey)
    # -----------------------------------------
    print("-" * 50)
    print("Fig 1B")
    landscape = True
    exclude_2dirs = False
    only_cross_fork = False
    cate_df = pd.DataFrame(
        [
            [2, False, "straight"],
            [2, True, "L-shape"],
            [3, True, "fork"],
            [4, True, "cross"],
        ],
        columns=["NextNum", "if_cross", "category"],
    )
    df_overlap_file = pd.read_pickle(
        "../../Data/constant/df_overlap_" + "omega" + ".pkl"
    ).append(pd.read_pickle("../../Data/constant/df_overlap_" + "patamon" + ".pkl"))
    df_overlap = extend_df_overlap(df_total, df_total.index > -1, df_overlap_file, )
    result_df, result_trial_df_list, result_game_df_list, result_day_df_list = go_to_most_beans(
        df_overlap,
        cate_df,
        "",
        only_cross_fork,
        exclude_2dirs,
        landscape,
    )
    print("Data shape : ", result_df.shape)
    with open(save_base + "{}_1.B.pkl".format(monkey), "wb") as file:
        pickle.dump(result_df, file)
    with open(save_base + "{}_1.B.trial.pkl".format(monkey), "wb") as file:
        pickle.dump(result_trial_df_list, file)
    with open(save_base + "{}_1.B.game.pkl".format(monkey), "wb") as file:
        pickle.dump(result_game_df_list, file)
    with open(save_base + "{}_1.B.day.pkl".format(monkey), "wb") as file:
        pickle.dump(result_day_df_list, file)
    # -----------------------------------------
    print("-" * 50)
    print("Fig 1C")
    df_sub = copy.deepcopy(df_total)
    df_sub = add_possible_dirs(df_sub)
    all_rs, result_trial_rs, result_game_rs, result_day_rs = toward_ghost_table(df_sub)
    print("Data shape : ", all_rs.shape)
    with open(save_base + "{}_1.C.pkl".format(monkey), "wb") as file:
        pickle.dump(all_rs, file)
    with open(save_base + "{}_1.C.trial.pkl".format(monkey), "wb") as file:
        pickle.dump(result_trial_rs, file)
    with open(save_base + "{}_1.C.game.pkl".format(monkey), "wb") as file:
        pickle.dump(result_game_rs, file)
    with open(save_base + "{}_1.C.day.pkl".format(monkey), "wb") as file:
        pickle.dump(result_day_rs, file)


def StrategyHeuristic(monkey):
    '''
    Analyze the fitted strategy heuristic.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("=" * 50)
    print("Strategy Heuristic for {} Data".format(monkey))
    df_total = _readData(monkey)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 2DE (1st - 2nd largest weight distribution) : ")
    largest_weights = [df_total.contribution.apply(lambda x: sorted(x)[-i]) for i in range(1, 4)]
    with open(save_base + "{}_2DE.detail_dist.pkl".format(monkey), "wb") as file:
        pickle.dump(largest_weights, file)
    values = df_total.contribution.apply(lambda x: sorted(x)).values
    data = pd.Series(np.asarray([each[-1] for each in values]) - np.asarray([each[-2] for each in values]))
    print("plot data shape : ", data.shape)
    data = {"values": values, "series": data}
    with open(save_base + "{}_2DE.pkl".format(monkey), "wb") as file:
        pickle.dump(data, file)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 2F (dominating strategy in different contexts): ")
    all_type = {
        "all": [],
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": [],
    } # different contexts
    all_X = copy.deepcopy(df_total)
    early_thr = 80 if monkey == "Omega" else 65
    ending_thr = 10 if monkey == "Omega" else 7
    end_index = all_X.beans.apply(lambda x: len(x) <= ending_thr if not isinstance(x, float) else True)
    end_index = np.where(end_index == True)[0]
    early_index = all_X.beans.apply(lambda x: len(x) >= early_thr if not isinstance(x, float) else False)
    early_index = np.where(early_index == True)[0]
    middle_index = all_X.beans.apply(
        lambda x: ending_thr < len(x) < early_thr if not isinstance(x, float) else False)
    middle_index = np.where(middle_index == True)[0]
    scared_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 > 3 or x.ifscared2 > 3, axis=1)
    scared_index = np.where(scared_index == True)[0]
    normal_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 or x.ifscared2 < 3, axis=1)
    normal_index = np.where(normal_index == True)[0]
    close_index = all_X[["pacmanPos", "ghost1Pos"]].apply(
        lambda x: True if x.pacmanPos == x.ghost1Pos else LOCS_DF_4_CONTEXT[x.pacmanPos][x.ghost1Pos] <= 10,
        axis=1
    )
    close_index = np.where(close_index == True)[0]
    all_type["early"].append(copy.deepcopy(early_index))
    all_type["middle"].append(copy.deepcopy(middle_index))
    all_type["end"].append(copy.deepcopy(end_index))
    all_type["close-normal"].append(copy.deepcopy(np.intersect1d(close_index, normal_index)))
    all_type["close-scared"].append(copy.deepcopy(np.intersect1d(close_index, scared_index)))
    all_type["all"].append(copy.deepcopy(np.arange(all_X.shape[0])))
    print("Finished assigning contexts.")
    result = {}
    for type in all_type:
        data = df_total.loc[all_type[type][0]]
        result[type] = data.labels.replace({"suicide": "approach"}).value_counts(
            normalize=True
        )
    with open(save_base + "{}_2F.pkl".format(monkey), "wb") as file:
        pickle.dump(result, file)


def Trajectory(monkey):
    '''
    Compare the monkey's trajectory with the best trajectory.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("="*50)
    print("Trajectory Comparison for {} Data".format(monkey))
    df_total = _readData(monkey)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 3C")
    global_lists = [
        list(i)
        for i in cg(df_total[df_total.labels == "global"].index)
    ]
    global_lists = list(filter(lambda x: len(x) >= 4, global_lists))
    # find the best trajectory
    df_temp = df_total.loc[
        [i[0] for i in global_lists], ["file", "rwd_cnt"]
    ].drop_duplicates(keep="first")
    df_temp["rwd_cnt"] = df_temp["rwd_cnt"] - 1
    df_temp["end_index"] = df_temp.merge(
        df_total[["file", "rwd_cnt"]].drop_duplicates(keep="first").reset_index(),
        on=["file", "rwd_cnt"],
        how="left",
    )["index"].values
    start_end_index = (
        df_temp.reset_index()[["index", "end_index"]]
            .dropna()
            .astype(int)
            .rename(columns={"index": "start_index"})
    )
    # find monkey's actual trajectory
    actual_routes = (
        df_total["pacmanPos"]
            .reset_index()
            .merge(
            df_total["pacmanPos"]
                .reset_index()
                .merge(start_end_index, left_on="index", right_on="start_index"),
            left_on="index",
            right_on="end_index",
            suffixes=["_end", "_start"],
        )
            .drop(columns=["index_end", "index_start"])
    )
    actual_routes["actual_length"] = actual_routes.end_index - actual_routes.start_index
    actual_routes = add_dis(actual_routes, "pacmanPos_start", "pacmanPos_end")
    # difference between actual and the best trajectory
    ylim = 10
    data = (
        (actual_routes.actual_length - actual_routes.dis)
            .where(lambda x: x <= ylim)
            .value_counts()
            .sort_index()
            .reset_index()
    )
    print("3C Data shape : ", data.shape)
    with open(save_base + "{}_3C.pkl".format(monkey), "wb") as file:
        pickle.dump(data, file)
    with open(save_base + "{}_3C.detail.pkl".format(monkey), "wb") as file:
        pickle.dump({"actual": actual_routes.actual_length, "dis": actual_routes.dis}, file)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 3D")
    # the number of turns of monkey's actual trajectory
    actual_routes["actual_turns"] = actual_routes.apply(
        lambda x: how_many_turns(df_total, x["start_index"], x["end_index"]), 1
    )
    actual_routes = actual_routes.merge(
        LOCS_DF[["pos1", "pos2", "path"]],
        left_on=["pacmanPos_start", "pacmanPos_end"],
        right_on=["pos1", "pos2"],
    )
    # the number of turns of the best trajectory
    actual_routes["fewest_turns"] = actual_routes.apply(
        lambda x: min([how_many_turns_poss(i) for i in x.path]), 1
    )
    data = (
        (actual_routes.actual_turns - actual_routes.fewest_turns)
            .where(lambda x: x >= 0)
            .dropna()
            .apply(lambda x: int(min(x, 5)))
            .value_counts()
            .sort_index()
            .reset_index()
    )
    print("3D Data shape : ", data.shape)
    with open(save_base + "{}_3D.pkl".format(monkey), "wb") as file:
        pickle.dump(data, file)
    with open(save_base + "{}_3D.detail.pkl".format(monkey), "wb") as file:
        pickle.dump({
            "actual": actual_routes.actual_turns,
            "fewest": actual_routes.fewest_turns,
            "path": actual_routes.path
        }, file)
    print("=" * 50)


def NonConflictingSaccade(monkey):
    '''
    Analyze the saccade fixation ratio.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("=" * 50)
    print("Non-Conflicting Saccade for {} Data".format(monkey))
    df_total = _readData(monkey)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 3E : ")
    state_list = ["global", "local", "evade", "approach", "energizer"]
    trial_saccade_num = {each: [] for each in state_list}
    for state in state_list:
        print("-" * 50)
        print(state)
        if state == "vague":
            sel_index = vague_consecutive_groups(df_total)
        elif state == "evade":
            sel_index = evade_consecutive_groups(df_total)
        else:
            sel_index = consecutive_groups(df_total, state)
        for w in [
            "ghost",
            "energizer_sacc",
            "beans_sacc",
        ]:
            if w == "ghost":
                tmp = (
                    (
                        pd.Series(sel_index)
                            .explode()
                            .rename("sec_level_1")
                            .reset_index()
                            .merge(
                            df_total["on_ghost_all"]
                                .reset_index(),
                            left_on="sec_level_1",
                            right_on="index",
                        )
                            .groupby("index_x")["on_ghost_all"]
                            .mean()
                    )
                )
            elif w == "on_map_saccade":
                tmp = (
                    (
                        pd.Series(sel_index)
                            .explode()
                            .rename("sec_level_1")
                            .reset_index()
                            .merge(
                            df_total["on_map_all"]
                            .reset_index(),
                            left_on="sec_level_1",
                            right_on="index",
                            )
                            .groupby("index_x")["on_map_all"]
                            .mean()
                    )
                )
            else:
                tmp = (
                    (
                        pd.Series(sel_index)
                            .explode()
                            .rename("sec_level_1")
                            .reset_index()
                            .merge(
                            df_total[w].reset_index(),
                            left_on="sec_level_1",
                            right_on="index",
                        )
                            .groupby("index_x")[w]
                            .mean()
                    )
                )
            trial_saccade_num[state].append(
                [w, np.sum(tmp), len(tmp), np.mean(tmp), np.std(tmp) / np.sqrt(len(tmp))])
    with open(save_base + "{}_3E.pkl".format(monkey), "wb") as file:
        pickle.dump(trial_saccade_num, file)
    print("="*50)


def PAvsAA(monkey):
    '''
    Compare planned attack with accidental assumoption.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("="*50)
    print("PA vs. AA for {} Data".format(monkey))
    df_total = _readData(monkey)
    non_conflicting_df_total = _readData(monkey)
    planned_lists, accidental_lists, planned_all, accidental_all = generate_planned_accidental(df_total)
    print("The num of planned data : ", len(planned_lists)) # PA 数据中最终吃到了鬼的数据数量
    print("The num of accidental data : ", len(accidental_lists)) # AA 数据中最终吃到了鬼的数据数量
    print("The num of planned all data : ", len(planned_all)) # 所有 PA 数据的数量
    print("The num of accidental all data : ", len(accidental_all)) # 所有 AA 数据的数量
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 5C:")
    df_reset_comb_extend = add_PEG_dis(
        df_total.drop(columns=["PE_dis", "EG1_dis", "EG2_dis"])
    )
    mapping = {1: "Accidentally Hunting", 2: "Planned Hunting"}
    ghost_data = {"1": None, "2": None}
    all_ghost_data = {"1": None, "2": None}
    for ghost in ["1", "2"]:
        i = 1
        df_status = pd.DataFrame()
        all_df_status = pd.DataFrame()
        for sel_list in [accidental_lists, planned_lists]:
            filter_sel_list = pd.Series(sel_list)[
                pd.Series(sel_list)
                    .explode()
                    .reset_index()
                    .set_index(0)
                    .reset_index()
                    .merge(df_reset_comb_extend.reset_index(), left_on=0, right_on="level_0")
                    .groupby("index_x")
                    .apply(lambda x: (x.next_eat_rwd.count() / len(x) <= 0.2))
                    .values
            ]
            x = df_reset_comb_extend.loc[
                (
                    df_reset_comb_extend.index.isin(
                        filter_sel_list.apply(lambda x: x[0]).values.tolist()
                    )
                )
                & (
                        (df_reset_comb_extend.ifscared1 <= 2)
                        | (df_reset_comb_extend.ifscared2 <= 2)
                )
                & (df_reset_comb_extend.PE_dis <= 10)
                & (df_reset_comb_extend["index"] > 0),
                ["distance" + ghost, "EG" + ghost + "_dis", "PE_dis"],
            ]
            df_status = df_status.append(x.reset_index().assign(cate=mapping[i]))

            all_sel_list = pd.Series(sel_list)[
                pd.Series(sel_list)
                    .explode()
                    .reset_index()
                    .set_index(0)
                    .reset_index()
                    .merge(df_reset_comb_extend.reset_index(), left_on=0, right_on="level_0")
                    .groupby("index_x")
                    .apply(lambda x: True)
                    .values
            ]
            all_x = df_reset_comb_extend.loc[
                (
                    df_reset_comb_extend.index.isin(
                        all_sel_list.apply(lambda x: x[0]).values.tolist()
                    )
                ),
                ["distance" + ghost, "EG" + ghost + "_dis", "PE_dis"],
            ]
            all_df_status = all_df_status.append(all_x.reset_index().assign(cate=mapping[i]))
            i += 1
        df_status = df_status.reset_index().drop(columns="level_0")
        all_df_status = all_df_status.reset_index().drop(columns="level_0")

        ghost_data[ghost] = df_status
        all_ghost_data[ghost] = all_df_status
        print("Ghost {} data shape {}".format(ghost, df_status.shape))
    with open(save_base + "{}_5B.pkl".format(monkey), "wb") as file:
        pickle.dump(ghost_data, file)
    with open(save_base + "{}_5B.all.pkl".format(monkey), "wb") as file:
        pickle.dump(all_ghost_data, file)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 5A:")
    planned_weight = np.zeros((len(planned_lists), 6, 20))
    planned_weight[planned_weight == 0] = np.nan
    for index, each in enumerate(planned_lists):
        # Before eating energizers
        prev_start = max(0, each[-1] - 10)
        contributions = df_total.contribution[np.arange(prev_start, each[-1])].values
        for i in range(contributions.shape[0]):
            contributions[i] = np.array(contributions[i]) / np.linalg.norm(np.array(contributions[i]))
            planned_weight[index, :, 10 - len(contributions) + i] = copy.deepcopy(contributions[i])
        # After eating energizers
        after_end = each[-1] + 10
        contributions = df_total.contribution[np.arange(each[-1], after_end)].values
        for i in range(contributions.shape[0]):
            contributions[i] = np.array(contributions[i]) / np.linalg.norm(np.array(contributions[i]))
            planned_weight[index, :, 10 + i] = copy.deepcopy(contributions[i])

    max_length = np.max([len(each) for each in accidental_lists])
    print("Max for accidental : ", max_length)
    accidental_weight = np.zeros((len(accidental_lists), 6, 20))
    accidental_weight[accidental_weight == 0] = np.nan
    for index, each in enumerate(accidental_lists):
        # Before eating energizers
        prev_start = max(0, each[-1] - 10)
        contributions = df_total.contribution[np.arange(prev_start, each[-1])].values
        for i in range(contributions.shape[0]):
            contributions[i] = np.array(contributions[i]) / np.linalg.norm(np.array(contributions[i]))
            accidental_weight[index, :, 10 - len(contributions) + i] = copy.deepcopy(contributions[i])
        # After eating energizers
        after_end = each[-1] + 10
        contributions = df_total.contribution[np.arange(each[-1], after_end)].values
        for i in range(contributions.shape[0]):
            contributions[i] = np.array(contributions[i]) / np.linalg.norm(np.array(contributions[i]))
            accidental_weight[index, :, 10 + i] = copy.deepcopy(contributions[i])
    print("Planned weight shape : ", planned_weight.shape)
    print("Accidental weight shape : ", accidental_weight.shape)
    with open(save_base + "{}_5A_planned_weight.pkl".format(monkey), "wb") as file:
        pickle.dump(planned_weight, file)
    with open(save_base + "{}_5A_accidental_weight.pkl".format(monkey), "wb") as file:
        pickle.dump(accidental_weight, file)
    # ----------------------------------
    print("-" * 50)
    mapping = {
        "Planned Attack": planned_lists,
        "Accidental Attack": accidental_lists
    }
    print("For Fig. 5C:")
    trial_saccade_num = {each: [] for each in mapping}
    for state in mapping:
        print("-" * 50)
        print(state)
        sel_index = mapping[state]
        for w in [
            "pacman_sacc",
            "on_ghost_all",
            "energizer_sacc"
        ]:
            tmp = (
                pd.Series(sel_index)
                    .explode()
                    .rename("sec_level_1")
                    .reset_index()
                    .merge(
                    non_conflicting_df_total[w].reset_index(), left_on="sec_level_1", right_on="index",
                )
                    .groupby("index_x")[w]
                    .mean()
            )
            trial_saccade_num[state].append(
                [
                    w,
                    np.sum(tmp),
                    len(tmp),
                    np.mean(tmp),
                    np.std(tmp) / np.sqrt(len(tmp)),
                    tmp.values
                ]
            )
    with open(save_base + "{}_5C.ghost_all.pkl".format(monkey), "wb") as file:
        pickle.dump(trial_saccade_num, file)
    # pupil size
    print("For Fig. 5D:")

    def rt_before_after_eye(last_index_list, df_total, rt, cutoff, col, cond_col=None):
        after_df, before_df = pd.DataFrame(), pd.DataFrame()
        for i in last_index_list:
            file, index = df_total.iloc[i]["file"], df_total.iloc[i]["index"]
            before = rt[(rt.file == file) & (rt["index"] < index)].iloc[-cutoff:][
                [col, cond_col]
            ]
            before = before[col].mask(before[cond_col] == 0)
            before = pd.Series(before.values, index=range(before.shape[0], 0, -1))
            before_df = pd.concat([before_df, before], 1)

            after = rt[(rt.file == file) & (rt["index"] > index)].iloc[:cutoff][
                [col, cond_col]
            ]
            after = after[col].mask(after[cond_col] == 0)
            after = pd.Series(after.values, index=range(1, after.shape[0] + 1))
            after_df = pd.concat([after_df, after], 1)
        return after_df, before_df

    accidental_all = [each[-1] for each in accidental_all]
    planned_all = [each[-1] for each in planned_all]

    df_total = z_score_wo_outlier2(df_total, "eye_size", 3, True).reset_index(drop=True)
    cutoff = 10
    all_data = {"accidental": None, "planned": None}
    name = ["accidental", "planned"]
    for index, compute_list in enumerate([accidental_all, planned_all]):
        after_df, before_df = rt_before_after_eye(
            compute_list,
            df_total,
            df_total,
            cutoff,
            "eye_size_z",
            cond_col="eye_size",
        )
        print(before_df.shape)
        print(after_df.shape)
        after_sts = pd.DataFrame(
            {
                "mean": after_df.mean(1).values,
                "std": after_df.std(1).values,
                "count": after_df.count(1).values,
            },
            index=range(1, cutoff + 1),
        )
        before_sts = pd.DataFrame(
            {
                "mean": before_df.mean(1).values,
                "std": before_df.std(1).values,
                "count": before_df.count(1).values,
            },
            index=range(-1, -cutoff - 1, -1),
        )
        df_plot = before_sts.append(after_sts).sort_index()
        print("Data shape of {} is {}".format(name[index], df_plot.shape))
        all_data[name[index]] = copy.deepcopy(df_plot)
    with open(save_base + "{}_5D.pkl".format(monkey), "wb") as file:
        pickle.dump(all_data, file)


def SuicidevsNormal(monkey):
    '''
    Compare suicide with failure death.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("=" * 50)
    print("Suicide vs. Normal for {} Data".format(monkey))
    df_total = _readData(monkey)
    non_conflicting_df_total = _readData(monkey)
    suicide_lists, normal_lists, other_lists, suicide_next_lists, normal_next_lists, other_next_lists = \
        generate_suicide_normal_next(
            df_total[["labels", "file", "game", "game_trial", "contribution", "ifscared1", "ifscared2"]],"normal"
        )
    print("The num of suicide data : ", len(suicide_lists))
    print("The num of normal data : ", len(normal_lists))
    print("The num of other data : ", len(other_lists))

    print("The num of suicide next data : ", len(suicide_next_lists))
    print("The num of normal next data : ", len(normal_next_lists))
    print("The num of other next data : ", len(other_next_lists))
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 6C : ")
    mapping = {
        "Suicide": suicide_lists,
        "Normal Die": normal_lists,
        "Others": other_lists,
    }
    # Saccade ratio
    print("Ratio")
    trial_saccade_num = {each: [] for each in mapping}
    for state in mapping:
        print("-" * 50)
        print(state)
        sel_index = mapping[state]
        for w in [
            "pacman_sacc",
            "on_ghost_all",
            "beans_sacc"
        ]:
            tmp = (
                pd.Series(sel_index)
                    .explode()
                    .rename("sec_level_1")
                    .reset_index()
                    .merge(
                    non_conflicting_df_total[w].reset_index(), left_on="sec_level_1", right_on="index",
                )
                    .groupby("index_x")[w]
                    .mean()
            )
            trial_saccade_num[state].append(
                [
                    w,
                    np.sum(tmp),
                    len(tmp),
                    np.mean(tmp),
                    np.std(tmp) / np.sqrt(len(tmp)),
                    tmp.values
                ]
            )
    with open(save_base + "{}_6C.ghost_all.pkl".format(monkey),"wb") as file:
        pickle.dump(trial_saccade_num, file)
    # Pupil Size
    print("Pupil size")
    df_total = z_score_wo_outlier2(df_total, "eye_size", 3, True).reset_index(drop=True)
    all_data = {"suicide": None, "normal": None, "other": None}
    name = ["suicide", "normal", "other"]
    xaxis = range(10, 1, -1)
    for index, compute_list in enumerate([suicide_lists, normal_lists, other_lists]):
        print("{} data num {}".format(name[index], len(compute_list)))
        data = [
            [
                df_total.iloc[j[-i]]["eye_size_z"]
                for j in compute_list
                if i <= len(j) and df_total.iloc[j[-i]]["eye_size"] != 0
            ]
            for i in xaxis
        ]
        print("data list")
        gpd = pd.DataFrame(
            [[np.nanmean(i), np.nanstd(i), len(i)] for i in data],
            columns=["mean", "std", "count"],
        )
        gpd.index = [round(-(i - 1) * 25 / 60, 2) for i in xaxis]
        all_data[name[index]] = gpd
        print("Data shape for {} : {}".format(name[index], gpd.shape))
    # save data
    with open(save_base + "{}_6D.pkl".format(monkey), "wb") as file:
        pickle.dump(all_data, file)


def statisticalTest(monkey):
    '''
    Statistical test for PA vs. AA and suicide vs. faiure.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("=" * 50)
    print("Statistical Test Data for {} Data".format(monkey))
    df_total = _readData(monkey)
    df_total = z_score_wo_outlier2(df_total, "eye_size", 3, True).reset_index(drop = True)
    non_conflicting_df_total = _readData(monkey)
    planned_lists, accidental_lists, planned_all, accidental_all = generate_planned_accidental(df_total)
    print("The num of planned data : ", len(planned_lists))
    print("The num of accidental data : ", len(accidental_lists))
    print("The num of planned all data : ", len(planned_all))
    print("The num of accidental all data : ", len(accidental_all))

    suicide_lists, normal_lists, other_lists, suicide_next_lists, normal_next_lists, other_next_lists = \
        generate_suicide_normal_next(
            df_total[["labels", "file", "game", "game_trial", "contribution", "ifscared1", "ifscared2"]], "normal"
        )
    print("The num of suicide data : ", len(suicide_lists))
    print("The num of normal data : ", len(normal_lists))
    print("The num of other data : ", len(other_lists))

    print("The num of suicide next data : ", len(suicide_next_lists))
    print("The num of normal next data : ", len(normal_next_lists))
    print("The num of other next data : ", len(other_next_lists))
    # ----------------------------------
    print("-" * 50)
    # Pupil size
    print("For 5D.detail:")

    def rt_before_after_eye(last_index_list, df_total, rt, cutoff, col, cond_col=None):
        after_df, before_df = pd.DataFrame(), pd.DataFrame()
        for i in last_index_list:
            file, index = df_total.iloc[i]["file"], df_total.iloc[i]["index"]
            before = rt[(rt.file == file) & (rt["index"] < index)].iloc[-cutoff:][
                [col, cond_col]
            ]
            before = before[col].mask(before[cond_col] == 0)
            before = pd.Series(before.values, index=range(before.shape[0], 0, -1))
            before_df = pd.concat([before_df, before], 1)

            after = rt[(rt.file == file) & (rt["index"] > index)].iloc[:cutoff][
                [col, cond_col]
            ]
            after = after[col].mask(after[cond_col] == 0)
            after = pd.Series(after.values, index=range(1, after.shape[0] + 1))
            after_df = pd.concat([after_df, after], 1)
        return after_df, before_df

    accidental_all = [each[-1] for each in accidental_all]
    planned_all = [each[-1] for each in planned_all]

    cutoff = 10
    all_data = {"accidental": None, "planned": None}
    name = ["accidental", "planned"]
    for index, compute_list in enumerate([accidental_all, planned_all]):
        after_df, before_df = rt_before_after_eye(
            compute_list,
            df_total,
            df_total,
            cutoff,
            "eye_size_z",
            cond_col="eye_size",
        )
        print(before_df.shape)
        print(after_df.shape)
        df_plot = before_df.set_index(np.arange(-len(before_df), 0, 1)).append(after_df).sort_index()
        print("Data shape of {} is {}".format(name[index], df_plot.shape))
        all_data[name[index]] = copy.deepcopy(df_plot)
    with open(save_base + "{}_5D.detail.pkl".format(monkey), "wb") as file:
        pickle.dump(all_data, file)
     # ----------------------------------
    # Pupil Size
    print("Pupil size")
    all_data = {"suicide": None, "normal": None, "other": None}
    name = ["suicide", "normal", "other"]
    xaxis = range(10, 1, -1)
    for index, compute_list in enumerate([suicide_lists, normal_lists, other_lists]):
        print("{} data num {}".format(name[index], len(compute_list)))
        data = [
            [
                df_total.iloc[j[-i]]["eye_size_z"]
                for j in compute_list
                if i <= len(j) and df_total.iloc[j[-i]]["eye_size"] != 0
            ]
            for i in xaxis
        ]
        print("data list")
        all_data[name[index]] = data
    # save data
    with open(save_base + "{}_6D.detail.pkl".format(monkey), "wb") as file:
        pickle.dump(all_data, file)
    # ----------------------------------
    print("-" * 50)
    print("For Fig. 6A.detail:")
    evade_lists = list(
        filter(
            lambda x: len(x) >= 3,
            [
                list(i)
                for i in cg(
                df_total[(df_total.labels == "evade_blinky") | (df_total.labels == "evade_clyde")].index
            )
            ],
        )
    )
    evade_indexes = [i[0] for i in evade_lists]

    df_temp = add_dis(
        df_total.loc[evade_indexes, :], "next_eat_rwd_fill", "pacmanPos",
    ).dis

    df1 = df_total.loc[evade_indexes, :]
    df1["resetpos"] = [(14, 27)] * df1.shape[0]

    df_temp_re = add_dis(df1, "resetpos", "next_eat_rwd_fill").dis

    suicide_lists = list(
        filter(
            lambda x: len(x) >= 3,
            [
                list(i)
                for i in cg(
                df_total[
                    (df_total.labels == "approach")
                    & (df_total[["ifscared1", "ifscared2"]].max(1) < 3)
                    ].index
            )
            ],
        )
    )
    suicide_start_index = [i[0] for i in suicide_lists]

    df_com = df_total[["file", "index", "next_eat_rwd_fill"]].merge(
        df_total.loc[suicide_start_index, "file"]
            .apply(
            lambda x: "-".join(
                [x.split("-")[0]] + [str(int(x.split("-")[1]) + 1)] + x.split("-")[2:]
            ),
        )
            .reset_index()
            .assign(
            index=[0] * len(suicide_start_index),
            reset_pos=[(14, 27)] * len(suicide_start_index),
            suicide_start_index=df_total.loc[suicide_start_index, "pacmanPos"].tolist(),
        ),
        on=["file", "index"],
    )

    df_com = add_dis(
        add_dis(df_com, "suicide_start_index", "next_eat_rwd_fill").rename(
            columns={"dis": "suicide_dis"}
        ),
        "next_eat_rwd_fill",
        "reset_pos",
    ).rename(columns={"dis": "reset_dis"})

    df_hist_suicide = (
        pd.cut(
            df_com.suicide_dis - df_com.reset_dis,
            bins=range(-38, 30, 2),
            labels=range(-36, 30, 2),
        )
            .value_counts(normalize=True)
            .rename("distance")
            .reset_index()
            .assign(category="suicide")
    )
    df_hist_suicide.category = "suicide > 0 ratio: " + str(
        round(df_hist_suicide[df_hist_suicide["index"] > 0].sum().distance, 2)
    )
    df_hist_evade = (
        pd.cut(df_temp - df_temp_re, bins=range(-38, 30, 2), labels=range(-36, 30, 2))
            .value_counts(normalize=True)
            .rename("distance")
            .reset_index()
            .assign(category="evade")
    )
    df_hist_evade.category = "evade > 0 ratio: " + str(
        round(df_hist_evade[df_hist_evade["index"] > 0].sum().distance, 2)
    )
    with open(save_base + "{}_6A.detail.pkl".format(monkey), "wb") as file:
        pickle.dump({"df_com": df_com, "df_temp": df_temp, "df_temp_re": df_temp_re}, file)
    # ----------------------------------
    evade_lists = list(
        filter(
            lambda x: len(x) >= 3,
            [
                list(i)
                for i in cg(
                df_total[(df_total.labels == "evade_blinky") | (df_total.labels == "evade_clyde")].index)
            ],
        )
    )
    evade_indexes = [i[0] - 1 for i in evade_lists]

    suicide_lists = list(
        filter(
            lambda x: len(x) >= 3,
            [
                list(i)
                for i in cg(
                df_total[
                    (df_total.labels == "approach")
                    & (df_total["ifscared1"] < 3)
                    & (df_total.file == df_total.file.shift())
                    ].index
            )
            ],
        )
    )
    suicide_start_index = [
        i[0] - 1
        for i in suicide_lists
        if (df_total.loc[i, "distance1"].diff() > 0).sum() < 1
    ]
    df_plot_all = {"evade": None, "suicide": None}
    PG_all = {"evade": None, "suicide": None}
    for k, target_index in {
        "evade": evade_indexes,
        "suicide": suicide_start_index,
    }.items():
        df_plot = (
            pd.Series(target_index)
                .explode()
                .rename("target_index")
                .reset_index()
                .merge(
                df_total["distance1"]
                    .reset_index()
                    .rename(columns={"index": "level_0"}),
                left_on="target_index",
                right_on="level_0",
            )
                .groupby("index")["distance1"]
                .mean()
                .rename(k)
        )
        df_plot_all[k] = copy.deepcopy(df_plot)
        PG_all[k] = df_total.iloc[target_index][["distance1", "distance2"]]
    # with open(save_base + "{}_6B.detail.pkl".format(monkey), "wb") as file:
    #     pickle.dump(df_plot_all, file)
    with open(save_base + "{}_6B.detail.PG.pkl".format(monkey), "wb") as file:
        pickle.dump(PG_all, file)


def strategyRatio(monkey):
    '''
    Computer strategy ratio in different contexts.
    :param monkey: (pandas.DataFrame) A table of monkey data.
    :return: VOID
    '''
    print("=" * 50)
    print("Strategy Ratio w.r.t. Descriptive Features")
    # Read data
    data = _readData(monkey)
    print("Data shape : ", data.shape)
    locs_df = readLocDistance("../../Data/constant/dij_distance_map.csv")
    strategy = data.contribution.apply(lambda x: _estimationVagueLabeling2(x)).values
    # Extract features
    print("-" * 50)
    print("Start extracting features...")
    scared_PG = data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _closestScaredDist(x.pacmanPos, x.ghost1Pos, x.ghost2Pos, x.ifscared1, x.ifscared2, locs_df),
        axis=1
    )
    min_scared_PG = np.nanmin(scared_PG[scared_PG < 999])
    max_scared_PG = np.nanmax(scared_PG[scared_PG < 999])
    print("| Scared PG | min = {}, max = {}".format(min_scared_PG, max_scared_PG))
    beans_10step = data[["pacmanPos", "beans"]].apply(
        lambda x: 0 if isinstance(x.beans, float) or len(x.beans) == 0
        else len(
            np.where(
                np.array([0 if x.pacmanPos == each
                          else locs_df[x.pacmanPos][each] for each in x.beans]) <= 10
            )[0]
        ),
        axis=1
    )
    min_beans_in_10 = int(np.nanmin(beans_10step))
    max_beans_in_10 = int(np.nanmax(beans_10step))
    # Assign strategy to features
    print("-" * 50)
    print("Begin assigning strategy to features...")
    agents = ["global", "local", "evade(Blinky)", "evade(Clyde)", "approach", "energizer", "vague"]
    scared_PG_bin = [[] for _ in range(0, 53 + 1)]
    for i, a in enumerate(scared_PG):
        if not pd.isna(a):
            if a < 999:
                scared_PG_bin[int(a - min_scared_PG)].append(agents.index(strategy[i]))
            else:
                continue
    beans_in10_bin = [[] for _ in range(0, 26 + 1)]
    for i, a in enumerate(beans_10step):
        if not pd.isna(a):
            beans_in10_bin[int(a - min_beans_in_10)].append(agents.index(strategy[i]))
    # save data
    np.save(
        save_base + "{}-strategy_ratio-resample.npy".format(monkey),
        {
            "scared-PG": scared_PG_bin,
            "beans_in10": beans_in10_bin,
        }
    )


def PupilSizeDynamic(monkey):
    print("="*50)
    print("Pupil Size for {} Data".format(monkey))
    df_total = _readData(monkey)
    # ----------------------------------
    print("-" * 50)
    print("Pupil Size Variation around Vague")
    for coef in [3, 4, 5]:
        print("*" * 50)
        print("STD coef : ", coef)
        df = z_score_wo_outlier2(df_total, "eye_size", coef, True).reset_index(drop=True)
        # Transition around vague
        vague_index, vague_trial_index = pupil_consecutive_groups(df, "vague")
        transition_type = {}
        length = 10
        for index, each in enumerate(vague_index):
            if len(each) == 0:
                continue
            vague_center = seq_center(each)
            # vague at the beginning or end of a trial
            if each[0] == vague_trial_index[index][0] or each[-1] == vague_trial_index[index][-1]:
                continue
            # otherwise
            before_center = list(
                np.arange(max(vague_trial_index[index][0], vague_center - length), vague_center))  # before center
            after_center = list(
                np.arange(vague_center, min(vague_center + length, vague_trial_index[index][-1])))  # after center
            if len(before_center) < length or len(after_center) < length:
                continue
            # transition type
            t_type = (df.loc[each[0] - 1].labels, df.loc[each[-1] + 1].labels)
            if t_type[0] == t_type[1] or isinstance(t_type[0], float) or isinstance(t_type[1], float):
                continue
            else:
                if t_type not in transition_type:
                    transition_type[t_type] = [[before_center, after_center]]
                else:
                    transition_type[t_type].append([before_center, after_center])
        transition_num = [(each, len(transition_type[each])) for each in transition_type]
        transition_num = sorted(transition_num, key=lambda x: x[1], reverse=True)
        print("Number of transition types : ", len(transition_num))

        print("Around vague : ")
        transition_pupil = {each: None for each in transition_type}
        for t_type in transition_type:
            t_index = transition_type[t_type]
            print("{} | {}".format(t_type, len(t_index)))
            type_vague_pupil = np.zeros((len(t_index), 2 * length))
            for ind, trial_index in enumerate(t_index):
                temp_pupil = np.zeros((2 * length,))
                temp_pupil[temp_pupil == 0] = np.nan
                before = list(trial_index[0])
                after = list(trial_index[1])
                t_data = df.loc[before + after]
                # eye_size_z = z_score(t_data, "eye_size")
                eye_size_z = t_data
                for i in range(len(before)):
                    if eye_size_z.loc[before[-1 - i], "eye_size_z"] != 0:
                        temp_pupil[length - 1 - i] = eye_size_z.loc[before[-1 - i], "eye_size_z"]
                for i in range(len(after)):
                    if eye_size_z.loc[after[i], "eye_size_z"] != 0:
                        temp_pupil[length + i] = eye_size_z.loc[after[i], "eye_size_z"]
                type_vague_pupil[ind] = temp_pupil
            transition_pupil[t_type] = type_vague_pupil

        # Transition without vague
        def transition_data(df_total, first_strategy, second_strategy, length):
            trial_index = df_total.groupby("file").apply(lambda x: x.index.to_list()).values
            sel_index = []
            for trial in trial_index:
                trial_labels = df_total.loc[trial, "labels"].values
                trans_point = []
                for i in trial[:-1]:
                    if trial_labels[i - trial[0]] == first_strategy and trial_labels[
                        i + 1 - trial[0]] == second_strategy:
                        trans_point.append(i)
                if len(trans_point) == 0:
                    continue
                else:
                    for t in trans_point:
                        before_trans = list(np.arange(max(trial[0], t - length), t))  # before vague
                        after_trans = list(np.arange(t, min(t + length, trial[-1])))  # after vague
                        if len(before_trans) < length or len(after_trans) < length:
                            continue
                        sel_index.append([before_trans, after_trans])
            return sel_index

        print("W/O vague : ")
        transition_pupil_wo_vague = {each: None for each in transition_pupil}
        for t_type in transition_type:
            t_index = transition_data(df_total, t_type[0], t_type[1], length)
            print("{} | {}".format(t_type, len(t_index)))
            type_transition_pupil = np.zeros((len(t_index), 2 * length))
            for ind, trial_index in enumerate(t_index):
                temp_pupil = np.zeros((2 * length,))
                temp_pupil[temp_pupil == 0] = np.nan
                before = list(trial_index[0])
                after = list(trial_index[1])
                t_data = df.loc[before + after]
                # eye_size_z = z_score(t_data, "eye_size")
                eye_size_z = t_data
                for i in range(len(before)):
                    if eye_size_z.loc[before[-1 - i], "eye_size_z"] != 0:
                        temp_pupil[length - 1 - i] = eye_size_z.loc[before[-1 - i], "eye_size_z"]
                for i in range(len(after)):
                    if eye_size_z.loc[after[i], "eye_size_z"] != 0:
                        temp_pupil[length + i] = eye_size_z.loc[after[i], "eye_size_z"]
                type_transition_pupil[ind] = temp_pupil
            transition_pupil_wo_vague[t_type] = type_transition_pupil
        with open(save_base + "{}_3F_{}std.pkl".format(monkey, coef), "wb") as file:
            pickle.dump({"vague": transition_pupil, "transition": transition_pupil_wo_vague}, file)

# =======================================================

def _isCorrect(df_q, df_dynamic):
    '''
    Compare estimated directions with true directions.
    '''
    true_dir = df_dynamic.next_pacman_dir_fill
    estimated_direction = df_q.estimated_direction
    is_correct = (true_dir == estimated_direction)
    df_q["is_correct"] = is_correct
    return df_q


def _randomFit(cur_pos, true_dir, adjacent_data):
    '''
    Randomly predicting directions.
    '''
    if cur_pos == (29, 18) or cur_pos == (0, 18):
        available_dir = ["left", "right"]
    else:
        adjacent_pos = adjacent_data[cur_pos]
        available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != adjacent_pos[dir] and not isinstance(adjacent_pos[dir], float):
                available_dir.append(dir)
    if 0 == len(available_dir) or 1 == len(available_dir):
        # raise ValueError("The position {} has {} adjacent positions.".format(cur_pos, len(available_dir)))
        return np.nan
    else:
        return np.random.choice(available_dir, 1).item() == true_dir

#TODO:数据路径
def diffModelCr():
    '''
    Compute estimation correct rate for different models.
    :return: VOID
    '''
    adjacent_data = readAdjacentMap("../../Data/constant/adjacent_map.csv")
    locs_df = readLocDistance("../../Data/constant/dij_distance_map.csv")

    # Read random model results
    random_data = np.load("/home/qlyang/jiaqi/Pacman-Analysis/common_data/special_case/100trial-all_random_is_correct.npy", allow_pickle=True).item()
    all_mean = np.array([np.mean(random_data["early"]), np.mean(random_data["middle"]), np.mean(random_data["end"])])
    all_size = np.array([len(random_data["early"]), len(random_data["middle"]), len(random_data["end"])])
    avg_random_cr = {each: np.nanmean(random_data[each]) for each in random_data}
    avg_random_cr["all"] = np.sum(all_mean * all_size) / np.sum(all_size)
    omega_cr = np.load("Omega_cr_index.npy", allow_pickle=True)
    patamon_cr = np.load("Patamon_cr_index.npy", allow_pickle=True)
    # Read pre-computed fitted data for the dynamic model
    all_is_correct = pd.read_pickle("/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/all_trials-merge_weight-dynamic-res.pkl")
    o_is_correct = []
    p_is_correct = []
    for i in range(len(all_is_correct)):
        if "Omega" in all_is_correct[i][0]:
            o_is_correct.append(all_is_correct[i])
        else:
            p_is_correct.append(all_is_correct[i])
    # o_is_correct = pd.read_pickle("./common_data/change_point_res/8478_trial_data_Omega-with_Q-clean-res.pkl")
    o_trial_length = [o_is_correct[i][1].shape[0] for i in range(len(o_is_correct))]
    # p_is_correct = pd.read_pickle("./common_data/change_point_res/7294_trial_data_Patamon-with_Q-clean-res.pkl")
    p_trial_length = [p_is_correct[i][1].shape[0] for i in range(len(p_is_correct))]
    moving_is_correct = list(np.array(o_is_correct)[omega_cr]) + list(np.array(p_is_correct)[patamon_cr])
    print("The num of trials is ", len(moving_is_correct))
    print("Dynamic data shape : ", sum([len(each[1]) for each in moving_is_correct]))
    # Read pre-computed fitted data for the static model
    o_hybird_trial_is_correct = []
    p_hybird_trial_is_correct = []
    o_hybrid_is_correct = np.load("/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/8478_trial_data_Omega-inf_Q-merge_weight-static_is_correct.npy", allow_pickle=True)
    p_hybrid_is_correct = np.load("/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/7294_trial_data_Patamon-inf_Q-merge_weight-static_is_correct.npy", allow_pickle=True)
    prev = 0
    for i in range(len(o_trial_length)):
        o_hybird_trial_is_correct.append(o_hybrid_is_correct[prev:prev+o_trial_length[i]])
        prev = prev + o_trial_length[i]
    o_hybird_trial_is_correct = np.array(o_hybird_trial_is_correct)[omega_cr]
    prev = 0
    for i in range(len(p_trial_length)):
        p_hybird_trial_is_correct.append(p_hybrid_is_correct[prev:prev + p_trial_length[i]])
        prev = prev + p_trial_length[i]
    p_hybird_trial_is_correct = np.array(p_hybird_trial_is_correct)[patamon_cr]
    hybird_trial_is_correct = np.array(list(o_hybird_trial_is_correct) + list(p_hybird_trial_is_correct))
    print("Static data shape after sampling : ", sum([len(each) for each in o_hybird_trial_is_correct]) + sum([len(each) for each in p_hybird_trial_is_correct]))
    # Read pre-computed fitted data for the perceptron model
    o_perceptron_is_correct = np.load("/home/qlyang/jiaqi/Pacman-Analysis/common_data/change_point_res/8478_trial_data_Omega-with_Q-clean-perceptron_is_correct.npy",allow_pickle=True)
    p_perceptron_is_correct = np.load("/home/qlyang/jiaqi/Pacman-Analysis/common_data/change_point_res/7294_trial_data_Patamon-with_Q-clean-perceptron_is_correct.npy", allow_pickle=True)
    o_perceptron_is_correct = np.array(o_perceptron_is_correct)[omega_cr]
    p_perceptron_is_correct = np.array(p_perceptron_is_correct)[patamon_cr]
    perceptron_is_correct = np.array(list(o_perceptron_is_correct) + list(p_perceptron_is_correct))
    print("Perceptron data shape after sampling : ", sum([len(each) for each in perceptron_is_correct]))

    trial_length = list(omega_cr) + list(patamon_cr)
    print("The num of used trials is ", len(trial_length))
    locs_df = readLocDistance("../../Data/constant/dij_distance_map.csv")
    # Different contexts
    all_type = {
        "all": [],
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": [],
    }
    for trial in moving_is_correct:
        early_thr = 80 if "Omega" in trial[0] else 65
        ending_thr = 10 if "Omega" in trial[0] else 7
        all_X = trial[1].reset_index(drop = True)
        end_index = all_X.beans.apply(lambda x: len(x) <= ending_thr if not isinstance(x, float) else True)
        end_index = np.where(end_index == True)[0]
        early_index = all_X.beans.apply(lambda x: len(x) >= early_thr if not isinstance(x, float) else False)
        early_index = np.where(early_index == True)[0]
        middle_index = all_X.beans.apply(lambda x: ending_thr < len(x) < early_thr if not isinstance(x, float) else False)
        middle_index = np.where(middle_index == True)[0]
        scared_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 > 3 or x.ifscared2 > 3, axis=1)
        scared_index = np.where(scared_index == True)[0]
        normal_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 or x.ifscared2 < 3, axis=1)
        normal_index = np.where(normal_index == True)[0]
        close_index = all_X[["pacmanPos", "ghost1Pos"]].apply(lambda x: True if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 10, axis=1)
        close_index = np.where(close_index == True)[0]
        all_type["early"].append(copy.deepcopy(early_index))
        all_type["middle"].append(copy.deepcopy(middle_index))
        all_type["end"].append(copy.deepcopy(end_index))
        all_type["close-normal"].append(copy.deepcopy(np.intersect1d(close_index, normal_index)))
        all_type["close-scared"].append(copy.deepcopy(np.intersect1d(close_index, scared_index)))
        all_type["all"].append(copy.deepcopy(np.arange(all_X.shape[0])))
    print("Finished extracting contexts!")
    # avg and std correct rate
    avg_hybrid_cr = {each:np.nanmean([np.nanmean(hybird_trial_is_correct[i][all_type[each][i]]) for i in range(len(trial_length))]) for each in all_type}
    std_hybrid_cr = {each: scipy.stats.sem([np.nanmean(hybird_trial_is_correct[i][all_type[each][i]]) for i in range(len(trial_length))], nan_policy="omit") for each in all_type}
    avg_moving_cr = {each: np.nanmean([np.nanmean(moving_is_correct[i][1].is_correct.values[all_type[each][i]]) for i in range(len(trial_length))]) for each in all_type}
    std_moving_cr = {each: scipy.stats.sem([np.nanmean(moving_is_correct[i][1].is_correct.values[all_type[each][i]]) for i in range(len(trial_length))],nan_policy="omit") for each in all_type}
    avg_preceptron_cr = {}
    std_perceptron_cr = {}
    for each in all_type:
        avg_tmp = []
        std_tmp = []
        for i in range(len(trial_length)):
            try:
                avg_tmp.append(np.nanmean(perceptron_is_correct[i][all_type[each][i]]))
                std_tmp.append(np.nanmean(perceptron_is_correct[i][all_type[each][i]]))
            except:
                continue
        avg_preceptron_cr[each] = copy.deepcopy(np.nanmean(avg_tmp))
        std_perceptron_cr[each] = copy.deepcopy(scipy.stats.sem(std_tmp, nan_policy="omit"))
    type_name = {"all": "All Stages", "early": "Early Stage", "end": "Ending Stage", "close-scared": "Scared Stage"}
    np.save(
        "all_2B.npy",
        (avg_hybrid_cr, std_hybrid_cr, avg_moving_cr, std_moving_cr, avg_preceptron_cr, std_perceptron_cr, type_name)
    )

    random_is_correct = []
    for j in range(len(moving_is_correct)):
        tmp = moving_is_correct[j][1][["pacmanPos", "next_pacman_dir_fill"]].apply(
            lambda x: _randomFit(x.pacmanPos, x.next_pacman_dir_fill, adjacent_data),
            axis=1
        )
        random_is_correct.append(copy.deepcopy(tmp.values))
    for type in all_type:
        type_idx = all_type[type]
        avg_random_cr = np.nanmean(
            [np.nanmean(random_is_correct[i][type_idx[i]]) for i in range(len(random_is_correct))])
        type_static_cr = np.asarray([np.nanmean(hybird_trial_is_correct[i][type_idx[i]]) for i in
                                range(len(trial_length))])
        type_static_cr = type_static_cr[~np.isnan(type_static_cr)]
        print("One sample t-test against chance level ({}) : ".format(type),
              scipy.stats.ttest_1samp(type_static_cr, avg_random_cr))

#TODO:数据路径
def diffModelCrDiffMonkey(monkey):
    '''
    Compute estimation correct rate for different models.
    :return: VOID
    '''
    # Read random model results
    random_data = np.load("/home/qlyang/jiaqi/Pacman-Analysis/common_data/special_case/100trial-all_random_is_correct.npy", allow_pickle=True).item()
    all_mean = np.array([np.mean(random_data["early"]), np.mean(random_data["middle"]), np.mean(random_data["end"])])
    all_size = np.array([len(random_data["early"]), len(random_data["middle"]), len(random_data["end"])])
    avg_random_cr = {each: np.nanmean(random_data[each]) for each in random_data}
    avg_random_cr["all"] = np.sum(all_mean * all_size) / np.sum(all_size)

    monkey_cr = np.load("{}_cr_index.npy".format(monkey), allow_pickle=True)
    # Read pre-computed fitted data for the dynamic model
    all_is_correct = pd.read_pickle(
        "/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/all_trials-merge_weight-dynamic-res.pkl")
    monkey_is_correct = []
    for i in range(len(all_is_correct)):
        if monkey in all_is_correct[i][0]:
            monkey_is_correct.append(all_is_correct[i])
    monkey_trial_length = [monkey_is_correct[i][1].shape[0] for i in range(len(monkey_is_correct))]
    moving_is_correct = list(np.array(monkey_is_correct)[monkey_cr])
    print("The num of trials is ", len(moving_is_correct))
    print("Dynamic data shape : ", sum([len(each[1]) for each in moving_is_correct]))
    # Read pre-computed fitted data for the static model
    monkey_hybird_trial_is_correct = []
    if monkey == "Omega":
        monkey_hybrid_is_correct = np.load("/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/8478_trial_data_Omega-inf_Q-merge_weight-static_is_correct.npy", allow_pickle=True)
    else:
        monkey_hybrid_is_correct = np.load("/home/qlyang/jiaqi/Decision_Making_PacMan/Data/fitted_data/7294_trial_data_Patamon-inf_Q-merge_weight-static_is_correct.npy", allow_pickle=True)
    prev = 0
    for i in range(len(monkey_trial_length)):
        monkey_hybird_trial_is_correct.append(monkey_hybrid_is_correct[prev:prev+monkey_trial_length[i]])
        prev = prev + monkey_trial_length[i]
    monkey_hybird_trial_is_correct = np.array(monkey_hybird_trial_is_correct)[monkey_cr]
    hybird_trial_is_correct = np.array(list(monkey_hybird_trial_is_correct))
    print("Static data shape after sampling : ", sum([len(each) for each in monkey_hybird_trial_is_correct]))
    # Read pre-computed fitted data for the perceptron model
    if monkey == "Omega":
        monkey_perceptron_is_correct = np.load("/home/qlyang/jiaqi/Pacman-Analysis/common_data/change_point_res/8478_trial_data_Omega-with_Q-clean-perceptron_is_correct.npy",allow_pickle=True)
    else:
        monkey_perceptron_is_correct = np.load("/home/qlyang/jiaqi/Pacman-Analysis/common_data/change_point_res/7294_trial_data_Patamon-with_Q-clean-perceptron_is_correct.npy", allow_pickle=True)
    monkey_perceptron_is_correct = np.array(monkey_perceptron_is_correct)[monkey_cr]
    perceptron_is_correct = np.array(list(monkey_perceptron_is_correct))
    print("Perceptron data shape after sampling : ", sum([len(each) for each in perceptron_is_correct]))

    trial_length = list(monkey_cr)
    print("The num of used trials is ", len(trial_length))
    locs_df = readLocDistance("../../Data/constant/dij_distance_map.csv")
    # Different contexts
    all_type = {
        "all": [],
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": [],
    }
    for trial in moving_is_correct:
        early_thr = 80 if "Omega" in trial[0] else 65
        ending_thr = 10 if "Omega" in trial[0] else 7
        all_X = trial[1].reset_index(drop = True)
        end_index = all_X.beans.apply(lambda x: len(x) <= ending_thr if not isinstance(x, float) else True)
        end_index = np.where(end_index == True)[0]
        early_index = all_X.beans.apply(lambda x: len(x) >= early_thr if not isinstance(x, float) else False)
        early_index = np.where(early_index == True)[0]
        middle_index = all_X.beans.apply(lambda x: ending_thr < len(x) < early_thr if not isinstance(x, float) else False)
        middle_index = np.where(middle_index == True)[0]
        scared_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 > 3 or x.ifscared2 > 3, axis=1)
        scared_index = np.where(scared_index == True)[0]
        normal_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 or x.ifscared2 < 3, axis=1)
        normal_index = np.where(normal_index == True)[0]
        close_index = all_X[["pacmanPos", "ghost1Pos"]].apply(lambda x: True if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 10, axis=1)
        close_index = np.where(close_index == True)[0]
        all_type["early"].append(copy.deepcopy(early_index))
        all_type["middle"].append(copy.deepcopy(middle_index))
        all_type["end"].append(copy.deepcopy(end_index))
        all_type["close-normal"].append(copy.deepcopy(np.intersect1d(close_index, normal_index)))
        all_type["close-scared"].append(copy.deepcopy(np.intersect1d(close_index, scared_index)))
        all_type["all"].append(copy.deepcopy(np.arange(all_X.shape[0])))
    print("Finished extracting contexts!")
    # avg and std correct rate
    avg_hybrid_cr = {each:np.nanmean([np.nanmean(hybird_trial_is_correct[i][all_type[each][i]]) for i in range(len(trial_length))]) for each in all_type}
    std_hybrid_cr = {each: scipy.stats.sem([np.nanmean(hybird_trial_is_correct[i][all_type[each][i]]) for i in range(len(trial_length))], nan_policy="omit") for each in all_type}
    avg_moving_cr = {each: np.nanmean([np.nanmean(moving_is_correct[i][1].is_correct.values[all_type[each][i]]) for i in range(len(trial_length))]) for each in all_type}
    std_moving_cr = {each: scipy.stats.sem([np.nanmean(moving_is_correct[i][1].is_correct.values[all_type[each][i]]) for i in range(len(trial_length))],nan_policy="omit") for each in all_type}
    avg_preceptron_cr = {}
    std_perceptron_cr = {}
    for each in all_type:
        avg_tmp = []
        std_tmp = []
        for i in range(len(trial_length)):
            try:
                avg_tmp.append(np.nanmean(perceptron_is_correct[i][all_type[each][i]]))
                std_tmp.append(np.nanmean(perceptron_is_correct[i][all_type[each][i]]))
            except:
                continue
        avg_preceptron_cr[each] = copy.deepcopy(np.nanmean(avg_tmp))
        std_perceptron_cr[each] = copy.deepcopy(scipy.stats.sem(std_tmp, nan_policy="omit"))
    type_name = {"all": "All Stages", "early": "Early Stage", "end": "Ending Stage", "close-scared": "Scared Stage"}
    np.save(
        "{}_2B.npy".format(monkey),
        (avg_hybrid_cr, std_hybrid_cr, avg_moving_cr, std_moving_cr, avg_preceptron_cr, std_perceptron_cr, type_name)
    )

# =======================================================

def _2Array(mat):
    '''
    Convert data format for compatibility.
    '''
    res = np.zeros((6, 4))
    for i in range(6):
        for j in range(4):
            res[i,j] = mat[i][j]
    return res


def _moveDir(pos1, pos2):
    '''
    Obtain moving directions given two positions.
    '''
    #
    if pos1 == (0, 18) and pos2 == (29, 18):
        return "left"
    if pos2 == (0, 18) and pos1 == (29, 18):
        return "right"
    #
    if pos1 == pos2:
        direction =  np.nan
    elif pos1[0] == pos2[0]:
        if pos1[1] < pos2[1]:
            direction = "down"
        else:
            direction = "up"
    elif pos1[1] == pos2[1]:
        if pos1[0] < pos2[0]:
            direction = "right"
        else:
            direction = "left"
    else:
        direction = np.nan
    return direction


def _getDir(ls):
    i = np.argmax(ls).item()
    return ["left", "right", "up", "down"][i]


def exampleTrial():
    '''
    Obtanin the example trial for Fig. 2A.
    '''
    # read fitted data
    trial_name = '25-1-Omega-17-Jul-2019-1.csv'  # (single trial: 2~40)
    ind_list = ["global_Q_norm", "local_Q_norm", 'evade_blinky_Q_norm', 'evade_clyde_Q_norm',
                'approach_Q_norm', 'energizer_Q_norm']
    data = _readData("Omega")
    df_monkey = data[data.file == trial_name].reset_index(drop = True)
    print("Example trial shape : ", df_monkey.shape)
    df_monkey.to_pickle("./example_trial.pkl")
    print("Finished saving data.")


if __name__ == '__main__':
    # =============================================================================
    # Note: Running some functions in this script requires using all the data.
    #       So running this script might raise exceptions.
    #       As an alternative, we provide all the analyzing results in the directory "Data/Behavior_Plot_Data".
    #       Please run "Analysis/Plotting/FigPlotting.py" to generate plots from precomputed data.
    #       Additionally, please backup "Data/Behavior_Plot_Data" before running this file, because the data could be o
    #       verwritten.
    # =============================================================================
    # rewardOrientation("all")
    # StrategyHeuristic("all")
    # Trajectory("all")
    # NonConflictingSaccade("all")
    # PAvsAA("all")
    # SuicidevsNormal("all")
    # statisticalTest("all")
    # strategyRatio("all")
    # PupilSizeDynamic("all")
    # # -------------------------
    # diffModelCr()
    # diffModelCrDiffMonkey("Omega")
    # diffModelCrDiffMonkey("Patamon")
    # # -------------------------
    # exampleTrial()

    pass