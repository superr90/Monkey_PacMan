'''
Description:
    Model fitting with pre-processed data.
'''
import numpy as np
import pandas as pd
import pickle
import scipy
import copy
import ruptures as rpt
import os
from sklearn.model_selection import KFold
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append("../../Utils")
from FileUtils import readAdjacentMap, readLocDistance

# =================================================
# Global variables
agents = [
        "global",
        "local",
        "evade_blinky",
        "evade_clyde",
        "approach",
        "energizer",
] # list of all the agents
all_dir_list = ["left", "right", "up", "down"]
adjacent_data = readAdjacentMap("../../Data/Constants/adjacent_map.csv")
inf_val = 100 # A large number representing the positive infinity
reborn_pos = (14, 27) # Pacman reborn position

# =================================================

def _makeChoice(prob):
    '''
    Chose a direction based on estimated Q values.
    :param prob: (list) Q values of four directions (lef, right, up, down).
    :return: (int) The chosen direction.
    '''
    copy_estimated = copy.deepcopy(prob)
    if np.any(prob) < 0:
        available_dir_index = np.where(prob != 0)
        copy_estimated[available_dir_index] = (
            copy_estimated[available_dir_index]
            - np.min(copy_estimated[available_dir_index])
            + 1
        )
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


def _oneHot(val):
    """
    Convert the direction into a one-hot vector.
    :param val: (str) The direction ("left", "right", "up", "down").
    :return: (list) One-hotted vector.
    """
    dir_list = ["left", "right", "up", "down"]
    # Type check
    if val not in dir_list:
        raise ValueError("Undefined direction {}!".format(val))
    if not isinstance(val, str):
        raise TypeError("Undefined direction type {}!".format(type(val)))
    # One-hot
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[dir_list.index(val)] = 1
    return onehot_vec


def _normalize(x):
    '''
    Normalization.
    :param x: (numpy.ndarray) Original data.
    :return: (numpy.ndarray) Normalized data.
    '''
    return (x) / (x).sum()


def _combine(cutoff_pts, dir):
    '''
    Combine cut off points when necessary.
    '''
    if len(cutoff_pts)>1:
        temp_pts = [cutoff_pts[0]]
        for i in range(1, len(cutoff_pts)):
            if cutoff_pts[i][1] - cutoff_pts[i][0] > 3:
                if np.all(dir.iloc[cutoff_pts[i][0]:cutoff_pts[i][1]].apply(lambda x:isinstance(x, float)) == True):
                    temp_pts[-1] = (temp_pts[-1][0], cutoff_pts[i][1])
                else:
                    temp_pts.append(cutoff_pts[i])
            else:
                temp_pts[-1] = (temp_pts[-1][0], cutoff_pts[i][1])
        cutoff_pts = temp_pts
    return cutoff_pts


def _positivePessi(pess_Q, offset, pos):
    '''
    Make evade agent Q values non-negative.
    '''
    non_zero = []
    if pos == (29, 18) or pos == (30, 18):
        pos = (28, 18)
    if pos == (0, 18) or pos == (-1, 18):
        pos = (1, 18)
    for dir in all_dir_list:
        if None != adjacent_data[pos][dir] and not isinstance(adjacent_data[pos][dir], float):
            non_zero.append(all_dir_list.index(dir))
    pess_Q[non_zero] = pess_Q[non_zero] - offset
    return _normalizeWithInf(pess_Q)

# =================================================

def negativeLikelihood(
    param, all_data, true_prob, agents_list, return_trajectory=False, suffix="_Q"
):
    """
    Compute the negative log-likelihood.
    :param param: (list) Model parameters, which are agent weights.
    :param all_data: (pandas.DataFrame) A table of data.
    :param agent_list: (list) Names of all the agents.
    :param return_trajectory: (bool) Set to True when making predictions.
    :return: (float) Negative log-likelihood
    """
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute negative log likelihood
    nll = 0
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    # raise KeyboardInterrupt
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]
    dir_Q_value = agent_Q_value @ agent_weight
    dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
    exp_prob = np.exp(dir_Q_value)
    for each_sample in range(num_samples):
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
        nll = nll - log_likelihood
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def negativeLikelihoodMergeGroup(
    param, all_data, true_prob, group_idx, agents_list, return_trajectory=False, suffix="_Q"
):
    """
    Compute the negative log-likelihood.
    :param param: (list) Model parameters, which are agent weights.
    :param all_data: (pandas.DataFrame) A table of data.
    :param agent_list: (list) Names of all the agents.
    :param return_trajectory: (bool) Set to True when making predictions.
    :return: (float) Negative log-likelihood
    """
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute negative log likelihood
    nll = 0
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    # raise KeyboardInterrupt
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]
    # merge Q values
    merg_Q_values = np.zeros((agent_Q_value.shape[0], agent_Q_value.shape[1], len(group_idx)))
    for i, g in enumerate(group_idx):
        merg_Q_values[:,:,i] = np.nanmean(agent_Q_value[:,:,g], axis=-1)
    dir_Q_value = merg_Q_values @ agent_weight
    dir_Q_value[np.isnan(dir_Q_value)] = -np.inf
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
    exp_prob = np.exp(dir_Q_value)
    for each_sample in range(num_samples):
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
        nll = nll - log_likelihood
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def caculate_correct_rate(result_x, all_data, true_prob, agents, suffix="_Q"):
    '''
    Compute the estimation correct rate of a fitted model.
    '''
    _, estimated_prob = negativeLikelihood(
        result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir) / len(estimated_dir)
    return correct_rate


def _calculate_is_correct(result_x, all_data, true_prob, agents, suffix="_Q"):
    '''
    Determine whether the estimation of each time step is correct.
    '''
    _, estimated_prob = negativeLikelihood(
        result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
    is_correct = (estimated_dir == true_dir)
    return is_correct


def change_dir_index(x):
    '''
    Find the position where the Pacman changes its direction.
    '''
    temp = pd.Series((x != x.shift()).where(lambda x: x == True).dropna().index)
    return temp[(temp - temp.shift()) > 1].values


def fit_func(df_monkey, cutoff_pts, suffix="_Q", is_match = False,
             agents = ["global","local","evade_blinky","evade_clyde","approach","energizer"]):
    '''
    Fit model parameters (i.e., agent weights).
    '''
    result_list = []
    is_correct = []
    bounds = [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]]
    params = [0.0] * 6
    cons = []  # construct the bounds in the form of constraints

    for par in range(len(bounds)):
        l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
        u = {"type": "ineq", "fun": lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)

    prev = 0
    total_loss = 0
    is_correct = np.zeros((df_monkey.shape[0],))
    is_correct[is_correct == 0] = np.nan

    for prev, end in cutoff_pts:
        all_data = df_monkey[prev:end]
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        ind = np.where(temp_data.nan_dir == False)[0] + prev

        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
        func = lambda params: negativeLikelihood(
            params, all_data, true_prob, agents, return_trajectory=False, suffix=suffix
        )
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints=cons,
        )
        if set(res.x) == {0}:
            print("Failed optimization at ({},{})".format(prev, end))
            params = [0.1]*6
            for i, a in enumerate(agents):
                if set(np.concatenate(all_data["{}{}".format(a, suffix)].values)) == {0}:
                    params[i] = 0.0
            res = scipy.optimize.minimize(
                func,
                x0=params,
                method="SLSQP",
                bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
                tol=1e-5,
                constraints=cons,
            )
        total_loss += negativeLikelihood(
            res.x / res.x.sum(),
            all_data,
            true_prob,
            agents,
            return_trajectory=False,
            suffix=suffix,
        )
        cr = caculate_correct_rate(res.x, all_data, true_prob, agents, suffix=suffix)
        result_list.append(res.x.tolist() + [cr] + [prev] + [end])
        phase_is_correct = _calculate_is_correct(res.x, all_data, true_prob, agents, suffix=suffix)
        is_correct[ind] = phase_is_correct
    if is_match:
        return result_list, total_loss, is_correct
    else:
        return result_list, total_loss


def merge_fit_func(df_monkey, cutoff_pts, suffix="_Q", is_match = False,
             agents = ["global","local","evade_blinky","evade_clyde","approach","energizer"]):
    '''
    Fit model parameters (i.e., agent weights).
    '''
    agent_Q_list = [each + suffix for each in agents]

    result_list = []
    is_correct = []

    prev = 0
    total_loss = 0
    is_correct = np.zeros((df_monkey.shape[0],))
    is_correct[is_correct == 0] = np.nan
    trial_same_dir_groups = []
    for prev, end in cutoff_pts:
        all_data = df_monkey[prev:end]
        temp_data = copy.deepcopy(all_data)
        temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
        all_data = all_data[temp_data.nan_dir == False]
        if all_data.shape[0] == 0:
            print("All the directions are nan from {} to {}!".format(prev, end))
            continue
        # -----
        agent_dirs = []
        agent_Q_data = all_data[agent_Q_list]
        for each in agent_Q_list:
            agent_Q_data[each] = agent_Q_data[each].apply(lambda x: np.nan if isinstance(x, list) else x).fillna(
                method="ffill").fillna(
                method="bfill").values
            tmp_dirs = agent_Q_data[each].apply(
                lambda x: np.argmax(x) if not np.all(x[~np.isinf(x)] == 0) else np.nan).values
            agent_dirs.append(tmp_dirs)
        agent_dirs = np.asarray(agent_dirs)
        # -----
        # Split into groups
        wo_nan_idx = np.asarray([i for i in range(agent_dirs.shape[0]) if not np.any(np.isnan(agent_dirs[i]))])
        if len(wo_nan_idx) <= 1:
            same_dir_groups = [[i] for i in range(len(agents))]
        else:
            same_dir_groups = [[i] for i in range(len(agents)) if i not in wo_nan_idx]
            wo_nan_is_same = np.asarray([[np.all(agent_dirs[i] == agent_dirs[j]) for j in wo_nan_idx] for i in wo_nan_idx], dtype=int)
            _, component_labels = connected_components(wo_nan_is_same, directed=False)
            for i in np.unique(component_labels):
                same_dir_groups.append(list(wo_nan_idx[np.where(component_labels==i)[0]]))
        trial_same_dir_groups.append(same_dir_groups)
        # construct reverse table
        reverse_group_idx = {each:[None, 0] for each in range(6)}
        for g_idx, g in enumerate(same_dir_groups):
            for i in g:
                reverse_group_idx[i][0] = g_idx
                for j in g:
                    reverse_group_idx[j][1] +=1
        # -----
        ind = np.where(temp_data.nan_dir == False)[0] + prev
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
        # -----
        bounds = [[0, 1000] for _ in range(len(same_dir_groups))]
        params = [0.0] * len(same_dir_groups)
        cons = []  # construct the bounds in the form of constraints
        for par in range(len(bounds)):
            l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
            u = {"type": "ineq", "fun": lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)
        func = lambda params: negativeLikelihoodMergeGroup(
            params, all_data, true_prob, same_dir_groups, agents, return_trajectory=False, suffix=suffix
        )
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints=cons,
        )
        if set(res.x) == {0}:
            print("Failed optimization at ({},{})".format(prev, end))
            params = [0.1] * len(same_dir_groups)
            for i, a in enumerate(agents):
                if set(np.concatenate(all_data["{}{}".format(a, suffix)].values)) == {0}:
                    params[i] = 0.0
            res = scipy.optimize.minimize(
                func,
                x0=params,
                method="SLSQP",
                bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
                tol=1e-5,
                constraints=cons,
            )
        total_loss += negativeLikelihoodMergeGroup(
            res.x / res.x.sum(),
            all_data,
            true_prob,
            same_dir_groups,
            agents,
            return_trajectory=False,
            suffix=suffix,
        )
        # -----
        reassigned_weights = [res.x[reverse_group_idx[i][0]] / reverse_group_idx[i][1] for i in range(6)]
        cr = caculate_correct_rate(reassigned_weights, all_data, true_prob, agents, suffix=suffix)
        result_list.append(reassigned_weights + [cr] + [prev] + [end])
        phase_is_correct = _calculate_is_correct(reassigned_weights, all_data, true_prob, agents, suffix=suffix)
        is_correct[ind] = phase_is_correct
    if is_match:
        return result_list, total_loss, is_correct, trial_same_dir_groups
    else:
        return result_list, total_loss


def normalize_weights(result_list, df_monkey):
    '''
    Normalize fitted agent weights.
    '''
    agents = [
        "global",
        "local",
        "pessimistic_blinky",
        "pessimistic_clyde",
        "suicide",
        "planned_hunting",
    ]
    df_result = (
        pd.DataFrame(
            result_list,
            columns=[i + "_w" for i in agents] + ["accuracy", "start", "end"],
        )
        .set_index("start")
        .reindex(range(df_monkey.shape[0]))
        .fillna(method="ffill")
    )
    df_plot = df_result.filter(regex="_w").divide(
        df_result.filter(regex="_w").sum(1), 0
    )
    return df_plot, df_result


def add_cutoff_pts(cutoff_pts, df_monkey):
    '''
    Initialize cut-off points at where the ghosts and energizers are eaten.
    '''
    eat_ghost = (
        (
            ((df_monkey.ifscared1 == 3) & (df_monkey.ifscared1.diff() < 0))
            | ((df_monkey.ifscared2 == 3) & (df_monkey.ifscared2.diff() < 0))
        )
        .where(lambda x: x == True)
        .dropna()
        .index.tolist()
    )
    eat_energizers = (
        (
            df_monkey.energizers.apply(
                lambda x: len(x) if not isinstance(x, float) else 0
            ).diff()
            < 0
        )
        .where(lambda x: x == True)
        .dropna()
        .index.tolist()
    )
    cutoff_pts = sorted(list(cutoff_pts) + eat_ghost + eat_energizers)
    return cutoff_pts

# =================================================

def _normalizeWithInf(x):
    res_x = x.copy()
    tmp_x_idx = np.where(~np.isinf(x))[0]
    if set(x[tmp_x_idx]) == {0}:
        res_x[tmp_x_idx] = 0
    else:
        res_x[tmp_x_idx] = res_x[tmp_x_idx] / np.max(res_x[tmp_x_idx])
    return res_x


def _readData(filename):
    '''
    Read data.
    '''
    print("Filename : ", filename)
    df = pd.read_pickle(filename)
    # -----
    filename_list = df.file.unique()
    selected_data = pd.concat([df[df.file == i] for i in filename_list]).reset_index(drop=True)
    df = selected_data
    # -----------------
    # Drop some columns
    # print(df.columns.values)
    if "global_Q" in df.columns and "global_inf_Q" in df.columns:
        # df = df.drop(columns=['global_Q', 'local_Q', 'evade_blinky_Q', 'evade_clyde_Q', 'approach_Q', 'energizer_Q',
        #                      'global_Q_norm', 'local_Q_norm', 'evade_blinky_Q_norm', 'evade_clyde_Q_norm',
        #                       'approach_Q_norm', 'energizer_Q_norm'])
        df = df.drop(columns=['global_Q', 'local_Q', 'evade_blinky_Q', 'evade_clyde_Q', 'approach_Q', 'energizer_Q',
                              'global_Q_norm', 'local_Q_norm', 'pessimistic_blinky_Q_norm', 'pessimistic_clyde_Q_norm',
                              'suicide_Q_norm', 'planned_hunting_Q_norm'])
        df = df.rename(columns={'global_inf_Q':"global_Q", 'local_inf_Q':'local_Q',
                                'evade_blinky_inf_Q':'evade_blinky_Q', 'evade_clyde_inf_Q':'evade_clyde_Q',
                                'approach_inf_Q':'approach_Q', 'energizer_inf_Q':'energizer_Q'})
    # -----------------
    if "DayTrial" in df.columns.values:
        df["file"] = df.DayTrial
    if "pacman_dir" in df.columns.values and "next_pacman_dir_fill" not in df.columns.values:
        df["next_pacman_dir_fill"] = df.pacman_dir.apply(lambda x: x if x is not None else np.nan)
    trial_name_list = np.unique(df.file.values)
    trial_data = []
    for each in trial_name_list:
        pac_dir = df[df.file == each].next_pacman_dir_fill
        if np.sum(pac_dir.apply(lambda x: isinstance(x, float)))==len(pac_dir):
            # all the directions are none
            print("({}) Pacman No Move ! Shape = {}".format(each, pac_dir.shape))
            continue
        else:
            trial_data.append(df[df.file == each])
    df = pd.concat(trial_data).reset_index(drop=True)
    for c in df.filter(regex="_Q").columns:
        if "evade" not in c:
            # df[c + "_norm"] = df[c].apply(
            #     lambda x: x / max(x) if set(x) != {0} else [0, 0, 0, 0]
            # )
            df[c + "_norm"] = df[c].apply(
                lambda x: _normalizeWithInf(x)
            )
        else:
            tmp_val = df[c].explode().values
            offset_num = np.min(tmp_val[tmp_val != -np.inf])
            # offset_num = df[c].explode().min()
            df[c + "_norm"] = df[[c, "pacmanPos"]].apply(
                lambda x: _positivePessi(x[c], offset_num, x.pacmanPos)
                if set(x[c]) != {0}
                else [0, 0, 0, 0],
                axis=1
            )
    return df


def _combinePhases(bkpt_idx, df, return_flag=False):
    agent_list = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]
    agent_Q_list = [each+"_Q_norm" for each in agent_list]
    agent_Q_data = df[agent_Q_list]
    true_dirs =df.next_pacman_dir_fill.fillna(method="bfill").fillna(method="ffill").apply(lambda x: ["left", "right", "up", "down"].index(x))
    print("The number of phases (before combination) is {}".format(len(bkpt_idx)))
    # -----
    agent_dirs = []
    for each in agent_Q_list:
        agent_Q_data[each] = agent_Q_data[each].apply(lambda x: np.nan if isinstance(x, list) else x).fillna(method="ffill").fillna(
            method="bfill").values
        tmp_dirs = agent_Q_data[each].apply(lambda x: np.argmax(x) if not np.all(x[~np.isinf(x)] == 0) else np.nan).values
        agent_dirs.append(tmp_dirs)
    agent_dirs = np.asarray(agent_dirs)
    same_as_true_dir = np.asarray([(agent_dirs[i]==true_dirs).values for i in range(6)])
    phase_same_as_true = [np.nanmean(same_as_true_dir[:, each[0]:each[1]], axis=1) for each in bkpt_idx]
    all_not_same = [np.all(each == 0) for each in phase_same_as_true]
    all_not_same_trial = [[df.file.values[0], bkpt_idx[i]] for i in np.where(all_not_same == True)[0]]
    # -----
    phase_directions = [agent_dirs[:, each[0]:each[1]] for each in bkpt_idx]
    phase_same_num = np.asarray([0 for _ in range(len(bkpt_idx))])
    for p in range(len(bkpt_idx)):
        same_num = 0
        for i in range(len(agent_list)):
            for j in range(i+1, len(agent_list)):
                if i == 2 and j == 3: # ignore two evade agents
                    continue
                # 如果有nan就跳过
                if np.any(np.isnan(phase_directions[p][i, :])) or np.any(np.isnan(phase_directions[p][j, :])):
                    continue
                if np.mean(phase_directions[p][i, :] == phase_directions[p][j, :]) == 1:
                    same_num += 1
        phase_same_num[p] = same_num
    phase_is_same = np.asarray(phase_same_num > 0, dtype=int)
    # -----
    if np.all(phase_is_same == 1):
        new_phase = [(0, bkpt_idx[-1][1])]
    else:
        new_phase = []
        iterator = 0
        while iterator < len(phase_is_same):
            if phase_is_same[iterator] == 0:
                new_phase.append(bkpt_idx[iterator])
            else:
                start_idx = bkpt_idx[iterator][0]
                while phase_is_same[iterator] == 1:
                    iterator += 1
                    if iterator >= len(phase_is_same):  # the last phase has same action sequences
                        start_idx = new_phase[-1][0]
                        end_idx = bkpt_idx[iterator - 1][1]
                        new_phase = new_phase[:-1]
                        break
                    else:
                        end_idx = bkpt_idx[iterator][1]
                new_phase.append((start_idx, end_idx))
            iterator += 1
    print("The number of phases (after combination) is {}".format(len(new_phase)))
    if return_flag:
        return new_phase, phase_is_same
    else:
        return new_phase, all_not_same_trial


def dynamicStrategyFitting(config):
    '''
    Dynamic strategy model fitting.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    # -----
    print("=== Dynamic Strategy Fitting ====")
    print("Start reading data...")
    df = _readData(config["filename"])
    suffix = "_Q_norm"
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    all_trial_record = []
    # all_not_same_list = []
    same_group_list = []
    for t, trial_name in enumerate(trial_name_list):
        if "level_0" in df.columns.values:
            df_monkey = df[df.file == trial_name].reset_index().drop(columns="level_0")
        else:
            df_monkey = df[df.file == trial_name].reset_index()
        print("| ({}) {} | Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points
        cutoff_pts = add_cutoff_pts(change_dir_index(df_monkey.next_pacman_dir_fill),df_monkey)  # Add eating ghost and eating energizer points
        cutoff_pts = list(zip([0] + list(cutoff_pts[:-1]), cutoff_pts))
        cutoff_pts = _combine(cutoff_pts, df_monkey.next_pacman_dir_fill)
        result_list, _ = fit_func(df_monkey, cutoff_pts, suffix=suffix, agents=agents)
        print("-" * 50)
        # breakpoints detection
        try:
            df_plot, df_result = normalize_weights(result_list, df_monkey)
        except:
            print("Error occurs in weight normalization.")
            continue
        signal = df_plot.filter(regex="_w").fillna(0).values
        algo = rpt.Dynp(model="l2", jump=2).fit(signal)
        nll_list = []
        bkpt_list = list(range(2, 21))
        this_bkpt_list = []
        bkpt_idx_list = []
        for index, n_bkpt in enumerate(bkpt_list):
            try:
                result = algo.predict(n_bkpt)
                result = list(zip([0] + result[:-1], result))
                result = _combine(result, df_monkey.next_pacman_dir_fill)
                result_list, total_loss = merge_fit_func(df_monkey, result, suffix=suffix, agents=agents)
                print(
                    "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',
                                                                                    total_loss + 0.5 * n_bkpt * 5,
                )
                nll_list.append(total_loss)
                this_bkpt_list.append(n_bkpt)
                bkpt_idx_list.append(result)
                # not_same_list.append(trial_not_same_trial)
            except:
                print("No admissible last breakpoints found.")
                break
        if len(nll_list) == 0:
            continue
        best_arg = np.argmin(nll_list)
        best_num_of_bkpt = this_bkpt_list[best_arg]
        best_log = nll_list[best_arg]
        best_bkpt_idx = bkpt_idx_list[best_arg]
        # best_not_same_list = not_same_list[best_arg]
        print("Least Log Likelihood value : {}, Best # of breakpoints {}".format(best_log, best_num_of_bkpt))
        # =============use best # of breakpoints to get weights and accuracy================
        result_list, total_loss, is_correct, trial_same_dir_groups = merge_fit_func(df_monkey, best_bkpt_idx, suffix=suffix, is_match=True, agents=agents)
        # append result to record
        trial_weight = []
        trial_contribution = []
        for res in result_list:
            weight = res[:len(agents)]
            prev = res[-2]
            end = res[-1]
            for _ in range(prev, end):
                trial_weight.append(weight)
                trial_contribution.append(weight/np.linalg.norm(weight))
        if len(trial_weight) != df_monkey.shape[0]:
            df_monkey["weight"] = [np.nan for _ in range(df_monkey.shape[0])]
            df_monkey["contribution"] = [np.nan for _ in range(df_monkey.shape[0])]
            df_monkey["is_correct"] = [np.nan for _ in range(df_monkey.shape[0])]
        elif len(trial_weight) > 0:
            df_monkey["weight"] = trial_weight
            df_monkey["contribution"] = trial_contribution
            df_monkey["is_correct"] = is_correct
        else:
            pass
        all_trial_record.append((trial_name, df_monkey, df_monkey.next_pacman_dir_fill, best_bkpt_idx))
        best_bkpt_list.append((trial_name, best_num_of_bkpt, best_bkpt_idx))
        same_group_list.extend(trial_same_dir_groups)
        print("=" * 50)
    # save data
    print("Finished fitting.")
    with open("{}/{}-merge_weight-dynamic-res.pkl".format(config["save_base"], config["filename"].split("/")[-1].split(".")[-2]), "wb") as file:
        pickle.dump(all_trial_record, file)
    np.save("{}/{}-merge_weight-dynamic-bkpt.npy".format(config["save_base"], config["filename"].split("/")[-1].split(".")[-2]), best_bkpt_list)
    np.save("{}/{}-merge_weight-dynamic-same_dir_groups.npy".format(config["save_base"], config["filename"].split("/")[-1].split(".")[-2]), same_group_list)
    print("Finished saving data.")


def staticStrategyFitting(config):
    '''
    Static strategy model fitting.
    '''
    print("=== Static Strategy Fitting ====")
    df = _readData(config["filename"])
    suffix = "_Q_norm"
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    all_data = copy.deepcopy(df)
    print("Shape of data : ", all_data.shape)

    is_correct = np.zeros((all_data.shape[0],))
    is_correct[is_correct == 0] = np.nan

    temp_data = copy.deepcopy(all_data)
    temp_data["nan_dir"] = temp_data.next_pacman_dir_fill.apply(lambda x: isinstance(x, float))
    all_data = all_data[temp_data.nan_dir == False]
    true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(_oneHot)
    ind = np.where(temp_data.nan_dir == False)[0]

    # Model selection with 5-fold cross-validation
    kf = KFold(n_splits=5)
    all_res = {}
    index = 1
    for train_index, test_index in kf.split(all_data):
        X_train, X_test = all_data.iloc[train_index], all_data.iloc[test_index]
        y_train, y_test = true_prob.iloc[train_index], true_prob.iloc[test_index]
        bounds = [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]]
        params = [0.0] * 6
        cons = []  # construct the bounds in the form of constraints
        for par in range(len(bounds)):
            l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
            u = {"type": "ineq", "fun": lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)
        # true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(oneHot)
        func = lambda params: negativeLikelihood(
            params, X_train, y_train, agents, return_trajectory=False, suffix=suffix
        )
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints=cons,
        )

        cr = caculate_correct_rate(res.x, X_test, y_test, agents, suffix=suffix)
        all_res[index] = [cr, copy.deepcopy(res.x)]
        print("|Fold {}| Avg correct rate : ".format(index), cr)
        index += 1
    print("Finished fitting.")
    print("="*50)
    best_model = sorted([[k, all_res[k][0]] for k in all_res], key = lambda x: x[1])
    print("Best model index is ", best_model[-1][0])
    best_par = all_res[best_model[-1][0]][1]
    phase_is_correct = _calculate_is_correct(best_par, all_data, true_prob, agents, suffix=suffix)
    is_correct[ind] = phase_is_correct
    np.save("{}/{}-merge_weight-static_is_correct.npy".format(config["save_base"], config["filename"].split("/")[-1].split(".")[-2]),
            is_correct)
    np.save("{}/{}-merge_weight-static_weight.npy".format(
        config["save_base"], config["filename"].split("/")[-1].split(".")[-2]),
            best_par)
    print("Finished saving data.")

# =================================================

def _adjacentBeans(pacmanPos, beans, type, locs_df):
    '''
    Compute the number of beans within 10 steps away from the Pacman.
    '''
    # Pacman in tunnel
    if pacmanPos == (29, 18):
        pacmanPos = (28, 18)
    if pacmanPos == (0, 18):
        pacmanPos = (1, 18)
    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent beans num
    if adjacent not in locs_df:
        bean_num = 0
    else:
        bean_num = (
            0 if isinstance(beans, float) else len(np.where(
                np.array([0 if adjacent == each else locs_df[adjacent][each] for each in beans]) <= 10)[0]
            )
        )
    return bean_num


def _adjacentDist(pacmanPos, ghostPos, type, adjacent_data, locs_df):
    '''
    The distance between the Pcaman and a ghost.
    '''
    # Pacman in tunnel
    if pacmanPos == (29, 18):
        pacmanPos = (28, 18)
    if pacmanPos == (0, 18):
        pacmanPos = (1, 18)
    if isinstance(adjacent_data[pacmanPos][type], float):
        return inf_val
    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent positions in the tunnel
    if adjacent == (29, 18):
        adjacent = (28, 18)
    elif adjacent == (0, 18):
        adjacent = (1, 18)
    else:
        pass
    return 0 if adjacent == ghostPos else locs_df[adjacent][ghostPos]


def extractFeatureWRTDir(all_data):
    '''
    Extract features with respect to four directions.
    '''
    locs_df = readLocDistance("../Data/constant/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("../Data/constant/adjacent_map.csv")
    trial_features = []
    trial_labels = []
    trial_names = []
    #
    trial_data = []
    trial_name_list = np.unique(all_data.file.values)
    for each in trial_name_list:
        each_trial = all_data[all_data.file == each].reset_index(drop=True)
        # True moving directions
        true_prob = each_trial.next_pacman_dir_fill
        # Fill nan direction for optimization use
        start_index = 0
        while pd.isna(true_prob[start_index]):
            start_index += 1
            if start_index == len(true_prob):
                break
        if start_index == len(true_prob):
            print("Moving direciton of trial {} is all nan.".format(each))
            continue
        if start_index > 0:
            true_prob[:start_index + 1] = true_prob[start_index + 1]
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(_oneHot(x)))
        trial_data.append([each, each_trial, true_prob])
    print("Finished separating trials.")
    #
    for trial_index, each in enumerate(trial_data):
        trial_names.append(each[0])
        trial = each[1]
        true_dir = each[2]
        # Features for the estimation

        PG1_left = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "left", adjacent_data, locs_df),
            axis = 1
        )
        PG1_right = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "right", adjacent_data, locs_df),
            axis = 1
        )
        PG1_up = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "up", adjacent_data, locs_df),
            axis = 1
        )
        PG1_down = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "down", adjacent_data, locs_df),
            axis = 1
        )

        PG2_left = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "left", adjacent_data, locs_df),
            axis=1
        )
        PG2_right = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "right", adjacent_data, locs_df),
            axis=1
        )
        PG2_up = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "up", adjacent_data, locs_df),
            axis=1
        )
        PG2_down = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "down", adjacent_data, locs_df),
            axis=1
        )

        PE_left = trial[["pacmanPos", "energizers"]].apply(
            lambda x : inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "left", adjacent_data, locs_df) for each in x.energizers]),
            axis = 1
        )
        PE_right = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "right", adjacent_data, locs_df) for each in x.energizers]),
            axis=1
        )
        PE_up = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "up", adjacent_data, locs_df) for each in x.energizers]),
            axis=1
        )
        PE_down = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "down", adjacent_data, locs_df) for each in x.energizers]),
            axis=1
        )

        PF_left = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x : inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "left", adjacent_data, locs_df),
            axis = 1
        )
        PF_right = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x: inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "right", adjacent_data, locs_df),
            axis=1
        )
        PF_up = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x: inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "up", adjacent_data, locs_df),
            axis=1
        )
        PF_down = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x : inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "down", adjacent_data, locs_df),
            axis = 1
        )
        beans_10step = trial[["pacmanPos", "beans"]].apply(
            lambda x : 0 if isinstance(x.beans, float)
            else len(
                np.where(
                    np.array([0 if x.pacmanPos == each
                              else locs_df[x.pacmanPos][each] for each in x.beans]) <= 10
                )[0]
            ),
            axis = 1
        )
        beans_over_10step = trial[["pacmanPos", "beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(
                np.where(
                    np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) > 10
                )[0]
            ),
            axis=1
        )
        beans_num = trial[["beans"]].apply(
            lambda x: 0 if isinstance(x.beans, float)
            else len(x.beans),
            axis=1
        )


        beans_left = trial[["pacmanPos", "beans"]].apply(
            lambda x: _adjacentBeans(x.pacmanPos, x.beans, "left", locs_df),
            axis=1
        )
        beans_right = trial[["pacmanPos", "beans"]].apply(
            lambda x: _adjacentBeans(x.pacmanPos, x.beans, "right", locs_df),
            axis=1
        )
        beans_up = trial[["pacmanPos", "beans"]].apply(
            lambda x: _adjacentBeans(x.pacmanPos, x.beans, "up", locs_df),
            axis=1
        )
        beans_down = trial[["pacmanPos", "beans"]].apply(
            lambda x: _adjacentBeans(x.pacmanPos, x.beans, "down", locs_df),
            axis=1
        )


        # beans_diff = trial[["pacmanPos", "beans"]].apply(
        #     lambda x : 0 if isinstance(x.beans, float)
        #     else np.sum(
        #         np.where(
        #             np.array([0 if reborn_pos == each else locs_df[reborn_pos][each] for each in x.beans]) <= 15
        #         )
        #     ) - np.sum(
        #         np.where(
        #             np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) <= 15
        #         )
        #     ),
        #     axis = 1
        # )
        # trial_data["PG1"], trial_data["PG2"], trial_data["min_PE"], trial_data["PF"], trial_data["beans_15step"], trial_data["beans_diff"] \
        #     = [PG1, PG2, min_PE, PF, beans_15step, beans_diff]
        processed_trial_data = pd.DataFrame(
            data=
            {
                "ifscared1" : trial.ifscared1,
                "ifscared2" : trial.ifscared2,

                "PG1_left" : PG1_left,
                "PG1_right": PG1_right,
                "PG1_up": PG1_up,
                "PG1_down": PG1_down,

                "PG2_left" : PG2_left,
                "PG2_right": PG2_right,
                "PG2_up": PG2_up,
                "PG2_down": PG2_down,

                "PE_left" : PE_left,
                "PE_right": PE_right,
                "PE_up": PE_up,
                "PE_down": PE_down,

                "PF_left" : PF_left,
                "PF_right": PF_right,
                "PF_up": PF_up,
                "PF_down": PF_down,

                # "beans_10step" : beans_10step,
                "beans_left": beans_left,
                "beans_right": beans_right,
                "beans_up": beans_up,
                "beans_down": beans_down,

                "beans_num" : beans_num,

                # "beans_over_10step" : beans_over_10step,

                # "beans_diff" : beans_diff,
            }
        )
        X = processed_trial_data
        y = true_dir.apply(lambda x : list(x).index(1))
        trial_features.append(copy.deepcopy(X))
        trial_labels.append(copy.deepcopy(y))
    print("Finished extracing features...")
    return trial_names, trial_features, trial_labels


def perceptron(config):
    '''
    Perceptron model fitting.
    '''
    print("=== Perceptron with Global Features ====")
    print("Start reading data...")
    df = _readData(config["filename"])
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("="*30)
    # extract features
    trial_names, trial_features, trial_labels = extractFeatureWRTDir(df)
    trial_num = len(trial_names)
    trial_index = list(range(trial_num))
    trial_names = [trial_names[each] for each in trial_index]
    trial_features = [trial_features[each] for each in trial_index]
    trial_labels = [trial_labels[each] for each in trial_index]

    all_features = np.concatenate(trial_features)
    all_labels = np.concatenate(trial_labels)
    # Parameter selection with 5-fold cross-validation
    kf = KFold(n_splits=5)
    hidden_size_range = [16, 32, 64, 128, 256]
    hyperpar_res = []
    for s_i, s in enumerate(hidden_size_range):
        print("="*50)
        print("Hidden layer size {}".format(s))
        # Model selection with 5-fold cross-validation
        all_res = {}
        index = 1
        for train_index, test_index in kf.split(all_features):
            X_train, X_test = all_features[train_index], all_features[test_index]
            y_train, y_test = all_labels[train_index], all_labels[test_index]

            print("|Fold {}| Start training... ".format(index))
            # model = Perceptron()
            model = MLPClassifier(hidden_layer_sizes=s, batch_size=128, activation="identity")
            model.fit(X_train, y_train)
            cr = model.score(X_test, y_test)
            all_res[index] = [cr, copy.deepcopy(model)]
            print("|Fold {}| Avg correct rate : ".format(index), cr)
            index += 1
        print("Finished fitting.")
        print("-" * 50)
        best_model = sorted([[k, all_res[k][0]] for k in all_res], key=lambda x: x[1])
        print("Best model index is ", best_model[-1][0])
        best_par = all_res[best_model[-1][0]][1]
        # computes correct rate for every trial
        all_cr = []
        all_is_correct = []
        for i in range(len(trial_features)):
            is_correct = np.array(best_par.predict(trial_features[i])) == np.array(trial_labels[i])
            cr = np.nanmean(is_correct)
            all_cr.append(cr)
            all_is_correct.append(copy.deepcopy(np.array(is_correct, dtype=np.int)))
        print("Acg cr over trials : ", np.mean(np.array(all_cr)))
        # print("-" * 50)
        hyperpar_res.append([s_i, np.mean(np.array(all_cr)), copy.deepcopy(best_par)])

    # Find the overall best model
    best_model = sorted(hyperpar_res, key=lambda x: x[1])
    print("Best model s is ", hidden_size_range[best_model[-1][0]])
    best_par = best_model[-1][-1]
    all_cr = []
    all_is_correct = []
    for i in range(len(trial_features)):
        is_correct = np.array(best_par.predict(trial_features[i])) == np.array(trial_labels[i])
        cr = np.nanmean(is_correct)
        all_cr.append(cr)
        all_is_correct.append(copy.deepcopy(np.array(is_correct, dtype=np.int)))
    # Save data
    if "Fitted_Data" not in os.listdir("../../Data"):
        os.mkdir("../../Data/Fitted_Data")
    np.save("../../Data/Fitted_Data/{}-perceptron_cr.npy".format(config["filename"].split("/")[-1].split(".")[-2]),
            all_cr)
    np.save("../../Data/Fitted_Data/{}-perceptron_weight.npy".format(
        config["filename"].split("/")[-1].split(".")[-2]), best_par)
    np.save("../../Data/Fitted_Data/{}-perceptron_is_correct.npy".format(
        config["filename"].split("/")[-1].split(".")[-2]),
        all_is_correct)
    print("Finished saving data.")



if __name__ == '__main__':
    config = {
        "filename": "../../Data/TestExample/10_trial_data_Omega-with_Q-inf.pkl",
        "save_base": "../../Data/TestExample/"
    }
    dynamicStrategyFitting(config)
    staticStrategyFitting(config)



