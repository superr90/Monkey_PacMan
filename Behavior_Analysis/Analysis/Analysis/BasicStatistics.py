'''
Description:
    Compute basic statistics (e.g., the number of rounds per game and the number of games per day) based on all the data.
'''
import numpy as np
import pickle
import copy

# The number of rewards for different entities (id:reward amount)
reward_amount = {
        1:2, # bean
        2:4, # energizer
        3:3, # cherry
        4:5, # strawberry
        5:8, # orange
        6:12, # apple
        7:17, # melon
        8:8, # ghost
    }

# ==================================================

def _rewardDiff(begin, end):
    '''
    Compute the number of pellets eaten on a Pacman moving path.
    :param begin: (2-tuple) The beginning position of path.
    :param end: (2-tuple) The end position of path.
    :return: (int) The number of eaten pellets.
    '''
    if isinstance(begin.beans, float):
        begin_num = 0
    else:
        begin_num = len(begin.beans)
    if isinstance(end.beans, float):
        end_num = 0
    else:
        end_num = len(end.beans)
    return  begin_num - end_num


def _ghostEaten(ghost_status):
    '''
    Compute the number of ghosts eaten in a game.
    :param ghost_status: (list) Status of ghosts.
    :return: (int) The number of ghosts eaten.
    '''
    ghost1_is_dead = ghost_status.ifscared1.apply(lambda x: int(x== 3)).values
    ghost1_diff = np.diff(ghost1_is_dead)
    ghost2_is_dead = ghost_status.ifscared2.apply(lambda x: int(x == 3)).values
    ghost2_diff = np.diff(ghost2_is_dead)
    return len(np.where(ghost1_diff==1)[0]) + len(np.where(ghost2_diff==1)[0])


def _energizerDiff(begin, end):
    '''
    Compute the number of energizers eaten on a Pacman moving path.
    :param begin: (2-tuple) The beginning position of path.
    :param end: (2-tuple) The end position of path.
    :return: (int) The number of eaten energizers.
    '''
    if isinstance(begin.energizers, float):
        begin_num = 0
    else:
        begin_num = len(begin.energizers)
    if isinstance(end.energizers, float):
        end_num = 0
    else:
        end_num = len(end.energizers)
    return  begin_num - end_num


def _fruitDiff(begin, end):
    '''
    Compute the reward amount of fruits eaten on a Pacman moving path.
    :param begin: (2-tuple) The beginning position of path.
    :param end: (2-tuple) The end position of path.
    :return: (int) The amount of rewards of eaten fruits.
    '''
    if isinstance(begin.fruitPos, float):
        begin_num = 0
    else:
        begin_num = 1
    if isinstance(end.fruitPos, float):
        end_num = 0
    else:
        end_num = 1
    if isinstance(begin.fruitPos, float) or isinstance(begin.Reward, float):
        reward = 0
    else:
        reward = reward_amount[int(begin.Reward)]
    return  reward * (begin_num - end_num)

# ==================================================

def computeStatistics(monkey_data, monkey):
    '''
    Compute basic statistics.
    :param monkey_data: (pandas.DataFrame) A tale of data.
    :param monkey: (str) The name of monkey ("Omega", "Patamon").
    :return: (dict) The basic statistics record.
    '''
    monkey_records = {}
    # The number of rounds
    monkey_rounds = {}
    monkey_trial_name = np.sort(np.unique(monkey_data.file.values))
    monkey_except = {}
    for t in monkey_trial_name:
        game_name = t.split(".")[0].split("-")
        round_name = game_name[1]
        parse_game_name = "-".join([game_name[0]] + game_name[2:-1])
        if parse_game_name not in monkey_rounds:
            monkey_rounds[parse_game_name] = [round_name]
        else:
            monkey_rounds[parse_game_name].append(round_name)
        if game_name[-1] != "1":
            tmp = "-".join(game_name[:-1])
            if tmp in monkey_except:
                if int(game_name[-1]) > monkey_except[tmp]:
                    monkey_except[tmp] = int(game_name[-1])
            else:
                monkey_except[tmp] = int(game_name[-1])
    monkey_records["round_num"] = monkey_rounds
    monkey_records["except"] = monkey_except
    monkey_num_rounds = [len(monkey_rounds[each]) for each in monkey_rounds]
    print("The maximum number of game rounds for {} : ".format(monkey), max(monkey_num_rounds))
    # The number of pelltes and energizers
    monkey_pellet_num = {each: [] for each in monkey_rounds}
    monkey_energizer_num = {each: [] for each in monkey_rounds}
    cnt = 0
    for g in monkey_rounds:
        tmp = g.split("-")
        if tmp[0] + "-1-" + "-".join(tmp[1:]) in monkey_except:
            tmp = tmp[0] + "-1-" + "-".join(tmp[1:]) + "-{}.csv".format(
                monkey_except[tmp[0] + "-1-" + "-".join(tmp[1:])])
        else:
            tmp = tmp[0] + "-1-" + "-".join(tmp[1:]) + "-1.csv"
        temp_data = monkey_data[monkey_data.file == tmp].reset_index(drop=True).iloc[0]
        monkey_pellet_num[g] = len(temp_data.beans)
        monkey_energizer_num[g] = len(temp_data.energizers)
    print(cnt)
    monkey_records["all_pellets"] = monkey_pellet_num
    monkey_records["all_energizers"] = monkey_energizer_num
    # Statistics for each round
    monkey_round_tile = {each: [] for each in monkey_rounds}  # Time for every game and round (25/60 * tiles)
    monkey_round_reward = {each: [] for each in monkey_rounds}  # Reward num for every round
    monkey_round_energizer = {each: [] for each in monkey_rounds}  # Reward num for every round
    monkey_round_eaten_ghost = {each: [] for each in monkey_rounds}  # Num of ghost eaten in a game
    monkey_round_drop = {each: [] for each in monkey_rounds}  # Drop num for every round
    for i, each in enumerate(monkey_rounds.keys()):
        # if i % 10 == 0:
        #     print("| Finished {} Omega Games |".format(i))
        for round in monkey_rounds[each]:
            round_name = "{}-{}-{}".format(each.split("-")[0], round, "-".join(each.split("-")[1:]))
            if round_name in monkey_except:
                id = monkey_except[round_name]
            else:
                id = 1
            # tmp = tmp[0] + "-1-" + "-".join(tmp[1:]) + "-{}.csv".format(id)
            temp = monkey_data[monkey_data.file == "{}-{}.csv".format(round_name, id)].reset_index(drop=True)
            monkey_round_tile[each].append(temp.shape[0])
            monkey_round_reward[each].append(_rewardDiff(temp.iloc[0], temp.iloc[-1]))
            monkey_round_energizer[each].append(_energizerDiff(temp.iloc[0], temp.iloc[-1]))
            monkey_round_eaten_ghost[each].append(_ghostEaten(temp[["ifscared1", "ifscared2"]]))
            monkey_round_drop[each].append(
                reward_amount[1] * _rewardDiff(temp.iloc[0], temp.iloc[-1]) +
                reward_amount[8] * _ghostEaten(temp[["ifscared1", "ifscared2"]]) +
                reward_amount[2] * _energizerDiff(temp.iloc[0], temp.iloc[-1]) +
                _fruitDiff(temp.iloc[0], temp.iloc[-1])
            )
    monkey_round_time = np.concatenate(list(monkey_round_tile.values())) * (25 / 60)
    monkey_records["round_tile"] = monkey_round_tile
    print("The maximum time within a round for {} : ".format(monkey), np.max(monkey_round_time))
    monkey_round_reward_all = np.concatenate(list(monkey_round_reward.values()))
    monkey_records["round_reward"] = monkey_round_reward
    monkey_records["round_energizer"] = monkey_round_energizer
    print("The maximum rewards for {} : ".format(monkey), np.max(monkey_round_reward_all))
    monkey_round_drop_all = np.concatenate(list(monkey_round_drop.values()))
    monkey_records["round_drop"] = monkey_round_drop
    print("The maximum drops for {} : ".format(monkey), np.max(monkey_round_drop_all))
    monkey_round_eaten_ghost_all = np.concatenate(list(monkey_round_eaten_ghost.values()))
    monkey_records["round_eaten_ghosts"] = monkey_round_eaten_ghost
    print("The maximum eaten ghosts for {} : ".format(monkey), np.max(monkey_round_eaten_ghost_all))
    # The number of games for every day
    monkey_game_name_list = monkey_rounds.keys()
    monkey_game_per_day = {}
    for each in monkey_game_name_list:
        temp = each.split("-")
        date = "-".join(temp[1:])
        if date not in monkey_game_per_day:
            monkey_game_per_day[date] = [temp[0]]
        else:
            monkey_game_per_day[date].append(temp[0])
    monkey_records["game_per_day"] = monkey_game_per_day
    monkey_num_game_per_day = [len(monkey_game_per_day[each]) for each in monkey_game_per_day]
    print("The maximum number of games/day for {} : ".format(monkey), max(monkey_num_game_per_day))
    # The number of rounds for every day
    monkey_rounds_name_list = monkey_data.file.unique()
    monkey_round_per_day = {}
    for each in monkey_rounds_name_list:
        temp = each.split("-")
        date = "-".join(temp[2:6])
        if date not in monkey_round_per_day:
            monkey_round_per_day[date] = ["-".join(temp[:2])]
        else:
            monkey_round_per_day[date].append("-".join(temp[:2]))
    monkey_records["round_per_day"] = monkey_round_per_day
    monkey_num_round_per_day = [len(monkey_round_per_day[each]) for each in monkey_round_per_day]
    print("The maximum number of round/day for {} : ".format(monkey), max(monkey_num_round_per_day))
    return monkey_records

# ==================================================

if __name__ == '__main__':
    # =================================================================================
    # Note: This main function needs all the data for computations.
    #       We provided a pre-computed record as "Data/Behavior_Plot_Data/basic_statistics.npy".
    # =================================================================================

    print("=" * 50)
    # Read data
    filename = "..." # path of all recorded data
    all_data = pickle.load(open(filename, "rb"))
    print("Finished reading data.")
    print("-" * 50)
    # Split data for two monkeys
    omega_data = copy.deepcopy(all_data[all_data.file.str.contains("Omega")]).sort_index()
    patamon_data = copy.deepcopy(all_data[all_data.file.str.contains("Patamon")]).sort_index()
    print("Finished splitting data.")
    print("Omega data shape : ", omega_data.shape)
    print("Patamon data shape : ", patamon_data.shape)
    print("-" * 50)
    # Compute basic statistics
    omega_records = None if len(omega_data) == 0 else computeStatistics(omega_data, "Omega")
    print("Finished statistics computing for Omega.")
    print("-" * 50)
    patamon_records = None if len(patamon_data) == 0 else computeStatistics(patamon_data, "Patamon")
    print("Finished statistics computing for Patamon.")
    print("-" * 50)
    # Save data
    np.save("../../../Data/Behavior_Plot_Data/basic_statistics.npy", {"Omega":omega_records, "Patamon":patamon_records})
    print("Finished saving data.")
    print("=" * 50)