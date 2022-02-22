'''
Description:
    Tool functions for the analysis.
'''

import numpy as np


def scaleOfNumber(num):
    '''
    Obtain the scale of a number.
    :param num: The number
    :return:
    '''
    if num >= 1:
        order = len(str(num).split(".")[0])
        return 10 ** (order - 1)
    elif num == 0:
        return 1
    else:
        order = str(num).split(".")[1]
        temp = 0
        for each in order:
            if each == "0":
                temp += 1
            else:
                break
        return 10 ** (-temp - 1)


def makeChoice(prob):
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


def _estimationVagueLabeling(contributions, all_agent_name):
    '''
    Determine the time-step strategy with the largest weight. The strategy is "vague" when the weight difference
    between the largest and the second largest one is no more than 0.1.
    :param contributions: (list) Fitted agent weights with the normalization.
    :param all_agent_name: (list) All the agents.
    :return: Time-step strategy.
    '''
    if isinstance(contributions, float):
        return np.nan
    sorted_contributions = np.sort(contributions)[::-1]
    if sorted_contributions[0] - sorted_contributions[1] <= 0.1 :
        return ["vague"]
    else:
        label = all_agent_name[np.argmax(contributions)]
        return [label]


def _estimationVagueLabeling2(contributions):
    if isinstance(contributions, float):
        return np.nan
    all_agent_name = ["global", "local", "evade(Blinky)", "evade(Clyde)", "approach", "energizer"]
    sorted_contributions = np.sort(contributions)[::-1]
    if sorted_contributions[0] - sorted_contributions[1] <= 0.1:
        return "vague"
    else:
        label = all_agent_name[np.argmax(contributions)]
        return label


def _closestScaredDist(pacmanPos, ghost1Pos, ghost2Pos, ifscared1, ifscared2, locs_df):
    PG1 = 0 if pacmanPos == ghost1Pos else locs_df[pacmanPos][ghost1Pos]
    PG2 = 0 if pacmanPos == ghost2Pos else locs_df[pacmanPos][ghost2Pos]
    # 处理一个鬼scared，一个鬼normal的情况。
    if ifscared1 <= 3:
        PG1 = 999
    if ifscared2 <= 3:
        PG2 = 999
    return min(PG1, PG2)


if __name__ == '__main__':
    print(scaleOfNumber(0.1204))
