from endplay.types import Deal, Contract, Denom, Vul, Player, Card
from endplay.dealer import generate_deal
from endplay.dds.ddtable import calc_dd_table
import numpy as np


def random_two_hands():
    d = generate_deal()
    return d.north, d.south


def random_remaining_hands(north, south):
    d = Deal()
    d.north, d.south = north, south
    return generate_deal(predeal=d)


def tricks_dd(d):
    return calc_dd_table(d)


def tricks_tb_to_score(tricks):
    scores = {Player.north: [0], Player.south: [0]}
    for c in [Contract(level=l, denom=d, declarer=dc)
              for l in range(1, 8)
              for d in [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
              for dc in [Player.north, Player.south]]:
        c.result = tricks[c.declarer, c.denom] - (c.level + 6)
        scores[c.declarer].append(c.score(Vul.none))

    scores[Player.north], scores[Player.south] =\
        np.array(scores[Player.north]), np.array(scores[Player.north])
    return scores


def mean_scores(north, south, N=5):
    deals = [random_remaining_hands(north, south) for _ in range(N)]
    score_results = [tricks_tb_to_score(tricks_dd(d))
                     for d in deals]
    score_means = {}
    for p in [Player.north, Player.south]:
        score_means[p] = np.mean(np.concat([r[p].reshape(1, -1) for r in score_results], axis=0),
                                 axis=0)
    return score_means


def score_to_imp(diff):
    assert diff >= 0
    imp_table = [
        (0, 0), (20, 1), (50, 2), (90, 3), (130, 4), (170, 5), (220, 6),
        (260, 7), (310, 8), (360, 9), (420, 10), (490, 11), (590, 12),
        (740, 13), (890, 14), (1090, 15), (1290, 16), (1490, 17), (1740, 18),
        (1990, 19), (2240, 20), (2490, 21), (2990, 22), (3490, 23), (3990, 24)
    ]

    for threshold, imp in imp_table:
        if diff <= threshold:
            return imp
    return 24


def costs_imp(scores):
    costs = {}
    for p in [Player.north, Player.south]:
        costs[p] = np.array([score_to_imp(diff)
                             for diff in np.max(scores[p]) - scores[p]])
    return costs


def reward_from_cost(imps):
    rewards = {}
    for p in [Player.north, Player.south]:
        rewards[p] = 1 - (imps[p] - np.min(imps[p]))/25
    return rewards


def one_hot_hands(hand):
    cards_index = {Card(name=f"{denom}{rank}"): i
                   for i, (denom, rank) in enumerate((d, r) for d in ["C", "D", "H", "S"]
                                                     for r in [2, 3, 4, 5, 6, 7, 8, 9, "T", "J", "Q", "K", "A"])}
    ohe = np.zeros(52)

    for card in hand:
        ohe[cards_index[card]] = 1
    return ohe


if __name__ == '__main__':
    north, south = random_two_hands()
    north.pprint()
    south.pprint()

    scores = mean_scores(north, south)
    costs = costs_imp(scores)
    rewards = reward_from_cost(costs)
    print(rewards)

    print(one_hot_hands(north))
    print(one_hot_hands(south))
