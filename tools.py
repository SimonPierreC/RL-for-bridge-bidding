from endplay.types import Deal, Contract, Denom, Vul, Player, Card, Hand
from endplay.dealer import generate_deal
from endplay.dds.ddtable import calc_dd_table
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count


def random_two_hands():
    """"Generate randomly two hands of the same game."""
    d = generate_deal()
    return d.north, d.south


def random_remaining_hands(north, south):
    """Generate randomly two heads to complete the deal"""
    d = Deal()
    d.north, d.south = north, south
    return generate_deal(predeal=d)


def tricks_dd(d):
    """Returns the table of the greatest number of tricks achievable for each trump and each declarer."""
    return calc_dd_table(d)


def tricks_tb_to_score(tricks):
    """Converts number of tricks to score, for north or south declarers only 
    and for each possible contract from 1 club to 7NT.
    None is vulnerable"""
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
    """"Compute the mean of the achievable scores for each contract, 
    north or souoth declarer, over five generation of east and west hands"""
    deals = [random_remaining_hands(north, south) for _ in range(N)]
    score_results = [tricks_tb_to_score(tricks_dd(d))
                     for d in deals]
    score_means = {}
    for p in [Player.north, Player.south]:
        score_means[p] = np.mean(np.concat([r[p].reshape(1, -1) for r in score_results], axis=0),
                                 axis=0)
    return score_means


def score_to_imp(diff):
    """Converts a score diff to imps."""
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
    """Convert the imps to costs"""
    costs = {}
    for p in [Player.north, Player.south]:
        costs[p] = np.array([score_to_imp(diff)
                             for diff in np.max(scores[p]) - scores[p]])
    return costs


def reward_from_cost(imps):
    """Turns costs into rewards between 0 and 1"""
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


def ohe_to_hand(ohe_hand):
    str_hand = ["", "", "", ""]
    index_to_card = ['2', '3', '4', '5', '6',
                     '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    for i in range(4):
        for j in range(13):
            if ohe_hand[i*13+j] == 1:
                # Spades are firsts
                str_hand[3-i] += index_to_card[j]
    return Hand(".".join(str_hand))


def generate_one_line():
    north, south = random_two_hands()
    rewards = reward_from_cost(
        costs_imp(mean_scores(north, south)))
    return np.concatenate([one_hot_hands(north),
                           one_hot_hands(south),
                           rewards[Player.north],
                           rewards[Player.south]])


def generate_df(N):
    lines = []
    for _ in tqdm(range(N)):
        lines.append(generate_one_line().reshape(1, -1))
    return pd.DataFrame(data=np.concatenate(lines, axis=0))


def line_parallel(_):
    return generate_one_line()


def generate_df_parallel(N):
    num_workers = cpu_count()
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(line_parallel, range(N)), total=N))
    return pd.DataFrame(data=np.array(results))


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
