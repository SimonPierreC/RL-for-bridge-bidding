import numpy as np
import torch
from tqdm import tqdm


# Stratégie epsilon-greedy pour explorer/exploiter les actions
def algo_e(Q, state, epsilon, all_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(all_actions)
    else:
        with torch.no_grad():
            return np.argmax(Q(state))

# Sélectionne une ligne aléatoire du dataset et extrait les informations


def peak_one(data):
    row = data.sample(n=1).iloc[0]  # Sélection aléatoire d’une ligne
    x1 = row[:52].values  # Main du joueur 1
    x2 = row[52:104].values  # Main du joueur 2
    scores = row[104:140].values  # Scores des 36 actions possibles joueur
    return x1, x2, scores


def update(D, state, C, j):
    D[j][0] = torch.concat([D[j][0], state.clone().detach().unsqueeze(0)])
    D[j][1] = torch.concat(
        [D[j][1], torch.tensor(C, dtype=torch.float32).unsqueeze(0)])
    return D


def bid_results(biddings, scores):
    assert np.any(biddings == 1)
    nb_bids = np.count_nonzero(biddings)
    highest_bid = np.max(np.where(biddings == 1))
    last_contract, last_layer = highest_bid, nb_bids - 1
    return scores[last_contract], last_layer


def perform_bids(x1, x2, history, Q_models, nb_layers):
    num_ones = np.count_nonzero(history == 1)
    bidding_history = history.copy()
    for i in range(num_ones, nb_layers):
        # Vérifier si les enchères sont terminées, c'est-à-dire s'il y a eu pass
        if bidding_history[0] == 1:  # Il y a eu pass
            break

        highest_bid = np.max(np.where(bidding_history == 1)) if np.any(
            bidding_history == 1) else -1

        # Déterminer l'enchère suivante
        # Détermine la main en fonction du joueur actif
        hand = x1 if (i+1) % 2 == 1 else x2
        state = torch.tensor(hand, dtype=torch.float32) if i == 0 else torch.tensor(
            np.concatenate([hand, bidding_history]), dtype=torch.float32)
        # calcul de la meilleure q_value légale
        with torch.no_grad():
            q_values = Q_models[i](state)
        q_values_masked = q_values.clone()
        q_values_masked[1:highest_bid+1] = -float('inf')
        next_a = torch.argmax(q_values_masked).item()
        # Mettre à jour l'historique des enchères
        bidding_history[next_a] = 1

    return bidding_history


def legal_bid(action, bidding_history):
    if np.count_nonzero(bidding_history) == 0:
        return True
    if bidding_history[0] == 1:
        return False
    last_bid = np.max(np.where(bidding_history == 1))
    if action > last_bid or action == 0:
        return True
    return False


def algo_p(action, x1, x2, scores, bidding_history, Q_models, nb_layers):
    updated_bid_history = bidding_history.copy()
    if not legal_bid(action, bidding_history):
        return -0.2
    updated_bid_history[action] = 1
    final_biddings = perform_bids(
        x1, x2, updated_bid_history, Q_models, nb_layers)
    final_score, _ = bid_results(final_biddings, scores)
    return final_score


def assess_models(Q_models, data_set):
    scores_layers = [0 for k in range(len(Q_models))]
    N_layers = [0 for k in range(len(Q_models))]
    for k in range(len(data_set)):
        x1, x2 = data_set.iloc[k, :52], data_set.iloc[k, 52:104]
        r = data_set.iloc[k, 104:140]
        bids = perform_bids(x1, x2, np.zeros(36), Q_models, len(Q_models))
        score, last_layer = bid_results(bids, r)
        scores_layers[last_layer] += score
        N_layers[last_layer] += 1
    return np.array(scores_layers)/np.array(N_layers)
