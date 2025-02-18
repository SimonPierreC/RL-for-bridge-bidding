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


def perform_bids(x1, x2, scores, history, Q_models, nb_layers):
    num_ones = np.count_nonzero(history == 1)
    bidding_history = history.copy()
    for i in range(num_ones, nb_layers):
        # Détermine la main en fonction du joueur actif
        hand = x1 if (i+1) % 2 == 1 else x2
        state = torch.tensor(hand, dtype=torch.float32) if i == 0 else torch.tensor(
            np.concatenate([hand, bidding_history]), dtype=torch.float32)

        highest_bid = np.max(np.where(bidding_history == 1)) if np.any(
            bidding_history == 1) else -1

        # passer l'état dans le modèle Q[i] et obtenir les valeurs Q[i]
        with torch.no_grad():
            # prédiction du modèle Q[i] pour le state
            q_values = Q_models[i](state)

        q_values_masked = q_values.clone()
        q_values_masked[:highest_bid+1] = -float('inf')

        # sélection greedy max(Q)   a mettre la regle next_a>derniere enchere
        next_a = torch.argmax(q_values_masked).item()
        bidding_history[next_a] = 1  # met à jour l'historique des enchères

        if next_a == len(bidding_history) - 1 or i == nb_layers - 1:
            last_action = next_a  # sauvegarde de la dernière action
            last_layer = i
            return scores[last_action], last_layer


def algo_p(action, x1, x2, scores, bidding_history, Q_models, nb_layers):
    updated_bid_history = bidding_history.copy()
    updated_bid_history[action] = 1
    if np.count_nonzero(updated_bid_history) == nb_layers:
        return scores[action]
    final_score, _ = perform_bids(
        x1, x2, scores, updated_bid_history, Q_models, nb_layers)
    return final_score


def assess_models(Q_models, data_set):
    scores_layers = [0 for k in range(len(Q_models))]
    N_layers = [0 for k in range(len(Q_models))]
    for k in range(len(data_set)):
        x1, x2 = data_set.iloc[k, :52], data_set.iloc[k, 52:104]
        r = data_set.iloc[k, 104:140]
        score, last_layer = perform_bids(
            x1, x2, r, np.zeros(36), Q_models, len(Q_models))
        scores_layers[last_layer] += score
        N_layers[last_layer] += 1
    return np.array(scores_layers)/np.array(N_layers)
