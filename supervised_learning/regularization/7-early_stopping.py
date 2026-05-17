#!/usr/bin/env python3
"""Module to determine if gradient descent should stop early."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determine if you should stop gradient descent early.

    Args:
        cost (float): The current validation cost of the neural network.
        opt_cost (float): The lowest recorded validation cost so far.
        threshold (float): The threshold used for early stopping.
        patience (int): The patience count used for early stopping.
        count (int): The count of how long the threshold has not been met.

    Returns:
        tuple: (bool, int) whether the network should be stopped early,
               followed by the updated count.
    """
    # On vérifie si le coût actuel s'est amélioré par rapport au coût optimal
    # d'une valeur supérieure au seuil (threshold) imposé.
    if opt_cost - cost > threshold:
        # Si oui, le modèle progresse bien ! On réinitialise le compteur.
        count = 0
    else:
        # Si non, la progression stagne ou régresse. On incrémente la patience.
        count += 1

    # Si le compteur atteint ou dépasse la patience autorisée, on s'arrête.
    if count >= patience:
        return True, count

    return False, count
