#!/usr/bin/env python3
"""Module to calculate the learning rate decay using inverse time decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using inverse time decay in a stepwise fashion.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate at which
            alpha will decay.
        global_step (int): The number of passes of gradient descent that
            have elapsed.
        decay_step (int): The number of passes of gradient descent that should
            occur before alpha is decayed further.

    Returns:
        float: The updated value for alpha.
    """
    step_bloc = global_step // decay_step
    return alpha / (1 + (decay_rate * step_bloc))
