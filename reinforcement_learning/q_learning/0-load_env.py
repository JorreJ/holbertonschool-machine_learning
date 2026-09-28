#!/usr/bin/env python3
"""Module that loads the Gymnasium FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the Gymnasium FrozenLake-v1 environment.

    Args:
        desc (list of list of char, optional): Custom map layout for the
            environment. Defaults to None.
        map_name (str, optional): Pre-existing map name (e.g., '4x4', '8x8').
            Defaults to None.
        is_slippery (bool, optional): Whether ice is slippery.
            Defaults to False.

    Returns:
        gym.Env: The loaded Gymnasium FrozenLake-v1 environment instance.
    """
    return gym.make(
        'FrozenLake-v1',
        render_mode='ansi',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
