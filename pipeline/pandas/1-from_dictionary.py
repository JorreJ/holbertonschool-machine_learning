#!/usr/bin/env python3
"""Creates and initializes a pandas DataFrame with specified indices."""

import pandas as pd


data = (
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["One", "Two", "Three", "Four"]
    }
)
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
