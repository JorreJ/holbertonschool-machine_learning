#!/usr/bn/env python3

import numpy as np
import pandas as pd
import string

def from_numpy(array):
    df = pd.DataFrame(array)
    df.columns = list(string.ascii_uppercase[:df.shape[1]])
    return df
