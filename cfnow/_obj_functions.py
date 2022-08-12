"""
In this module, we can specify the objective functions which the CF generator can use to optimize.
The optimizer will try to get the lowest value possible (minimization) of the objective function under the
condition the CF still a CF.

The requirements for a valid objective function are:
    - Lower values should be better.
    - It should not return negative values or zero. (obj > 0)
"""
import numpy as np


def _obj_manhattan(factual_np, c_cf):
    return sum(np.abs(factual_np - c_cf))
