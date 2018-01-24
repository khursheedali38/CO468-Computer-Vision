# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:45:56 2018

@author: Khursheed Ali
"""

import math as m

def quantize(x):
    x = m.degrees(x)
    if (0 <= x < 22.5) or (157.5 <= x < 180):
        x = 0
    elif (22.5 <= x < 67.5):
        x = 45
    elif (67.5 <= x < 112.5):
        x = 90
    elif (112.5 <= x < 157.5):
        x = 135
    return x