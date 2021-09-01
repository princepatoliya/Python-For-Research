"""
Created on Sat Sep 26 14:36:09 2020

@author: SHRIYU
"""

import pandas as pd
import numpy as np
whisky = pd.read_csv("whiskies.txt")
whisky["regions"] = pd.read_csv("regions.txt")
print(whiskey.head)
print(whisky.head)
flavors = whisky.iloc[:, 2:14]
print(flavors)

corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_falvors")
