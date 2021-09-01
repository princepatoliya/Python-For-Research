# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:15:06 2021

@author: SHRIYU
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 100 # our points
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size = n) # create random veriables size = 100(n)
# beta_0 - constance and beta_1 - multiplly with x
# norm.rvs - loc -means norm is zero
y = beta_0 + beta_1 * x + ss.norm.rvs(loc = 0, scale = 1, size = n) 


plt.figure()
plt.plot(x, y, "o", ms=5)
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel("x")
plt.ylabel("y")

# MCQ

def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))
def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x
rss = compute_rss(estimate_y(x, beta_0, beta_1), y)

#MCQ 

rss = []
slopes = np.arange(-10, 15, 0.01)
slopes = np.arange(-10, 15, 0.001) # for MCQ
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x)**2))

ind_min = np.argmin(rss)

plt.figure()
plt.plot(slopes, rss)
plt.xlabel("SLOPES")
plt.ylabel("RSS")



