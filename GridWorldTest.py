#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gridworld environment development script.

This script tests the gridworld environment currently in development. This
is a simple grid environment that can be used for discrete rl algorithms. The
environment consists of a 2D grid of possible states. Most states have a
small negative reward, and there is a "cliff" with a large negative reward.

Created on Sun Feb  9 12:53:25 2020

@author: alandegenhart
"""

#%% Setup

import numpy as np
import rl

#%% Define Gridworld object

G = rl.env.GridWorld()
G.reset()

#%% Test out the environment

s, r, d, info = G.step(3)
print('s: {}, r: {}, d: {}, x: {}, y: {}'.format(s, r, d, G.loc[0], G.loc[1]))