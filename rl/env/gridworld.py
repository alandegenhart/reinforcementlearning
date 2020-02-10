#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gridworld environment

This module defines a basic grid-style environment used to test and evaluate
discrete reinforcement learning algorithms. The environment consists of a grid
of squares.
"""

# Imports
import numpy as np
import gym
from gym.envs.toy_text import discrete


class GridWorld(discrete.DiscreteEnv):
    
    def __init__(self):
        # Define grid dimensions
        self.n_x = 12
        self.n_y = 4

        # Define mappings from actions to x-y movement on the grid. Actions are
        # coded around as angles, such that 0 corresponds to RIGHT, 1 
        # corresponds to UP, etc.
        self.nA = 4  # Property of the DiscreteEnv class
        self.action_loc_map = np.full((self.nA, 2), 0, dtype=int)
        self.action_loc_map[0, :] = [1, 0]  # RIGHT
        self.action_loc_map[1, :] = [0, 1]  # UP
        self.action_loc_map[2, :] = [-1, 0]  # LEFT
        self.action_loc_map[3, :] = [0, -1]  # DOWN

        # Define properties -- DiscreteEnv properties
        self.nS = self.n_x * self.n_y # Number of states
        self.P = []  # Transitions
        self.isd = []  # Initial state distribution

        # Define properties -- GridWorld
        self.define_grid()
       
    def reset(self):
        """Reset the environment."""
        self.loc = np.full((2,), 0, dtype=int)

    def step(self, action):
        """Take the specified action."""

        # Add the action to the current location
        new_loc = self.loc + self.action_loc_map[action, :]

        # Check to see if the updated location is outside of the bounds. Note
        # that the location is in cartesian coordinates, which is [x, y], which
        # is opposite of the standard [row, col] indexing
        if ((new_loc[0] in range(self.n_x)) and 
            (new_loc[1] in range(self.n_y))):
            self.loc = new_loc

        # Check to see if the cliff was encountered. If so, move back to the
        # start.
        r = self.reward_grid[self.loc[0], self.loc[1]]
        if r == self.cliff_val:
            self.loc = [0, 0]

        # Outputs
        s = self.loc_to_state()
        d = self.terminal_grid[self.loc[0], self.loc[1]]
        info = {}

        return (s, r, d, info)
        
    def render(self):
        """Display the current state of the map."""
        return None
        
    def define_grid(self):
        """Define grid for environment.

        Here we setup a grid and specify the reward for each location. In the
        default implementation, the reward for most spaces is -1, while that of
        the cliff is -100.

        Note that the layout of the grid matrices is flipped such that they can
        be indexed using the x and y position (e.g., reward = reward_grid[x,y]).
        """

        # Initialize reward grid. This defines the reward for each location in
        # the grid.
        self.cliff_val = -100
        self.reward_grid = np.full((self.n_x, self.n_y), -1, dtype=int)
        self.reward_grid[1:-1, 0] = self.cliff_val

        # Define the terminal state grid. This defines whether or not each state
        # in the grid is terminal or not.
        self.terminal_grid = np.full(
            (self.n_x, self.n_y), False, dtype=bool)
        self.terminal_grid[-1, 0] = True  # Only the end is a terminal state

    def state_to_loc(self):
        """Map state value to grid location."""
        x = []
        y = []
        return loc

    def loc_to_state(self):
        """Map location to state value.
        
        State values are ordered from the start in ascending order by columns.
        E.g., a location of [3, 0] has a state value of 3, while a location of
        [1, 2] has a value of 7.
        """
        return self.loc[0] * self.n_y + self.loc[1]
