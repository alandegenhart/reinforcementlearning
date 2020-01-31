# Import libraries
import numpy as np
import gym


class OnPolicyTDControl:
	"""On-policy temporal difference (TD) control (aka Sarsa)."""

	def __init__(self, env):
		# Add environment to the class and determine the size of the action and
		# observation spaces
		self.env = env
		self.n_actions = env.action_space.n
		self.n_states = env.observation_space.n

		# Initialize policy and action values
		self.P = np.ones([self.n_states, self.n_actions]) / self.n_actions
		self.Q = np.zeros([self.n_states, self.n_actions])


	def run(self, episodes,
		block_size=100,
		max_episode_steps=100,
		alpha=0.1,
		gamma=0.95,
		eps=0.05):
		"""Run learning algorithm."""

		# Add parameters to object. This makes it much easier to access these
		# values from the various methods.
		self.alpha = alpha
		self.gamma = gamma
		self.eps = eps
		self.block_size = block_size

		# Initialize matrices used to record results
		self.episode_steps = np.zeros(episodes)
		self.episode_reward = np.zeros(episodes)

		# Iterate over episodes
		for i in range(episodes):
			# Step through an episode
			s = self.env.reset()
			a = self.sample_action(s)
			for t in range(max_episode_steps):
				# Take action
				s_next, r, done, info = self.env.step(a)

				# Choose next action based on the current policy
				a_next = self.sample_action(s_next)

				# Update action value and policy
				self.update_action_value(s, a, r, s_next, a_next)
				self.update_policy(s)

				# Update current state and action
				a = a_next
				s = s_next

				# Check if the state is terminal
				if done:
					# Keep track of the outcome of the episode
					self.episode_steps[i] = t + 1
					self.episode_reward[i] = r  # Note this is the last reward
					break

			# Take any actions at the end of an episode (display status, etc.)
			self.episode_finished(i)

		return self.episode_steps, self.episode_reward


	def sample_action(self, s):
		"""Sample an action from the current policy."""
		return np.random.choice(range(self.n_actions), p=self.P[s, :])


	def update_action_value(self, s, a, r, s_next, a_next):
		"""Update action values.
		
		Inputs:
		-- s -- state at time t
		-- a -- action taken at time t
		-- r -- reward received from taking action a
		-- s_next -- new state after taking action a
		-- a_next -- the next action according to the current policy

		Currently this method implements Sarsa. Eventually this method will be
		overridden by child classes implementing other methods (e.g.,
		Q-learning, expected sarsa, etc.).
		"""

		# Calculate target, update action value
		targ = r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a]
		self.Q[s, a] = self.Q[s, a] + self.alpha * targ


	def update_policy(self, s):
		"""Epsilon-greedy policy update given current action values."""
		mask = np.arange(self.n_actions) == np.argmax(self.Q[s, :])
		self.P[s, mask] = 1 - self.eps + (self.eps/self.n_actions)
		self.P[s, ~mask] = self.eps/self.n_actions


	def episode_finished(self, episode):
		"""Take actions at the end of an episode."""
		# Display a status update each block
		if (episode + 1) % self.block_size == 0:
			# Calculate reward rate over the last block_size episode_steps
			reward_rate = sum(self.episode_reward[episode - self.block_size + 1:
				episode + 1]) / self.block_size
			# Display status message
			print('Episode {} finished: reward rate = {}'.format(episode + 1, 
				reward_rate))



""" To-do:
- it might be helpful to add methods to initialize an episode and to take a
step in an episode. This would make it easier to create child classes that
implement slightly different algorithms (Q-learning for example)
- plotting of results/action values?
- function to save checkpoint
- some way to schedule epsilon
"""



