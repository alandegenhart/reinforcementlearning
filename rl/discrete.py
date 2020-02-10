# Import libraries
import numpy as np


class TDControl:
	"""Base class for temporal difference (TD) control."""

	def __init__(self, env, init_mode='random'):
		# Add environment to the class and determine the size of the action and
		# observation spaces
		self.env = env
		self.n_actions = env.action_space.n
		self.n_states = env.observation_space.n

		# Initialize policy
		self.P = np.ones([self.n_states, self.n_actions]) / self.n_actions

		# Check to see if initialization option is valid
		if init_mode not in ['zeros', 'random', 'optimistic']:
			print('Warning: invalid init_mode specified. Defaulting to random.')
			init_mode = 'random'

		# Initialize action values
		if init_mode is 'zeros':
			self.Q = np.zeros([self.n_states, self.n_actions])
		elif init_mode is 'random':
			self.Q = np.random.uniform(0, 1, (self.n_states, self.n_actions))
		elif init_mode is 'optimistic':
			self.Q = np.ones([self.n_states, self.n_actions]) * 100

		# Set terminal state to 0
		self.Q[-1,:] = 0

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
			
			# Perform initialization procedures
			self.s = self.env.reset()
			self.episode_init()

			# Iterate over steps in the episode
			for t in range(max_episode_steps):

				# Perform operations for current step
				done = self.episode_step()

				# Check if the state is terminal
				if done:
					# If the state is terminal, make sure that Q is zero.
					# NOTE: this might cause problems if the episode times out
					self.Q[self.s,:] = 0

					# Keep track of the outcome of the episode
					self.episode_steps[i] = t + 1
					self.episode_reward[i] = self.r  # Outcome of episode
					break

			# Take any actions at the end of an episode (display status, etc.)
			self.episode_finished(i)

		return self.episode_steps, self.episode_reward

	def episode_init(self):
		"""Perform initialization operations for an episode."""

	def episode_step(self):
		"""Perform operations for a single episode step.
		
		This is an empty method for the base class. It is expected that this
		will be over-ridden by child classes.

		Returns:
		:done 	Boolean indicate whether a terminal state was reached
		"""
		return False

	def sample_action(self, s):
		"""Sample an action from the current policy."""
		return np.random.choice(range(self.n_actions), p=self.P[s, :])

	def update_action_value(self):
		"""Update action values.
		
		This is an empty method for the base class. It is expected that this
		will be over-ridden by child classes.
		"""

	def update_policy(self, s):
		"""Epsilon-greedy policy update given current action values."""
		mask = np.arange(self.n_actions) == np.argmax(self.Q[s, :])
		self.P[s, mask] = 1 - self.eps + (self.eps/self.n_actions)
		self.P[s, ~mask] = self.eps/self.n_actions

	def episode_finished(self, episode):
		"""Take actions at the end of an episode."""

		# Display a status update at the end of each block
		if (episode + 1) % self.block_size == 0:
			# Calculate reward rate over the last block_size episode_steps
			reward_rate = sum(self.episode_reward[episode - self.block_size + 1:
				episode + 1]) / self.block_size
			# Display status message
			print('Episode {} finished: reward rate = {}'.format(episode + 1, 
				reward_rate))

	def info(self):
		"""Test print function for testing inheritance"""
		print('Class: TDControl (base class)')


class Sarsa(TDControl):
	"""On-policy temporal difference control (Sarsa).
	
	"""

	def episode_init(self):
		"""Perform initialization operations for an episode.

		For Sarsa, we need to sample an action at the start of an episode. This
		is needed b/c it is simpler if the main iteration for an episode samples
		actions at the end of each step, meaning that we need to start with an
		initial value before we step into the loop.
		"""
		self.a = self.sample_action(self.s)

	def episode_step(self):
		"""Perform operations for a single episode step."""
		# Take action and choose next action based on the current policy
		self.s_next, self.r, done, info = self.env.step(self.a)
		self.a_next = self.sample_action(self.s_next)

		# Update action value and policy
		self.update_action_value()
		self.update_policy(self.s)

		# Update current state and action
		self.a = self.a_next
		self.s = self.s_next

		return done

	def update_action_value(self):
		"""Update action values."""
		# Calculate target, update action value
		targ = (self.r + self.gamma * self.Q[self.s_next, self.a_next] - 
			self.Q[self.s, self.a])
		self.Q[self.s, self.a] = self.Q[self.s, self.a] + self.alpha * targ


	def info(self):
		"""Test print function for testing inheritance"""
		print('Class: Sarsa')


class QLearning(TDControl):
	"""Off-policy TD control (Q-learning)

	The algorithm for Q-learning is very similar to that of Sarsa but with a few
	exceptions:
	- The action value update is different (off-policy)
	- Because the action value update for Sarsa uses the value of the next
	action selected, it makes sense to do this after taking the action each
	step. For Q-learning, it makes more sense to select the action before taking
	a step each iteration.
	"""

	def episode_step(self):
		"""Perform operations for a single episode step.

		The algorithm for Sarsa assumes that we have already selected an action
		as part of the previous step. This is done because the action value
		update uses this action, so it is most appropriate just to keep the
		selected action.
		"""
		# Choose next action based on the current policy and take action
		self.a = self.sample_action(self.s)
		self.s_next, self.r, done, info = self.env.step(self.a)

		# Update action value and policy
		self.update_action_value()
		self.update_policy(self.s)

		# Update current state and action
		self.s = self.s_next

		return done

	def update_action_value(self):
		"""Update action values.

		"""
		# Calculate target, update action value
		targ = (self.r + self.gamma * np.max(self.Q[self.s_next, :]) - 
			self.Q[self.s, self.a])
		self.Q[self.s, self.a] = self.Q[self.s, self.a] + self.alpha * targ

	def info(self):
		"""Test print function for testing inheritance"""
		print('Class: Q-learning')


class ExpectedSarsa(QLearning):
	"""Expected Sarsa

	Expected Sarsa is identical to Q-learning except the target uses the
	expected value over the next state-action pairs. Thus, we only need to
	override the 'update_action_value' method.
	"""

	def update_action_value(self):
		"""Update action values.

		"""
		# Calculate target, update action value
		target = (self.r + 
			self.gamma * self.P[self.s_next, :] @ self.Q[self.s_next, :].T)
		self.Q[self.s, self.a] = (self.Q[self.s, self.a] + 
			self.alpha * (target - self.Q[self.s, self.a]))

	def info(self):
		"""Test print function for testing inheritance"""
		print('Class: Expected Sarsa')