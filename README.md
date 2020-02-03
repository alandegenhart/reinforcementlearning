## Reinforcement learning toolkit

This repository contains examples of simple reinforcement learning concepts using OpenAI Gym environments.

**Note:** This repository is in active development, so things are undergoing frequent changes.

### Organization

Currently, this repository consists of: (1) Jupyter notebooks containing example code, plots, etc., and (2) modules that implement various concepts.

Notebooks:
1. `MonteCarloControl.ipynb` -- A working example of Monte Carlo control using a 'frozen lake' grid environment. This notebook is fully self-contained, in that it doesn't rely on any custom modules.
2. `TDControl.ipynb` -- Examples of various temporal difference (TD) learning algorithms (Sarsa, Q-Learning, etc.). This notebook currently uses the `discrete` module, which contains implementations of the various TD algorithms.

Modules:
1. `discrete.py` -- Implementations of temporal difference learning algorithms, currently including Sarsa, Q-learning, and Expected Sarsa.  This will eventually be expanded to include Monte Carlo control and Double-Q learning.
