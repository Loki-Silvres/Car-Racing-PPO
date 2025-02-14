# Car Racing PPO

This repository provides an implementation of the Proximal Policy Optimization (PPO) algorithm tailored for training an agent to master the [CarRacing-v3](https://gym.openai.com/envs/CarRacing-v0/) environment from OpenAI Gym. It aims to offer a straightforward yet powerful setup for both training and evaluating reinforcement learning agents in continuous control tasks.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (or any virtual environment tool)
- Python 3.11
- Trained on RTX 4060 Laptop GPU (optional)

## Installation

Set up your environment and install dependencies by running:

```bash
conda create -n CRP python=3.11
conda activate CRP
pip install -r requirements.txt
```

## Training the Agent

To begin training PPO agent on the CarRacing environment, execute:

```bash
python3 src/train.py
```

This script will initialize the training process, log performance metrics, and save model checkpoints periodically. You can customize hyperparameters (like learning rate, discount factor, and clip range) directly within the script.

## Evaluating the Agent

After training—or if you have a pre-trained model—you can evaluate your agent’s performance with:

```bash
python3 src/eval.py
```

This evaluation script loads the saved model and runs it in the environment, providing insights into its racing capabilities.

## Demo

View a demonstration of the trained agent in action:

[CarRacingV3Demo](https://github.com/user-attachments/assets/742933df-e748-4c3e-a2b6-a03c3adef26d)

## Project Structure

```
├── src
│   ├── train.py        # Script for training the PPO agent
│   ├── eval.py         # Script for evaluating the trained model
│   └── ...             # Additional modules and utilities
├── requirements.txt    # List of Python dependencies
└── README.md           # This file
```

## Customization and Hyperparameters

To modify training settings within `src/train.py` to experiment with different configurations. Parameters such as learning rate, batch size, discount factor, and clip range can be adjusted to suit needs.

## Acknowledgements

- [OpenAI Gym](https://gym.openai.com/) for providing the simulation environment.
- [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) for PPO implementation.
