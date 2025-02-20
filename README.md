# **Project: Implementing the Algorithm from "Automatic Bridge Bidding Using Deep Reinforcement Learning"**

## **Project Overview**
The objective of our project is to implement and test the algorithm described in the paper *Automatic Bridge Bidding Using Deep Reinforcement Learning*. This paper proposes a deep reinforcement learning (DRL) approach to automate the bidding phase in the game of bridge. Our goal is to train an agent capable of making optimal bidding decisions based on training data and simulations.

## **Short Description of Bridge and the Bidding Phase**
Bridge is a trick-taking card game played by four players in two partnerships. The game consists of two main phases:
1. **Bidding Phase**: Players take turns making bids to establish the contract, which determines the number of tricks their partnership must win and the trump suit (if any).
2. **Play Phase**: Players play their cards to fulfill the contract, aiming to win as many tricks as possible.

The bidding phase is critical as it sets the foundation for the play phase. It involves a structured auction where players communicate information about their hands using a predefined bidding system.

As in the article we considered a subproblem of the bidding phase: the bidding without competition. Bidding without competition assumes that the opponent’s team always calls PASS during bidding, and hence information-exchanging would not be blocked.

## **Methodology**
We followed a structured approach consisting of multiple steps:

### **1. Understanding the Algorithm**
We began by analyzing the paper in detail to understand:
- How the bridge bidding problem is modeled as a Markov Decision Process (MDP).
- The reinforcement learning algorithm used (DQN).
- The representation of game states and possible actions.
- The techniques used for training and evaluating the agent.

### **2. Preparing the Data and Environment**
Next, we set up the environment to train our agent and implement a double dummy analysis to simulate the playing phase of the game:
- **State Representation**: We encoded each hand and bidding history as feature vectors suitable for training.
- **Action Representation**: The possible actions correspond to valid bids in a bridge game.
- **Simulation Engine**: We developed a simulation framework allowing the agent to learn from historical examples.

**Double dummy analysis** is a technique that attempts to compute the number of tricks taken by each team in the playing phase under perfect information and optimal playing strategy, and is generally considered to be a solved problem in the art of bridge bidding AI. Although the analysis is done with an optimistic assumption of perfect information, it has been shown to achieve considerable accuracy with a more rapid analysis than an actual play.

### **3. Implementing the Algorithm**
We implemented the algorithm using **Python** and basic libraries with the additional endplay library which is a Python library designed for analyzing and simulating bridge games. We used it to implement the double dummy analysis.

Here is the structure of the code:

- **tools.py**: Contains utility functions for data processing and simulation.
- **Q_networks.ipynb**: Jupyter notebook for defining and training the Q-networks.
- **nn_tools.py**: Contains neural network-related tools and helper functions.
- **main.ipynb**: Jupyter notebook for running the main training and evaluation loops.
- **learning_tools.py**: Contains functions related to the learning algorithm and training process.
- **Learning_algo.ipynb**: Jupyter notebook for implementing and testing the learning algorithm.
- **deals_generation.ipynb**: Jupyter notebook for generating and analyzing bridge deals.
- **README.md**: This file, providing an overview and explanation of the project.

### **4. Training and Optimization**
We trained models by playing simulated games:
- Using a **reward function** based on bidding scores.
- Adjusting **hyperparameters** (learning rate, gamma, exploration/exploitation balance).

## **Results and Key Learnings**
- We only have predictions for the two layers. Our models don't go further in the bidding phase
- We didn't manage to get the results showed in the article. We'll try with the UCB algorithm and we will  update the database D by deleting the firt predictions. Nevertheless it wasn't mentionned in the article. 

