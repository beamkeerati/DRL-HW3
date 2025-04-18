# DRL-HW3

## Table of contents
- [Task](#cart-pole-hw3)
- [Algorithm Summary](#algorithm-summary)
- [Hyperparameters Summary](#rl-hyperparameter-summary)
- [Hyperparameters Evaluation](#hyperparameters-evaluation)
- [Algorithm Evaluation](#algorithm-evaluation)

## Cart Pole HW3

After reviewing the updated `Cart-Pole` [instruction](https://github.com/S-Tuchapong/FRA503-Deep-Reinforcement-Learning-for-Robotics/tree/main/CartPole_4.5.0), you are now ready to proceed with the final homework.

Similar to the previous homework, this assignment focuses on the **Stabilizing Cart-Pole Task**, but using function approximation-based RL approaches instead of table-based RL approaches.

Additionally, as in the previous homework, the `CartPole` extension repository includes configurations for the **Swing-up Cart-Pole Task** as an optional resource for students seeking a more challenging task.

### Learning Objectives:
1. Understand how **function approximation** works and how to implement it.

2. Understand how **policy-based RL** works and how to implement it.

3. Understand how advanced RL algorithms balance exploration and exploitation.

4. Be able to differentiate RL algorithms based on stochastic or deterministic policies, as well as value-based, policy-based, or Actor-Critic approaches. 

5. Gain insight into different reinforcement learning algorithms, including Linear Q-Learning, Deep Q-Network (DQN), the REINFORCE algorithm, and the Actor-Critic algorithm. Analyze their strengths and weaknesses.

### Part 1: Understanding the Algorithm
In this homework, you have to implement 4 different function approximation-based RL algorithms:

- **Linear Q-Learning**

- **Deep Q-Network** (DQN)

- **REINFORCE algorithm**

- One algorithm chosen from the following Actor-Critic methods:
    - **Deep Deterministic Policy Gradient** (DDPG)
    - **Advantage Actor-Critic** (A2C)
    - **Proximal Policy Optimization** (PPO)
    - **Soft Actor-Critic** (SAC)

For each algorithm, describe whether it follows a value-based, policy-based, or Actor-Critic approach, specify the type of policy it learns (stochastic or deterministic), identify the type of observation space and action space (discrete or continuous), and explain how each advanced RL method balances exploration and exploitation.
 

### Part 2: Setting up `Cart-Pole` Agent.

Similar to the previous homework, you will implement a common components that will be the same in most of the function approximation-based RL in the `RL_base_function.py`.The core components should include, but are not limited to:

#### 1. RL Base class

This class should include:

- **Constructor `(__init__)`** to initialize the following parameters:

    - **Number of actions**: The total number of discrete actions available to the agent.

    - **Action range**: The minimum and maximum values defining the range of possible actions.

    - **Discretize state weight**: Weighting factor applied when discretizing the state space for learning.

    - **Learning rate**: Determines how quickly the model updates based on new information.

    - **Initial epsilon**: The starting probability of taking a random action in an ε-greedy policy.

    - **Epsilon decay rate**: The rate at which epsilon decreases over time to favor exploitation over exploration.

    - **Final epsilon**: The lowest value epsilon can reach, ensuring some level of exploration remains.

    - **Discount factor**: A coefficient (γ) that determines the importance of future rewards in decision-making.

    - **Buffer size**: Maximum number of experiences the buffer can hold.

    - **Batch size**: Number of experiences to sample per batch.

- **Core Functions**
    - `scale_action()`: scale the action (if it is computed from the sigmoid or softmax function) to the proper length.

    - `decay_epsilon()`: Decreases epsilon over time and returns the updated value.

Additional details about these functions are provided in the class file. You may also implement additional functions for further analysis.

#### 2. Replay Buffer Class

A class use to store state, action, reward, next state, and termination status from each timestep in episode to use as a dataset to train neural networks. This class should include:

- **Constructor `(__init__)`** to initialize the following parameters:
  
    - **memory**: FIFO buffer to store the trajectory within a certain time window.
  
    - **batch_size**: Number of data samples drawn from memory to train the neural network.

- **Core Functions**
  
    - `add()`: Add state, action, reward, next state, and termination status to the FIFO buffer. Discard the oldest data in the buffer
    
    - `sample()`: Sample data from memory to use in the neural network training.
 
  Note that some algorithms may not use all of the data mentioned above to train the neural network.

#### 3. Algorithm folder

This folder should include:

- **Linear Q Learning class**

- **Deep Q-Network class**

- **REINFORCE Class**

- One class chosen from the Part 1.

Each class should **inherit** from the `RL Base class` in `RL_base_function.py` and include:

- A constructor which initializes the same variables as the class it inherits from.

- Superclass Initialization (`super().__init__()`).

- An `update()` function that updates the agent’s learnable parameters and advances the training step.

- A `select_action()` function select the action according to current policy.

- A `learn()` function that train the regression or neural network.


### Part 3: Trainning & Playing to stabilize `Cart-Pole` Agent.

You need to implement the `training loop` in train script and `main()` in the play script (in the *"Can be modified"* area of both files). Additionally, you must collect data, analyze results, and save models for evaluating agent performance.

#### Training the Agent

1. `Stabilizing` Cart-Pole Task

    ```
    python scripts/Function_based/train.py --task Stabilize-Isaac-Cartpole-v0 
    ```

2. `Swing-up` Cart-Pole Task (Optional)
    ```
    python scripts/Function_based/train.py --task SwingUp-Isaac-Cartpole-v0
    ```

#### Playing

1. `Stabilize` Cart-Pole Task

    ```
    python scripts/Function_based/play.py --task Stabilize-Isaac-Cartpole-v0 
    ```

2. `Swing-up` Cart-Pole Task (Optional)
    ```
    python scripts/Function_based/play.py --task SwingUp-Isaac-Cartpole-v0 
    ```

### Part 4: Evaluate `Cart-Pole` Agent performance.

You must evaluate the agent's performance in terms of **learning efficiency** (i.e., how well the agent learns to receive higher rewards) and **deployment performance** (i.e., how well the agent performs in the Cart-Pole problem). Analyze and visualize the results to determine:

1. Which algorithm performs best?
2. Why does it perform better than the others?

---

---

---

## Algorithm Summary

1. Deep Q-Network (DQN)   

   - **Type**: Off-policy, value-based
   - **Description**: DQN uses Q-learning with a neural network to approximate the Q-value function. It is one of the most successful algorithms for handling high-dimensional, continuous input spaces. The agent interacts with the environment, storing experiences in a replay buffer and periodically updates the Q-function.
   - **Key Features**:

      - Experience Replay: Stores experiences in a buffer and samples random batches to break correlations in consecutive experiences.

      - Target Network: A separate target network that updates less frequently, providing more stable Q-value updates.

      - Exploration: Uses epsilon-greedy strategy, balancing exploration and exploitation.

2. Linear Q (Linear Q-learning)
    - **Type**: Off-policy, value-based

    - **Description**: A simpler, linear approximation of the Q-function, where the Q-values are modeled as a linear combination of features from the state-action pair. It does not use a neural network like DQN, making it more efficient for simpler problems but less capable for high-dimensional spaces.

    - **Key Features**:

      - Linear Function Approximation: Q-values are computed as a linear combination of state-action features.

      - Off-Policy: Uses past experiences stored in a buffer to update the Q-values.

      - Exploration Strategy: Typically uses epsilon-greedy to balance exploration and exploitation.

3. Monte Carlo REINFORCE
   - **Type**: On-policy, policy gradient

   - **Description**: The REINFORCE algorithm is a policy gradient method where the agent directly learns the policy by computing gradients of the expected return. It uses Monte Carlo sampling to estimate the returns and adjust the policy based on those estimates.

   - **Key Features**:

     - Monte Carlo Estimation: Uses complete episodes to estimate returns, providing a low-variance but high-bias estimator.

     - Policy Gradient: The agent learns a direct mapping from states to actions, improving the policy over time.

     - On-Policy: The agent only uses experiences generated from the current policy to update its parameters.

4. Actor-Critic (AC)
   - **Type**: On-policy, actor-critic

   - **Description**: The Actor-Critic method combines both value-based and policy-based approaches. The actor updates the policy, while the critic evaluates the actions taken by the actor using a value function. This combination allows the agent to both improve its policy and evaluate its actions effectively.

   - **Key Features**:

     - Actor: A neural network that directly outputs a probability distribution over actions.

     - Critic: A value function that estimates the expected return for a given state, used to evaluate the actor's actions.

     - On-Policy: Uses the current policy to generate experiences for training.

     - Advantage Estimation: Often uses the Generalized Advantage Estimation (GAE) to reduce bias and variance in updates.


 
## RL Hyperparameter Summary

The following table summarizes the key hyperparameters for different function-based reinforcement learning algorithms used in this project. It outlines which parameters are common to all algorithms and which ones are specifically used by DQN, Linear Q, Monte Carlo REINFORCE, and Actor–Critic (AC) methods.

| **Parameter**         | **Purpose**                                               | **DQN** | **Linear Q** | **MC_REINFORCE**       | **Actor–Critic (AC)**         |
|-----------------------|-----------------------------------------------------------|:-------:|:------------:|:----------------------:|:-----------------------------:|
| **num_of_action**     | Defines the discrete action space                         |   ✓     |      ✓       |          ✓           |              ✓              |
| **action_range**      | Scales or limits the magnitude of actions                 |   ✓     |      ✓       |          ✓           |              ✓              |
| **learning_rate**     | Step size for gradient descent                            |   ✓     |      ✓       |          ✓           |              ✓              |
| **hidden_dim**        | Network capacity (hidden layers)                          |   ✓     |      –       |          ✓           |              ✓              |
| **n_episodes**        | Total number of training episodes                         |   ✓     |      ✓       |          ✓           |              ✓              |
| **initial_epsilon**   | Starting value for ε-greedy exploration                   |   ✓     |      ✓       |    – (not used)      |  – (typically not used)       |
| **epsilon_decay**     | Decay rate for ε-greedy exploration                       |   ✓     |      ✓       |    – (not used)      |  – (typically omitted)        |
| **final_epsilon**     | Minimum exploration probability                           |   ✓     |      ✓       |    – (not used)      |  – (typically omitted)        |
| **discount**          | Discount factor for future rewards                        |   ✓     |      ✓       |          ✓           |              ✓              |
| **buffer_size**       | Size of the experience replay buffer                      |   ✓     |      ✓       |          –           |   (sometimes used in variants)|
| **batch_size**        | Mini-batch size for training updates                      |   ✓     |      ✓       |          –           |   (sometimes used in variants)|
| **dropout**           | Dropout rate for neural network regularization            |   ✓     |      –       |          ✓           |              ✓              |
| **tau**               | Soft update rate for target network                       |   ✓     |      –       |          –           |   (sometimes used in variants)|
| **clip_ratio**        | Clipping ratio for PPO-like policy updates                |   –     |      –       |          –           |              ✓              |
| **entropy_coef**      | Coefficient for entropy bonus (encourages exploration)    |   –     |      –       |          –           |              ✓              |
| **value_coef**        | Weight for the critic’s loss in AC algorithms             |   –     |      –       |          –           |              ✓              |
| **update_epochs**     | Number of epochs per update cycle (policy updates)        |   –     |      –       |          –           |              ✓              |
| **gae_lambda**        | Lambda for Generalized Advantage Estimation (bias/variance trade-off) |   –  |      –       |          –           |              ✓              |

> **Legend:**  
> ✓ — Parameter is typically used for that algorithm.  
> – — Parameter is not applicable or generally omitted.

## Hyperparameters Evaluation
Since hyperparameters interact dynamically, we tried to used a thourough gridsearch with these parameters on DQN algorithm with this script file => [DQN_Script](/CartPole_4.5.0/DQN_tuning_run_script.sh) which will tune: 

batch_sizes=(32 64 128)   
hidden_dims=(64 128 256)   
target_update_rates=(0.001 0.005 0.01)   
discount_factors=(0.9 0.95 0.99)

### Discount Factor

In this experiment, we observed the impact of varying the discount factor (γ) on the agent's performance in terms of loss and reward. The three γ values tested were 0.9, 0.95, and 0.99, each of which led to distinct learning behaviors in the agent.

![DQN_0.9](/image/DQN_Discount_Factor_0.9.png) ![DQN_0.9_R](/image/DQN_Discount_Factor_0.9_reward.png)

**γ = 0.9 (Low Discount Factor)**:   
At γ = 0.9, the agent focused primarily on short-term rewards, meaning it heavily prioritized immediate payoffs rather than future rewards. As a result, the loss remained very low and stable, suggesting that the agent was quickly converging to a policy that maximized short-term gains. The reward, in turn, was stable throughout training, indicating that the agent’s policy was effective for the task, but without any major fluctuations. This behavior is typical for tasks that can be optimized through immediate rewards without much need for long-term consideration.

![DQN_0.95](/image/DQN_Discount_Factor_0.95.png) ![DQN_0.95_R](/image/DQN_Discount_Factor_0.95_reward.png)

**γ = 0.95 (Moderate Discount Factor)**:   
With γ = 0.95, the agent began to balance short-term rewards and long-term outcomes. This led to moderate loss and reward fluctuations, indicating that the agent was experimenting more with different actions, sometimes sacrificing immediate rewards to potentially gain higher future rewards. While the agent performed decently, the fluctuations in reward reflect the agent’s exploration of a broader set of strategies that consider both immediate and delayed rewards. This is beneficial in more complex environments where the agent needs to make trade-offs between short-term and long-term optimization.

![DQN_0.99](/image/DQN_Discount_Factor_0.99.png) ![DQN_0.99_R](/image/DQN_Discount_Factor_0.99_reward.png)

**γ = 0.99 (High Discount Factor):**   
For γ = 0.99, the agent placed greater emphasis on long-term rewards, even at the expense of immediate payoffs. Initially, this caused the loss to skyrocket as the agent explored actions that provided no immediate rewards but were expected to yield better long-term results. The reward initially plummeted, reflecting the agent’s struggles to optimize for future gains, but eventually began to converge to a stable value as the agent refined its policy to balance the trade-off between immediate and future rewards. This behavior is common in environments requiring long-term planning, where the agent needs to forgo short-term rewards for a more beneficial long-term strategy, though this comes at the cost of slower convergence and higher volatility in early training phases.

---

### Hidden Dimension

We tried hidden dimension range of 64 128 and 256. With other arguments scrambled we found one thing in common which is the reward trend as shown below.

![HD64](/image/DQN_Hidden_Dimension_64.png)

**Reward: 0.3**

Smaller model (64): With a smaller hidden dimension, the agent may have been exploring the environment more effectively, possibly because the model wasn't overcomplicating its value function and was more able to act randomly (explore). This could have allowed it to discover rewarding states, giving it a higher reward.

![HD128](/image/DQN_Hidden_Dimension_128.png)

**Reward: 0.2**

Medium model (128): With a larger hidden dimension, the model likely started exploiting its capacity too early, memorizing the rewards from certain actions but failing to generalize effectively, leading to worse exploration and lower rewards.

![HD256](/image/DQN_Hidden_Dimension_256.png)

**Reward: 0.3**

Larger model (256): Increasing the hidden dimension further (to 256) allowed the model to strike a better balance between exploration and exploitation. It was able to learn a more generalized function, allowing it to explore better and find more optimal policies, leading to a recovery of the reward.

---

### Batch Size

In this experiment, the effect of varying the batch size on the agent's learning performance was explored. Specifically, batch sizes of 32 and 64 were tested to observe their impact on initial reward, training stability, and convergence behavior.

![BS32](/image/DQN_Batch_size_32.png)

At batch size = 32, the agent processed fewer samples per update. This led to a slower convergence as the agent had to rely on smaller, noisier updates for each step. Initial reward was slightly lower, starting around `0.35`, reflecting that smaller batches resulted in less accurate gradient estimates and slower learning of optimal policies. The loss function might have been more volatile in the earlier stages, as the agent relied on fewer data points per update, leading to potentially noisier learning and greater variance in the reward curve. Despite this, the agent eventually converged to a similar final reward value as batch size 64.

Smaller batch sizes generally result in more frequent updates, but each update is based on less data, which can make the learning process more erratic in the short term. However, this can also allow the agent to react quicker to changes in the environment, though it might struggle more in the early stages to optimize the policy.

![BS64](/image/DQN_Batch_size_64.png)

At batch size = 64, the agent processed a larger number of samples per update, leading to more stable and less noisy gradient estimates. As a result, the agent’s learning was smoother, and it was able to converge faster compared to batch size 32. The initial reward was higher, starting around `0.4`, indicating that the larger batch allowed for more accurate updates, leading to quicker improvements in the policy. The learning process was less affected by the noise that smaller batches introduce, which is reflected in the smoother initial reward curve.

Larger batch sizes tend to lead to smoother updates, which result in less variance in the agent’s learning process. This can cause the agent to have more stable progress early in training, but it can also mean the agent may be slower to adapt in certain situations, as it relies on larger batches of data before updating the policy.

---


### Actor Critic Hyperparameters

After tuning all Actor Critic hyperparameters using the grid search method with the [Actor Critic tune Script](/CartPole_4.5.0/AC_tuning_run_script.sh), there was minimal divergence observed across all aspects, which demonstrates the stability of the algorithm. However, this consistency may not necessarily be a positive outcome, as it could also indicate that the reward values were not high enough to detect meaningful differences in this dynamic environment.

---

Epsilon and other parameter's behavior has already been studied at [DRL_HW2](https://github.com/beamkeerati/DRL-HW2).

## Algorithm Evaluation

### 1. **Actor Critic (PPO)**
- **Performance**: Actor Critic consistently achieved relatively high rewards.
- **Reward Behavior**: The reward stayed relatively stable and high over time, showing good convergence even with scrambled hyperparameters.
- **Key Insight**: The Actor Critic method showed the most robust performance across all configurations. It appears to handle the instability of other algorithms better, maintaining high rewards even when hyperparameters were varied.

![AC_reward](/image/AC_Reward.png)

### 2. **Linear Q**
- **Performance**: Linear Q demonstrated higher rewards than Actor Critic but exhibited much more variance throughout training.
- **Reward Behavior**: The reward fluctuated significantly, with occasional sharp peaks and valleys, which suggests that Linear Q has the potential to achieve higher rewards but lacks stability.
- **Key Insight**: While **Linear Q** can outperform **Actor Critic** in terms of peak reward, its inconsistency could make it less reliable for long-term training. The higher variance in reward indicates that the model may benefit from further tuning and stabilization techniques to fully realize its potential in more complex environments.

![LQ_reward](/image/LQ_Reward.png)

### 3. **DQN**
   - **Performance**: DQN exhibited an exponential decay in its rewards.
   - **Reward Behavior**: The reward initially increased but then rapidly decayed over time, showing an unstable learning process with high initial rewards followed by a sharp decline.
   - **Key Insight**: While DQN can show some promise in certain setups, it seems highly sensitive to hyperparameters like learning rate and discount factor. This sensitivity results in an exponential decay in rewards, indicating that DQN might require more fine-tuning to maintain stable performance.

![DQN_reward](/image/DQN_Reward.png)

### 4. **Monte Carlo**
- **Performance**: Monte Carlo exhibited some improvements in reward generation after adjustments, though it still struggled to match the stability and convergence of other algorithms. The reward showed a gradual increase but did not achieve the same level of performance as Actor Critic or Linear Q.

- **Key Insight**: While Monte Carlo had some success in reward generation, it still faces challenges with stability and efficiency in this environment. The results suggest that Monte Carlo may be more sensitive to the specific configuration and hyperparameters, and may not perform as reliably in dynamic environments like the one used in this experiment. Due to time constraints, only one graph was generated for this method, which limited further exploration of its full potential. Further fine-tuning of hyperparameters could potentially improve its performance, but it requires careful attention to detail and better handling of episodic rewards.

![MC_reward](/image/MC_Reward.png)

---

## Summary of Findings

- **Actor Critic** provides the best performance with the highest reward, showing stable convergence with minimal divergence across all aspects. Its consistency, however, might indicate that the reward values were not high enough to detect differences in this dynamic environment.

- **Linear Q** exhibits similar behavior to Actor Critic but with a higher degree of reward fluctuation. While it achieves higher rewards than Actor Critic in some cases, the reward variance makes it less stable overall.

- **DQN** shows unstable behavior with an exponential decay in rewards, likely due to its sensitivity to hyperparameters, leading to less reliable performance compared to other algorithms.

- **Monte Carlo** failed to generate meaningful results during this experiment, but this was due to issues in running the algorithm on our end and not a limitation of the algorithm itself. **Only one graph of Monte Carlo was obtained** due to time constraints, and the results were not sufficient for a full evaluation.

In conclusion, **Actor Critic (PPO)** is the best-performing algorithm in this experiment, providing the highest reward with stable convergence. **Linear Q** performs similarly but with more reward fluctuation, while **DQN** exhibits instability. **Monte Carlo** was not evaluated successfully, but its failure was due to issues with running the algorithm, not the algorithm's inherent capabilities.

Training Algorithms: [Actor Critic](/CartPole_4.5.0/AC_tuning_run_script.sh), [DQN](/CartPole_4.5.0/DQN_tuning_run_script.sh), [DQN2](/CartPole_4.5.0/DQN_tuning_run_script_2.sh), [Monte Carlo Reinforced](/CartPole_4.5.0/MCR_tuning_run%20_script.sh) and [Linear Q](/CartPole_4.5.0/LQ_tuning_run_script.sh).


---

Author: 
- Keerati Ubolmart 65340500003
- Manaswin Anekvisudwong 65340500049

---