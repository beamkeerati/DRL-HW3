#!/bin/bash

# Set environment variable to run Isaac Sim in headless mode
export OMNIVERSE_HEADLESS=true

# Define hyperparameter values to loop over
learning_rates=(1e-4 1e-3 1e-2)
epsilons=(1.0 0.9 0.8)
dropouts=(0.2 0.3 0.4)  # Added dropout as a tunable hyperparameter
discount_factors=(0.9 0.95 0.99)  # Added discount factor (gamma)

# Define fixed hyperparameters
batch_size=64
hidden_dim=128
target_update_rate=0.005  # Fixed tau (target network update rate)

# Define the base task and algorithm
TASK="Stabilize-Isaac-Cartpole-v0"
ALGORITHM="DQN"

# Loop through discount factor values first (outer loop)
for discount in "${discount_factors[@]}"; do
  # Then loop through other hyperparameter values
  for lr in "${learning_rates[@]}"; do
    for epsilon in "${epsilons[@]}"; do
      for dropout in "${dropouts[@]}"; do
        echo "Running with learning_rate=${lr}, epsilon=${epsilon}, dropout=${dropout}, discount=${discount}"

        # Run the Python training script with the current set of hyperparameters and headless mode
        python3 scripts/Function_based/train.py \
          --task $TASK \
          --algorithm $ALGORITHM \
          --learning_rate $lr \
          --initial_epsilon $epsilon \
          --batch_size $batch_size \
          --hidden_dim $hidden_dim \
          --tau $target_update_rate \
          --discount $discount \
          --dropout $dropout \
          --headless
      done
    done
  done
done
