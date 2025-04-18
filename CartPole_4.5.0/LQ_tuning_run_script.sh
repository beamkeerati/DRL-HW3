#!/bin/bash

# Set environment variable to run Isaac Sim in headless mode
export OMNIVERSE_HEADLESS=true

# Define hyperparameter values to loop over (extended range)
learning_rates=(1e-5 1e-4 1e-3 1e-2 1e-1)  # Expanded learning rates to tune
discount_factors=(0.8 0.85 0.9 0.95 0.99)  # Expanded discount factor values to tune (gamma)

# Define fixed hyperparameters
batch_size=64
hidden_dim=128  # Size of hidden layers in Linear Q

# Define the base task and algorithm
TASK="Stabilize-Isaac-Cartpole-v0"
ALGORITHM="Linear_Q"  # Using Linear Q algorithm

# Loop through discount factor values first (outer loop)
for discount in "${discount_factors[@]}"; do
  # Then loop through other hyperparameter values
  for lr in "${learning_rates[@]}"; do
    echo "Running with learning_rate=${lr}, discount=${discount}"

    # Run the Python training script with the current set of hyperparameters and headless mode
    python3 scripts/Function_based/train.py \
      --task $TASK \
      --algorithm $ALGORITHM \
      --learning_rate $lr \
      --batch_size $batch_size \
      --hidden_dim $hidden_dim \
      --discount $discount \
      --headless
  done
done
