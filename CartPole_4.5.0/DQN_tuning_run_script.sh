#!/bin/bash

# Set environment variable to run Isaac Sim in headless mode
export OMNIVERSE_HEADLESS=true

# Define hyperparameter values to loop over
learning_rates=(1e-4 1e-3 1e-2)
epsilons=(1.0 0.9 0.8)
batch_sizes=(32 64 128)
hidden_dims=(64 128 256)
target_update_rates=(0.001 0.005 0.01)  # Added tau (target network update rate)
discount_factors=(0.9 0.95 0.99)  # Added discount factor (gamma)

# Define the base task and algorithm
TASK="Stabilize-Isaac-Cartpole-v0"
ALGORITHM="DQN"

# Loop through different hyperparameter values
for lr in "${learning_rates[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for hidden_dim in "${hidden_dims[@]}"; do
        for tau in "${target_update_rates[@]}"; do
          for discount in "${discount_factors[@]}"; do
            echo "Running with learning_rate=${lr}, epsilon=${epsilon}, batch_size=${batch_size}, hidden_dim=${hidden_dim}, tau=${tau}, discount=${discount}"

            # Run the Python training script with the current set of hyperparameters and headless mode
            python3 scripts/Function_based/train.py \
              --task $TASK \
              --algorithm $ALGORITHM \
              --learning_rate $lr \
              --initial_epsilon $epsilon \
              --batch_size $batch_size \
              --hidden_dim $hidden_dim \
              --tau $tau \
              --discount $discount \
              --headless
          done
        done
      done
    done
  done
done
