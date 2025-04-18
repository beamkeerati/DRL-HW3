#!/bin/bash

# Set environment variable to run Isaac Sim in headless mode
export OMNIVERSE_HEADLESS=true

# Define PPO-specific hyperparameter values to loop over (4 arguments only)
clip_ratios=(0.1 0.2)  # Clip ratio values to tune (2 values only)
entropy_coefs=(0.01 0.02 0.05)  # Entropy coefficient values to tune
value_coefs=(0.5 0.6 0.7)  # Value coefficient values to tune
update_epochs=(4 8)  # Number of update epochs to tune (2 values only)

# Define fixed hyperparameters
learning_rate=1e-3
epsilons=1.0
batch_size=64
hidden_dim=128
target_update_rate=0.005  # Fixed tau (target network update rate)

# Define the base task and algorithm
TASK="Stabilize-Isaac-Cartpole-v0"
ALGORITHM="AC"  # Use Actor-Critic (PPO)

# Loop through other PPO-specific hyperparameter values
for clip_ratio in "${clip_ratios[@]}"; do
  for entropy_coef in "${entropy_coefs[@]}"; do
    for value_coef in "${value_coefs[@]}"; do
      for update_epoch in "${update_epochs[@]}"; do
            echo "Running with learning_rate=${learning_rate}, epsilon=${epsilon}, clip_ratio=${clip_ratio}, entropy_coef=${entropy_coef}, value_coef=${value_coef}, update_epochs=${update_epoch}"

            # Run the Python training script with the current set of hyperparameters and headless mode
            python3 scripts/Function_based/train.py \
              --task $TASK \
              --algorithm $ALGORITHM \
              --clip_ratio $clip_ratio \
              --entropy_coef $entropy_coef \
              --value_coef $value_coef \
              --update_epochs $update_epoch \
              --headless
      done
    done
  done
done
