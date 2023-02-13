import os
import torch
import wandb

import matplotlib.pyplot as plt

from pathlib import Path

from torchrl.envs import TransformedEnv, ObservationNorm, Compose, DoubleToFloat, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs

import ppo

from loading import save_model

# TODO is this really working?

if __name__ == '__main__':
    config = {
        "env_name": "InvertedDoublePendulum-v4",
        "algorithm": "ppo",
        "device": "cpu" if not torch.has_cuda else "cuda:0",
        "actor_width": 16,  # number of neurons in each layer
        "actor_num_hidden": 3,  # number of hidden layers
        "critic_width": 16,  # number of neurons in each layer
        "critic_num_hidden": 3,  # number of hidden layers
        "learning rate": 3e-3,  # learning rate
        "max_grad_norm": 1.0,  # gradient clipping max value
        "frame_skip": 1,  # repeating the same action multiple times over the course of a trajectory
        "sub_batch_size": 64,  # gradient descent mini-batch size
        "num_epochs": 10,  # number of gradient descent updates(sample from the replay buffer that many times)
        "clip_epsilon": 0.2,  # clip value for PPO loss: see the equation in the intro for more
        "gamma": 0.99,  # discount factor
        "lambda": 0.95,  # a ppo parameter
        "entropy_eps": 1e-4,  # a ppo parameter,
        "max_frames_per_trajectory": 1000  # after how many frames to reset the environment; default = -1
    }

    # a frame is a timestep
    config["frames_per_batch"] = 100 // config["frame_skip"]  # how many frames to take from the environment
    config["total_frames"] = 50_000 // config["frame_skip"]  # total number of frames to get from the environment

    wandb.init(project=config["env_name"] + '_' + config["algorithm"])  # log using weights and biases
    # TODO update config with wandb config in case of sweeps

    # load environment
    base_env = GymEnv(env_name=config["env_name"], device=config["device"], frame_skip=config["frame_skip"])
    print("\nInit td:\n", base_env.reset())

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"], ),
            StepCounter()
        )
    )

    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)  # get mean and std from the environment to normalize samples

    print("\nObservation loc(mean):\n", env.transform[0].loc)
    print("\nObservation scale(std):\n", env.transform[0].scale)

    config["loc"] = env.transform[0].loc
    config["scale"] = env.transform[0].scale

    check_env_specs(env)

    rollout = env.rollout(7)
    print("\nrollout of three steps:\n", rollout)
    print("\nShape of the rollout TensorDict:\n", rollout.batch_size)

    # train
    actor_net, value_net, logs = ppo.train(config, env, vebose=True)

    # save
    env_path = config["env_name"].replace('-', '_')
    Path(env_path).mkdir(exist_ok=True)

    algorithm_path = os.path.join(env_path, config["algorithm"])
    Path(algorithm_path).mkdir(exist_ok=True)

    filepath = save_model(config, actor_net, value_net, filename=os.path.join(algorithm_path, 'model_state.p'))
    wandb.save(filepath)

    # plot
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()
