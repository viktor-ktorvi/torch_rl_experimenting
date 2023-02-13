import random
import os
import torch
import wandb

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from torchrl.envs.utils import check_env_specs

import ppo

from loading import init_environment, load_yaml, save_model

# TODO is this really working?

if __name__ == '__main__':
    # TODO learn how to use hydra
    config = load_yaml("ppo_config_LinearSystem.yaml")

    # a frame is a timestep
    config["frames_per_batch"] = config["frames_per_batch_init"] // config["frame_skip"]  # how many frames to take from the environment
    config["total_frames"] = config["total_frames_init"] // config["frame_skip"]  # total number of frames to get from the environment

    wandb.init(project=config["env_name"] + '_' + config["algorithm"])  # log using weights and biases
    # TODO update config with wandb config in case of sweeps

    env = init_environment(config)

    # TODO num_iter should be in the config/defaults
    env.transform[0].init_stats(num_iter=2000, reduce_dim=0, cat_dim=0)  # get mean and std from the environment to normalize samples

    # set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    env.set_seed(config['random_seed'])

    print("\nObservation loc(mean):\n", env.transform[0].loc)
    print("\nObservation scale(std):\n", env.transform[0].scale)

    config["loc"] = env.transform[0].loc
    config["scale"] = env.transform[0].scale

    check_env_specs(env)

    rollout = env.rollout(7)
    print("\nrollout of three steps:\n", rollout)
    print("\nShape of the rollout TensorDict:\n", rollout.batch_size)

    # TODO add max_eval_steps from config
    # train
    actor_net, value_net, logs = ppo.train(config, env, verbose=True)

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
