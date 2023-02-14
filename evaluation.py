import os
import torch

import numpy as np

from torchrl.envs.utils import set_exploration_mode

from matplotlib import pyplot as plt

from loading import init_environment, load_config, load_model

if __name__ == '__main__':
    filepath = os.path.join("LinearSystem", "ppo", "model_state.p")
    config = load_config(filepath)

    env = init_environment(config)

    env.transform[0].loc = config["loc"].to(config["device"])
    env.transform[0].scale = config["scale"].to(config["device"])

    policy_module, _ = load_model(filepath, env)

    with set_exploration_mode("mean"), torch.no_grad():
        # execute a rollout with the trained policy
        eval_rollout = env.rollout(config["max_eval_steps"], policy_module)

    # should be around 9.25 for InvertedDoublePendulum_v4
    print("Mean evaluation reward: {:2.4f}".format(eval_rollout["reward"].mean().item()))

    if config["env_name"] == "LinearSystem":
        states = eval_rollout["next"]["observation"].cpu().numpy().T
        states -= config["loc"].cpu().numpy()[:, np.newaxis]
        states /= config["scale"].cpu().numpy()[:, np.newaxis]
        y_env = env.C @ states
        y_env = y_env.flatten()

        env_time_axis = np.arange(start=0, step=env.dt, stop=len(y_env) * env.dt)

        plt.figure()
        plt.title("Output")
        plt.step(env_time_axis, y_env, label='output')
        plt.axhline(env.ref, color='r', label='reference')
        plt.xlabel("t[s]")
        plt.ylabel("y")
        plt.legend()

        plt.figure()
        plt.title("Reward")
        plt.step(env_time_axis, eval_rollout["reward"].cpu().numpy().flatten(), label='reward')
        plt.xlabel('t[s]')
        plt.ylabel("r")
        plt.legend()

        plt.show()
