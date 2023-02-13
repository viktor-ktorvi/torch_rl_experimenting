import torch
import yaml

from torchrl.envs import TransformedEnv, ObservationNorm, Compose, DoubleToFloat, StepCounter
from torchrl.envs.libs.gym import GymEnv

import ppo

from custom_environment import controllable_linear_system, SystemEnvironment


def save_model(config, actor, critic, filename='model_state.p'):
    # TODO I should learn how to write Typing
    """
    :param config: dict; must contain key "algorithm"(referring to the name of the algorithm) and all the hyperparameters used by the algorithm.
    :param actor: torch.nn.Module
    :param critic: torch.nn.Module
    :param filename: str
    :return:
    """
    objs = {"config": config}
    if config["algorithm"] == "ppo":
        objs["actor_state_dict"] = actor.state_dict()
        objs["critic_state_dict"] = critic.state_dict()
    else:
        raise ValueError("Saving algorithm '{:s}' is not supported.".format(config["algorithm"]))

    torch.save(objs, filename)
    return filename


def load_config(filename):
    objs = torch.load(filename)

    return objs["config"]


def load_model(filename, env):
    objs = torch.load(filename)

    config = objs["config"]

    # TODO check for conflicts between config env specs and env specs

    if config["algorithm"] == "ppo":
        return ppo.load_policy(config, objs, env)
    else:
        raise ValueError("Loading algorithm '{:s}' is not supported.".format(config["algorithm"]))


def init_environment(config):
    # load environment
    if config["env_name"] in ["InvertedDoublePendulum-v4"]:
        base_env = GymEnv(env_name=config["env_name"], device=config["device"], frame_skip=config["frame_skip"])
    elif config["env_name"] == "LinearSystem":
        A, B, C, D = controllable_linear_system(a=tuple(config["characteristic_polynomial_coeffs"]))
        base_env = SystemEnvironment(A, B, C, D, config["dt"], device=config["device"])
    else:
        raise ValueError("Environment named '{:s}' is not supported.".format(config["env_name"]))
    # print("\nInit td:\n", base_env.reset())

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"], ),
            StepCounter()
        )
    )

    return env


def load_yaml(filepath):
    with open(filepath, 'r') as stream:
        dictionary = yaml.safe_load(stream)

    for key in dictionary.keys():
        dictionary[key] = dictionary[key]["value"]

    return dictionary


if __name__ == '__main__':
    pass
