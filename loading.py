import torch

import ppo


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


if __name__ == '__main__':
    pass
