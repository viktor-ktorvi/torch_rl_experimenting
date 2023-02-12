import torch.nn as nn

from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule

from torchrl.modules import TanhNormal, ProbabilisticActor, ValueOperator
from torchvision.ops import MLP


# TODO docs
def make_actor(config):
    actor_net = nn.Sequential(
        MLP(in_channels=config["observation shape"][-1],
            hidden_channels=[config["actor_width"]] * config["actor_num_hidden"] + [2 * config["action shape"][-1]]),
        NormalParamExtractor(),
    )

    return actor_net


def make_critic(config):
    value_net = MLP(in_channels=config["observation shape"][-1], hidden_channels=[config["critic_width"]] * config["critic_num_hidden"] + [1])
    return value_net


def load_policy(config, objs, env):
    actor_net = make_actor(config)
    actor_net.load_state_dict(objs["actor_state_dict"])
    actor_net.to(config["device"])

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={"min": config["action minimum"],
                             "max": config["action maximum"]},
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = make_critic(config)
    value_net.load_state_dict(objs["critic_state_dict"])
    value_net.to(config["device"])

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return policy_module, value_module
