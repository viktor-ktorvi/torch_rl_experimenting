import torch
import wandb

import torch.nn as nn

from collections import defaultdict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.tensor_specs import CompositeSpec
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import TanhNormal, ProbabilisticActor, ValueOperator
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE

from torchvision.ops import MLP

from tqdm import tqdm


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


def train(config, env, verbose=True):
    # TODO update config with wandb in case of a sweep

    if type(env.observation_spec) == CompositeSpec:
        config["observation shape"] = env.observation_spec._specs['observation'].shape
    else:
        config["observation shape"] = env.observation_spec.shape

    config["action shape"] = env.action_spec.shape
    config["action maximum"] = env.action_spec.space.maximum
    config["action minimum"] = env.action_spec.space.minimum

    actor_net = make_actor(config)
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
    value_net.to(config["device"])

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=config["frames_per_batch"],
        total_frames=config["total_frames"],
        split_trajs=False,
        device=config["device"],
        max_frames_per_traj=config["max_frames_per_trajectory"]
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config["frames_per_batch"], device=config["device"]),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=config["gamma"], lmbda=config["lambda"], value_network=value_module,
        average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        advantage_key="advantage",
        clip_epsilon=config["clip_epsilon"],
        entropy_bonus=bool(config["entropy_eps"]),
        entropy_coef=config["entropy_eps"],
        # these keys match by default, but we set this for completeness
        value_target_key=advantage_module.value_target_key,
        critic_coef=1.0,
        gamma=0.99,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), config["learning rate"])

    # TODO maybe the scheduler should be customizable in the future
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        config["total_frames"] // config["frames_per_batch"] * 1.5,
        0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=config["total_frames"] * config["frame_skip"])
    eval_str = ""

    wandb.config.update(config, allow_val_change=True)

    if verbose:
        print("\nPolicy module v1:\n", policy_module)

        print("\nPolicy module v2:\n", policy_module)
        print("\nValue module:\n", value_module)

        print("\nRunning policy\n:", policy_module(env.reset()))
        print("\nRunning value\n:", value_module(env.reset()))

        print("\nCollector:\n", collector)

        print("\nReplay buffer:\n", replay_buffer)
        print("\nAdvantage module:\n", advantage_module)

        print("\nLoss module:\n", loss_module)

        print("\nConfig:\n", config)

    # TODO why does it start fast and then slows down?
    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for k in range(config["num_epochs"]):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            for j in range(config["frames_per_batch"] // config["sub_batch_size"]):
                subdata, *_ = replay_buffer.sample(config["sub_batch_size"])
                loss_vals = loss_module(subdata)
                loss_value = loss_vals["loss_objective"] + loss_vals[
                    "loss_critic"] + loss_vals["loss_entropy"]

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory, but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(),
                    config["max_grad_norm"]
                )
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["reward"].mean().item())

        pbar.update(tensordict_data.numel() * config["frame_skip"])
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(tensordict_data['step_count'].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]['lr'])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        wandb.log({"train reward": logs["reward"][-1],
                   "step count": logs["step_count"][-1],
                   "learning rate": logs["lr"][-1],
                   "frames": i * config["frames_per_batch"]}, step=i)

        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our env horizon).
            # The ``rollout`` method of the env can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_mode("mean"), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(config["max_eval_steps"], policy_module)
                logs["eval reward"].append(eval_rollout["reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["reward"].sum().item()
                )
                logs["eval step_count"].append(
                    eval_rollout["step_count"].max().item()
                )
                logs["eval action mean"].append(eval_rollout["action"].mean().item())
                logs["eval action std"].append(eval_rollout["action"].std().item())

                # TODO log validation reward and mean action and std etc.
                wandb.log({"evaluation return": logs["eval reward (sum)"][-1],
                           "evaluation reward": logs["eval reward"][-1],
                           "evaluation action mean": logs["eval action mean"][-1],
                           "evaluation action std": logs["eval action std"][-1]
                           }, step=i)

                eval_str = f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} (init: {logs['eval reward (sum)'][0]: 4.4f}), eval step-count: {logs['eval step_count'][-1]}"
                del eval_rollout
        pbar.set_description(
            ", ".join(
                [
                    eval_str,
                    cum_reward_str,
                    stepcount_str,
                    lr_str
                ]
            )
        )

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    return actor_net, value_net, logs
