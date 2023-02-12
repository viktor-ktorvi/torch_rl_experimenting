import os
import torch
import wandb

import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path

from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import TransformedEnv, ObservationNorm, Compose, \
    DoubleToFloat, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import set_exploration_mode, check_env_specs
from torchrl.modules import TanhNormal, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import CompositeSpec

from tqdm import tqdm

from loading import save_model
from ppo import make_actor, make_critic

# TODO is this really working?
# TODO how to make own environment
# TODO render to see if it's working

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
        "entropy_eps": 1e-4,  # a ppo parameter
    }

    # a frame is a timestep
    config["frames_per_batch"] = 100 // config["frame_skip"]  # how many frames to take from the environment
    config["total_frames"] = 50_000 // config["frame_skip"]  # total number of frames to get from the environment

    wandb.init(project=config["env_name"] + '_' + config["algorithm"])  # log using weights and biases

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

    if type(env.observation_spec) == CompositeSpec:
        config["observation shape"] = env.observation_spec._specs['observation'].shape
    else:
        config["observation shape"] = env.observation_spec.shape

    config["action shape"] = env.action_spec.shape
    config["action maximum"] = env.action_spec.space.maximum
    config["action minimum"] = env.action_spec.space.minimum

    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)  # get mean and std from the environment to normalize samples

    print("\nObservation loc(mean):\n", env.transform[0].loc)
    print("\nObservation scale(std):\n", env.transform[0].scale)

    config["loc"] = env.transform[0].loc
    config["scale"] = env.transform[0].scale

    check_env_specs(env)

    rollout = env.rollout(7)
    print("\nrollout of three steps:\n", rollout)
    print("\nShape of the rollout TensorDict:\n", rollout.batch_size)

    print(rollout['observation'])
    print(rollout['next']['observation'])

    actor_net = make_actor(config)
    actor_net.to(config["device"])

    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"],
        out_keys=["loc", "scale"]
    )

    print("\nPolicy module v1:\n", policy_module)

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

    print("\nPolicy module v2:\n", policy_module)

    value_net = make_critic(config)
    value_net.to(config["device"])

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    print("\nValue module:\n", value_module)

    print("\nRunning policy\n:", policy_module(env.reset()))
    print("\nRunning value\n:", value_module(env.reset()))

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=config["frames_per_batch"],
        total_frames=config["total_frames"],
        split_trajs=False,
        device=config["device"],
    )

    print("\nCollector:\n", collector)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config["frames_per_batch"], device=config["device"]),
        sampler=SamplerWithoutReplacement(),
    )

    print("\nReplay buffer:\n", replay_buffer)

    advantage_module = GAE(
        gamma=config["gamma"], lmbda=config["lambda"], value_network=value_module,
        average_gae=True
    )

    print("\nAdvantage module:\n", advantage_module)

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

    print("\nLoss module:\n", loss_module)

    optim = torch.optim.Adam(loss_module.parameters(), config["learning rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        config["total_frames"] // config["frames_per_batch"],
        0.0
    )

    logs = defaultdict(list)
    pbar = tqdm(total=config["total_frames"] * config["frame_skip"])
    eval_str = ""

    wandb.config.update(config, allow_val_change=True)

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
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["reward"].sum().item()
                )
                logs["eval step_count"].append(
                    eval_rollout["step_count"].max().item()
                )

                wandb.log({"evaluation return": logs["eval reward (sum)"][-1]}, step=i)

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

    env_path = config["env_name"].replace('-', '_')
    Path(env_path).mkdir(exist_ok=True)

    algorithm_path = os.path.join(env_path, config["algorithm"])
    Path(algorithm_path).mkdir(exist_ok=True)

    filepath = save_model(config, actor_net, value_net, filename=os.path.join(algorithm_path, 'model_state.p'))
    wandb.save(filepath)

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
