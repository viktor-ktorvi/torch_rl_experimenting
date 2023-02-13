import torch

from torchrl.envs import TransformedEnv, ObservationNorm, Compose, \
    DoubleToFloat, StepCounter
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import set_exploration_mode

from loading import load_config, load_model

if __name__ == '__main__':
    filepath = 'model_state.p'
    config = load_config(filepath)

    base_env = GymEnv(env_name=config["env_name"], device=config["device"], frame_skip=config["frame_skip"])

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(in_keys=["observation"], ),
            StepCounter()
        )
    )

    env.transform[0].loc = config["loc"]
    env.transform[0].scale = config["scale"]

    policy_module, _ = load_model(filepath, env)

    with set_exploration_mode("mean"), torch.no_grad():
        # execute a rollout with the trained policy
        eval_rollout = env.rollout(1000, policy_module)

    # should be around 9.25 for InvertedDoublePendulum_v4
    print("Mean evaluation reward: {:2.4f}".format(eval_rollout["reward"].mean().item()))

    # TODO evaluate LinearSystem
