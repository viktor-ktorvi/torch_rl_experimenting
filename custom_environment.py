import control
import torch

import numpy as np

from tensordict import TensorDict
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

from matplotlib import pyplot as plt


def controllable_linear_system(a=(2.0, 1.0), C=None, dtype=np.float32):
    """
    Represents a (SISO)system with f(s) = s^n + a_n-1 s^n-1 + ... + a_1 s + a_0,
    n being the length of 'a', in the controllable canonical form.
    Returns the state space matrices
    :param a: tuple; eigen-polynomial coefficients
    :param C: tuple; what states to observe
    :param dtype: numpy data type
    :return: tuple(np.ndarray); returns the state space matrices.
    """
    n = 1 if type(a) != tuple else len(a)

    zero_column = np.zeros((n - 1, 1))
    eye = np.eye(n - 1)
    negative_coeffs = -np.array(a).reshape(1, n)
    A = np.vstack((np.hstack((zero_column, eye)), negative_coeffs))

    B = np.zeros((n, 1))
    B[-1, 0] = 1

    if C is None:
        C = np.zeros((1, n))
        C[0, 0] = 1
    else:
        if type(C) != tuple:
            raise ValueError("C needs to be a tuple, but got '{:s}' instead".format(str(type(C))))

        C = np.array(C).reshape(1, n)

    D = np.zeros((1, 1))

    return A.astype(dtype), B.astype(dtype), C.astype(dtype), D.astype(dtype)


class SystemEnvironment(EnvBase):

    def __init__(self, A, B, C, D, dt, ref=1, device="cpu"):
        super(SystemEnvironment, self).__init__()
        self.dtype = np.float32

        self.A, self.B, self.C, self.D, self.dt, self.ref = A, B, C, D, dt, ref
        self.device = device

        self.state_size = self.A.shape[0]
        self.action_size = self.B.shape[1]

        self.state = np.zeros((self.state_size, 1), dtype=self.dtype)

        self.action_spec = BoundedTensorSpec(minimum=-1, maximum=1, shape=torch.Size([self.action_size]))

        observation_spec = UnboundedContinuousTensorSpec(shape=torch.Size([self.state_size]))
        self.observation_spec = CompositeSpec(observation=observation_spec)

        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]))

    def _reset(self, tensordict, **kwargs):
        out_tensordict = TensorDict({}, batch_size=torch.Size())

        self.state = np.zeros((self.state_size, 1), dtype=self.dtype)
        out_tensordict.set("observation", torch.tensor(self.state.flatten(), device=self.device))

        return out_tensordict

    def _step(self, tensordict):
        action = tensordict["action"]
        action = action.cpu().numpy().reshape((self.action_size, 1))

        self.state += self.dt * (self.A @ self.state + self.B @ action)

        y = self.C @ self.state + self.D @ action

        error = (self.ref - y) ** 2

        reward = -error

        out_tensordict = TensorDict({"observation": torch.tensor(self.state.astype(self.dtype).flatten(), device=self.device),
                                     "reward": torch.tensor(reward.astype(np.float32), device=self.device),
                                     "done": False}, batch_size=torch.Size())

        return out_tensordict

    def _set_seed(self, seed):
        pass


if __name__ == '__main__':
    dt = 1e-1  # sec; the sampling period

    # side note: f = 1/dt should be 10 to 40 times larger than the natural frequency fn of the system(open loop I think) which in out case
    # boils down to fn = sqrt(a[0]) / 2 / pi

    A, B, C, D = controllable_linear_system(a=(0.5, 1.0))

    sys = control.ss(A, B, C, D)

    transfer_func = control.tf(sys)

    print("\nTransfer function\n", transfer_func)

    time_axis, y_out = control.step_response(sys)

    plt.figure()
    plt.title("Step response")
    plt.plot(time_axis, y_out, label="control library")

    env = SystemEnvironment(A, B, C, D, dt)
    check_env_specs(env)

    env_time_axis = np.arange(start=0, step=env.dt, stop=time_axis[-1])

    rollout = env.rollout(len(env_time_axis), policy=lambda _: TensorDict({"action": 1.0}, batch_size=torch.Size()))
    print("\nThis is what the rollout returns:\n", rollout)
    print("\nThis is the action we've applies:\n", rollout["action"])

    y_env = env.C @ rollout["next"]["observation"].cpu().numpy().T
    y_env = y_env.flatten()

    plt.step(env_time_axis, y_env, color="r", label='custom environment')
    plt.legend()

    plt.show()

    # TODO train ppo on this :)
