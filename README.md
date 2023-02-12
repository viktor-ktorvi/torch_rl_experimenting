# TorchRL exampels

## Environment installation

```
conda create -n torchrl_examples_env python=3.8
conda activate torchrl_examples_env

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchrl
pip install gym
pip install matplotlib
pip install tqdm
pip install wandb
pip install gym[mujoco]
```