# Categorical DQN.

Attempt at implementing the algorithm in *A distributional Perspective on
Reinforcement Learning*. Right now it is training on Catch but it is rather
instable and sensible to hyperparameter choice.

## Dependencies

```
git clone https://github.com/floringogianu/gym_fast_envs
cd gym_fast_envs

pip install -r requirements.txt
pip install -e .
```

## Training

Train a DQN baseline with `python main.py -cf catch_dqn`.

Train the Categorical DQN with `python main.py -cf catch_dev`.
