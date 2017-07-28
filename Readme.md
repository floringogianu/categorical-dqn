# Categorical DQN.

Implementation of the **Categorical DQN** as described in *A distributional
Perspective on Reinforcement Learning*.

Thanks to [@tudor-berariu](https://github.com/tudor-berariu) for optimisation
and training tricks and for catching two nasty bugs.

## Dependencies

You can take a look in the [env export file](categorical.yaml) for the full
list of dependencies.

```
git clone https://github.com/floringogianu/gym_fast_envs
cd gym_fast_envs

pip install -r requirements.txt
pip install -e .
```

## Training

Train the Categorical DQN with `python main.py -cf catch_categorical`.

Train a DQN baseline with `python main.py -cf catch_dqn`.



## To Do

- [ ] Add some training curves.
- [ ] Run on Atari.
