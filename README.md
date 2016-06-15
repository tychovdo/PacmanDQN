# PacmanDQN
Deep Reinforcement Learning in Pac-man

## Usage

Play 4000 games (3000 training, 1000 testing)

```
$ python3 pacman.py -p PacmanDQN -n 4000 -x 3000 -l smallGrid
```
### Parameters

Parameters can be found in the `paramters` dictionary in `pacmanDQN_Agents.py`.__
__
Models are saved as "checkpoint" files in the `\saves` directory.__
Load and save filenames can be set using the `load_file` and `save_file` parameters.__
__
Episodes before training starts: `train_start`__
Size of replay memory batch size: `batch_size`__
Amount of experience tuples in replay memory: `mem_size`__
Discount rate (gamma value): `discount`__
Learning rate: `lr`__
RMS Prop decay rate: `rms_decay`__
RMS Prop epsilon value: `rms_eps`__
__
Exploration/Exploitation (ε-greedy):__
Epsilon start value: `eps`__
Epsilon final value: `eps_final`__
Number of steps between start and final epsilon value (linear): `eps_step`__


## Requirements

- `python==3.5.1`
- `tensorflow==0.8rc`

## Acknoledgemenets

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))
