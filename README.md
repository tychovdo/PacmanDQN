# PacmanDQN
Deep Reinforcement Learning in Pac-man

## Usage

Play 4000 games (3000 training, 1000 testing)

```
$ python3 pacman.py -p PacmanDQN -n 4000 -x 3000 -l smallGrid
```
### Parameters

Parameters can be found in the `paramters` dictionary in `pacmanDQN_Agents.py`.

Models are saved as "checkpoint" files in the `\saves` directory.
Load and save filenames can be set using the `load_file` and `save_file` parameters.

Episodes before training starts: `train_start`
Size of replay memory batch size: `batch_size`
Amount of experience tuples in replay memory: `mem_size`
Discount rate (gamma value): `discount`
Learning rate: `lr`
RMS Prop decay rate: `rms_decay`
RMS Prop epsilon value: `rms_eps`

Exploration/Exploitation (ε-greedy):
Epsilon start value: `eps`
Epsilon final value: `eps_final`
Number of steps between start and final epsilon value (linear): `eps_step`


## Requirements

- `python==3.5.1`
- `tensorflow==0.8rc`

## Acknoledgemenets

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))
