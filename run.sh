#!/bin/bash

# Total number of episodes:          1000000
# Total number of training episodes: 100000
# Level:                             mediumClassic

python3 pacman.py -p PacmanDQN -n 1000000 -x 100000 -l mediumClassic
