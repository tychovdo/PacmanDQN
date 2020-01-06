# Layouts
All the grids that can be used in the game

## Grids

- capsuleClassic<br>
![capsuleClassic](../images/capsuleClassic.png)
- contestClassic<br>
![contestClassic](../images/contestClassic.png)
- mediumClassic<br>
![mediumClassic](../images/mediumClassic.png)
- mediumGrid<br>
![mediumGrid](../images/mediumGrid.png)
- minimaxClassic<br>
![minimaxClassic](../images/minimaxClassic.png)
- openClassic<br>
![openClassic](../images/openClassic.png)
- originalClassic<br>
![originalClassic](../images/originalClassic.png)
- smallClassic<br>
![smallClassic](../images/smallClassic.png)
- smallGrid<br>
![smallGrid](../images/smallGrid.png)
- testClassic<br>
![testClassic](../images/testClassic.png)
- trappedClassic<br>
![trappedClassic](../images/trappedClassic.png)
- trickyClassic<br>
![trickyClassic](../images/trickyClassic.png)

# How to chose the layout

When you run a model, put the name of the layout with the option '-l'
Example: 

```
$ python3 pacman.py -p PacmanDQN -n 6000 -x 5000 -l originalClassic
```

# Layouts
Different layouts can be found and created in the `layouts` directory

# Creation
You can create a layout easily by creating a '.lay' file in this directory.
You can draw your own grid with some special characters.
Example (smallGrid.lay): 

```
%%%%%%%%%%%%%%%%%%%%
%......%G  G%......%
%.%%...%%  %%...%%.%
%.%o.%........%.o%.%
%.%%.%.%%%%%%.%.%%.%
%........P.........%
%%%%%%%%%%%%%%%%%%%%
```

- % draws a wall
- G draws the spawn of the ghosts
- P draws the spawn of pacman
- o draws an energizer item (allow pacman to eat the ghosts during a period of time)
- . draws the dots pacman has to eat