# TrackletGraphs
Repository constructing graphs of tracklets (doublets, triplets, quadruplets, etc...) for particle physics applications.
Graph construction of each tracklet type is followed by filtering and ambiguity graph contruction sub-stage.
And finally disconnecting the graphs to get well-defined tracks.

## Cloning Instruction:
```
git clone git@github.com:tkar-git/TrackletGraphs.git
```

## How to use: (work in progress, the following may be subject to changes)
Similar to acorn: tracklet [discon, eval] config_file.yaml

discon: Performs the whole disconnection process with the filters given in the .yaml file (in progress)

eval: Enters eval stage to print performance metrics (not yet implemented)

## How to structure the config file:
TBD