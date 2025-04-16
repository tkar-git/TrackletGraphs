# TrackletGraphs
Repository constructing graphs of tracklets (doublets, triplets, quadruplets, etc...) for particle physics applications.
Graph construction of each tracklet type is followed by filtering and ambiguity graph contruction sub-stage.
And finally disconnecting the graphs to get well-defined tracks.

## Cloning Instruction:
```
git clone git@github.com:tkar-git/TrackletGraphs.git
```

## How to use:
Install via:    python setup.py sdist    
                pip install ./dist/disconnecting_framework-0.0.1.tar.gz
            
Similar to acorn: tracklet [discon, eval] config_file.yaml

discon: Takes in 

eval: Enters eval stage to print performance metrics (NOT YET IMPLEMENTED)