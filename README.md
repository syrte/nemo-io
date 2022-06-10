# snapio
Load n-body simulation snapshots of NEMO format


## Examples

Open the snapshots file
```
snap = open_nemo('nbody.nemo', file_meta=False)

print(snap.History.data)
```

Load the data of the 10th snapshot
```
t = snap['SnapShot/10/Parameters/Time'].data
x = snap['SnapShot/10/Particles/Position'].data
v = snap['SnapShot/10/Particles/Velocity'].data
m = snap['SnapShot/10/Particles/Mass'].data
```

One can also use 
```
t = snap['SnapShot'][10]['Parameters']['Time'].data
```


Generate meta files for all *.snap files in a given directory.

```python
from multiprocessing import Pool
from pathlib import Path

def func(file):
    print(file, flush=True)
    snap = open_nemo(file, file_meta=True)


files = list(Path('data').glob('*.snap'))

with Pool(10) as pool:
    pool.map_async(func, files).get(1e6)
```
