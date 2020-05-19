## Docker 

```
docker build -t rlsynergy .
docker run --gpus all --rm -it --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/(local path)/synergyenvs:/root/synergyenvs" rlsynergy
```

## Test a training environment

```
python test_graspbox.py
```