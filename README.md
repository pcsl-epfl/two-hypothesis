# bandit

## requirements

- [`pytorch=1.8.1`](https://github.com/pytorch/pytorch/releases/tag/v1.8.1)
- [`gradientflow=0.0.1`](https://github.com/mariogeiger/gradientflow/releases/tag/0.0.1)

## Reproduce the figures

Execute:
```
python plots.py
```

It will run the simulations and plot the results.

To execute many simulation (let say 100) at the same time using a slurm scheduler execute:
```
python plots.py --thread 100 --python "srun python"
```

To execute the simulations longer (during 1 hour), execute:
```
# delete previous shorter time simulations
rm -rf glassy ram_opt_mem ram_opt_reset
python plots.py --thread 100 --python "srun python" --wall 3600
```

## Full install instructions for conda user

```
conda create -n test python=3.8
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install matplotlib
pip install git+https://github.com/mariogeiger/gradientflow

# clone this (bandit) repository
python plots.py
```