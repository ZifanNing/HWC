# Hebbian Weight Consolidation

This is the pure-software version of HWC code of paper "Neuromorphic Incremental on-chip Learning with Hebbian Weight Consolidation".


## Installation & requirements
The current version of the code has been tested with `Python 3.5.2` on several Linux operating systems with the following versions of PyTorch and Torchvision:
* `pytorch 1.1.0`
* `torchvision 0.2.2`

The versions that were used for other Python-packages are listed in `requirements.txt`.




## Demos

Default mode: SNN

To run ANN, change "spiking = False" in Classifier.py

#### Demo 1: SNN with hebbian weight consolidation
```bash
./main_cl.py --experiment=splitMNIST --scenario=task  --depth=3 --iters=200 --masking
```


#### Demo 2: On chip configuration
```bash
./main_cl.py --experiment=splitMNIST --scenario=class --depth=5 --masking --iters=50 --hard_masking  
```

#### Demo 3: label masking/ task masking
```bash
./main_cl.py --experiment=splitMNIST --scenario=class --depth=5 --masking --iters=50 --hard_masking --masking_label="task" 
```


## Running custom experiments
Using `main_cl.py`, it is possible to run custom individual experiments. The main options for this script are:
- `--experiment`: which task protocol? (`splitMNIST`|`permMNIST`|`CIFAR100`)
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)
- `--tasks`: how many tasks?

To run specific methods, use the following:
- Context-dependent-Gating (XdG): `./main_cl.py --xdg --xdg-prop=0.8`
- Elastic Weight Consolidation (EWC): `./main_cl.py --ewc --lambda=5000`
- Online EWC:  `./main_cl.py --ewc --online --lambda=5000 --gamma=1`
- Synaptic Intelligenc (SI): `./main_cl.py --si --c=0.1`
- Learning without Forgetting (LwF): `./main_cl.py --replay=current --distill`
- Generative Replay (GR): `./main_cl.py --replay=generative`
- Brain-Inspired Replay (BI-R): `./main_cl.py --replay=generative --brain-inspired`
- Hebbian Weight Consolidation: `./main_cl.py --masking --hard_masking`

For information on further options: `./main_cl.py -h`.

