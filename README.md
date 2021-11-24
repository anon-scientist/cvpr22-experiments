# Main experiments for anonymoud CVPR submission "A new perspective on probabilistic image modeling"

## Requirements besides Python3 and pip (install using OS):
* Ubuntu 20.4
* TensorFlow 2.5.0
* PyTorch 1.5.1

Install using 

```
  python3 -m pip install -r requirements.txt
```

## Setting up
We assume you cloned this repository to <path>. 

### Download and install Einsumnetworks
```
  cd <path>
  git clone https://github.com/cambridge-mlg/EinsumNetworks.git
  export PYTHONPATH=$PYTHONPATH:<path>/EinsumNetworks/src
```

### Install libspn-keras
  python3 -m pip install libspn-keras


### Install DCGMM
```
  git clone https://github-com/anon-scientist/DCGMM
  cd DCGMM
  python3 -m pip install .
  cd ..
```

### Setting up a path for the results
```
  cd <path>
  mkdir results
```

## Running experiments
Generally, this code serves to reproduce certain experiments in a simple fashion. 
Grid search and cluster distribution are not included. Parameters given through command line are best parameters found by grid search.


### 5.1 Training dynamics. Plot is stored in ./results
You can change the dataset between mnist and fashion_mnist by modifying the --dataset_name parameter in scripts/training-F.bash. A loss plot is produced in eng.png.

```
  source scripts/training-F.bash
  python3 evalLoss.py results/dynamics_log.json
```

### 5.3 outlier detection: model comparison. Graphs are stored in ./results
```
  source exp1/commands.bash        # rat-spn
  source exp2/commands.bash        # pd-spn
  source exp3/commands.bash        # dgcspn
  source scripts/outliers-E.bash   # DCGMM-E
  python3 -m DCGMM.utils.outliersFromJson --make_plots False --json results/outliers_dgcspn_log.json
  python3 -m DCGMM.utils.outliersFromJson --make_plots False --json results/outliers_rat_log.json
  python3 -m DCGMM.utils.outliersFromJson --make_plots False --json results/outliers_pd_log.json
  python3 -m DCGMM.utils.outliersFromJson --make_plots False --json results/outliers_dcgmm-e_log.json
```

### 5.4 DCGMM sampling and sharpening. PNGs are stored in ./results
```
  source scripts/sharpening-C.bash
  source scripts/sharpening-D.bash
```
The use of sharpening is controlled though the parameters L1_sharpening_iterations,L4_sharpening_iterations (300=on, 0=off) for sharpening-C.bash, 
and through L1_sharpening_iterations, L4_sharpening_iterations, L7_sharpening_iterations for sharpening-D.bash.
Results are stored in results/sharp-c_sharp.png, results/sharp-d_sharp.png.

### 5.7 SVHN sampling
  ## training and, and sampling from, one class of SVHN. A PNG will be procued in ./results
```
  source scripts/sampling-svhn-B.bash
```



