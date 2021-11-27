# Main experiments for anonymous CVPR submission "A new perspective on probabilistic image modeling"

## Main dependencies:
* Ubuntu 20.4
* TensorFlow 2.5.0
* PyTorch 1.5.1
* tensorflow-datasets 4.4.0


## Setting up
All datasets are downloaded via tfds or the analogous PyTorch functionalities, so be sure to have an active Internet connection on your PC.

We assume you cloned this repository to "cvpr22-experiments" via https. Then, we chdir to this path and stay there:
```
git clone https://github.com/anon-scientist/cvpr22-experiments
cd cvpr22-experiments
```

### Install pip dependencies

If Python3 and pip are not yet installed, install them using:
```
sudo apt-get install python3-pip
```
Then, use pip for installing the main dependencies:
```
python3 -m pip install -r requirements.txt
```

If you want to use a GPU with tensorflow, you will need to install a cuda driver and the cuda toolkit, as well as the cuDNN library.
See https://www.tensorflow.org/install/gpu for details

### Download and install EinsumNetworks
```
git clone https://github.com/cambridge-mlg/EinsumNetworks.git
export PYTHONPATH=$PYTHONPATH:<path>/EinsumNetworks/src
```

### Install libspn-keras
```
python3 -m pip install libspn-keras
```

### Install DCGMM
```
git clone https://github.com/anon-scientist/cvpr22-dcgmm
cd cvpr22-dcgmm
python3 -m pip install .
cd ..
```

### Setting up a path for the results
```
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
python3 -m DCGMM.utils.outliersFromJson --make_plots True --json results/outliers_dgcspn_log.json
python3 -m DCGMM.utils.outliersFromJson --make_plots True --json results/outliers_rat_log.json
python3 -m DCGMM.utils.outliersFromJson --make_plots True --json results/outliers_pd_log.json
python3 -m DCGMM.utils.outliersFromJson --make_plots True --json results/outliers_dcgmm-e_log.json
```

### 5.4 DCGMM sampling and sharpening. PNGs are stored in ./results
Training on, and sampling from, MNIST using DCGMM C and D (pooling architectures). 
```
source scripts/sharpening-C.bash
source scripts/sharpening-D.bash
```
The use of sharpening is controlled though the parameters L1_sharpening_iterations,L4_sharpening_iterations (300=on, 0=off) for sharpening-C.bash, 
and through L1_sharpening_iterations, L4_sharpening_iterations, L7_sharpening_iterations for sharpening-D.bash.
Results are stored in results/sharp-c_sharp.png, results/sharp-d_sharp.png.

### 5.7 SVHN sampling
Training on, and sampling from, one class of SVHN. A PNG will be procued in ./results
```
source scripts/sampling-svhn-B.bash
```

### 5.9 Generative-discriminative learning
```
source scripts/classifier.bash
```
You can observe the classification error on the console output ("accuracy")

### Appendix: generating the visual alphabet
```
source scripts/alphabet.bash
python3 -m DCGMM.utils.vis --what mus --prefix results/L2_ --out ./results/alphabet.png
```
View the result in results/alphabet.png. BTW, DCGMM.utils.vis is a flexible tool to visualize DCGMM centroids, variances etc. into png or pdf files

## A word on the ./bash and ./scripts subdirectories
The .bash subdir contains the "naked" definitions of parameters for the different DCGMM architectures A though G, in the form of executable script files.
These should be sourced or executed using the ./<script_file> syntax. 

For several experiments, only a few of the parameters are adapted (e.g., classes, epochs, sampling flags, ..). It seemed excessive to write a new script file 
for each new experiment. Each script in ./scripts therefore takes a file from ./bash, copies it to a file tmp.bash while replacing a select few parameters, leaving the rest untouched.
The command line arguments given in any ./scripts file are therefore only those that need to added/replaced w.r.t. to the respecive file in ./bash. 
What is finally executed is the generated file ./tmp.bash.

Formally, you should execute experiments from <path>. for example:
```
source scripts/sampling-svhn-B.bash
```



