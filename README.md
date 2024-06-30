# chemprop_VP

## Environment Setup

### For CentOS 7 (Linux) Users
Run the following command to set up the environment:
```sh
conda env create -f environment.yml
```

### For Windows Users
Run the following command to set up the environment:
```sh
conda env create -f environment_win.yml
```

**Note:** These environments include a CPU-only version of PyTorch. If GPU acceleration is desired, please refer to the [PyTorch homepage](https://pytorch.org) for the correct version and installation command. The recommended PyTorch version is 1.9.0.

## Testing the Model

After installing the environment, you can test the model's performance by executing the `ensemble_predict.sh` script. This script ensembles the models across 10 folds and provides the vapor pressure prediction by averaging the outputs of these 10 models.
