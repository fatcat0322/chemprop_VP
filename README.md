# chemprop_VP

To set up the environment, run the following command on CentOS 7 (Linux):
```sh
conda env create -f environment.yml
```
Note that this environment includes a cpu-only version of pytorch.
If GPU acceleration is desired, please check the pytorch homepage for the correct version and installing command.
After installing the environment, you can test the model's performance by executing the `ensemble_predict.sh` script. This script ensembles the models across 10 folds and provides the vapor pressure prediction by averaging the outputs of these 10 models.
