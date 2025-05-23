from collections import OrderedDict
import csv
from os import sep
from typing import List, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

from .evaluate import evaluate_predictions
from .predict import predict
from chemprop.spectra_utils import normalize_spectra, roundrobin_sid
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset, StandardScaler, divmaxScaler
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_reaction, set_explicit_h
from chemprop.models import MoleculeModel

import matplotlib.pyplot as plt
import math


def pred_true_curve(args: PredictArgs, pred_vals, true_vals, cur_name: str, unit_name: str):

    fig, ax = plt.subplots()
    x, y = np.array(true_vals), np.array(pred_vals)
    base = []
    pred = np.hstack(y)
    true_v = np.hstack(x)
    mini = min(min(pred), min(true_v))
    maxi = max(max(pred), max(true_v))
    base.append(mini)
    base.append(maxi)
    plt.plot(x, y, '.')
    plt.plot(base, base, '--')
    ax.set_title(f'{cur_name}')
    plt.ylabel('Predicted ' + unit_name)
    plt.xlabel('Reference ' + unit_name)
    plt.xlim(math.floor(mini), math.ceil(maxi))
    plt.ylim(math.floor(mini), math.ceil(maxi))
    plt.savefig(f'{args.checkpoint_dir}/parity_plot.png',
                dpi=300, format='png', bbox_inches='tight')


def load_model(args: PredictArgs, generator: bool = False):
    """
    Function to load a model or ensemble of models from file. If generator is True, a generator of the respective model and scaler 
    objects is returned (memory efficient), else the full list (holding all models in memory, necessary for preloading).

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param generator: A boolean to return a generator instead of a list of models and scalers.
    :return: A tuple of updated prediction arguments, training arguments, a list or generator object of models, a list or 
                 generator object of scalers, the number of tasks and their respective names.
    """
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    # Load model and scalers
    models = (load_checkpoint(checkpoint_path, device=args.device)
              for checkpoint_path in args.checkpoint_paths)
    scalers = (load_scalers(checkpoint_path)
               for checkpoint_path in args.checkpoint_paths)
    if not generator:
        models = list(models)
        scalers = list(scalers)

    return args, train_args, models, scalers, num_tasks, task_names


def load_data(args: PredictArgs, smiles: List[List[str]], train_args: TrainArgs):
    """
    Function to load data from a list of smiles or a file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: A list of list of smiles, or None if data is to be read from file
    :return: A tuple of a :class:`~chemprop.data.MoleculeDataset` containing all datapoints, a :class:`~chemprop.data.MoleculeDataset` containing only valid datapoints,
                 a :class:`~chemprop.data.MoleculeDataLoader` and a dictionary mapping full to valid indices. 
    """
    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=train_args.target_columns, ignore_columns=train_args.ignore_columns, temperature_columns=train_args.temperature_columns,
                             skip_invalid_smiles=False, args=args, store_row=not args.drop_extra_columns)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i]
                                for i in sorted(full_to_valid_indices.keys())])

    print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    return full_data, test_data, test_data_loader, full_to_valid_indices


def set_features(args: PredictArgs, train_args: TrainArgs):
    """
    Function to set extra options.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    """
    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_features_path is not None:
        set_extra_bond_fdim(train_args.bond_features_size)

    # set explicit H option and reaction option
    set_explicit_h(train_args.explicit_h)
    set_reaction(train_args.reaction, train_args.reaction_mode)


def predict_and_save(args: PredictArgs, train_args: TrainArgs, test_data: MoleculeDataset,
                     task_names: List[str], num_tasks: int, test_data_loader: MoleculeDataLoader, full_data: MoleculeDataset,
                     full_to_valid_indices: dict, models: List[MoleculeModel], scalers: List[List[StandardScaler]]):
    """
    Function to predict with a model and save the predictions to file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param task_names: A list of task names.
    :param num_tasks: Number of tasks.
    :param test_data_loader: A :class:`~chemprop.data.MoleculeDataLoader` to load the test data.
    :param full_data:  A :class:`~chemprop.data.MoleculeDataset` containing all (valid and invalid) datapoints.
    :param full_to_valid_indices: A dictionary dictionary mapping full to valid indices.
    :param models: A list or generator object of :class:`~chemprop.models.MoleculeModel`\ s.
    :param scalers: A list or generator object of :class:`~chemprop.features.scaler.StandardScaler` objects.
    :return:  A list of lists of target predictions.
    """
    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros(
            (len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))
    if args.ensemble_variance or args.individual_ensemble_predictions:
        if args.dataset_type == 'multiclass':
            all_preds = np.zeros(
                (len(test_data), num_tasks, args.multiclass_num_classes, len(args.checkpoint_paths)))
        else:
            all_preds = np.zeros(
                (len(test_data), num_tasks, len(args.checkpoint_paths)))

    # Partial results for variance robust calculation.
    print(
        f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, (model, scaler_list) in enumerate(tqdm(zip(models, scalers), total=len(args.checkpoint_paths))):
        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler, T_scaler = scaler_list

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(
                    atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                test_data.normalize_features(
                    bond_feature_scaler, scale_bond_features=True)

        if T_scaler is not None:
            T_scaler: divmaxScaler
            test_data.reset_features_and_targets()
            test_ts = test_data.temperatures()
            scaled_ts = T_scaler.transform(test_ts).tolist()
            test_data.set_ts(scaled_ts)

        test_targets = test_data.targets()

        # Make predictions
        model_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )
        if not args.predict_only:
            test_scores = evaluate_predictions(
                preds=model_preds,
                targets=test_targets,
                num_tasks=train_args.num_tasks,
                metrics=train_args.metrics,
                dataset_type=train_args.dataset_type
            )
            print(f'Model {index} test {train_args.metric}: {test_scores}')
        else:
            test_scores='NAN'
        if args.dataset_type == 'spectra':
            model_preds = normalize_spectra(
                spectra=model_preds,
                phase_features=test_data.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=float('nan')
            )
        sum_preds += np.array(model_preds,dtype=float)
        if args.ensemble_variance or args.individual_ensemble_predictions:
            if args.dataset_type == 'multiclass':
                all_preds[:, :, :, index] = model_preds
            else:
                all_preds[:, :, index] = model_preds

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)

    if not args.predict_only:
        ens_scores = evaluate_predictions(
            preds=avg_preds,
            targets=test_targets,
            num_tasks=train_args.num_tasks,
            metrics=train_args.metrics,
            dataset_type=train_args.dataset_type
        )
        if 'train' in args.test_path:
            mode = 'Train'
        elif 'val' in args.test_path:
            mode = 'Validation'
        else:
            mode = 'Test'
        pred_true_curve(args, avg_preds, test_targets,
                        cur_name='lnP ' + mode + '-True parity plot', unit_name='lnP (P in Pa)')
        print(f'Overall test {train_args.metric}: {ens_scores}')
    else:
        ens_scores='NAN'
    

    if args.ensemble_variance:
        if args.dataset_type == 'spectra':
            all_epi_uncs = roundrobin_sid(all_preds)
        else:
            all_epi_uncs = np.var(all_preds, axis=2)
            all_epi_uncs = all_epi_uncs.tolist()

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    if args.ensemble_variance:
        assert len(test_data) == len(all_epi_uncs)
    makedirs(args.preds_path, isfile=True)

    # Set multiclass column names, update num_tasks definition for multiclass
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(
            args.multiclass_num_classes)]
        num_tasks = num_tasks * args.multiclass_num_classes

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else [
            'Invalid SMILES'] * num_tasks
        if args.ensemble_variance:
            if args.dataset_type == 'spectra':
                epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else [
                    'Invalid SMILES']
            else:
                epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else [
                    'Invalid SMILES'] * num_tasks
        if args.individual_ensemble_predictions:
            ind_preds = all_preds[valid_index] if valid_index is not None else [
                ['Invalid SMILES'] * len(args.checkpoint_paths)] * num_tasks

        # Reshape multiclass to merge task and class dimension, with updated num_tasks
        if args.dataset_type == 'multiclass':
            preds = preds.reshape((num_tasks))
            if args.ensemble_variance or args. individual_ensemble_predictions:
                ind_preds = ind_preds.reshape(
                    (num_tasks, len(args.checkpoint_paths)))

        # If extra columns have been dropped, add back in SMILES columns
        if args.drop_extra_columns:
            datapoint.row = OrderedDict()

            smiles_columns = args.smiles_columns

            for column, smiles in zip(smiles_columns, datapoint.smiles):
                datapoint.row[column] = smiles

        # Add predictions columns
        for pred_name, pred in zip(task_names, preds):
            datapoint.row[pred_name] = pred
        if args.individual_ensemble_predictions:
            for pred_name, model_preds in zip(task_names, ind_preds):
                for idx, pred in enumerate(model_preds):
                    datapoint.row[pred_name+f'_model_{idx}'] = pred
        if args.ensemble_variance:
            if args.dataset_type == 'spectra':
                datapoint.row['epi_unc'] = epi_uncs
            else:
                for pred_name, epi_unc in zip(task_names, epi_uncs):
                    datapoint.row[pred_name+'_epi_unc'] = epi_unc

    # Save
    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
        writer.writeheader()

        for datapoint in full_data:
            writer.writerow(datapoint.row)

    avg_preds = avg_preds.tolist()

    return avg_preds


@timeit()
def make_predictions(args: PredictArgs, smiles: List[List[str]] = None,
                     model_objects: Tuple[PredictArgs, TrainArgs, List[MoleculeModel], List[StandardScaler], int, List[str]] = None) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param model_objects: Tuple of output of load_model function which can be called separately.
    :return: A list of lists of target predictions.
    """
    if model_objects:
        args, train_args, models, scalers, num_tasks, task_names = model_objects
    else:
        args, train_args, models, scalers, num_tasks, task_names = load_model(
            args, generator=True)

    set_features(args, train_args)

    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(
        args, smiles, train_args)

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    avg_preds = predict_and_save(args, train_args, test_data, task_names, num_tasks,
                                 test_data_loader, full_data, full_to_valid_indices, models, scalers)

    return avg_preds


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())
