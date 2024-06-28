from re import S
from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN, MPNEncoder
from .equations import get_equation
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.fp_method = args.fp_method
        self.output_size = args.num_tasks
        self.device = args.device
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.ffn_A = self.create_ffn(args)
        self.Tconcat = args.Tconcat
        if args.equation is not None:
            self.equation = get_equation(args)
            self.eqn_name=args.equation
        if not self.Tconcat:
            self.ffn_B = self.create_ffn(args)
            self.ffn_C = self.create_ffn(args)
            self.ffn_D = self.create_ffn(args)
            self.ffn_E = self.create_ffn(args)
            # self.ffn_fp = self.create_ffn(args, fp=True)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def create_ffn(self, args: TrainArgs, fp: bool = False) -> nn.Sequential:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.fp_method == 'hybrid_dim1':
                first_linear_dim *= 2
            if args.use_input_features:
                first_linear_dim += args.features_size
            if args.Tconcat:
                first_linear_dim += 1

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers ['molecular', 'atomic', 'hybrid_dim0', 'hybrid_dim1']
        if self.fp_method == 'molecular':
            if args.ffn_num_layers == 1:
                ffn = [
                    dropout,
                    nn.Linear(first_linear_dim, self.output_size)
                ]
            else:
                ffn = [
                    dropout,
                    nn.Linear(first_linear_dim, args.ffn_hidden_size)
                ]
                for _ in range(args.ffn_num_layers - 2):
                    ffn.extend([
                        activation,
                        dropout,
                        nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                    ])
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(
                        args.ffn_hidden_size, self.output_size if not fp else args.ffn_hidden_size),
                ])
        elif self.fp_method in ['atomic', 'hybrid_dim0', 'hybrid_dim1']:
            if args.ffn_num_layers == 1:
                ffn = [
                    TimeDistributed_wrapper(dropout),
                    # TODO: important bias set to False !!
                    TimeDistributed_wrapper(
                        nn.Linear(first_linear_dim, self.output_size))
                ]
            else:
                ffn = [
                    TimeDistributed_wrapper(dropout),
                    # TODO: important bias set to False !!
                    TimeDistributed_wrapper(
                        nn.Linear(first_linear_dim, args.ffn_hidden_size, bias=True))
                ]
                for _ in range(args.ffn_num_layers - 2):
                    ffn.extend([
                        TimeDistributed_wrapper(activation),
                        TimeDistributed_wrapper(dropout),
                        # TODO: important bias set to False !!
                        TimeDistributed_wrapper(
                            nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size)),
                    ])
                ffn.extend([
                    TimeDistributed_wrapper(activation),
                    TimeDistributed_wrapper(dropout),
                    # TODO: important bias set to False !!
                    TimeDistributed_wrapper(
                        nn.Linear(args.ffn_hidden_size, self.output_size if not fp else args.ffn_hidden_size, bias=False)),
                    #LambdaLayer(lambda x: torch.sum(x, 1)),
                ])

        # If spectra model, also include spectra activation
        if args.dataset_type == 'spectra':
            if args.spectra_activation == 'softplus':
                spectra_activation = nn.Softplus()
            else:  # default exponential activation which must be made into a custom nn module
                class nn_exp(torch.nn.Module):
                    def __init__(self):
                        super(nn_exp, self).__init__()

                    def forward(self, x):
                        return torch.exp(x)
                spectra_activation = nn_exp()
            ffn.append(spectra_activation)

        # Create FFN model
        ffnn = nn.Sequential(*ffn)

        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                # Freeze weights and bias for given number of layers
                for param in list(ffnn.parameters())[0:2*args.frzn_ffn_layers]:
                    param.requires_grad = False

        return ffnn

    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                    features_batch: List[np.ndarray] = None,
                    atom_descriptors_batch: List[np.ndarray] = None,
                    atom_features_batch: List[np.ndarray] = None,
                    bond_features_batch: List[np.ndarray] = None,
                    fingerprint_type='MPN') -> torch.FloatTensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model. 

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently 
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == 'MPN':
            return self.encoder(batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)
        elif fingerprint_type == 'last_FFN':
            return self.ffn_A[:-1](self.encoder(batch, features_batch, atom_descriptors_batch,
                                                atom_features_batch, bond_features_batch))
        else:
            raise ValueError(
                f'Unsupported fingerprint type {fingerprint_type}.')

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None,
                T_batch: torch.FloatTensor = None,
                smiles_batch: List[List[str]] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param sum_contribution: Return the summed value or the individual atomic contribution.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """
        hidden = self.encoder(batch, features_batch, atom_descriptors_batch,
                              atom_features_batch, bond_features_batch, smiles_batch)
        if self.Tconcat:
            # Temperature concatenated as fingerprint.
            if self.fp_method == 'molecular':
                T_batch = T_batch.reshape(-1, 1)
                hidden = torch.cat([hidden, T_batch], dim=1)
                output = self.ffn_A(hidden)
            else:
                # print(f'Init_fp:{hidden}')
                mask = self.encoder.encoder[0].mol_masks
                # mask = 1
                # print(f'mask:{mask}')
                temp_atomic = torch.Tensor([]).to(self.device)
                for n in range(hidden.size(0)):
                    temp = []
                    for _ in range(hidden.size(1)):
                        temp.append(T_batch[n])
                    temp = torch.Tensor(temp).unsqueeze(
                        0).unsqueeze(-1).to(self.device)
                    temp_atomic = torch.cat([temp_atomic, temp], dim=0)
                hidden = torch.cat([hidden, temp_atomic], dim=2)
                masked = hidden * mask
                ffn_output = self.ffn_A(masked)
                ffn_output: torch.FloatTensor
                # print(f'T_con_fp:{masked}')
                # print(f'ffn_output: {ffn_output}')
                # masked_output = ffn_output*mask
                # print(f'masked_output: {masked_output}')
                # output = torch.sum(masked_output, 1)
                output = torch.sum(ffn_output, 1)

        else:
            # hidden = self.ffn_fp(hidden)
            Pvap = torch.clone(T_batch)
            # print(f'T_batch:{T_batch.shape},{T_batch}')
            A = self.ffn_A(hidden)
            B = self.ffn_B(hidden)
            C = self.ffn_C(hidden)
            D = self.ffn_D(hidden)
            E = self.ffn_E(hidden)
            if self.fp_method == 'atomic' or self.fp_method == 'hybrid_dim0' or self.fp_method == 'hybrid_dim1':
                # mask = self.encoder.encoder[0].mol_masks
                mask = 1
                A = torch.sum(A * mask, 1)
                B = torch.sum(B * mask, 1)
                C = torch.sum(C * mask, 1)
                D = torch.sum(D * mask, 1)
                E = torch.sum(E * mask, 1)
            # print(f'Pvap:{Pvap.shape},{Pvap}')
            # print(f'A:{A.shape},{A}')
            # print(f'B:{B.shape},{B}')
            # print(f'C:{C.shape},{C}')
            # print(f'D:{D.shape},{D}')
            # print(f'E:{E.shape},{E}')
            # for i in range(hidden.size(0)):
            #     Pvap[i] = self.equation(A[i], B[i], C[i], D[i], E[i], T_batch[i])
            Pvap = self.equation(A, B, C, D, E, T_batch)
            output = Pvap.reshape(-1, 1)
            # print(f'output:{output.shape}')
            if self.eqn_name=='Tu':
                output = torch.sub(output, torch.log(self.encoder.MW()))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            # batch size x num targets x num classes per target
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
                output = self.multiclass_softmax(output)

        return output


class LambdaLayer(nn.Module):
    def __init__(self, lambda_function):
        super(LambdaLayer, self).__init__()
        self.lambda_function = lambda_function

    def forward(self, x):
        return self.lambda_function(x)


class TimeDistributed_wrapper(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed_wrapper, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        # reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))
        return y


if __name__ == "__main__":
    args = TrainArgs()
    print(args)
