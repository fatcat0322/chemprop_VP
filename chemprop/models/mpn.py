from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph, mol2morgan
from chemprop.nn_utils import index_select_ND, get_activation_function

from chemprop.rdkit_mol import make_mol
from rdkit.Chem.Descriptors import MolWt


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.fp_method = args.fp_method

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(
            self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def padding(self, mol_vector, padding_final_size=100):
        """ zero-padding tensor, ONLY used for atomic and hybrid method. """
        if self.fp_method == 'hybrid_dim0':
            padding_final_size = 101
        num_atoms_index = mol_vector.shape[0]
        num_features_index = mol_vector.shape[1]
        padding_tensor = torch.zeros((padding_final_size, num_features_index))
        padding_tensor[:num_atoms_index, :] = mol_vector
        return padding_tensor

    def mask(self, mol_vector, padding_final_size=100):
        """ mask for atomic method """
        if self.fp_method == 'hybrid_dim0':
            padding_final_size = 101
        num_atoms_index = mol_vector.shape[0]
        mask = torch.zeros((padding_final_size, 1))
        mask[:num_atoms_index, :] = 1
        return mask

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = [
                np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch
            atom_descriptors_batch = torch.from_numpy(np.concatenate(
                atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(
            self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2a)
                # num_atoms x max_num_bonds x bond_fdim
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                # num_atoms x max_num_bonds x hidden + bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                # num_atoms x hidden + bond_fdim
                message = nei_message.sum(dim=1)
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        # num_atoms x max_num_bonds x hidden
        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        # num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(
                    f'The number of atoms is different from the length of the extra atom features')

            # num_atoms x (hidden + descriptor size)
            atom_hiddens = torch.cat(
                [atom_hiddens, atom_descriptors_batch], dim=1)
            # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        self.mol_masks = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec_molar = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec_molar = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec_molar = mol_vec.sum(dim=0) / self.aggregation_norm
                # mol_vec_molar is molecular fingerprint
                if self.fp_method == 'molecular':
                    mol_vecs.append(mol_vec_molar)
                else:  # atomic or hybrid method
                    mol_vec_molar = torch.reshape(
                        mol_vec_molar, (1, list(mol_vec_molar.size())[0]))
                    if self.fp_method == 'hybrid_dim0':
                        new_mol_vec = torch.cat(
                            (mol_vec, mol_vec_molar), dim=0)
                    elif self.fp_method == 'hybrid_dim1':
                        mol_vec_molar_ensemble = mol_vec_molar
                        for i in range(0, list(mol_vec.size())[0]-1):
                            mol_vec_molar_ensemble = torch.cat(
                                (mol_vec_molar_ensemble, mol_vec_molar), dim=0)
                        new_mol_vec = torch.cat(
                            (mol_vec, mol_vec_molar_ensemble), dim=1)
                    elif self.fp_method == 'atomic':
                        new_mol_vec = mol_vec
                    # padding and collect
                    mol_mask = self.mask(new_mol_vec)
                    new_mol_vec = self.padding(new_mol_vec)
                    self.mol_masks.append(mol_mask)
                    mol_vecs.append(new_mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        mol_vecs = mol_vecs.to(self.device)
        if self.fp_method != 'molecular':
            self.mol_masks = torch.stack(self.mol_masks).to(self.device)
        return mol_vecs  # [num_molecules, hidden] if fp_method == 'molecular'
        # [num_molecules, num_atoms(42 or 43), hidden_size] if fp_method in ['atomic', 'hybrid_dim0', 'hybrid_dim1']


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features
        self.morgan = True if args.fp_method == 'morgan' else False
        self.hidden_size = args.hidden_size

        if self.features_only or self.morgan:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList(
                [MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None,
                smiles_batch: List[List[str]] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        self.SMILES = smiles_batch
        # Save SMILES batch for calculating molecular weight.

        if self.morgan:
            # Directly output Morgan fingerprint.
            output = mol2morgan(batch, self.hidden_size).to(self.device)
            return output

        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(
                np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch)
                         for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output

    def MW(self) -> torch.FloatTensor:
        '''
        Calculate molecular weight for group contribution methods.
        '''
        MWs = []
        for SMILE in self.SMILES:
            # print(f'SMILES:{SMILE[0]}')
            MW = MolWt(make_mol(SMILE[0], keep_h=True))
            MWs.append(MW)
        output = torch.Tensor(MWs).to(self.device).unsqueeze(-1)
        # print(f'MW:{output.shape}')
        return output


if __name__ == "__main__":
    from chemprop.args import TrainArgs
    args = TrainArgs()
    args.fp_method = 'molecular'
    MPN_test = MPN(args)
    smiles = [['CCCC'], ['ClC']]
    print('output:')
    outputs = MPN_test(smiles)
    print(outputs)
    print('output shape:')
    print(outputs.shape)
