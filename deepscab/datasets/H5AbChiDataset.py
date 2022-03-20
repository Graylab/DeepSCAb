import h5py
import pickle
import math
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm
from deeph3.util.tensor import pad_data_to_same_shape
from deeph3.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins, get_rotamer_bins
from deeph3.util.preprocess import bin_dist_angle_matrix, bin_value_matrix


class H5AbChiDataset(data.Dataset):
    def __init__(self,
                 filename,
                 onehot_prim=True,
                 num_bins=36,
                 max_seq_len=None,
                 bin_labels=True):
        """
        :param filename: The h5 file for the antibody data.
        :param onehot_prim:
            Whether or not to onehot-encode the primary structure data.
        :param num_bins:
            The number of bins to discretize the distance matrix into. If None,
            then the distance matrix remains continuous.
        """
        super(H5AbChiDataset, self).__init__()

        self.onehot_prim = onehot_prim
        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape
        self.dist_bins = get_dist_bins(num_bins) if bin_labels else None
        self.omega_bins = get_dihedral_bins(num_bins) if bin_labels else None
        self.theta_bins = get_dihedral_bins(num_bins) if bin_labels else None
        self.phi_bins = get_planar_bins(num_bins) if bin_labels else None
        self.chi_one_bins = get_rotamer_bins(num_bins) if bin_labels else None
        self.chi_two_bins = get_dihedral_bins(num_bins) if bin_labels else None
        self.chi_three_bins = get_dihedral_bins(
            num_bins) if bin_labels else None
        self.chi_four_bins = get_dihedral_bins(
            num_bins) if bin_labels else None
        self.chi_five_bins = get_dihedral_bins(
            num_bins) if bin_labels else None

        # Filter out sequences beyond the max length
        self.max_seq_len = max_seq_len
        self.valid_indices = None
        if max_seq_len is not None:
            self.valid_indices = self.get_valid_indices()
            self.num_proteins = len(self.valid_indices)

        self.bin_labels = bin_labels

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')

        if self.valid_indices is not None:
            index = self.valid_indices[index]

        id_ = self.h5file['id'][index]
        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        light_seq_len = self.h5file['light_chain_seq_len'][index]
        total_seq_len = heavy_seq_len + light_seq_len

        # Get the attributes from a protein and cut off zero padding
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]
        light_prim = self.h5file['light_chain_primary'][index, :light_seq_len]

        # Convert to torch tensors
        heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        light_prim = torch.Tensor(light_prim).type(dtype=torch.uint8)

        # Get CDR loops
        h3 = self.h5file['h3_range'][index]

        if self.onehot_prim:
            heavy_prim = F.one_hot(heavy_prim.long())
            light_prim = F.one_hot(light_prim.long())

        # Try to get the distance matrix from memory
        try:
            dist_angle_mat = self.h5file['dist_angle_mat'][
                index][:4, :total_seq_len, :total_seq_len]
            dist_angle_mat = torch.Tensor(dist_angle_mat).type(
                dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')

        try:
            chi_one_mat = self.h5file['chi_one_mat'][index][:1, :total_seq_len]
            chi_one_mat = torch.Tensor(chi_one_mat).type(dtype=torch.float)
            chi_two_mat = self.h5file['chi_two_mat'][index][:1, :total_seq_len]
            chi_two_mat = torch.Tensor(chi_two_mat).type(dtype=torch.float)
            chi_three_mat = self.h5file['chi_three_mat'][
                index][:1, :total_seq_len]
            chi_three_mat = torch.Tensor(chi_three_mat).type(dtype=torch.float)
            chi_four_mat = self.h5file['chi_four_mat'][
                index][:1, :total_seq_len]
            chi_four_mat = torch.Tensor(chi_four_mat).type(dtype=torch.float)
            chi_five_mat = self.h5file['chi_five_mat'][
                index][:1, :total_seq_len]
            chi_five_mat = torch.Tensor(chi_five_mat).type(dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')

        # Bin output matrices for classification or leave real values for regression

        dist_angle_mat = bin_dist_angle_matrix(dist_angle_mat, self.dist_bins,
                                               self.omega_bins,
                                               self.theta_bins,
                                               self.phi_bins).long()
        chi_one_mat = bin_value_matrix(chi_one_mat, self.chi_one_bins).long()
        chi_two_mat = bin_value_matrix(chi_two_mat, self.chi_two_bins).long()
        chi_three_mat = bin_value_matrix(chi_three_mat,
                                         self.chi_three_bins).long()
        chi_four_mat = bin_value_matrix(chi_four_mat,
                                        self.chi_four_bins).long()
        chi_five_mat = bin_value_matrix(chi_five_mat,
                                        self.chi_five_bins).long()

        return id_, heavy_prim, light_prim, dist_angle_mat, chi_one_mat, chi_two_mat, chi_three_mat, chi_four_mat, chi_five_mat, h3

    def get_valid_indices(self):
        """Gets indices with proteins less than the max sequence length
        :return: A list of indices with sequence lengths less than the max
                 sequence length.
        :rtype: list
        """
        valid_indices = []
        for i in range(self.h5file['heavy_chain_seq_len'].shape[0]):
            h_len = self.h5file['heavy_chain_seq_len'][i]
            l_len = self.h5file['light_chain_seq_len'][i]
            total_seq_len = h_len + l_len
            if total_seq_len < self.max_seq_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return H5AbChiBatch(zip(*samples)).data()


class H5AbChiBatch:
    def __init__(self, batch_data):
        (self.id_, self.heavy_prim, self.light_prim, self.dist_angle_mat,
         self.chi_one_mat, self.chi_two_mat, self.chi_three_mat,
         self.chi_four_mat, self.chi_five_mat, self.h3) = batch_data

    def data(self):
        return self.features(), self.labels()

    def features(self):
        """Gets the one-hot encoding of the sequences with a feature that
        delimits the chains"""
        X = [torch.cat(_, 0) for _ in zip(self.heavy_prim, self.light_prim)]
        X = pad_data_to_same_shape(X, pad_value=0).float()

        # Add chain delimiter
        X = F.pad(X, (0, 1, 0, 0, 0, 0))
        for i, h_prim in enumerate(self.heavy_prim):
            X[i, len(h_prim) - 1, X.shape[2] - 1] = 1

        # Switch shape from [batch, timestep/length, filter/channel]
        #                to [batch, filter/channel, timestep/length]
        return X.transpose(1, 2).contiguous()

    def labels(self):
        label_mat_2D = pad_data_to_same_shape(self.dist_angle_mat,
                                              pad_value=-999).transpose(0, 1)
        label_mat_1D_one = pad_data_to_same_shape(self.chi_one_mat,
                                                  pad_value=-999).transpose(
                                                      0, 1)
        label_mat_1D_two = pad_data_to_same_shape(self.chi_two_mat,
                                                  pad_value=-999).transpose(
                                                      0, 1)
        label_mat_1D_three = pad_data_to_same_shape(self.chi_three_mat,
                                                    pad_value=-999).transpose(
                                                        0, 1)
        label_mat_1D_four = pad_data_to_same_shape(self.chi_four_mat,
                                                   pad_value=-999).transpose(
                                                       0, 1)
        label_mat_1D_five = pad_data_to_same_shape(self.chi_five_mat,
                                                   pad_value=-999).transpose(
                                                       0, 1)

        return label_mat_2D[0], label_mat_2D[1], label_mat_2D[2], label_mat_2D[
            3], label_mat_1D_one[0], label_mat_1D_two[0], label_mat_1D_three[
                0], label_mat_1D_four[0], label_mat_1D_five[0]

    def batch_mask(self):
        """Gets the mask data of the batch with zero padding"""
        '''Code to use when masks are added
        masks = self.mask
        masks = pad_data_to_same_shape(masks, pad_value=0)
        return masks
        '''
        raise NotImplementedError(
            'Masks have not been added to antibodies yet')


def h5_antibody_dataloader(filename,
                           batch_size=1,
                           max_seq_len=None,
                           num_bins=36,
                           **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update(dict(collate_fn=H5AbChiDataset.merge_samples_to_minibatch))
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(
        H5AbChiDataset(filename, num_bins=num_bins, max_seq_len=max_seq_len),
        **kwargs)
