import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiBPEDataset(Dataset):
    """Characterizes a MultiBPE Dataset."""

    def __init__(
        self,
        file_path,
        embed_indices,
        output_name,
        group_name=None,
        ct_feature=False,
        tf_feature=False,
        exclude_groups=None,
        include_groups=None,
        no_target=False,
    ):
        self.__CT_FEATURE_NAME = "ct_feature"
        self.__TF_FEATURE_NAME = "tf_feature"

        self.file_path = file_path
        self.embed_indices = embed_indices
        self.output_name = output_name
        self.group_name = group_name
        self.ct_feature = ct_feature
        self.tf_feature = tf_feature
        self.exclude_groups = exclude_groups
        self.include_groups = include_groups

        self.no_target = no_target
        if self.no_target:
            self.__find_embed_ids()

        self.dset_length = self.__initialize_index()

    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.dset_length

    def __getitem__(self, index):
        """Generates one sample of size [seq_length, num_features]."""
        with h5py.File(self.file_path, "r") as dset:
            # Get embedding indices (and target).
            if self.no_target:
                itarget = None
                ilabel = self.ilabel
                iid = index
                tf = self.tf
                ct = self.ct
            else:
                item = self.__retrieve_target(dset, index)
                itarget, ilabel, iid = item.split([item.numel() - 3, 2, 1])
                tf, ct = self.__find_embed_labels(ilabel)

            # Get feature.
            input_group = dset["input"][str(int(iid))]
            ifeature = torch.from_numpy(input_group["fasta"][:]).float()
            if self.ct_feature:
                ifeature = self.__add_feature(
                    ifeature, input_group, self.__CT_FEATURE_NAME, ct
                )
            if self.tf_feature:
                ifeature = self.__add_feature(
                    ifeature, input_group, self.__TF_FEATURE_NAME, tf
                )

        return ifeature, ilabel, itarget

    def __initialize_index(self):
        """Initialize sample index arrays and self.dset_length."""
        with h5py.File(self.file_path, "r") as dset:
            if self.no_target:
                if self.group_name is None:
                    raise ValueError(
                        "group_name can not be null when target labels are "
                        "unavailable."
                    )
                return len(dset["input"])
            else:
                path_list = []
                index_list = []
                for output_name in self.output_name:
                    self.__initialize_output_name_index(
                        dset, output_name, path_list, index_list
                    )
                self.path_ary = np.array(path_list)
                self.index_ary = np.array(index_list)
                return len(self.path_ary)

    def __initialize_output_name_index(self, dset, output_name, path, index):
        """Initialize sample indices for one output_name."""
        group_names = set()
        dset_group = dset[output_name]
        if self.group_name is not None:
            self.__add_index(dset_group[self.group_name], path, index)
        else:
            for group_name in dset_group.keys():
                if self.exclude_groups is not None:
                    if group_name in self.exclude_groups:
                        continue
                if self.include_groups is not None:
                    if group_name not in self.include_groups:
                        continue
                group_names.add(group_name)
                self.__add_index(dset_group[group_name], path, index)
        print(group_names)

    def __add_index(self, group, path, index):
        """Add a sample index to index arrays."""
        p = group.name
        for j in range(len(group)):
            path.append(p)
            index.append(j)

    def __retrieve_target(self, dset, index):
        """Retrieve a sample target label."""
        p = self.path_ary[index]
        j = self.index_ary[index]
        return torch.from_numpy((dset[p][j])).float()

    def __add_feature(self, res, dset_group, feature_name, feature_id):
        """Retrieve a type of feature signal from dataset."""
        feature_np = dset_group[feature_name][feature_id][:]
        feature = torch.from_numpy(feature_np).float()

        # Check feature and res dimension compatibility
        fdim = feature.dim()
        rdim = res.dim()
        if fdim == rdim - 1:
            return torch.cat([res, feature.unsqueeze(-1)], dim=1)
        elif fdim == rdim:
            return torch.cat([res, feature], dim=1)
        else:
            raise RuntimeError(
                "Invalid feature dimensions: expect {} or {}, "
                "got {}.".format(rdim - 1, rdim, fdim)
            )

    def __find_embed_labels(self, arr):
        """Find the embedding labels from embedding indices."""
        tf = self.embed_indices["tf"][int(arr[0])]
        ct = self.embed_indices["ct"][int(arr[1])]
        return tf, ct

    def __find_embed_ids(self):
        """Find the embedding indices from embedding labels."""
        self.tf, self.ct = self.group_name.split(".")
        tf_id = self.embed_indices["tf"][self.tf]
        ct_id = self.embed_indices["ct"][self.ct]
        self.ilabel = torch.tensor([tf_id, ct_id])


def collate_samples(batch):
    """Collate samples in a minibatch."""
    feature = torch.stack([b[0] for b in batch])
    label = torch.stack([b[1] for b in batch]).long()

    target = [b[2] for b in batch]
    if None in target:
        target = None
    else:
        target = torch.stack(target).squeeze().long()
    return ((feature, label), target)


def push_to_device(batch, device):
    """Push tensors to device."""
    def _push_var(var, device):
        if var is not None:
            return var.to(device)

    inputs, targets = batch
    return (
        (_push_var(inputs[0], device), _push_var(inputs[1], device)),
        _push_var(targets, device),
    )


def get_group_names(
    h5_path,
    output_keys,
    exclude_groups,
    include_groups,
    no_target,
    predict_groups,
):
    """Get eligible group names."""
    if no_target:
        if predict_groups is None or len(predict_groups) == 0:
            raise ValueError(
                "--predict_groups cannot be null or empty when "
                "target labels are unavailable."
            )
        return predict_groups
    else:
        group_names = set()
        with h5py.File(h5_path, "r") as dset:
            for output_key in output_keys:
                for group_name in dset[output_key].keys():
                    if exclude_groups is not None:
                        if group_name in exclude_groups:
                            continue
                    if include_groups is not None:
                        if group_name not in include_groups:
                            continue
                    group_names.add(group_name)
        return group_names
