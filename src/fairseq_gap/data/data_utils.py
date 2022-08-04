# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import itertools

import torch


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            # FIXME: Unclear why this assert is necessary. We should not use
            # last element anyway
            # assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def load_indexed_dataset(path, dictionary, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets.
    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    from . import indexed_dataset

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else '')

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        print('| loaded {} examples from: {}'.format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


def collate_embeddings(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""

    # longest sentence size
    size = max(v.size(0) for v in values)

    # embedding size
    emb_dim = list(set(v.size(1) for v in values))
    assert len(emb_dim) == 1
    emb_dim = emb_dim[0]

    # final tensor after collating all
    res = values[0].new(len(values), size, emb_dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            # FIXME: Unclear why this assert is necessary. We should not use
            # last element anyway
            # assert src[-1] == eos_idx
            dst[0, :] = eos_idx
            dst[1:, :] = src[:-1, :]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size - len(v):, :] if left_pad else res[i][:len(v), :]
        )
    return res


def collate_wp_idx(values, pad_idx, eos_idx, left_pad,
                   move_eos_to_beginning=False, reverse=False):
    """
    Convert a list of 1d tensor indices into a padded 2d tensor.
    Note that indices are also affected by padding and need to be changed
    """

    # maximum size
    size = max(v.size(0) for v in values)
    # tensor of collated tensors (has maximum size as dim 1)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            # FIXME: Unclear why this assert is necessary. We should not use
            # last element anyway
            # assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    # iterate over tensors
    for i, source_tensor in enumerate(values):
        if left_pad:
            # word indices need to be shifted by pad and indices to not average
            # pad pre-appended. We also target the entire tensor
            target_slice = res[i]
            if reverse:
                pad_size = size - len(source_tensor)
                index = torch.arange(
                    source_tensor[0] + 1,
                    source_tensor[0] + pad_size + 1
                ).flip(0)
                pad_size = 0
            else:
                index = torch.arange(pad_size)
                # for the left pad
                pad_size = size - len(source_tensor)
            copy_tensor(
                torch.cat((index, source_tensor + pad_size)),
                target_slice
            )
        else:
            raise NotImplementedError()
            target_slice = res[i][:len(source_tensor)]
            copy_tensor(source_tensor, target_slice)
    return res
