# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

from .swig import build as build_swig
from .swig import processed_build as build_processed_swig

def build_dataset(image_set, args):
    if args.dataset_file == 'swig':
        return build_swig(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def build_processed_dataset(image_set, args, neighbors_dict=None):
    if args.dataset_file == 'swig':
        return build_processed_swig(image_set, args, neighbors_dict=neighbors_dict)
    raise ValueError(f'dataset {args.dataset_file} not supported')