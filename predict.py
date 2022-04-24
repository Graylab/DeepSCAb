import os
import argparse
from datetime import datetime
from glob import glob
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import pyrosetta

import deepscab
from deepscab.models.AbChiResNet.AbChiResNet import load_model
from deepscab.models.ModelEnsemble import ModelEnsemble
from deepscab.constraints import get_constraint_residue_pairs, get_filtered_constraint_file
from deepscab.util.util import get_heavy_seq_len

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"


def prog_print(text):
    print("*" * 50)
    print(text)
    print("*" * 50)


def _get_args():
    """Gets command line arguments"""
    project_path = os.path.abspath(os.path.join(deepscab.__file__, "../.."))

    desc = ('''
        Script for predicting antibody Fv structures from heavy and light chain sequences.
        ''')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("fasta_file",
                        type=str,
                        help="""
        Fasta file containing Fv heavy and light chain sequences.
        Heavy and light chain sequences should be truncated at Chothia positions 112 and 109.
    """)

    now = str(datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
    default_pred_dir = os.path.join(project_path, "pred_{}".format(now))
    parser.add_argument("--pred_dir",
                        type=str,
                        default=default_pred_dir,
                        help="Directory where results should be saved.")

    default_model_dir = "pretrained_models"
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help="Directory containing pretrained model files (in .p format).")

    parser.add_argument("--target",
                        type=str,
                        default="pred",
                        help="Identifier for predicted structure naming.")
    parser.add_argument(
        "--renumber",
        default=False,
        action="store_true",
        help="Convert final predicted structure to Chothia format using AbNum."
    )

    return parser.parse_args()


def _cli():
    args = _get_args()

    fasta_file = args.fasta_file
    pred_dir = args.pred_dir
    model_dir = args.model_dir
    target = args.target

    model_files = list(glob(os.path.join(model_dir, "*.p")))
    if len(model_files) == 0:
        exit("No model files found at: {}".format(model_dir))

    model = ModelEnsemble(model_files=model_files,
                          load_model=load_model,
                          eval_mode=True)

    prog_print("Generating constraints")
    heavy_seq_len = get_heavy_seq_len(fasta_file)
    residue_pairs = get_constraint_residue_pairs(model, fasta_file,
                                                 heavy_seq_len)
    get_filtered_constraint_file(residue_pairs=residue_pairs,
                                 constraint_dir=pred_dir,
                                 local=True)


if __name__ == '__main__':
    _cli()