import os
import math
from typing import Iterable, List
from tqdm import tqdm
import torch

import torch

from deeph3.constraints import Constraint, ConstraintType, Residue, ResiduePair, constraint_type_generator_dict
from deeph3.constraints.custom_filters import cb_glycine_filter, no_max_distance_filter, local_interaction_filter, rotamer_filter
from deeph3.constraints.rosetta_constraint_generators import neg_log_prob_to_energy
from deeph3.models.H3ResNet import H3ResNet
from deeph3.models.ExpandedGeometryResNet import ExpandedGeometryResNet
from deeph3.models.AbChiResNet import AbChiResNet
from deeph3.models.AbResNet import AbResNet
from deeph3.models.ModelEnsemble import ModelEnsemble
from deeph3.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins, get_rotamer_bins, get_bin_values
from deeph3.util.model_out import get_probs_from_model
from deeph3.util.util import load_full_seq
from deeph3.util.model_out import get_logits_from_model, get_probs_from_model, bin_matrix, binned_mat_to_values
from deeph3.util.util import load_full_seq

model_out_constraint_dict = {
    H3ResNet: [
        ConstraintType.cbca_distance, ConstraintType.omega_dihedral,
        ConstraintType.theta_dihedral, ConstraintType.phi_planar
    ],
    ExpandedGeometryResNet: [
        ConstraintType.cbca_distance, ConstraintType.omega_dihedral,
        ConstraintType.theta_dihedral, ConstraintType.phi_planar,
        ConstraintType.bb_phi_dihedral, ConstraintType.bb_psi_dihedral
    ],
    AbResNet: [
        ConstraintType.ca_distance, ConstraintType.cb_distance,
        ConstraintType.no_distance, ConstraintType.omega_dihedral,
        ConstraintType.theta_dihedral, ConstraintType.phi_planar
    ],
    AbChiResNet: [
        ConstraintType.cbca_distance, ConstraintType.omega_dihedral,
        ConstraintType.theta_dihedral, ConstraintType.phi_planar,
        ConstraintType.chi_one_dihedral, ConstraintType.chi_two_dihedral,
        ConstraintType.chi_three_dihedral, ConstraintType.chi_four_dihedral,
        ConstraintType.chi_five_dihedral
    ]
}

pairwise_constraint_types = [
    ConstraintType.cbca_distance, ConstraintType.ca_distance,
    ConstraintType.cb_distance, ConstraintType.no_distance,
    ConstraintType.omega_dihedral, ConstraintType.theta_dihedral,
    ConstraintType.phi_planar
]

asymmetric_constraint_types = [
    ConstraintType.no_distance, ConstraintType.theta_dihedral,
    ConstraintType.phi_planar
]

rotamer_constraint_types = [
    ConstraintType.chi_one_dihedral, ConstraintType.chi_two_dihedral,
    ConstraintType.chi_three_dihedral, ConstraintType.chi_four_dihedral,
    ConstraintType.chi_five_dihedral
]


def get_constraint_bin_value_dict(num_out_bins: int,
                                  mask_distant_orientations: bool = True):
    masked_bin_num = 1 if mask_distant_orientations else 0
    dist_bin_values = get_bin_values(get_dist_bins(num_out_bins))
    dihedral_bin_values = get_bin_values(
        get_dihedral_bins(num_out_bins - masked_bin_num, rad=True))
    planar_bin_values = get_bin_values(
        get_planar_bins(num_out_bins - masked_bin_num, rad=True))
    rotamer_bin_values = get_bin_values(
        get_rotamer_bins(num_out_bins, rad=True))

    constraint_bin_value_dict = {
        ConstraintType.cbca_distance: dist_bin_values,
        ConstraintType.ca_distance: dist_bin_values,
        ConstraintType.cb_distance: dist_bin_values,
        ConstraintType.no_distance: dist_bin_values,
        ConstraintType.omega_dihedral: dihedral_bin_values,
        ConstraintType.theta_dihedral: dihedral_bin_values,
        ConstraintType.phi_planar: planar_bin_values,
        ConstraintType.bb_phi_dihedral: dihedral_bin_values,
        ConstraintType.bb_psi_dihedral: dihedral_bin_values,
        ConstraintType.chi_one_dihedral: rotamer_bin_values,
        ConstraintType.chi_two_dihedral: dihedral_bin_values,
        ConstraintType.chi_three_dihedral: dihedral_bin_values,
        ConstraintType.chi_four_dihedral: dihedral_bin_values,
        ConstraintType.chi_five_dihedral: dihedral_bin_values
    }

    return constraint_bin_value_dict


def get_constraint_residue_pairs(model: torch.nn.Module,
                                 fasta_file: str,
                                 heavy_seq_len: int,
                                 constraint_bin_value_dict: dict = None,
                                 mask_distant_orientations: bool = False,
                                 use_logits: bool = False):
    seq = load_full_seq(fasta_file)

    model_type = type(
        model) if not type(model) == ModelEnsemble else model.model_type()
    model_out_constraint_types = model_out_constraint_dict[model_type]
    if constraint_bin_value_dict == None:
        constraint_bin_value_dict = get_constraint_bin_value_dict(
            model._num_out_bins,
            mask_distant_orientations=mask_distant_orientations)

    if use_logits:
        logits = [
            p.permute(1, 2, 0)
            for p in get_logits_from_model(model, fasta_file)
        ]
        preds = logits

        ca_dist_mat = binned_mat_to_values(
            bin_matrix(preds[0].permute(2, 0, 1), are_logits=use_logits),
            bins=get_dist_bins(model._num_out_bins))
        y_scale = 1 / (ca_dist_mat * ca_dist_mat)
    else:
        preds = get_probs_from_model(model, fasta_file)
        y_scale = torch.ones((len(seq), len(seq)))

    residue_pairs = []
    for i in tqdm(range(len(seq))):
        residue_i = Residue(identity=seq[i], index=i + 1)

        # Extract rotamer constraints
        ii_constraints = []
        for pred_i, constraint_type in enumerate(model_out_constraint_types):
            if constraint_type in rotamer_constraint_types:
                y_vals = preds[pred_i][i]
                if constraint_type == ConstraintType.chi_one_dihedral and not use_logits:
                    inv_bin_width = torch.tensor([
                        1.0 / (b[1] - b[0])
                        for b in get_rotamer_bins(model._num_out_bins,
                                                  rad=True)
                    ])
                    scale_mat = torch.eye(len(inv_bin_width)) * inv_bin_width
                    y_vals = torch.matmul(y_vals, scale_mat)
                    y_vals = y_vals / y_vals.sum(-1, keepdim=True)

                ii_constraints += [
                    Constraint(
                        constraint_type=constraint_type,
                        residue_1=residue_i,
                        residue_2=residue_i,
                        x_vals=constraint_bin_value_dict[constraint_type],
                        y_vals=y_vals,
                        are_logits=use_logits)
                ]

        residue_pairs.append(
            ResiduePair(residue_1=residue_i,
                        residue_2=residue_i,
                        constraints=ii_constraints))

        # Extract inter-residue constraints
        for j in range(i):
            residue_j = Residue(identity=seq[j], index=j + 1)

            ij_constraints = []
            for pred_i, constraint_type in enumerate(
                    model_out_constraint_types):

                if constraint_type in pairwise_constraint_types:
                    if preds[pred_i][i, j].argmax().item() >= len(
                            constraint_bin_value_dict[constraint_type]):
                        continue
                    ij_constraints += [
                        Constraint(
                            constraint_type=constraint_type,
                            residue_1=residue_i,
                            residue_2=residue_j,
                            x_vals=constraint_bin_value_dict[constraint_type],
                            y_vals=preds[pred_i][i, j]
                            [:len(constraint_bin_value_dict[constraint_type])],
                            are_logits=use_logits,
                            y_scale=y_scale[i, j])
                    ]

                    if constraint_type in asymmetric_constraint_types:
                        ij_constraints += [
                            Constraint(
                                constraint_type=constraint_type,
                                residue_1=residue_j,
                                residue_2=residue_i,
                                x_vals=constraint_bin_value_dict[
                                    constraint_type],
                                y_vals=preds[pred_i][j, i][:len(
                                    constraint_bin_value_dict[constraint_type]
                                )],
                                are_logits=use_logits,
                                y_scale=y_scale[i, j])
                        ]
                elif constraint_type == ConstraintType.bb_phi_dihedral and i - 1 == j and (
                        i < heavy_seq_len or j > heavy_seq_len):
                    ij_constraints += [
                        Constraint(
                            constraint_type=constraint_type,
                            residue_1=residue_j,
                            residue_2=residue_i,
                            x_vals=constraint_bin_value_dict[constraint_type],
                            y_vals=preds[pred_i][i]
                            [:len(constraint_bin_value_dict[constraint_type])],
                            are_logits=use_logits,
                            y_scale=y_scale[i, j])
                    ]
                elif constraint_type == ConstraintType.bb_psi_dihedral and i - 1 == j and (
                        i < heavy_seq_len or j > heavy_seq_len):
                    ij_constraints += [
                        Constraint(
                            constraint_type=constraint_type,
                            residue_1=residue_j,
                            residue_2=residue_i,
                            x_vals=constraint_bin_value_dict[constraint_type],
                            y_vals=preds[pred_i][j]
                            [:len(constraint_bin_value_dict[constraint_type])],
                            are_logits=use_logits,
                            y_scale=y_scale[i, j])
                    ]

            residue_pairs.append(
                ResiduePair(residue_1=residue_i,
                            residue_2=residue_j,
                            constraints=ij_constraints))

    return residue_pairs


def get_filtered_constraint_file(residue_pairs: List[ResiduePair],
                                 constraint_dir: str,
                                 threshold: float = 0.1,
                                 res_range: Iterable = None,
                                 max_separation: int = math.inf,
                                 local: bool = False,
                                 heavy_seq_len=None,
                                 heavy_only: bool = False,
                                 light_only: bool = False,
                                 interchain: bool = False,
                                 constraint_types: List[ConstraintType] = None,
                                 constraint_filters: List = None,
                                 prob_to_energy=neg_log_prob_to_energy):
    if not os.path.exists(constraint_dir):
        os.mkdir(constraint_dir)

    histogram_dir = os.path.join(constraint_dir, "histograms")
    if not os.path.exists(histogram_dir):
        os.mkdir(histogram_dir)

    if constraint_filters is None:
        constraint_filters = []

    # Add default constraint filters
    constraint_filters += [
        # Filter out CB distances and angles on pairs with a glycine
        cb_glycine_filter,
        # Filter out rotamer constraints for inappropriate residues
        rotamer_filter,
        # Filter out pairs predicted to be in last distance bin
        no_max_distance_filter,
        # Filter out constraints with greater sequence separation than max
        lambda rp, _: abs(rp.residue_1.index - rp.residue_2.index) <=
        max_separation,
    ]

    if not res_range == None:
        assert len(res_range) == 2
        constraint_filters.append(
            # Filter out pairs without residue in given range
            lambda rp, _: (res_range[0] <= rp.residue_1.index - 1 <= res_range[
                1] or res_range[0] <= rp.residue_2.index - 1 <= res_range[1]))
    if local:
        constraint_filters.append(local_interaction_filter)
    if heavy_only and not heavy_seq_len == None:
        constraint_filters.append(
            # Filter out light chain pairs
            lambda rp, _: (rp.residue_1.index - 1 < heavy_seq_len and rp.
                           residue_2.index - 1 < heavy_seq_len))
    if light_only and not heavy_seq_len == None:
        constraint_filters.append(
            # Filter out light chain pairs
            lambda rp, _: (rp.residue_2.index - 1 >= heavy_seq_len and rp.
                           residue_1.index - 1 >= heavy_seq_len))
    if interchain and not heavy_seq_len == None:
        constraint_filters.append(
            # Filter out intra-chain pairs
            lambda rp, _: (rp.residue_1.index - 1 <= heavy_seq_len and rp.
                           residue_2.index - 1 > heavy_seq_len) or
            (rp.residue_2.index - 1 <= heavy_seq_len and rp.residue_1.index - 1
             > heavy_seq_len))
    if not constraint_types == None:
        constraint_filters.append(
            lambda _, c: c.constraint_type in constraint_types)

    constraints = []
    for residue_pair in residue_pairs:
        constraints += residue_pair.get_constraints(
            custom_filters=constraint_filters)

    constraints = [c for c in constraints if c.modal_y >= threshold]

    constraint_file = os.path.join(constraint_dir, "constraints.cst")
    with open(constraint_file, "w") as f:
        for c in constraints:
            f.write(constraint_type_generator_dict[c.constraint_type](
                c, histogram_dir, prob_to_energy=prob_to_energy))

    return constraint_file
