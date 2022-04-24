import numpy as np

from deepscab.util.util import _aa_1_3_dict, r2_dict, r3_dict, r4_dict, r5_dict, r6_dict

from .Constraint import Constraint
from .ConstraintType import ConstraintType
from .ResiduePair import ResiduePair


def cb_glycine_filter(_: ResiduePair, constraint: Constraint):
    filters = [
        lambda _, c:
        (c.constraint_type == ConstraintType.cb_distance or
         (c.residue_1.identity != "G" and c.residue_2.identity != "G")),
        lambda _, c: not (c.constraint_type == ConstraintType.cb_distance and (
            c.residue_1.identity == "G" or c.residue_2.identity == "G"))
    ]

    result = np.all([rf(_, constraint) for rf in filters])

    return result


def no_max_distance_filter(residue_pair: ResiduePair, _: Constraint):
    """
    Filter on ResiduePair that returns false if the predicted distance
    falls in the last distance bin
    """

    if ConstraintType.cb_distance in residue_pair.constraint_types:
        cb_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.cb_distance
        ][0]

        if cb_constraint.modal_x == cb_constraint.x_vals[-1]:
            return False

    return True


def local_interaction_filter(residue_pair: ResiduePair,
                             _: Constraint,
                             local_distance: float = 12):
    """
    Filter on ResiduePair that returns false if the predicted distance
    is greater than 12 Å (default)
    """

    if ConstraintType.ca_distance in residue_pair.constraint_types:
        ca_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.ca_distance
        ][0]

        if ca_constraint.modal_x > local_distance:
            return False
    elif ConstraintType.cb_distance in residue_pair.constraint_types:
        cb_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.cb_distance
        ][0]

        if cb_constraint.modal_x > local_distance:
            return False
    elif ConstraintType.cbca_distance in residue_pair.constraint_types:
        cbca_constraint = [
            c for c in residue_pair.constraints
            if c.constraint_type == ConstraintType.cbca_distance
        ][0]

        if cbca_constraint.modal_x > local_distance:
            return False

    return True


def hb_dist_filter(_: ResiduePair, constraint: Constraint):
    """
    Filter on constraint that returns false for no_distance constraints
    with distance greater than 5 Å
    Note: 5 Å selected to provide generous cutoff for hbonds
    """

    hbond_distance = 5
    if constraint.constraint_type == ConstraintType.no_distance and constraint.modal_x < hbond_distance:
        return True

    return False


def rotamer_filter(residue_pair: ResiduePair, constraint: Constraint):
    rotamer_filters = [
        lambda rp, c: (c.constraint_type != ConstraintType.chi_one_dihedral or
                       _aa_1_3_dict[rp.residue_1.identity] in r2_dict),
        lambda rp, c: (c.constraint_type != ConstraintType.chi_two_dihedral or
                       _aa_1_3_dict[rp.residue_1.identity] in r3_dict),
        lambda rp, c: (c.constraint_type != ConstraintType.chi_three_dihedral
                       or _aa_1_3_dict[rp.residue_1.identity] in r4_dict),
        lambda rp, c: (c.constraint_type != ConstraintType.chi_four_dihedral or
                       _aa_1_3_dict[rp.residue_1.identity] in r5_dict),
        lambda rp, c: (c.constraint_type != ConstraintType.chi_five_dihedral or
                       _aa_1_3_dict[rp.residue_1.identity] in r6_dict)
    ]

    result = np.all([rf(residue_pair, constraint) for rf in rotamer_filters])

    return result