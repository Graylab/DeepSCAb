import enum


class ConstraintType(enum.Enum):
    cbca_distance = 1
    ca_distance = 2
    cb_distance = 3
    no_distance = 4
    omega_dihedral = 5
    theta_dihedral = 6
    phi_planar = 7
    bb_phi_dihedral = 8
    bb_psi_dihedral = 9
    chi_one_dihedral = 10
    chi_two_dihedral = 11
    chi_three_dihedral = 12
    chi_four_dihedral = 13
    chi_five_dihedral = 14
