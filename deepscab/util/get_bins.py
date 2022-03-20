import math


def get_dist_bins(num_bins, interval=0.5):
    bins = [(interval * i, interval * (i + 1)) for i in range(num_bins - 1)]
    bins.append((bins[-1][1], float('Inf')))
    return bins


def get_dihedral_bins(num_bins, rad=False):
    first_bin = -180
    bin_width = 2 * 180 / num_bins
    bins = [(first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)]

    if rad:
        bins = deg_bins_to_rad(bins)

    return bins


def get_planar_bins(num_bins, rad=False):
    first_bin = 0
    bin_width = 180 / num_bins
    bins = [(first_bin + bin_width * i, first_bin + bin_width * (i + 1))
            for i in range(num_bins)]

    if rad:
        bins = deg_bins_to_rad(bins)

    return bins


def get_rotamer_bins(num_bins, bin_size=6, rad=False):
    gauch_pos = []
    for i in range(-5, 5):
        gauch_pos.append((60 + i * bin_size, 60 + (i + 1) * bin_size))
    gauch_neg = []
    for i in range(-5, 5):
        gauch_neg.append((-60 + i * bin_size, -60 + (i + 1) * bin_size))
    trans_neg = []
    for i in range(5):
        trans_neg.append((-180 + i * bin_size, -180 + (i + 1) * bin_size))
    trans_pos = []
    for i in range(5):
        trans_pos.append((180 - (i + 1) * bin_size, 180 - i * bin_size))
    trans_pos = list(reversed(trans_pos))
    connect_trans_gauche_neg = [(trans_neg[-1][1], trans_neg[-1][1] + int(
        (gauch_neg[0][0] - trans_neg[-1][1]) / 2)),
                                (trans_neg[-1][1] + int(
                                    (gauch_neg[0][0] - trans_neg[-1][1]) / 2),
                                 gauch_neg[0][0])]
    connect_gauch_gauche = [(gauch_neg[-1][1], gauch_neg[-1][1] + int(
        (gauch_pos[0][0] - gauch_neg[-1][1]) / 2)),
                            (gauch_neg[-1][1] + int(
                                (gauch_pos[0][0] - gauch_neg[-1][1]) / 2),
                             gauch_pos[0][0])]
    connect_gauche_trans_pos = [(gauch_pos[-1][1], gauch_pos[-1][1] + int(
        (trans_pos[0][0] - gauch_pos[-1][1]) / 2)),
                                (gauch_pos[-1][1] + int(
                                    (trans_pos[0][0] - gauch_pos[-1][1]) / 2),
                                 trans_pos[0][0])]
    bins = trans_neg + connect_trans_gauche_neg + gauch_neg + \
        connect_gauch_gauche + gauch_pos + connect_gauche_trans_pos + trans_pos

    if rad:
        bins = deg_bins_to_rad(bins)

    return bins


def deg_bins_to_rad(bins):
    return [(v[0] * math.pi / 180, v[1] * math.pi / 180) for v in bins]


def get_bin_values(bins):
    bin_values = [(b[1] - b[0]) / 2 + b[0] for b in bins]
    bin_values[0] = bins[0][1] - (bin_values[1] - bins[1][0])
    bin_values[-1] = bins[-1][0] + (bin_values[-2] - bins[-2][0])

    return bin_values
