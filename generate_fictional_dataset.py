import numpy as np
import pandas as pd
import pandapower as pp
import power_grid_model as pgm


# cable parameter per km
# 630Al XLPE 10 kV with neutral conductor
cable_param = {
    'r1': 0.063,
    'x1': 0.103,
    'c1': 0.4e-6,
    'r0': 0.156,
    'x0': 0.1,
    'c0': 0.66e-6,
    'tan1': 0.0,
    'tan0': 0.0
}

# standard u rated
u_rated = 10e3


def generate_fictional_grid(
    n_feeder: int, n_node_per_feeder: int,
    cable_length_km_min: float, cable_length_km_max: float,
    load_p_w_max: float, load_p_w_min: float,
    pf: float):
    n_node = n_feeder * n_node_per_feeder + 1

    pp_net = pp.create_empty_network(f_hz=50.0)
    pgm_dataset = {}

    # node
    pgm_dataset['node'] = pgm.initialize_array('input', 'node', n_node)
    pgm_dataset['node']['id'] = np.arange(n_node, dtype=np.int32)
    pgm_dataset['node']['u_rated'] = u_rated

    return {
        'pgm_dataset': pgm_dataset,
        'pp_net': pp_net
    }

