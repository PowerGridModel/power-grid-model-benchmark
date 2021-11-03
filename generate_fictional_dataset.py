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
    'tan0': 0.0,
    'i_n': 1e3
}
cable_param_pp = {
    "c_nf_per_km": cable_param['c1'] * 1e9,
    "r_ohm_per_km": cable_param['r1'],
    "x_ohm_per_km": cable_param['x1'],
    "c0_nf_per_km": cable_param['c0'] * 1e9,
    "r0_ohm_per_km": cable_param['r0'],
    "x0_ohm_per_km": cable_param['x0'],
    "max_i_ka": cable_param['i_n'] * 1e-3
}

# standard u rated
u_rated = 10e3


def generate_fictional_grid(
        n_feeder: int, n_node_per_feeder: int,
        cable_length_km_min: float, cable_length_km_max: float,
        load_p_w_max: float, load_p_w_min: float, pf: float,
        seed=0):
    np.random.seed(seed)

    n_node = n_feeder * n_node_per_feeder + 1
    pp_net = pp.create_empty_network(f_hz=50.0)
    pgm_dataset = {}

    # node
    # pgm
    pgm_dataset['node'] = pgm.initialize_array('input', 'node', n_node)
    pgm_dataset['node']['id'] = np.arange(n_node, dtype=np.int32)
    pgm_dataset['node']['u_rated'] = u_rated
    # pp
    for i in pgm_dataset['node']['id']:
        pp.create_bus(pp_net, vn_kv=u_rated * 1e-3, type='n', index=i)

    # line
    n_line = n_node - 1
    to_node_feeder = np.arange(1, n_node_per_feeder + 1, dtype=np.int32)
    to_node_feeder = to_node_feeder.reshape(1, -1) + np.arange(0, n_feeder).reshape(-1, 1) * n_node_per_feeder
    to_node = to_node_feeder.ravel()
    from_node_feeder = np.arange(1, n_node_per_feeder, dtype=np.int32)
    from_node_feeder = from_node_feeder.reshape(1, -1) + np.arange(0, n_feeder).reshape(-1, 1) * n_node_per_feeder
    from_node_feeder = np.concatenate((np.zeros(shape=(n_feeder, 1), dtype=np.int32), from_node_feeder), axis=1)
    from_node = from_node_feeder.ravel()
    length = np.random.uniform(low=cable_length_km_min, high=cable_length_km_max, size=n_line)
    # pgm
    pgm_dataset['line'] = pgm.initialize_array('input', 'line', n_line)
    pgm_dataset['line']['id'] = np.arange(n_node, n_node + n_line, dtype=np.int32)
    pgm_dataset['line']['from_node'] = from_node
    pgm_dataset['line']['to_node'] = to_node
    pgm_dataset['line']['from_status'] = 1
    pgm_dataset['line']['to_status'] = 1
    for attr_name, attr in cable_param.items():
        if attr_name in ['i_n', 'tan1', 'tan0']:
            pgm_dataset['line'][attr_name] = attr
        else:
            pgm_dataset['line'][attr_name] = attr * length
    # pp
    pp.create_std_type(pp_net, cable_param_pp, name="630Al", element="line")
    for seq, (idx, l) in enumerate(zip(pgm_dataset['line']['id'], length)):
        pp.create_line(pp_net, from_bus=from_node[seq], to_bus=to_node[seq], length_km=l, index=idx, std_type='630Al')
    
    return {
        'pgm_dataset': pgm_dataset,
        'pp_net': pp_net
    }

