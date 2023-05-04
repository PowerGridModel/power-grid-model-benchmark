# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd
import pandapower as pp
import power_grid_model as pgm
from pandapower.timeseries.data_sources.frame_data import DFData
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR / "data"
TEMPLATE_PATH = FILE_DIR / "dss_template.dss"

JINJA_ENV = Environment(loader=FileSystemLoader(FILE_DIR))

# standard u rated
u_rated = 10e3
frequency = 50.0

# source
source_sk = 1e20
source_rx = 0.1
source_01 = 1.0
source_u_ref = 1.05
source_node = 0

# opendss monitor
opendss_monitor_mode = 32  # 0 for voltage, +32 for magnitude only

# cable parameter per km
# 630Al XLPE 10 kV with neutral conductor
cable_type = "630Al"
cable_param = {
    "r1": 0.063,
    "x1": 0.103,
    "c1": 0.4e-6,
    "r0": 0.156,
    "x0": 0.1,
    "c0": 0.66e-6,
    "tan1": 0.0,
    "tan0": 0.0,
    "i_n": 1e3,
}
cable_param_pp = {
    "c_nf_per_km": cable_param["c1"] * 1e9,
    "r_ohm_per_km": cable_param["r1"],
    "x_ohm_per_km": cable_param["x1"],
    "g_us_per_km": cable_param["tan1"] * cable_param["c1"] * 2 * np.pi * frequency * 1e6,
    "c0_nf_per_km": cable_param["c0"] * 1e9,
    "r0_ohm_per_km": cable_param["r0"],
    "x0_ohm_per_km": cable_param["x0"],
    "g0_us_per_km": cable_param["tan0"] * cable_param["c0"] * 2 * np.pi * frequency * 1e6,
    "max_i_ka": cable_param["i_n"] * 1e-3,
}


def generate_fictional_grid(
    n_feeder: int,
    n_node_per_feeder: int,
    cable_length_km_min: float,
    cable_length_km_max: float,
    load_p_w_max: float,
    load_p_w_min: float,
    pf: float,
    n_step: int,
    load_scaling_min: float,
    load_scaling_max: float,
    seed=0,
):
    dss_dict = {"frequency": frequency, "basekv": u_rated * 1e-3}

    DATA_DIR.mkdir(exist_ok=True)

    np.random.seed(seed)

    n_node = n_feeder * n_node_per_feeder + 1
    pp_net = pp.create_empty_network(f_hz=frequency)
    pgm_dataset = dict()

    # node
    # pgm
    pgm_dataset["node"] = pgm.initialize_array("input", "node", n_node)
    pgm_dataset["node"]["id"] = np.arange(n_node, dtype=np.int32)
    pgm_dataset["node"]["u_rated"] = u_rated
    # pp
    pp.create_buses(
        pp_net, nr_buses=n_node, vn_kv=pgm_dataset["node"]["u_rated"] * 1e-3, index=pgm_dataset["node"]["id"]
    )

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
    pgm_dataset["line"] = pgm.initialize_array("input", "line", n_line)
    pgm_dataset["line"]["id"] = np.arange(n_node, n_node + n_line, dtype=np.int32)
    pgm_dataset["line"]["from_node"] = from_node
    pgm_dataset["line"]["to_node"] = to_node
    pgm_dataset["line"]["from_status"] = 1
    pgm_dataset["line"]["to_status"] = 1
    for attr_name, attr in cable_param.items():
        if attr_name in ["i_n", "tan1", "tan0"]:
            pgm_dataset["line"][attr_name] = attr
        else:
            pgm_dataset["line"][attr_name] = attr * length
    # pp
    pp.create_std_type(pp_net, cable_param_pp, name="630Al", element="line")
    pp.create_lines_from_parameters(
        pp_net,
        from_buses=from_node,
        to_buses=to_node,
        length_km=length,
        index=pgm_dataset["line"]["id"] - n_node,
        **cable_param_pp,
    )
    # dss
    dss_dict["Line"] = {
        line["id"]: {
            "Phases": 3,
            "Bus1": line["from_node"],
            "Bus2": line["to_node"],
            "LineCode": cable_type,
            "Length": line_length,
            "Units": "km",
        }
        for line, line_length in zip(pgm_dataset["line"], length)
    }

    # add asym load
    n_load = n_node - 1
    # pgm
    pgm_dataset["asym_load"] = pgm.initialize_array("input", "asym_load", n_load)
    pgm_dataset["asym_load"]["id"] = np.arange(n_node + n_line, n_node + n_line + n_load, dtype=np.int32)
    pgm_dataset["asym_load"]["node"] = pgm_dataset["node"]["id"][1:]
    pgm_dataset["asym_load"]["status"] = 1
    pgm_dataset["asym_load"]["type"] = pgm.LoadGenType.const_power
    pgm_dataset["asym_load"]["p_specified"] = np.random.uniform(
        low=load_p_w_min / 3.0, high=load_p_w_max / 3.0, size=(n_load, 3)
    )
    pgm_dataset["asym_load"]["q_specified"] = pgm_dataset["asym_load"]["p_specified"] * np.sqrt(1 - pf**2) / pf
    # pp
    asym_load_df = pd.DataFrame(
        {
            "name": [None] * n_load,
            "bus": pgm_dataset["asym_load"]["node"],
            "p_a_mw": pgm_dataset["asym_load"]["p_specified"][:, 0] * 1e-6,
            "p_b_mw": pgm_dataset["asym_load"]["p_specified"][:, 1] * 1e-6,
            "p_c_mw": pgm_dataset["asym_load"]["p_specified"][:, 2] * 1e-6,
            "q_a_mvar": pgm_dataset["asym_load"]["q_specified"][:, 0] * 1e-6,
            "q_b_mvar": pgm_dataset["asym_load"]["q_specified"][:, 1] * 1e-6,
            "q_c_mvar": pgm_dataset["asym_load"]["q_specified"][:, 2] * 1e-6,
            "sn_mva": np.full(shape=(n_load,), fill_value=np.nan, dtype=np.float64),
            "scaling": np.full(shape=(n_load,), fill_value=1.0, dtype=np.float64),
            "in_service": np.full(shape=(n_load,), fill_value=True, dtype=np.bool_),
            "type": np.full(shape=(n_load,), fill_value="wye", dtype=np.object_),
        },
        index=pgm_dataset["asym_load"]["id"] - n_line - n_node,
    )
    pp_net.asymmetric_load = asym_load_df
    # dss
    dss_dict["Load"] = {
        f"{load['id']}_{phase + 1}": {
            "Bus1": f"{load['node']}.{phase + 1}",
            "Phases": 1,
            "Conn": "wye",
            "Model": 1,
            "Kv": u_rated / np.sqrt(3) * 1e-3,
            "Kw": load["p_specified"][phase] * 1e-3,
            "kvar": load["q_specified"][phase] * 1e-3,
            "Vmaxpu": 2.0,
            "Vminpu": 0.1,
            "Daily": f"LoadShape_{load['id']}_{phase + 1}",
        }
        for load in pgm_dataset["asym_load"]
        for phase in range(3)
    }

    # source
    # pgm
    source_id = n_node + n_line + n_load
    pgm_dataset["source"] = pgm.initialize_array("input", "source", 1)
    pgm_dataset["source"]["id"] = source_id
    pgm_dataset["source"]["node"] = source_node
    pgm_dataset["source"]["status"] = 1
    pgm_dataset["source"]["u_ref"] = source_u_ref
    pgm_dataset["source"]["sk"] = source_sk
    pgm_dataset["source"]["rx_ratio"] = source_rx
    pgm_dataset["source"]["z01_ratio"] = source_01
    # pp
    pp.create_ext_grid(
        pp_net,
        bus=0,
        vm_pu=source_u_ref,
        va_degree=0.0,
        index=0,
        s_sc_max_mva=source_sk * 1e-6,
        rx_max=source_rx,
        r0x0_max=source_rx,
        x0x_max=source_01,
    )
    # dss
    dss_dict["source_name"] = source_id
    dss_dict["source_dict"] = {
        "Bus1": source_node,
        "basekv": u_rated * 1e-3,
        "pu": source_u_ref,
        "MVAsc3": source_sk * 1e-6,
        "MVAsc1": source_sk * 1e-6 * 3.0 / (2.0 + source_01),
        "x1r1": 1.0 / source_rx,
        "x0r0": 1.0 / source_rx,
    }

    # generate time series
    np.random.seed(seed)

    # pgm
    n_load = pgm_dataset["asym_load"].size
    scaling = np.random.uniform(low=load_scaling_min, high=load_scaling_max, size=(n_step, n_load, 3))
    asym_load_profile = pgm.initialize_array("update", "asym_load", (n_step, n_load))
    asym_load_profile["id"] = pgm_dataset["asym_load"]["id"].reshape(1, -1)
    asym_load_profile["p_specified"] = pgm_dataset["asym_load"]["p_specified"].reshape(1, -1, 3) * scaling
    asym_load_profile["q_specified"] = pgm_dataset["asym_load"]["q_specified"].reshape(1, -1, 3) * scaling

    # pp
    pp_dataset = {}
    for x, y in zip(["p", "q"], ["mw", "mvar"]):
        for i, p in enumerate(["a", "b", "c"]):
            name = f"{x}_{p}_{y}"
            pp_dataset[name] = DFData(
                pd.DataFrame(
                    asym_load_profile[f"{x}_specified"][..., i] * 1e-6,
                    index=np.arange(n_step),
                    columns=pp_net.asymmetric_load.index,
                )
            )

    # dss
    dss_dict["LoadShape"] = {
        f"LoadShape_{load['id']}_{phase + 1}": {
            "Npts": n_step,
            "Interval": 1.0,
            "Mult": "(" + " ".join(map(str, scale[:, phase])) + ")",
        }
        for load, scale in zip(pgm_dataset["asym_load"], np.transpose(scaling, (1, 0, 2)))
        for phase in range(3)
    }
    # monitor
    dss_dict["Monitor"] = {
        f"VSensor_{source_node}": {"Element": "Vsource.Source", "Terminal": 1, "Mode": opendss_monitor_mode}
    }
    for load in pgm_dataset["asym_load"]:
        for phase in range(3):
            dss_dict["Monitor"][f"VSensor_{load['node']}_{phase + 1}"] = {
                "Element": f"Load.{load['id']}_{phase + 1}",
                "Terminal": 1,
                "Mode": opendss_monitor_mode,
            }

    # generate dss
    # jinja expects a string, representing a relative path with forward slashes
    template_path_str = str(TEMPLATE_PATH.relative_to(FILE_DIR)).replace("\\", "/")
    template = JINJA_ENV.get_template(template_path_str)
    output = template.render(dss_dict)
    output_path = DATA_DIR / "fictional_grid.dss"

    with (output_path).open(mode="w", encoding="utf-8") as output_file:
        output_file.write(output)

    # return values
    return {
        "pgm_dataset": pgm_dataset,
        "pp_net": pp_net,
        "pgm_update_dataset": {"asym_load": asym_load_profile},
        "pp_time_series_dataset": pp_dataset,
        "dss_file": output_path,
    }
