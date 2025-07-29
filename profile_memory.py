import copy
import time
import psutil
import warnings

from memory_profiler import profile

from power_grid_model import PowerGridModel
from power_grid_model.data_types import SingleDataset, BatchDataset

from generate_fictional_dataset import generate_fictional_grid


warnings.filterwarnings("ignore")


START_TIME = time.time()


def print_time():
    current_time = time.time()
    elapsed_time = current_time - START_TIME
    return f"timestamp = {current_time} ({elapsed_time:.2f} seconds elapsed since start of benchmark)"


# Fictional grid parameters
class FictionalGridParams:
    # fictional grid parameters

    n_node_per_feeder = 10
    n_feeder = 100

    cable_length_km_min = 0.8
    cable_length_km_max = 1.2
    load_p_w_max = 0.4e6 * 0.8
    load_p_w_min = 0.4e6 * 1.2
    pf = 0.95

    load_scaling_min = 0.5
    load_scaling_max = 1.5
    n_step = 1000

    # benchmark parameters
    use_lightsim2grid = True
    # when running a single scenario, repeat this many times to get a good estimate of the average time. NOTE: not needed for update data, because it already contains many scenarios.
    n_single_scenario_repeats = 5

    def __init__(self):
        # derived values
        self.n_node = self.n_node_per_feeder * self.n_feeder + 1
        self.n_line = self.n_node_per_feeder * self.n_feeder
        self.n_load = self.n_node_per_feeder * self.n_feeder


class SmallFictionalGridParams(FictionalGridParams):
    n_node_per_feeder = 3
    n_feeder = 2
    n_step = 10


def fictional_grid(params: FictionalGridParams):
    fictional_dataset = generate_fictional_grid(
        n_node_per_feeder=params.n_node_per_feeder,
        n_feeder=params.n_feeder,
        cable_length_km_min=params.cable_length_km_min,
        cable_length_km_max=params.cable_length_km_max,
        load_p_w_max=params.load_p_w_max,
        load_p_w_min=params.load_p_w_min,
        pf=params.pf,
        n_step=params.n_step,
        load_scaling_min=params.load_scaling_min,
        load_scaling_max=params.load_scaling_max,
    )
    input_data = copy.deepcopy(fictional_dataset["pgm_dataset"])
    update_data = copy.deepcopy(fictional_dataset["pgm_update_dataset"])

    del fictional_dataset

    return input_data, update_data


def input_fictional_grid(params: FictionalGridParams):
    input_data, update_data = fictional_grid(params)
    del update_data
    return input_data


def create_model(*, input_data: SingleDataset):
    print(f"Create model: {print_time()}")
    print(f"\tPowerGridModel: {print_time()}")
    model = PowerGridModel(input_data=input_data)
    print(f"\tdel model: {print_time()}")
    del model
    print(f"\tdel input_data: {print_time()}")
    del input_data
    print(f"\tDone: {print_time()}")


def single_calculation(*, input_data: SingleDataset, symmetric: bool = True):
    print(f"Single calculation (symmetric={symmetric}): {print_time()}")
    print(f"\tPowerGridModel: {print_time()}")
    model = PowerGridModel(input_data=input_data)
    print(f"\tmodel.calculate_power_flow: {print_time()}")
    model.calculate_power_flow(symmetric=symmetric)
    print(f"\tdel model: {print_time()}")
    del model
    print(f"\tdel input_data: {print_time()}")
    del input_data
    print(f"\tDone: {print_time()}")


def batch_calculation(*, input_data: SingleDataset, update_data: BatchDataset, symmetric: bool = True):
    print(f"Batch calculation (symmetric={symmetric}): {print_time()}")
    print(f"\tPowerGridModel: {print_time()}")
    model = PowerGridModel(input_data=input_data)
    print(f"\tmodel.calculate_power_flow: {print_time()}")
    model.calculate_power_flow(update_data=update_data, symmetric=symmetric)
    print(f"\tdel model: {print_time()}")
    del model
    print(f"\tdel input_data: {print_time()}")
    del input_data
    print(f"\tdel update_data: {print_time()}")
    del update_data
    print(f"\tDone: {print_time()}")


def main(params: FictionalGridParams):
    input_data, update_data = fictional_grid(params)
    create_model(input_data=input_data)
    single_calculation(input_data=input_data, symmetric=True)
    single_calculation(input_data=input_data, symmetric=False)
    batch_calculation(input_data=input_data, update_data=update_data, symmetric=True)
    batch_calculation(input_data=input_data, update_data=update_data, symmetric=False)


if __name__ == "__main__":
    try:
        # print(f"Running small fictional grid benchmark: {print_time()}")
        # main(SmallFictionalGridParams())
        print(f"Running large fictional grid benchmark: {print_time()}")
        main(FictionalGridParams())
    except psutil.NoSuchProcess:
        pass
