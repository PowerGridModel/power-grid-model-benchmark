# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from matplotlib import pyplot as plt


class BenchmarkPlotter:
    def __init__(self, n_steps=1000):
        self.data_pgm_ic = []
        self.data_pgm_nr = []
        self.data_tpf_tf = []
        self.n_nodes = []
        self.n_steps = n_steps

    def add(self, res, n_nodes):
        res_pgm = res["result pgm"]
        res_tpf = res["result tpf"]
        self.data_pgm_ic.append(res_pgm[0])
        self.data_pgm_nr.append(res_pgm[1])
        self.data_tpf_tf.append(res_tpf[0])
        self.n_nodes.append(n_nodes)

    def plot(self, log_scale=False):
        plt.figure(figsize=(8, 5))
        _, ax = plt.subplots()
        data_lists = [
            self.data_pgm_ic,
            self.data_pgm_nr,
            self.data_tpf_tf,
        ]
        labels = ["pgm ic", "pgm nr", "tpf"]
        styles = ["--", "--", "--", "--", "-"]
        for data_list, label, style in zip(data_lists, labels, styles):
            if log_scale:
                ax.semilogy(self.n_nodes, data_list, label=label, linestyle=style)
            else:
                ax.plot(self.n_nodes, data_list, label=label, linestyle=style)

        ax.set_title(f"PGM vs TPF {self.n_steps} steps")
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Execution Time (s)")

        ax.legend()
        # plt.savefig(f"data/benchmark_{self.n_steps}.pdf")
        plt.show()
