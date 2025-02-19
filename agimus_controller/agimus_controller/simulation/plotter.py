import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, simulation, ocp_params, n_iter=100):
        plt.ion()

        self._simulation = simulation
        self._n_iter = n_iter
        self._fig = plt.figure()
        self._ax_kkt, self._ax_time, self._ax_pred_q = self._fig.subplots(
            3, 1, sharex=True
        )
        self._ax_kkt.set_ylabel("KKT norm")
        self._ax_kkt.set_yscale("log")
        self._ax_kkt.set_ylim(1e-8, 1e-2)
        self._ax_time.set_ylabel("resolution time")
        self._ax_time.set_ylim(0, 20)
        self._ax_pred_q.set_ylabel("norm of prediction error")
        self._ax_pred_q.set_ylim(0, 1)

        self._iterations = []
        self._kkt = []
        self._time = []
        self._pred_q = []
        (self._line_kkt,) = self._ax_kkt.plot(self._iterations, self._kkt)
        (self._line_time,) = self._ax_time.plot(self._iterations, self._time)
        (self._line_pred_q,) = self._ax_pred_q.plot(self._iterations, self._pred_q)

        self._ax_kkt.axhline(
            y=ocp_params.termination_tolerance, color="red", linestyle="--"
        )
        self._ax_time.axhline(y=ocp_params.dt * 1e3, color="red", linestyle="--")

    @property
    def _lines(self):
        yield self._line_kkt
        yield self._line_time
        yield self._line_pred_q

    @property
    def _axes(self):
        yield self._ax_kkt
        yield self._ax_time
        yield self._ax_pred_q

    @property
    def _ydata(self):
        yield self._kkt
        yield self._time
        yield self._pred_q

    def simulation_callback(self, iteration):
        if not hasattr(self._simulation, "last_control"):
            return
        dgb_data = self._simulation.mpc.mpc_debug_data

        self._iterations.append(iteration)
        self._kkt.append(dgb_data.ocp.kkt_norm)
        self._time.append(dgb_data.duration_iteration_ns * 1e-6)
        cur_state = self._simulation._simulation.x
        pred_state = (
            self._simulation._simulation.x_expected
        )  # mpc._ocp.ocp_results.states[1]
        self._pred_q.append(np.linalg.norm(cur_state - pred_state))
        if len(self._iterations) > self._n_iter:
            self._iterations.pop(0)
            for ydata in self._ydata:
                ydata.pop(0)

        if iteration % 10 == 0:
            for ax, line, ydata in zip(self._axes, self._lines, self._ydata):
                line.set_xdata(self._iterations)
                line.set_ydata(ydata)
                ax.set_xlim(iteration - self._n_iter, iteration + 1)

            plt.draw()
            plt.pause(0.001)
