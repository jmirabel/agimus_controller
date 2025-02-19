import time
import typing as T

import crocoddyl
import numpy as np
import pinocchio

from agimus_controller.factory.robot_model import RobotModels
from agimus_controller.ocp.ocp_croco_generic import OCPCrocoGeneric
from agimus_controller.ocp.ocp_croco_goal_reaching import OCPCrocoGoalReaching
from agimus_controller.ocp_param_base import OCPParamsBaseCroco
from agimus_controller.warm_start_shift_previous_solution import (
    WarmStartShiftPreviousSolution,
)
from agimus_controller.mpc import MPC
from agimus_controller.warm_start_reference import WarmStartReference
import sys

from agimus_controller.trajectory import (
    TrajectoryBuffer,
    TrajectoryPoint,
    WeightedTrajectoryPoint,
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


log_level = 0


def logerr(msg):
    if log_level < 1:
        return
    sys.stderr.write(bcolors.FAIL)
    sys.stderr.write(msg)
    sys.stderr.write(bcolors.ENDC)
    sys.stderr.write("\n")


def logwarn(msg):
    if log_level < 2:
        return
    sys.stderr.write(bcolors.WARNING)
    sys.stderr.write(msg)
    sys.stderr.write(bcolors.ENDC)
    sys.stderr.write("\n")


def logmsg(msg):
    if log_level < 3:
        return
    sys.stderr.write(msg)
    sys.stderr.write("\n")


def create_simulation_model(rmodel: pinocchio.Model, armature: np.ndarray):
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    cost = crocoddyl.CostModelSum(state)
    diff_model = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, cost
    )
    diff_model.armature = armature
    model = crocoddyl.IntegratedActionModelEuler(diff_model, 0.0)
    return model


def disturb_model(model: pinocchio.Model, mass_noise_level: float):
    if mass_noise_level > 0:
        for j in range(1, model.njoints):
            dmass = (2 * np.random.rand() - 1) * mass_noise_level
            model.inertias[j].mass = max(
                0.1 * model.inertias[j].mass, model.inertias[j].mass + dmass
            )
            assert model.inertias[j].mass > 0
    return model


class LowLevelController:
    def __init__(
        self,
        rmodel: pinocchio.Model,
        velocity_gain: float = 1000.0,
        position_gain: float = 1000.0,
    ):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.velocity_gain = velocity_gain
        self.position_gain = position_gain

    def __call__(self, x: np.ndarray, xdes: np.ndarray, u: np.ndarray):
        nq = self.rmodel.nq
        M = pinocchio.crba(self.rmodel, self.rdata, x[:nq])
        dv = xdes[nq:] - x[nq:]
        dq = xdes[:nq] - x[:nq]
        Gv = self.velocity_gain  # * np.exp(-np.linalg.norm(dv))
        Gq = self.position_gain  # * np.exp(-np.linalg.norm(dq))
        da = Gv * dv + Gq * dq
        return u + M @ da


class Simulation:
    def __init__(
        self,
        ocp_model: crocoddyl.ActionModelAbstract,
        sim_model: crocoddyl.ActionModelAbstract,
        dt: float,
        x0: np.ndarray,
        nsteps: int = 1,
        u_noise_level: float = 0.0,
    ):
        self._ocp_model = ocp_model.copy()
        self._ocp_data = ocp_model.createData()
        self._sim_model = sim_model.copy()
        self._sim_data = sim_model.createData()
        self._dt = dt
        self._dts = [dt / nsteps] * nsteps
        self._u_noise_level = u_noise_level

        self._x = x0.copy()
        self._x_expected = x0.copy()

        self._low_level_controller = self._default_controller

    def _default_controller(self, x, xdes, u):
        return u

    def set_low_level_controller(
        self,
        func: T.Callable[[np.ndarray, np.ndarray, T.Optional[np.ndarray]], np.ndarray],
    ):
        """The low level controller is a function that takes as parameters:
        - current state (q,v)
        - desired torque
        - time step
        - expected current state
        """
        self._low_level_controller = func

    def integrate_torque(
        self, u: np.ndarray, xs_expected: T.Optional[T.List[np.ndarray]] = None
    ):
        if xs_expected is None:
            xs_expected = [None] * (len(self._dts) + 1)
        assert len(xs_expected) == len(self._dts) + 1
        for dt, xdes in zip(self._dts, xs_expected):
            self._sim_model.dt = dt
            u_corrected = self._low_level_controller(self._x, xdes, u)
            if self._u_noise_level > 0:
                u_corrected += self._u_noise_level * (
                    2 * np.random.random(u_corrected.shape) - 1
                )
            self._sim_model.calc(self._sim_data, self._x, u_corrected)
            self._x_expected = xdes
            self._x = self._sim_data.xnext

    def interpolate(self, x0, u):
        xs = [x0.copy()]
        integration_times = np.cumsum(self._dts)
        integration_times[-1] = self._dt
        for integration_time in integration_times:
            self._ocp_model.dt = integration_time
            self._ocp_model.calc(self._ocp_data, x0, u)
            xs.append(self._ocp_data.xnext.copy())
        return xs

    @property
    def q(self) -> np.ndarray:
        return self._x[: self._sim_model.state.nq]

    @property
    def v(self) -> np.ndarray:
        return self._x[self._sim_model.state.nq :]

    @property
    def a(self) -> np.ndarray:
        return self._sim_data.differential.xout

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def x_expected(self) -> np.ndarray:
        """The position expected by the controller"""
        return self._x_expected


class SimulationNode:
    def __init__(
        self,
        robot_models: RobotModels,
        ocp_params: OCPParamsBaseCroco,
        x0: np.ndarray,
        viewer=None,
        use_warm_start_shift_prev_sol: bool = False,
        u_noise_level: float = 0.0,
        mass_noise_level: float = 0.0,
        simulate_delay: bool = True,
        n_simu_steps: int = 5,
    ):
        self.robot_models = robot_models
        self.ocp_params = ocp_params
        self.traj_buffer = TrajectoryBuffer(ocp_params.dt_factor_n_seq)

        self._viewer = viewer
        self._delay = simulate_delay

        if use_warm_start_shift_prev_sol:
            self._ws = WarmStartShiftPreviousSolution()
            self._ws.setup(self.robot_models, ocp_params)
            self._warmstart_is_initialized = False
        else:
            self._ws = WarmStartReference()
            self._ws.setup(self.robot_models.robot_model)
            self._warmstart_is_initialized = True

        self._min_buffer_size = 2 * sum(
            factor * nb
            for factor, nb in zip(
                ocp_params.dt_factor_n_seq.factors, ocp_params.dt_factor_n_seq.dts
            )
        )
        self._simulation_is_on = True
        self._simulation_callback = None

        # Ability to test with a different model in simulation
        sim_integration = create_simulation_model(
            disturb_model(robot_models.robot_model.copy(), mass_noise_level),
            robot_models.armature,
        )
        ocp_integration = create_simulation_model(
            robot_models.robot_model, robot_models.armature
        )
        self._simulation = Simulation(
            ocp_integration,
            sim_integration,
            ocp_params.dt,
            x0,
            nsteps=n_simu_steps,
            u_noise_level=u_noise_level,
        )

    def setup_mpc(self, goal_reaching: bool, yaml_file: str):
        """Creates mpc, ocp, warmstart"""
        if goal_reaching:
            ocp = OCPCrocoGoalReaching(self.robot_models, self.ocp_params)
        else:
            ocp = OCPCrocoGeneric(self.robot_models, self.ocp_params, yaml_file)
        self.mpc = MPC()
        self.mpc.setup(ocp=ocp, warm_start=self._ws, buffer=self.traj_buffer)

    def set_low_level_controller(self, position_gain, velocity_gain):
        self._simulation.set_low_level_controller(
            LowLevelController(
                self.robot_models.robot_model, position_gain, velocity_gain
            )
        )

    def append_reference_point(self, ref_point: WeightedTrajectoryPoint) -> None:
        """Fill the new point msg in the trajectory buffer."""
        self.traj_buffer.append(ref_point)

    def iteration(self, *args) -> None:
        if len(self.traj_buffer) < self._min_buffer_size:
            logwarn(
                f"Not enough point in trajectory buffer. Has {len(self.traj_buffer)}, needs at least {self._min_buffer_size}."
            )
            return
        if self.mpc is None:
            logmsg("initializing MPC controller")
            self.setup_mpc()

        # Create trajectory point from simulation
        now_ns = time.time_ns()
        x0 = self._simulation.x
        if self._delay and hasattr(self, "last_control"):
            # Estimate the start point of the OCP. The simulation below will integrate `last_control`.
            model = self._simulation._ocp_model
            data = self._simulation._ocp_data
            model.dt = self._simulation._dt
            model.calc(data, x0, self.last_control)
            x0 = data.xnext
        x0_traj_point = TrajectoryPoint(
            time_ns=now_ns,
            robot_configuration=x0[: self.robot_models.robot_model.nq],
            robot_velocity=x0[self.robot_models.robot_model.nq :],
            robot_acceleration=self._simulation.a,
        )
        if not self._warmstart_is_initialized:
            logmsg("Initialize warmstart")
            ws_ref = WarmStartReference()
            ws_ref.setup(self.robot_models.robot_model)

            reference_trajectory = self.traj_buffer.horizon
            self.mpc._ocp.set_reference_weighted_trajectory(reference_trajectory)
            reference_trajectory_points = [el.point for el in reference_trajectory]
            x0, x_init, u_init = ws_ref.generate(
                x0_traj_point, reference_trajectory_points
            )
            self.mpc._ocp.solve(x0, x_init, u_init)
            self._ws.update_previous_solution(self.mpc._ocp.ocp_results)
            self._warmstart_is_initialized = True
            return

        ocp_res = self.mpc.run(
            initial_state=x0_traj_point,
            current_time_ns=now_ns,
        )
        if ocp_res is None:
            return

        # dbg = self.mpc.mpc_debug_data
        # if not dbg.ocp.problem_solved:
        #     logerr("OCP did not converge")
        # logmsg(f"{dbg.ocp.kkt_norm:7.5f}  {dbg.ocp.nb_iter:6d}  {self.mpc._ocp._solver.merit:.4e}")

        if self._delay:
            # The control is applied with a delay of ocp_params.dt
            skip = not hasattr(self, "last_control")
            if not skip:
                u_simu = self.last_control.copy()
            self.last_control = ocp_res.feed_forward_terms[0].copy()
            if skip:
                return
        else:
            u_simu = ocp_res.feed_forward_terms[0].copy()
            self.last_control = u_simu.copy()
        xs_expected = self._simulation.interpolate(self._simulation.x, u_simu)
        self._simulation.integrate_torque(u_simu, xs_expected)

        if self._viewer is not None:
            self._viewer.display(self._simulation.q)

    def stop_simu(self):
        self._simulation_is_on = False

    def set_simulation_callback(self, cb):
        self._simulation_callback = cb

    def loop(self):
        i = 0
        while self._simulation_is_on:
            start = time.time()
            self.iteration()
            stop = time.time()
            if self._simulation_callback is not None:
                self._simulation_callback(i)
            t = stop - start
            i += 1
            time.sleep(max(0, self.ocp_params.dt - t))
