import pinocchio as pin
import numpy as np
import numpy.typing as npt
import abc

from agimus_controller.trajectories.trajectory_base import TrajectoryBase
from agimus_controller.trajectory import (
    TrajectoryPoint,
    TrajectoryPointWeights,
    WeightedTrajectoryPoint,
)


class TrajectorySmootherBase:
    def smooth(
        self,
        trajectory: list[TrajectoryPoint],
        smoothing_indices: None | list[float],
    ) -> tuple[
        list[TrajectoryPoint],
        list[float],
    ]:
        if smoothing_indices is None:
            weights_factor = [
                1.0,
            ] * len(trajectory)
            return trajectory, weights_factor
        else:
            return self.perform_smooth(trajectory, smoothing_indices)

    @abc.abstractmethod
    def perform_smooth(
        self,
        trajectory: list[TrajectoryPoint],
        smoothing_indices: list[float],
    ) -> tuple[
        list[TrajectoryPoint],
        list[float],
    ]:
        raise NotImplementedError


class _NoSmoothing(TrajectorySmootherBase):
    def perform_smooth(
        self,
        trajectory: list[TrajectoryPoint],
        smoothing_indices: list[float],
    ) -> tuple[
        list[TrajectoryPoint],
        list[float],
    ]:
        weights_factor = [
            1.0,
        ] * len(trajectory)
        return trajectory, weights_factor


class TrajectorySmoothingWindowSubsampling(TrajectorySmootherBase):
    def __init__(
        self,
        window: int,
        subsampling: int,
        weight_factor: float,
    ):
        self.window = window
        self.subsampling = subsampling
        self.weight_factor = weight_factor

    def perform_smooth(
        self,
        trajectory: list[TrajectoryPoint],
        smoothing_indices: list[float],
    ) -> tuple[
        list[TrajectoryPoint],
        list[float],
    ]:
        new_trajectory = list()
        weights_factor = list()
        N = len(trajectory)
        prev_b = 0
        for i in smoothing_indices:
            # a and b are the start (included) and end (excluded) of the interval of
            # indices
            a = max(i - self.window, 0)
            b = min(i + self.window, N)
            # Add the points that are not changed.
            new_trajectory.extend(trajectory[prev_b:a])
            weights_factor.extend([1.0] * (a - prev_b))
            # Smoothen between a and b
            shortcut = trajectory[a : self.subsampling : b]
            new_trajectory.extend(shortcut)
            weights_factor.extend([self.weight_factor] * len(shortcut))
            prev_b = b

        # Add the points that are not changed.
        new_trajectory.extend(trajectory[b:])
        weights_factor.extend([1.0] * (N - b))
        return new_trajectory, weights_factor


class GenericTrajectory(TrajectoryBase):
    """Trajectory class that awaits for trajectory inputs by the user."""

    def __init__(
        self,
        ee_frame_name,
        w_q,
        w_qdot,
        w_qddot,
        w_robot_effort,
        w_pose,
    ):
        super().__init__(ee_frame_name)
        self.trajectory: None | list(TrajectoryPoint) = None
        self.weights_factor: None | list(float) = None
        self.traj_idx = 0
        self.w_q = w_q
        self.w_qdot = w_qdot
        self.w_qddot = w_qddot
        self.w_robot_effort = w_robot_effort
        self.w_pose = w_pose
        self._smoother: TrajectorySmootherBase = _NoSmoothing()

    def set_smoothing_trajectory(self, smoother: TrajectorySmootherBase):
        self._smoother = smoother

    def build_trajectory_from_q_dq_ddq_arrays(
        self,
        q_array: list[npt.NDArray[np.float64]],
        dq_array: list[npt.NDArray[np.float64]],
        ddq_array: list[npt.NDArray[np.float64]],
    ) -> list[TrajectoryPoint]:
        """Builds list of Trajectory points based on given trajectory of q,dq and ddq."""
        assert len(q_array) == len(dq_array) and len(q_array) == len(ddq_array)
        length = len(q_array)
        trajectory = []
        for idx in range(length):
            robot_effort = pin.rnea(
                self.pin_model,
                self.pin_data,
                q_array[idx],
                dq_array[idx],
                ddq_array[idx],
            )
            pin.forwardKinematics(self.pin_model, self.pin_data, self.q)
            pin.updateFramePlacement(self.pin_model, self.pin_data, self.ee_frame_id)
            ee_pose = pin.SE3ToXYZQUAT(self.pin_data.oMf[self.ee_frame_id])
            trajectory.append(
                TrajectoryPoint(
                    robot_configuration=q_array[idx],
                    robot_velocity=dq_array[idx],
                    robot_acceleration=ddq_array[idx],
                    robot_effort=robot_effort,
                    end_effector_poses={self.ee_frame_name: ee_pose},
                )
            )
        return trajectory

    def add_trajectory(
        self,
        trajectory: list[TrajectoryPoint],
        smoothen_indices: None | list[float],
    ) -> None:
        """Initialize the trajectory if it wasn't, otherwise extend the trajectory."""
        self.trajectory_is_done = False
        trajectory, weights_factor = self._smoother.smooth(trajectory, smoothen_indices)

        if self.trajectory is None:
            self.trajectory = list(trajectory)
            self.weights_factor = weights_factor
        else:
            self.trajectory.extend(list(trajectory))
            self.weights_factor.extend(weights_factor)

    def get_traj_point_at_t(self, t: np.float64) -> WeightedTrajectoryPoint:
        traj_point = self.trajectory[self.traj_idx]
        self.trajectory_is_done = self.traj_idx == len(self.trajectory) - 1
        self.traj_idx = min(self.traj_idx + 1, len(self.trajectory) - 1)
        traj_weights = TrajectoryPointWeights(
            w_robot_configuration=self.w_q,
            w_robot_velocity=self.w_qdot,
            w_robot_acceleration=self.w_qddot,
            w_robot_effort=self.w_robot_effort,
            w_end_effector_poses={self.ee_frame_name: self.w_pose},
        )
        return WeightedTrajectoryPoint(point=traj_point, weights=traj_weights)
