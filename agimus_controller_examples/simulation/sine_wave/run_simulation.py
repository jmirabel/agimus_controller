#!/usr/bin/env python

from pathlib import Path

from argparse import ArgumentParser
import numpy as np
from agimus_controller.trajectory import (
    TrajectoryPoint,
    WeightedTrajectoryPoint,
    TrajectoryPointWeights,
)
from agimus_controller.simulation import SimulationNode, Plotter
from agimus_controller.factory.robot_model import RobotModels, RobotModelParameters
from agimus_controller.ocp_param_base import OCPParamsBaseCroco, DTFactorsNSeq
from ament_index_python import get_package_share_directory
import xacro
import pinocchio
from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer

np.set_printoptions(linewidth=200)


def create_robot_model_parameters(moving_joint_names, armature):
    franka_description_path = Path(get_package_share_directory("franka_description"))
    srdf_path = franka_description_path / "robots" / "fer" / "fer.srdf"
    xacro_path = str(
        franka_description_path / "robots" / "fer" / "fer.urdf.xacro",
    )
    urdf_xml = xacro.process_file(
        xacro_path,
        mappings={"with_sc": "true"},
    ).toxml()
    env_xml = """
    <robot name="box">
    <material name="grey">
        <color rgba=".5 .5 .5 1"/>
    </material>
    <link name="base_link"/>
    <joint name="obstacle_joint" type="fixed">
        <parent link="base_link"/>
        <child link="obstacle"/>
        <origin xyz="0.6 0.15 0.8" rpy="0 0 0" />
    </joint>
    <link name="obstacle">
        <inertial>
            <mass value="0.6"/>
            <inertia ixx="0.001" ixy="0.0" ixz="0.0"
                iyy="0.001" iyz="0.0"
                izz="0.001" />
        </inertial>
        <visual>
            <geometry>
                <sphere radius="0.05"/>
            </geometry>
            <color name="grey"/>
        </visual>
        <collision>
            <geometry>
                <sphere radius="0.05"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    # Hack for the moving joint name
    model = pinocchio.buildModelFromXML(urdf_xml)
    robot_model_params = RobotModelParameters(
        q0=np.zeros(model.nq),
        free_flyer=False,
        moving_joint_names=moving_joint_names,
        robot_urdf=urdf_xml,
        env_urdf=env_xml,
        srdf=srdf_path,
        collision_as_capsule=True,
        urdf_meshes_dir=franka_description_path,
        armature=np.array(armature),
        collision_pairs=[("fer_hand_sc_capsule_0", "obstacle_0")],
    )
    return robot_model_params


class TargetGeneration:
    def __init__(self, robot_models, ocp_params, ee_frame_name):
        self._ocp_params = ocp_params
        self._rmodel = robot_models.robot_model
        self._rdata = self._rmodel.createData()
        self._ee_name = ee_frame_name
        self._ee_id = self._rmodel.getFrameId(ee_frame_name)
        self._scale_duration = 0.2

        self._x0 = np.concatenate(
            (
                # pinocchio.randomConfiguration(self._rmodel),
                # pinocchio.neutral(self._rmodel),
                # np.zeros(self._rmodel.nq),
                [0, 0, 0, -np.pi / 2, 0, np.pi, 0],
                np.zeros(self._rmodel.nv),
            )
        )
        self._amp = 0.1 * np.ones(self._rmodel.nv)
        self._w = np.pi / 2

    @property
    def x0(self):
        return self._x0

    def generate(self, t):
        assert t >= 0
        if t < self._scale_duration:
            # scale the amplitude from 0 to full in such a way that the velocity and acceleration are zero at start.
            s = t / self._scale_duration
            scale_amp = 10 * s**3 - 15 * s**4 + 6 * s**5
            amp = scale_amp * self._amp
        else:
            amp = self._amp

        q = self._x0[: self._rmodel.nq] + amp * np.sin(self._w * t)
        v = amp * self._w * np.cos(self._w * t)
        a = -amp * self._w * self._w * np.sin(self._w * t)
        return q, v, a

    def set_simulation(self, simulation):
        self._simulation = simulation

    def simulation_callback(self, iteration: int):
        if False and hasattr(self._simulation, "last_control"):
            print(self._simulation.last_control)
            print(self._simulation.mpc._ocp._ocp_results.states[1])
            input("press enter to continue")
        t = iteration * self._ocp_params.dt
        q, v, a = self.generate(t)

        # Extract the end-effector position and orientation
        pinocchio.forwardKinematics(self._rmodel, self._rdata, q)
        pinocchio.updateFramePlacement(self._rmodel, self._rdata, self._ee_id)

        ee_pose = self._rdata.oMf[self._ee_id]

        u = pinocchio.computeGeneralizedGravity(self._rmodel, self._rdata, q)

        nv = self._rmodel.nv
        ref_point = WeightedTrajectoryPoint(
            point=TrajectoryPoint(
                time_ns=int(t * 1e-9),
                robot_configuration=q,
                robot_velocity=v,
                robot_acceleration=a,
                robot_effort=u,
                end_effector_poses={self._ee_name: ee_pose},
            ),
            weights=TrajectoryPointWeights(
                w_robot_configuration=[10.0] * np.ones(nv),
                w_robot_velocity=[0e-2] * np.ones(nv),
                w_robot_acceleration=[1e-6] * np.ones(nv),
                w_robot_effort=[1e-4] * np.ones(nv),
                w_end_effector_poses={self._ee_name: [1e-10] * np.ones(6)},
            ),
        )
        # print(ref_point.point.robot_configuration)
        # print(ref_point.point.robot_velocity)
        # print(ref_point.point.robot_acceleration)
        self._simulation.append_reference_point(ref_point)


parser = ArgumentParser()
parser.add_argument(
    "--u-noise-level",
    default=0.0,
    type=float,
    help="Noise level added to the applied control.",
)
parser.add_argument(
    "--mass-noise-level",
    default=0.0,
    type=float,
    help="Noise level added to the robot link masses.",
)
parser.add_argument(
    "--ocp",
    default="",
    type=str,
    help="If not empty, it should be a path to a yaml file describing the OCP to be used. Otherwise, the goal reaching OCP is used.",
)
parser.add_argument(
    "--ws-reference",
    action="store_true",
    help="This flag enables the use of WarmStartReference",
)
parser.add_argument(
    "--use-low-level-controller",
    action="store_true",
    help="This flag enables the use of the low level controller",
)
args = parser.parse_args()
robot_models = RobotModels(
    create_robot_model_parameters(
        [
            "fer_joint1",
            "fer_joint2",
            "fer_joint3",
            "fer_joint4",
            "fer_joint5",
            "fer_joint6",
            "fer_joint7",
        ],
        1e-2 * np.ones(7),
    )
)
for cp in robot_models.collision_model.collisionPairs:
    print(
        [
            robot_models.collision_model.geometryObjects[i].name
            for i in (cp.first, cp.second)
        ]
    )

ocp_params = OCPParamsBaseCroco(
    dt=0.01,
    solver_iters=7,
    nb_threads=6,
    # dt_factor_n_seq=DTFactorsNSeq([1], [19]),
    # dt_factor_n_seq=DTFactorsNSeq([1, 2, 4, 6], [5, 5, 5, 4]),
    # horizon_size=19,
    dt_factor_n_seq=DTFactorsNSeq([1, 2, 4, 6], [5, 8, 8, 9]),
    horizon_size=30,
    callbacks=False,
)
viewer = MeshcatVisualizer(
    robot_models.robot_model, robot_models.collision_model, robot_models.visual_model
)
viewer.initViewer(zmq_url="tcp://127.0.0.1:6000")
viewer.loadViewerModel()
viewer.displayCollisions(True)
print("Horizon total time:", sum(ocp_params.timesteps))
input("Press enter to continue...")

target = TargetGeneration(robot_models, ocp_params, "fer_link8")
simulation = SimulationNode(
    robot_models,
    ocp_params,
    target.x0,
    viewer,
    use_warm_start_shift_prev_sol=not args.ws_reference,
    u_noise_level=args.u_noise_level,
    mass_noise_level=args.mass_noise_level,
    n_simu_steps=10,
)
if args.use_low_level_controller:
    simulation.set_low_level_controller(1000, 1000)
plotter = Plotter(simulation, ocp_params, n_iter=100)
target.set_simulation(simulation)
simulation.setup_mpc(
    goal_reaching=args.ocp == "",
    yaml_file=args.ocp,
)


def callback(iter):
    target.simulation_callback(iter)
    plotter.simulation_callback(iter)


simulation.set_simulation_callback(callback)

simulation.loop()
