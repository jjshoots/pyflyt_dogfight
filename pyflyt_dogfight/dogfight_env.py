from os import path

import numpy as np
from gymnasium import spaces
from PyFlyt.core import Aviary

# fix numpy buggy cross
np_cross = lambda x, y: np.cross(x, y)


class DogfightEnv:
    """Base Dogfighting Environment for the Aggressor model using custom environment API."""

    def __init__(
        self,
        spawn_height: float = 15.0,
        flight_dome_size: float = 150.0,
        max_duration_seconds: float = 60.0,
        agent_hz: int = 30,
        damage_per_hit: float = 0.02,
        lethal_distance: float = 15.0,
        lethal_angle_radians: float = 0.1,
        assisted_flight: bool = True,
        render: bool = False,
        human_camera: bool = True,
    ):
        """__init__.

        Args:
            spawn_height (float): spawn_height
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            agent_hz (int): agent_hz
            damage_per_hit (float): damage_per_hit
            lethal_distance (float): lethal_distance
            lethal_angle_radians (float): how close must the nose of the aircraft be to the opponents body to be considered a hit
            assisted_flight (bool): whether to fly using RPYT controls or manual control of all actuators
            render (bool): whether to render the environment
            human_camera (bool): to allow the image from `render` to be available without the PyBullet display
        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        """SPACES"""
        high = np.ones(4 if assisted_flight else 6)
        low = high * -1.0
        low[-1] = 0.0
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        state_shape = 13  # 12 states + health
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * state_shape + self.action_space.shape[0],),
        )

        """CONSTANTS"""
        self.env: Aviary
        self.num_drones = 2
        self.to_render = render
        self.human_camera = human_camera
        self.max_steps = int(agent_hz * max_duration_seconds) if not render else np.inf
        self.env_step_ratio = int(120 / agent_hz)
        self.flight_dome_size = flight_dome_size
        self.aggressor_filepath = path.join(path.dirname(__file__), "./models")

        self.assisted_flight = assisted_flight
        self.damage_per_hit = damage_per_hit
        self.spawn_height = spawn_height
        self.lethal_distance = lethal_distance
        self.lethal_angle = lethal_angle_radians

    def reset(self) -> tuple[np.ndarray, dict]:
        """Resets the environment

        Args:

        Returns:
            tuple[np.ndarray, dict()]:
        """
        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        # reset learning parameters
        self.step_count = 0
        self.termination = np.zeros((2), dtype=bool)
        self.truncation = np.zeros((2), dtype=bool)
        self.reward = np.zeros((2))
        self.state = np.zeros((2, self.observation_space.shape[0]))
        self.prev_actions = np.zeros((2, 4))
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["d1_win"] = False
        self.info["d2_win"] = False
        self.info["healths"] = np.ones((2))

        # reset runtime parameters
        self.health = np.ones((2))
        self.in_cone = np.zeros((2,), dtype=bool)
        self.in_range = np.zeros((2,), dtype=bool)
        self.chasing = np.zeros((2,), dtype=bool)
        self.current_hits = np.zeros((2), dtype=bool)
        self.current_angles = np.zeros((2))
        self.current_offsets = np.zeros((2))
        self.current_distance = np.zeros((2))
        self.previous_hits = np.zeros((2), dtype=bool)
        self.previous_angles = np.zeros((2))
        self.previous_offsets = np.zeros((2))
        self.previous_distance = np.zeros((2))

        # randomize starting position and orientation
        # constantly regenerate starting position if they are too close
        # fix height to 20 meters
        start_pos = np.zeros((2, 3))
        while np.linalg.norm(start_pos[0] - start_pos[1]) < self.flight_dome_size * 0.2:
            start_pos = (np.random.rand(2, 3) - 0.5) * self.flight_dome_size * 0.5
            start_pos[:, -1] = self.spawn_height
        start_orn = (np.random.rand(2, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])

        # start_pos = np.array([[0, 0, 5], [5, 5, 10]])
        # start_orn = np.zeros_like(start_pos)
        # start_orn[1, -1] = np.pi
        # start_orn[1, 0] = np.pi / 2

        _, start_vec = self.compute_rotation_forward(start_orn)
        start_vec *= 10.0

        # define all drone options
        drone_options = [dict(), dict()]
        for i in range(len(drone_options)):
            drone_options[i]["model_dir"] = self.aggressor_filepath
            drone_options[i]["drone_model"] = "aggressor"
            drone_options[i]["starting_velocity"] = start_vec[i]
        drone_options[0]["use_camera"] = self.human_camera or self.to_render
        drone_options[0]["camera_resolution"] = np.array([240, 320])

        # start the environment
        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=self.to_render,
            drone_type="fixedwing",
            drone_options=drone_options,
        )

        # render settings
        if self.to_render:
            self.env.drones[0].camera.camera_position_offset = [-10, 0, 5]

        # set flight mode and register all bodies
        self.env.register_all_new_bodies()
        self.env.set_mode(0 if self.assisted_flight else -1)

        # wait for env to stabilize
        for _ in range(3):
            self.env.step()

        return self.state, self.info

    def step(
        self, actions
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """step.

        Args:
            actions: an [n, 4] array of each drone's action

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        # set the actions, reset the reward
        self.env.set_all_setpoints(actions)
        self.prev_actions = actions.copy()
        self.reward *= 0.0

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination.any() or self.truncation.any():
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        # colour the gunsights conditionally
        if self.render and not np.all(self.previous_hits == self.current_hits):
            self.previous_hits = self.current_hits.copy()
            hit_colour = np.array([1.0, 0.0, 0.0, 0.2])
            norm_colour = np.array([0.0, 0.0, 0.0, 0.2])
            for i in range(2):
                self.env.changeVisualShape(
                    self.env.drones[i].Id,
                    7,
                    rgbaColor=(hit_colour if self.current_hits[i] else norm_colour),
                )

        return self.state, self.reward, self.termination, self.truncation, self.info

    def compute_state(self):
        """compute_state."""
        # get the states of both drones
        self.attitudes = np.array(self.env.all_states)

        """COMPUTE HITS"""
        # get the rotation matrices and forward vectors
        # offset the position to be on top of the main wing
        rotation, forward_vecs = self.compute_rotation_forward(self.attitudes[:, 1])
        self.attitudes[:, -1] -= forward_vecs * 0.35

        # compute the vectors of each drone to each drone
        separation = self.attitudes[::-1, -1] - self.attitudes[:, -1]
        self.previous_distance = self.current_distance.copy()
        self.current_distance = np.linalg.norm(separation[0])

        # compute engagement angles
        self.previous_angles = self.current_angles.copy()
        self.current_angles = np.arccos(
            np.sum(separation * forward_vecs, axis=-1) / self.current_distance
        )

        # compute engagement offsets
        self.previous_offsets = self.current_offsets.copy()
        self.current_offsets = np.linalg.norm(
            np_cross(separation, forward_vecs), axis=-1
        )

        # whether we're lethal or chasing or have opponent in cone
        self.in_cone = self.current_angles < self.lethal_angle
        self.in_range = self.current_distance < self.lethal_distance
        self.chasing = np.abs(self.current_angles) < (np.pi / 2.0)

        # compute whether anyone hit anyone
        self.current_hits = self.in_cone & self.in_range & self.chasing

        # update health based on hits
        self.health -= self.damage_per_hit * self.current_hits[::-1]

        """COMPUTE THE STATE VECTOR"""
        # form the opponent state matrix
        opponent_attitudes = np.zeros_like(self.attitudes)

        # opponent angular rate is unchanged
        opponent_attitudes[:, 0] = self.attitudes[::-1, 0]

        # oponent angular position is relative to ours
        opponent_attitudes[:, 1] = self.attitudes[::-1, 1] - self.attitudes[:, 1]

        # opponent velocity is relative to ours in our body frame
        ground_velocities = (
            rotation @ np.expand_dims(self.attitudes[:, -2], axis=-1)
        ).reshape(2, 3)
        opponent_velocities = (
            np.expand_dims(ground_velocities, axis=1)[::-1] @ rotation
        ).reshape(2, 3)
        opponent_attitudes[:, 2] = opponent_velocities - self.attitudes[:, 2]

        # opponent position is relative to ours in our body frame
        opponent_attitudes[:, 3] = (
            np.expand_dims(separation, axis=1) @ rotation
        ).reshape(2, 3)

        # flatten the attitude and opponent attitude, expand dim of health
        flat_attitude = self.attitudes.reshape(2, -1)
        flat_opponent_attitude = opponent_attitudes.reshape(2, -1)
        health = np.expand_dims(self.health, axis=-1)

        # form the state vector
        self.state = np.concatenate(
            [
                flat_attitude,
                health,
                flat_opponent_attitude,
                health[::-1],
                self.prev_actions,
            ],
            axis=-1,
        )

    def compute_term_trunc_reward(self):
        """compute_term_trunc_reward."""
        collisions = self.env.contact_array.sum(axis=0) > 0.0
        collisions = collisions[np.array([d.Id for d in self.env.drones])]
        out_of_bounds = (
            np.linalg.norm(self.attitudes[:, -1], axis=-1) > self.flight_dome_size
        )
        out_of_bounds |= self.attitudes[:, -1, -1] <= 0.0

        # terminate if out of bounds, no health, or collision
        self.termination |= out_of_bounds
        self.termination |= self.health <= 0.0
        self.termination |= collisions

        # truncation is just end
        self.truncation |= self.step_count > self.max_steps

        # reward for closing the distance
        self.reward += (
            np.clip(
                self.previous_distance - self.current_distance, a_min=0.0, a_max=None
            )
            * (~self.in_range & self.chasing)
            * 1.0
        )

        # reward for progressing to engagement
        self.reward += (
            (self.previous_angles - self.current_angles) * self.in_range * 10.0
        )

        # reward for engaging the enemy
        self.reward += 3.0 / (self.current_angles + 0.1) * self.in_range

        # reward for hits
        self.reward += 30.0 * self.current_hits

        # penalty for being hit
        self.reward -= 20.0 * self.current_hits[::-1]

        # penalty for crashing
        self.reward -= 3000.0 * collisions

        # penalty for out of bounds
        self.reward -= 3000.0 * out_of_bounds

        # all the info things
        self.info["out_of_bounds"] = out_of_bounds
        self.info["collision"] = collisions
        self.info["wins"] = self.health <= 0.0
        self.info["healths"] = self.health

    def render(self) -> np.ndarray:
        return self.env.drones[0].rgbaImg

    @staticmethod
    def compute_rotation_forward(orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the rotation matrix and forward vector of an aircraft given its orientation.

        Args:
            orn (np.ndarray): an [n, 3] array of each drone's orientation

        Returns:
            np.ndarray: an [n, 3, 3] rotation matrix of each aircraft
            np.ndarray: an [n, 3] forward vector of each aircraft
        """
        c, s = np.cos(orn), np.sin(orn)
        eye = np.stack([np.eye(3)] * orn.shape[0], axis=0)

        rx = eye.copy()
        rx[:, 1, 1] = c[..., 0]
        rx[:, 1, 2] = -s[..., 0]
        rx[:, 2, 1] = s[..., 0]
        rx[:, 2, 2] = c[..., 0]
        ry = eye.copy()
        ry[:, 0, 0] = c[..., 1]
        ry[:, 0, 2] = s[..., 1]
        ry[:, 2, 0] = -s[..., 1]
        ry[:, 2, 2] = c[..., 1]
        rz = eye.copy()
        rz[:, 0, 0] = c[..., 2]
        rz[:, 0, 1] = -s[..., 2]
        rz[:, 1, 0] = s[..., 2]
        rz[:, 1, 1] = c[..., 2]

        forward_vector = np.stack(
            [c[..., 2] * c[..., 1], s[..., 2] * c[..., 1], -s[..., 1]], axis=-1
        )

        # order of operations for multiplication matters here
        return rz @ ry @ rx, forward_vector
