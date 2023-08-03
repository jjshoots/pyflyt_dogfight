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
        flight_dome_size: float = 100.0,
        max_duration_seconds: float = 100.0,
        agent_hz: int = 40,
        damage_per_hit: float = 0.1,
        lethal_angle_radian: float = 0.1,
        lethal_offset: float = 0.15,
        render: bool = False
    ):
        """__init__.

        Args:
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            agent_hz (int): agent_hz
            lethal_angle_radian (float): the width of the weapons engagement cone
            lethal_offset (float): how close must the nose of the aircraft be to the opponents body to be considered a hit
            render (bool): whether to render the environment
        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        """SPACES"""
        state_shape = 13  # 12 states + health
        action_shape = 4  # roll, pitch, yaw, thrust
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * state_shape + action_shape,)
        )

        high = np.array([1.0, 1.0, 1.0, 1.0])
        low = np.array([-1.0, -1.0, -1.0, 0.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        """CONSTANTS"""
        self.env: Aviary
        self.num_drones = 2
        self.render = render
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        self.flight_dome_size = flight_dome_size
        self.aggressor_filepath = path.join(path.dirname(__file__), "./models")

        self.damage_per_hit = damage_per_hit
        self.lethal_angle = lethal_angle_radian
        self.lethal_offset = lethal_offset

    def reset(self) -> tuple[np.ndarray, dict()]:
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
        self.current_angles = np.zeros((2))
        self.current_offsets = np.zeros((2))
        self.current_distance = np.zeros((2))
        self.previous_angles = np.zeros((2))
        self.previous_offsets = np.zeros((2))
        self.previous_distance = np.zeros((2))
        self.hits = np.zeros((2), dtype=bool)

        # randomize starting position and orientation
        # constantly regenerate starting position if they are too close
        start_pos = np.zeros((2, 3))
        while np.linalg.norm(start_pos[0] - start_pos[1]) < self.flight_dome_size * 0.3:
            start_pos = (np.random.rand(2, 3) - 0.5) * self.flight_dome_size
            start_pos[:, -1] = np.clip(start_pos[:, -1], a_min=10.0, a_max=None)
        start_orn = (np.random.rand(2, 3) - 0.5) * 2.0 * np.array([1.5, 1.0, 2 * np.pi])
        start_vec = self.compute_forward_vec(start_orn) * 10.0

        # define all drone options
        drone_options = [dict(), dict()]
        for i in range(len(drone_options)):
            drone_options[i]["model_dir"] = self.aggressor_filepath
            drone_options[i]["drone_model"] = "aggressor"
            drone_options[i]["starting_velocity"] = start_vec[i]

        # start the environment
        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            render=self.render,
            drone_type="fixedwing",
            drone_options=drone_options,
        )

        # set flight mode and register all bodies
        self.env.register_all_new_bodies()
        self.env.set_mode(0)

        # wait for env to stabilize
        for _ in range(3):
            self.env.step()

        return self.state, self.info

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
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

        return self.state, self.reward, self.termination, self.truncation, self.info

    def compute_state(self):
        """compute_state.
        """
        # get the states of both drones
        self.attitudes = np.array(self.env.all_states)

        """COMPUTE HITS"""
        # get the positions, orientations, and forward vectors
        # offset the position to be on top of the main wing
        orientations, positions = self.attitudes[:, 1], self.attitudes[:, -1]
        forward_vecs = self.compute_forward_vec(orientations)
        positions -= forward_vecs * 0.35

        # compute the vectors of each drone to each drone
        separation = positions[::-1] - positions
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

        # compute whether anyone hit anyone
        self.hits = self.current_angles < self.lethal_angle
        self.hits &= self.current_offsets < self.lethal_offset

        # update health based on hits
        self.health -= self.damage_per_hit * self.hits[::-1]

        """COMPUTE THE STATE VECTOR"""
        # opponent attitude is relative to ours
        attitude = np.reshape(self.attitudes, (2, -1))
        opponent_attitude = attitude[::-1] - attitude

        # form the state vector
        health = np.expand_dims(self.health, axis=-1)
        self.state = np.concatenate(
            [
                attitude,
                health,
                opponent_attitude,
                health[::-1],
                self.prev_actions,
            ],
            axis=-1,
        )

    def compute_term_trunc_reward(self):
        """compute_term_trunc_reward.
        """
        collisions = self.env.contact_array.sum(axis=0) > 0.0
        collisions = collisions[np.array([d.Id for d in self.env.drones])]
        out_of_bounds = np.linalg.norm(self.attitudes[:, -1], axis=-1) > self.flight_dome_size

        # terminate if out of bounds, no health, or collision
        self.termination |= out_of_bounds
        self.termination |= self.health <= 0.0
        self.termination |= collisions

        # truncation is just end
        self.truncation |= self.step_count > self.max_steps

        # reward for bringing the opponent closer to engagement
        self.reward += (
            np.clip(self.previous_angles - self.current_angles, a_min=0.0, a_max=None)
            * 10.0
        )
        self.reward += (
            np.clip(self.previous_offsets - self.current_offsets, a_min=0.0, a_max=None)
            * 1.0
        )

        # reward for being close to bringing weapons to engagement
        self.reward += 0.1 / (self.current_angles + 0.1)
        self.reward += 0.1 / (self.current_offsets + 0.1)

        # reward for hits
        self.reward += 20.0 * self.hits

        # penalty for being hit
        self.reward -= 20.0 * self.hits[::-1]

        # penalty for crashing
        self.reward -= 1000.0 * collisions

        # penalty for out of bounds
        self.reward -= 1000.0 * out_of_bounds

        # all the info things
        self.info["out_of_bounds"] = out_of_bounds.any()
        self.info["collision"] = collisions.any()
        self.info["d1_win"] = self.health[1] <= 0.0
        self.info["d2_win"] = self.health[1] <= 0.0
        self.info["healths"] = np.ones((2))

    @staticmethod
    def compute_forward_vec(orn: np.ndarray) -> np.ndarray:
        """compute_forward_vec.

        Args:
            orn (np.ndarray): an [n, 3] array of each drone's orientation

        Returns:
            np.ndarray: an [n, 3] array of each drone's forward vector
        """
        # computes a forward vector given an orientation
        c, s = np.cos(orn), np.sin(orn)
        # c_phi = c[..., 1]
        # s_phi = s[..., 1]
        # c_psi = c[..., 2]
        # s_psi = s[..., 2]

        return np.stack(
            [c[..., 2] * c[..., 1], s[..., 2] * c[..., 1], -s[..., 1]], axis=-1
        )
