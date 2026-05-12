#!/usr/bin/env python

# Copyright (c) 2026 Robert Bosch GmbH and its subsidiaries.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This implementation was independently reimplemented based on the publicly
# available E2E Parking CARLA project:
#   https://github.com/qintonguav/e2e-parking-carla
#
# No original source code from the upstream project has been copied.

__author__      = "Min Hee Jo"
__copyright__   = "Copyright 2026, Robert Bosch GmbH"
__license__     = "AGPL"
__version__     = "3.0" 
__email__       = "minhee.jo@de.bosch.com"

import random
from collections import deque

import carla

_SLOTS_PER_ROW = 16
_LOCATION_MATCH_EPS = 0.01
_LEFT_EDGE_SLOTS = (0, 16, 32, 48)
_RIGHT_EDGE_SLOTS = (15, 31, 47, 63)

town04_bound = {
    "x_min": 264.0,
    "x_max": 304.0,
    "y_min": -241.0,
    "y_max": -178.0,
}

slot_id = [
    '2-1',   # 0
    '2-3',
    '2-5',
    '2-7',
    '2-9',
    '2-11',
    '2-13',
    '2-15',
    '3-1',
    '3-3',
    '3-5',
    '3-7',
    '3-9',
    '3-11',
    '3-13',
    '3-15',  # 15
]

parking_vehicle_locations_Town04 = [
    # row 1
    carla.Location(x=298.5, y=-235.73, z=0.3),  # 1-1
    carla.Location(x=298.5, y=-232.73, z=0.3),  # 1-2
    carla.Location(x=298.5, y=-229.53, z=0.3),  # 1-3
    carla.Location(x=298.5, y=-226.43, z=0.3),  # 1-4
    carla.Location(x=298.5, y=-223.43, z=0.3),  # 1-5
    carla.Location(x=298.5, y=-220.23, z=0.3),  # 1-6
    carla.Location(x=298.5, y=-217.23, z=0.3),  # 1-7
    carla.Location(x=298.5, y=-214.03, z=0.3),  # 1-8
    carla.Location(x=298.5, y=-210.73, z=0.3),  # 1-9
    carla.Location(x=298.5, y=-207.30, z=0.3),  # 1-10
    carla.Location(x=298.5, y=-204.23, z=0.3),  # 1-11
    carla.Location(x=298.5, y=-201.03, z=0.3),  # 1-12
    carla.Location(x=298.5, y=-198.03, z=0.3),  # 1-13
    carla.Location(x=298.5, y=-194.90, z=0.3),  # 1-14
    carla.Location(x=298.5, y=-191.53, z=0.3),  # 1-15
    carla.Location(x=298.5, y=-188.20, z=0.3),  # 1-16

    # row 2
    carla.Location(x=290.9, y=-235.73, z=0.3),  # 2-1
    carla.Location(x=290.9, y=-232.73, z=0.3),  # 2-2
    carla.Location(x=290.9, y=-229.53, z=0.3),  # 2-3
    carla.Location(x=290.9, y=-226.43, z=0.3),  # 2-4
    carla.Location(x=290.9, y=-223.43, z=0.3),  # 2-5
    carla.Location(x=290.9, y=-220.23, z=0.3),  # 2-6
    carla.Location(x=290.9, y=-217.23, z=0.3),  # 2-7
    carla.Location(x=290.9, y=-214.03, z=0.3),  # 2-8
    carla.Location(x=290.9, y=-210.73, z=0.3),  # 2-9
    carla.Location(x=290.9, y=-207.30, z=0.3),  # 2-10
    carla.Location(x=290.9, y=-204.23, z=0.3),  # 2-11
    carla.Location(x=290.9, y=-201.03, z=0.3),  # 2-12
    carla.Location(x=290.9, y=-198.03, z=0.3),  # 2-13
    carla.Location(x=290.9, y=-194.90, z=0.3),  # 2-14
    carla.Location(x=290.9, y=-191.53, z=0.3),  # 2-15
    carla.Location(x=290.9, y=-188.20, z=0.3),  # 2-16

    # row 3
    carla.Location(x=280.0, y=-235.73, z=0.3),  # 3-1
    carla.Location(x=280.0, y=-232.73, z=0.3),  # 3-2
    carla.Location(x=280.0, y=-229.53, z=0.3),  # 3-3
    carla.Location(x=280.0, y=-226.43, z=0.3),  # 3-4
    carla.Location(x=280.0, y=-223.43, z=0.3),  # 3-5
    carla.Location(x=280.0, y=-220.23, z=0.3),  # 3-6
    carla.Location(x=280.0, y=-217.23, z=0.3),  # 3-7
    carla.Location(x=280.0, y=-214.03, z=0.3),  # 3-8
    carla.Location(x=280.0, y=-210.73, z=0.3),  # 3-9
    carla.Location(x=280.0, y=-207.30, z=0.3),  # 3-10
    carla.Location(x=280.0, y=-204.23, z=0.3),  # 3-11
    carla.Location(x=280.0, y=-201.03, z=0.3),  # 3-12
    carla.Location(x=280.0, y=-198.03, z=0.3),  # 3-13
    carla.Location(x=280.0, y=-194.90, z=0.3),  # 3-14
    carla.Location(x=280.0, y=-191.53, z=0.3),  # 3-15
    carla.Location(x=280.0, y=-188.20, z=0.3),  # 3-16

    # row 4
    carla.Location(x=272.5, y=-235.73, z=0.3),  # 4-1
    carla.Location(x=272.5, y=-232.73, z=0.3),  # 4-2
    carla.Location(x=272.5, y=-229.53, z=0.3),  # 4-3
    carla.Location(x=272.5, y=-226.43, z=0.3),  # 4-4
    carla.Location(x=272.5, y=-223.43, z=0.3),  # 4-5
    carla.Location(x=272.5, y=-220.23, z=0.3),  # 4-6
    carla.Location(x=272.5, y=-217.23, z=0.3),  # 4-7
    carla.Location(x=272.5, y=-214.03, z=0.3),  # 4-8
    carla.Location(x=272.5, y=-210.73, z=0.3),  # 4-9
    carla.Location(x=272.5, y=-207.30, z=0.3),  # 4-10
    carla.Location(x=272.5, y=-204.23, z=0.3),  # 4-11
    carla.Location(x=272.5, y=-201.03, z=0.3),  # 4-12
    carla.Location(x=272.5, y=-198.03, z=0.3),  # 4-13
    carla.Location(x=272.5, y=-194.90, z=0.3),  # 4-14
    carla.Location(x=272.5, y=-191.53, z=0.3),  # 4-15
    carla.Location(x=272.5, y=-188.20, z=0.3),  # 4-16
]


def clamp_xy_to_bounds(x, y, bounds=town04_bound, buffer=0.0):
    min_x = bounds["x_min"] + buffer
    max_x = bounds["x_max"] - buffer
    min_y = bounds["y_min"] + buffer
    max_y = bounds["y_max"] - buffer
    clamped_x = min(max(x, min_x), max_x)
    clamped_y = min(max(y, min_y), max_y)
    return clamped_x, clamped_y


def get_next_to_target(target_idx):
    if target_idx in _LEFT_EDGE_SLOTS:
        return [target_idx + 1]
    if target_idx in _RIGHT_EDGE_SLOTS:
        return [target_idx - 1]
    return [target_idx - 1, target_idx + 1]


def get_row_col_from_parking_idx(idx: int):
    return divmod(idx, _SLOTS_PER_ROW)


def get_idx_from_location(location: carla.Location):
    for idx, loc in enumerate(parking_vehicle_locations_Town04):
        if abs(loc.x - location.x) < _LOCATION_MATCH_EPS and abs(loc.y - location.y) < _LOCATION_MATCH_EPS:
            return idx
    return None


class EgoPosTown04:
    def __init__(self, rng=None, deterministic=False):
        self.x = 285.600006   # 2-1 slot.x
        self.y = -243.729996  # 2-1 slot.y - 8.0
        self.z = 0.32682
        self.yaw = 90.0

        self._rng = rng if rng is not None else random.Random()
        self._deterministic = deterministic

        self._min_goal_y_gap = 3.5
        self._y_queue = deque()
        self._yaw_queue = deque()

        self.yaw_to_r = 90.0
        self.yaw_to_l = -90.0

        self.goal_y = None
        self.y_max = None
        self.y_min = None

    def set_seed(self, seed):
        self._rng.seed(seed)

    def get_cur_ego_transform(self):
        return carla.Transform(carla.Location(x=self.x, y=self.y, z=self.z),
                               carla.Rotation(pitch=0.0, yaw=self.yaw, roll=0.0))

    def get_init_ego_transform(self):
        return self.get_cur_ego_transform()

    def update_y_scope(self, goal_y, buffer=2.0):
        self.goal_y = goal_y
        self.y_max = min(self.goal_y + 8, town04_bound["y_max"] - buffer)
        self.y_min = max(self.goal_y - 8, town04_bound["y_min"] + buffer)

    def update_yaw_scope(self):
        self.yaw_max = 135
        self.yaw_min = 45

    def update_data_gen_goal_y(self, goal_y):
        self.update_y_scope(goal_y)

    def update_eva_goal_y(self, goal_y, every_parking_num):
        self.update_y_scope(goal_y)

        if self._deterministic:
            self._y_queue, self._yaw_queue = self._build_deterministic_queues(every_parking_num)
        else:
            self._y_queue = self._build_y_queue(every_parking_num)
            self._yaw_queue = deque()  # empty; yaw chosen randomly in get_eva_ego_transform

    def get_data_gen_ego_transform(self):
        self.y = self._sample_valid_y()
        self.yaw = self._rng.choice([self.yaw_to_r, self.yaw_to_l])
        return self.get_cur_ego_transform()

    def get_eva_ego_transform(self):
        if self._yaw_queue:
            self.yaw = self._yaw_queue.popleft()
        else:
            self.yaw = self._rng.choice([self.yaw_to_r, self.yaw_to_l])

        if self._y_queue:
            self.y = self._y_queue.popleft()

        return self.get_cur_ego_transform()

    def _build_y_queue(self, every_parking_num):
        candidates = []

        while len(candidates) < every_parking_num:
            candidates.append(self._sample_valid_y())

        return deque(candidates)

    def _sample_valid_y(self):
        temp_y = self._rng.uniform(self.y_min, self.y_max)
        while abs(temp_y - self.goal_y) < self._min_goal_y_gap:
            temp_y = self._rng.uniform(self.y_min, self.y_max)
        return temp_y

    def _build_deterministic_queues(self, every_parking_num):
        """Build evenly spaced y positions and deterministic yaw based on parking index."""
        y_candidates = []
        yaw_candidates = []

        if every_parking_num > 1:
            y_step = (self.y_max - self.y_min) / (every_parking_num - 1)
            for parking_idx in range(every_parking_num):
                y_candidates.append(self.y_min + parking_idx * y_step)
                yaw = self.yaw_to_r if parking_idx < (every_parking_num / 2) else self.yaw_to_l
                yaw_candidates.append(yaw)
        else:
            y_candidates.append(self.goal_y)
            yaw_candidates.append(self.yaw_to_r)

        return deque(y_candidates), deque(yaw_candidates)
