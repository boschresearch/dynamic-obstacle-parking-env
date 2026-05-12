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

__author__      = "Min Hee Jo"
__copyright__   = "Copyright 2026, Robert Bosch GmbH"
__license__     = "AGPL"
__version__     = "3.0" 
__email__       = "minhee.jo@de.bosch.com"

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

import carla

from . import parking_layout as parking_position

DIRECTIONS = {
    "east": 90.0,
    "south": 180.0,
    "west": -90.0,
    "north": 0.0,
}

# Backward-compatible alias used by prior code versions.
directions = DIRECTIONS


def get_direction(value):
    closest_key = None
    closest_error = None
    for key, angle in DIRECTIONS.items():
        error = abs((value - angle + 180.0) % 360.0 - 180.0)
        if closest_error is None or error < closest_error:
            closest_error = error
            closest_key = key
    return closest_key


def get_closer_horizontal_direction(value):
    east_error = abs((value - DIRECTIONS["east"] + 180.0) % 360.0 - 180.0)
    west_error = abs((value - DIRECTIONS["west"] + 180.0) % 360.0 - 180.0)
    return "east" if east_error <= west_error else "west"


@dataclass
class DriveOutState:
    actor: object
    origin_transform: object
    blueprint: object
    direction: str = "east"
    phase: str = "forward"
    step: int = 0
    last_steer: float = 0.0


@dataclass
class BlockState:
    actor: Optional[object] = None
    init_done: bool = False
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    direction: str = "east"
    blocking_tick: int = 0
    blocking_timeout: int = 0
    phase: str = "drive"


@dataclass
class FollowState:
    actor: Optional[object] = None
    init_done: bool = False
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    direction: str = "east"
    mode: str = "accelerate"
    mode_ticks: int = 0


class NpcManager:
    """Manages NPC vehicles in static and dynamic parking scenarios."""

    def __init__(self, carla_world, parking_spawn_points, actor_list, perfect_parking=True, npc_mode="none", rng=None):
        self._world = carla_world
        self._parking_spawn_points = parking_spawn_points
        self._actor_list = actor_list

        self.perfect_parking = perfect_parking
        self.npc_mode = npc_mode

        self._rng = rng if rng is not None else random.Random()
        self._slot_seed = None

        self._drive_out_npc: Optional[DriveOutState] = None
        self._init_drive_out_npc = True

        self._block_npc: Optional[BlockState] = None
        self._init_block_npc = True
        self._block_blueprint = None

        self._follow_npc: Optional[FollowState] = None
        self._init_follow_npc = True
        self._follow_blueprint = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_seed(self, seed):
        self._rng.seed(seed)

    def set_attempt_seed(self, attempt_index):
        if self._slot_seed is None:
            return
        self._rng.seed(self._slot_seed * 1009 + attempt_index)

    def init_npc(self, seed, target_index, player=None):
        del player

        self._slot_seed = seed

        # Keep static generation deterministic regardless of dynamic mode.
        layout_rng = random.Random(seed * 9973 + 11)
        blueprint_rng = random.Random(seed * 9973 + 13)
        pose_rng = random.Random(seed * 9973 + 17)

        # Separate stream for dynamic behaviour.
        self.set_seed(seed * 10007 + 23)

        target_goal = self._parking_spawn_points[target_index]
        static_vehicle_num = layout_rng.randint(
            int(len(self._parking_spawn_points) / 3),
            len(self._parking_spawn_points) - 1,
        )

        parking_points = self._parking_spawn_points.copy()
        layout_rng.shuffle(parking_points)

        blueprints = self._get_vehicle_blueprints("vehicle")
        self._block_blueprint = self._rng.choice(blueprints)
        self._follow_blueprint = self._rng.choice(blueprints)

        occupied_slots = set()
        near_target_locations = self._locations_next_to_target(target_index)

        for spawn_point in parking_points[:static_vehicle_num]:
            if spawn_point == target_goal:
                continue

            npc_bp = blueprint_rng.choice(blueprints)
            npc, transform = self._try_spawn_npc(spawn_point, npc_bp, rng=pose_rng)
            if npc is None:
                continue

            npc.set_simulate_physics(False)
            self._actor_list.append(npc)
            occupied_slots.add(spawn_point)

            if (
                self.npc_mode == "drive_out"
                and spawn_point in near_target_locations
                and self._drive_out_npc is None
            ):
                npc.set_simulate_physics(True)
                self._drive_out_npc = DriveOutState(
                    actor=npc,
                    origin_transform=transform,
                    blueprint=npc_bp,
                    direction=self._rng.choice(["east", "west"]),
                )

        if self.npc_mode == "drive_out" and self._drive_out_npc is None and near_target_locations:
            occupied_location = self._spawn_guaranteed_drive_out_npc(near_target_locations, blueprint_rng, pose_rng)
            static_vehicle_num += 1
            occupied_slots.add(occupied_location)

        all_parking_goals = [
            point for point in self._parking_spawn_points if point not in occupied_slots
        ]

        logging.info("spawn %d static vehicle in parking lot", static_vehicle_num)
        logging.info("set %d parking goal", len(all_parking_goals))

        return all_parking_goals

    def update(self, target_index, player):
        if self.npc_mode == "drive_out":
            self._update_drive_out_npc(player)

        if self.npc_mode == "block":
            self._update_block_npc(target_index, player)

        if self.npc_mode == "follow":
            self._update_follow_npc(target_index, player)

    def reset_dynamic_npcs(self):
        self._drive_out_npc = None
        self._init_drive_out_npc = True
        self._block_npc = None
        self._init_block_npc = True
        self._block_blueprint = None
        self._follow_npc = None
        self._init_follow_npc = True
        self._follow_blueprint = None

    def clear_block_npc(self):
        if self._block_npc and self._block_npc.actor is not None:
            self._destroy_actor(self._block_npc.actor)
        self._block_npc = None
        self._init_block_npc = True

    def clear_follow_npc(self):
        if self._follow_npc and self._follow_npc.actor is not None:
            self._destroy_actor(self._follow_npc.actor)
        self._follow_npc = None
        self._init_follow_npc = True

    def clear_drive_out_npc(self):
        mover = self._drive_out_npc
        if mover is None:
            return

        self._destroy_actor(mover.actor)

        npc = None
        while npc is None:
            npc = self._world.try_spawn_actor(mover.blueprint, mover.origin_transform)

        npc.set_simulate_physics(True)
        self._actor_list.append(npc)

        mover.actor = npc
        mover.phase = "forward"
        mover.step = 0
        mover.last_steer = 0.0
        self._init_drive_out_npc = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def init_drive_out_npc(self):
        return self._init_drive_out_npc

    @init_drive_out_npc.setter
    def init_drive_out_npc(self, value):
        self._init_drive_out_npc = value

    @property
    def init_block_npc(self):
        return self._init_block_npc

    @init_block_npc.setter
    def init_block_npc(self, value):
        self._init_block_npc = value

    @property
    def init_follow_npc(self):
        return self._init_follow_npc

    @init_follow_npc.setter
    def init_follow_npc(self, value):
        self._init_follow_npc = value

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _locations_next_to_target(self, target_index):
        idxs = parking_position.get_next_to_target(target_index)
        return [self._parking_spawn_points[idx] for idx in idxs]

    def _get_vehicle_blueprints(self, pattern):
        blueprints = self._world.get_blueprint_library().filter(pattern)
        valid = [bp for bp in blueprints if self._valid_vehicle(bp)]
        return sorted(valid, key=lambda bp: bp.id)

    def _spawn_guaranteed_drive_out_npc(self, candidate_locations, blueprint_rng, pose_rng):
        location = self._rng.choice(candidate_locations)
        blueprint = blueprint_rng.choice(self._get_vehicle_blueprints("vehicle"))

        npc = None
        transform = None
        while npc is None:
            npc, transform = self._try_spawn_npc(location, blueprint, rng=pose_rng)

        npc.set_simulate_physics(True)
        self._actor_list.append(npc)
        self._drive_out_npc = DriveOutState(actor=npc, origin_transform=transform, blueprint=blueprint)
        return location

    def _destroy_actor(self, actor):
        if actor is None:
            return
        try:
            actor.destroy()
        except RuntimeError:
            pass
        if actor in self._actor_list:
            self._actor_list.remove(actor)

    def _try_spawn_npc(self, spawn_point, blueprint, rng=None):
        rng = rng if rng is not None else self._rng
        original_index = parking_position.get_idx_from_location(spawn_point)
        yaw = self._get_reverse_parking_direction(original_index)

        if self.perfect_parking:
            yaw_jitter = 0.0
            location = spawn_point
        else:
            yaw_jitter = rng.uniform(-6.0, 6.0)
            x_jitter = rng.uniform(-0.3, 0.3)
            y_jitter = rng.uniform(-0.8, 0.8)
            location = carla.Location(
                x=spawn_point.x + x_jitter,
                y=spawn_point.y + y_jitter,
                z=spawn_point.z,
            )

        transform = carla.Transform(location, rotation=carla.Rotation(yaw=yaw + yaw_jitter))
        return self._world.try_spawn_actor(blueprint, transform), transform

    @staticmethod
    def _get_reverse_parking_direction(idx):
        row, _ = parking_position.get_row_col_from_parking_idx(idx)
        return 0 if row in (0, 2) else 180

    @staticmethod
    def _valid_vehicle(vehicle_bp):
        if int(vehicle_bp.get_attribute("number_of_wheels")) != 4:
            return False
        vehicle_id = vehicle_bp.id.lower()
        blocked = ("firetruck", "bus", "truck", "ambulance", "carlacola", "fusorosa", "sprinter")
        return not any(word in vehicle_id for word in blocked)

    # ------------------------------------------------------------------
    # Drive-out mode
    # ------------------------------------------------------------------

    def _update_drive_out_npc(self, player):
        mover = self._drive_out_npc
        if mover is None:
            return

        if self._init_drive_out_npc:
            self._reinitialize_drive_out(mover)
            return

        actor = mover.actor
        origin_loc = mover.origin_transform.location
        distance_to_ego = _bbox_distance_2d(player, actor)

        if distance_to_ego > 15.0 or distance_to_ego < 1.8:
            throttle, steer, brake = 0.0, mover.last_steer, 1.0
        elif mover.phase == "stop":
            return
        elif mover.phase == "forward":
            throttle, steer, brake = self._drive_out_forward_step(mover, actor, origin_loc)
        elif mover.phase == "turn":
            throttle, steer, brake = self._drive_out_turn_step(mover, actor)
        else:
            return

        mover.last_steer = steer
        actor.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

    def _reinitialize_drive_out(self, mover):
        npc = mover.actor
        if npc.get_transform() != mover.origin_transform:
            self._destroy_actor(npc)
            npc = self._world.try_spawn_actor(mover.blueprint, mover.origin_transform)
            npc.set_simulate_physics(True)
            self._actor_list.append(npc)
            mover.actor = npc

        mover.direction = self._rng.choice(["east", "west"])
        mover.phase = "forward"
        mover.step = 0
        mover.last_steer = 0.0
        self._init_drive_out_npc = False

    def _drive_out_forward_step(self, mover, actor, origin_loc):
        cur_loc = actor.get_transform().location
        moved = math.hypot(cur_loc.x - origin_loc.x, cur_loc.y - origin_loc.y)

        if moved >= 3.0:
            mover.phase = "turn"
            mover.step = 0
            return 0.0, 0.0, 1.0

        return 0.4, 0.0, 0.0

    def _drive_out_turn_step(self, mover, actor):
        mover.step += 1

        current_yaw = actor.get_transform().rotation.yaw
        target_yaw = DIRECTIONS[mover.direction]

        signed_diff = (target_yaw - current_yaw + 180.0) % 360.0 - 180.0
        yaw_diff = abs(signed_diff)

        if yaw_diff < 4.0 or mover.step > 220:
            mover.phase = "stop"
            return 0.0, 0.0, 1.0

        throttle = 0.2 if yaw_diff < 10.0 else 0.3
        steer = -0.65 if signed_diff > 0 else 0.65
        return throttle, steer, 0.0

    # ------------------------------------------------------------------
    # Block mode
    # ------------------------------------------------------------------

    def _update_block_npc(self, target_index, player):
        if self._block_npc is None:
            self._block_npc = BlockState()

        if self._init_block_npc:
            self._initialize_or_validate_block_npc(target_index, player)
            return

        mover = self._block_npc
        actor = mover.actor
        if actor is None:
            return

        target_y = self._parking_spawn_points[target_index].y
        cur_y = actor.get_location().y
        distance_to_ego = _bbox_distance_2d(player, actor)

        throttle = 0.0
        steer = 0.0
        brake = 1.0
        reverse = False

        if mover.phase == "drive":
            throttle = self._rng.uniform(0.4, 0.5)
            brake = 0.0
            if distance_to_ego < 3.0 or abs(cur_y - target_y) < 0.5:
                mover.phase = "blocking"
                throttle = 0.0
                brake = 1.0

        elif mover.phase == "blocking":
            if distance_to_ego < 2.0:
                mover.blocking_tick += 1
            if mover.blocking_tick > mover.blocking_timeout:
                mover.phase = "drive_further"

        elif mover.phase == "drive_further":
            throttle = self._rng.uniform(0.1, 0.25)
            brake = 0.0
            reverse = self._is_ego_in_front(actor, player)

        actor.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=reverse))

    def _initialize_or_validate_block_npc(self, target_index, player):
        block_state = self._block_npc

        if block_state.init_done is False and block_state.actor is not None:
            self._validate_block_candidate(block_state, player)
            return

        self._destroy_actor(block_state.actor)
        self._block_npc = self._create_block_candidate(target_index, player)

    def _validate_block_candidate(self, state, player):
        npc = state.actor
        if npc is None:
            self._block_npc = BlockState()
            return

        dist = _bbox_distance_2d(player, npc)
        logging.debug(
            "candidate block npc at (%.2f, %.2f) dir=%s, bbox distance to ego: %.2f",
            state.spawn_x,
            state.spawn_y,
            state.direction,
            dist,
        )

        if dist < 1.5:
            logging.debug("too close to ego (%.2f m). destroying candidate and retrying.", dist)
            self._destroy_actor(npc)
            self._block_npc = BlockState()
            return

        npc.set_simulate_physics(True)
        self._actor_list.append(npc)

        state.blocking_tick = 0
        state.blocking_timeout = self._rng.randint(50, 80)
        state.phase = "drive"
        state.init_done = True

        logging.info(
            "Spawned block npc %s at (%.2f, %.2f) with distance %.2f to ego.",
            npc.type_id,
            state.spawn_x,
            state.spawn_y,
            dist,
        )
        self._init_block_npc = False

    def _create_block_candidate(self, target_index, player):
        target_goal = self._parking_spawn_points[target_index]
        row, _ = parking_position.get_row_col_from_parking_idx(target_index)

        spawn_x = target_goal.x + (4.0 if row in (0, 2) else -4.0)
        spawn_y = target_goal.y + self._rng.uniform(-8.0, 8.0)

        ego_y = player.get_location().y
        while abs(spawn_y - ego_y) < 2.0:
            spawn_y = target_goal.y + self._rng.uniform(-8.0, 8.0)
            spawn_x, spawn_y = parking_position.clamp_xy_to_bounds(spawn_x, spawn_y, buffer=1.0)

        direction = "west" if spawn_y >= target_goal.y else "east"
        transform = carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=0.3),
            carla.Rotation(yaw=DIRECTIONS[direction]),
        )

        npc = self._world.try_spawn_actor(self._block_blueprint, transform)
        if npc is None:
            logging.debug("try_spawn_actor failed for block npc at (%.2f, %.2f). will retry.", spawn_x, spawn_y)
            return BlockState()

        npc.set_simulate_physics(False)
        return BlockState(
            actor=npc,
            init_done=False,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            direction=direction,
        )

    @staticmethod
    def _is_ego_in_front(npc_actor, ego_actor):
        npc_tf = npc_actor.get_transform()
        npc_loc = npc_tf.location
        ego_loc = ego_actor.get_location()
        forward = npc_tf.get_forward_vector()
        rel_x = ego_loc.x - npc_loc.x
        rel_y = ego_loc.y - npc_loc.y
        return (rel_x * forward.x + rel_y * forward.y) >= 0.0

    # ------------------------------------------------------------------
    # Follow mode
    # ------------------------------------------------------------------

    def _update_follow_npc(self, target_index, player):
        if self._follow_npc is None:
            self._follow_npc = FollowState()

        if self._init_follow_npc:
            self._initialize_or_validate_follow_npc(target_index, player)
            return

        mover = self._follow_npc
        actor = mover.actor
        if actor is None:
            return

        mover.mode_ticks -= 1
        if mover.mode_ticks <= 0:
            self._flip_follow_mode(mover)

        if mover.mode == "accelerate":
            throttle = self._rng.uniform(0.2, 0.5)
            brake = 0.0
        else:
            throttle = 0.0
            brake = self._rng.uniform(0.25, 0.6)

        actor.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0, brake=brake))

    def _initialize_or_validate_follow_npc(self, target_index, player):
        follow_state = self._follow_npc

        if follow_state.init_done is False and follow_state.actor is not None:
            self._validate_follow_candidate(follow_state, player)
            return

        self._destroy_actor(follow_state.actor)
        self._follow_npc = self._create_follow_candidate(target_index, player)

    def _validate_follow_candidate(self, state, player):
        npc = state.actor
        if npc is None:
            self._follow_npc = FollowState()
            return

        dist = _bbox_distance_2d(player, npc)
        logging.debug(
            "candidate follow npc at (%.2f, %.2f) dir=%s, bbox distance to ego: %.2f",
            state.spawn_x,
            state.spawn_y,
            state.direction,
            dist,
        )

        if dist < 1.0:
            logging.debug("too close to ego (%.2f m). destroying candidate and retrying.", dist)
            self._destroy_actor(npc)
            self._follow_npc = FollowState()
            return

        npc.set_simulate_physics(True)
        self._actor_list.append(npc)

        state.init_done = True
        state.mode = "accelerate"
        state.mode_ticks = self._rng.randint(20, 60)

        logging.info(
            "Spawned follow npc %s at (%.2f, %.2f) with distance %.2f to ego.",
            npc.type_id,
            state.spawn_x,
            state.spawn_y,
            dist,
        )
        self._init_follow_npc = False

    def _create_follow_candidate(self, target_index, player):
        target_goal = self._parking_spawn_points[target_index]

        ego_loc = player.get_location()
        ego_tf = player.get_transform()
        ego_direction = get_direction(ego_tf.rotation.yaw)

        spawn_x = ego_loc.x + self._rng.uniform(-1.2, 1.2)
        gap = 7.0

        if ego_direction == "east":
            if target_goal.y + 3.0 > ego_loc.y:
                spawn_y = ego_loc.y + gap
                spawn_direction = "east"
            else:
                spawn_y = ego_loc.y - gap
                spawn_direction = "west"
        elif ego_direction == "west":
            if target_goal.y - 3.0 < ego_loc.y:
                spawn_y = ego_loc.y - gap
                spawn_direction = "west"
            else:
                spawn_y = ego_loc.y + gap
                spawn_direction = "east"
        else:
            spawn_direction = get_closer_horizontal_direction(ego_tf.rotation.yaw)
            spawn_y = ego_loc.y + gap if target_goal.y >= ego_loc.y else ego_loc.y - gap

        transform = carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=0.3),
            carla.Rotation(yaw=DIRECTIONS[spawn_direction]),
        )

        npc = self._world.try_spawn_actor(self._follow_blueprint, transform)
        if npc is None:
            logging.debug("try_spawn_actor failed for follow npc at (%.2f, %.2f). will retry.", spawn_x, spawn_y)
            return FollowState()

        npc.set_simulate_physics(False)
        return FollowState(
            actor=npc,
            init_done=False,
            spawn_x=spawn_x,
            spawn_y=spawn_y,
            direction=spawn_direction,
        )

    def _flip_follow_mode(self, state):
        if state.mode == "accelerate":
            state.mode = "break" if self._rng.random() < 0.35 else "accelerate"
        else:
            state.mode = "accelerate"
        state.mode_ticks = self._rng.randint(1, 5)


# ------------------------------------------------------------------
# Module-level geometry helpers (shared with world.py via import)
# ------------------------------------------------------------------

def _bbox_distance_2d(ego_actor, other_actor):
    poly_a = _carla_actor_obb_corners_2d(ego_actor)
    poly_b = _carla_actor_obb_corners_2d(other_actor)
    if _polygons_intersect_sat(poly_a, poly_b):
        return 0.0
    return _polygons_min_distance(poly_a, poly_b)


def _carla_actor_obb_corners_2d(actor):
    tf = actor.get_transform()
    loc = tf.location
    yaw_rad = math.radians(tf.rotation.yaw)

    bbox = actor.bounding_box
    ex, ey = bbox.extent.x, bbox.extent.y
    off = bbox.location

    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)

    center_x = loc.x + off.x * c - off.y * s
    center_y = loc.y + off.x * s + off.y * c

    local = [(+ex, +ey), (+ex, -ey), (-ex, -ey), (-ex, +ey)]
    return [(center_x + x * c - y * s, center_y + x * s + y * c) for x, y in local]


def _edges(poly):
    return list(zip(poly, poly[1:] + poly[:1]))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _norm2(v):
    return v[0] * v[0] + v[1] * v[1]


def _project(poly, axis):
    vals = [_dot(p, axis) for p in poly]
    return min(vals), max(vals)


def _polygons_intersect_sat(poly_a, poly_b):
    for poly in (poly_a, poly_b):
        for p1, p2 in _edges(poly):
            edge = _sub(p2, p1)
            axis = (-edge[1], edge[0])
            a_min, a_max = _project(poly_a, axis)
            b_min, b_max = _project(poly_b, axis)
            if a_max < b_min or b_max < a_min:
                return False
    return True


def _point_segment_distance(p, a, b):
    ab = _sub(b, a)
    ap = _sub(p, a)
    ab_len2 = _norm2(ab)

    if ab_len2 == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])

    t = max(0.0, min(1.0, _dot(ap, ab) / ab_len2))
    closest = (a[0] + t * ab[0], a[1] + t * ab[1])
    return math.hypot(p[0] - closest[0], p[1] - closest[1])


def _polygons_min_distance(poly_a, poly_b):
    min_d = float("inf")
    for p in poly_a:
        for b1, b2 in _edges(poly_b):
            min_d = min(min_d, _point_segment_distance(p, b1, b2))
    for p in poly_b:
        for a1, a2 in _edges(poly_a):
            min_d = min(min_d, _point_segment_distance(p, a1, a2))
    return float(min_d)
