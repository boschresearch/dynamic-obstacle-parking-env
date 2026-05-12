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

import glob
import logging
import math
import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from . import parking_layout as parking_position
from .world import World


class NetworkEvaluator:
    """Coordinates benchmark lifecycle, slot progression, and metric export."""

    def __init__(self, carla_world, args):
        self._init_seed_state(args.random_seed)

        self._world = World(carla_world, args)

        # Evaluation settings
        self._render_bev = args.show_eva_imgs
        self._eva_epochs = args.eva_epochs
        self._eva_task_nums = args.eva_task_nums
        self._eva_parking_nums = args.eva_parking_nums

        # Goal setup: evaluate odd slots from index 16 onward.
        self._parking_goal_index = 16
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]

        deterministic = args.deterministic_ego_spawn
        self._ego_transform_generator = parking_position.EgoPosTown04(deterministic=deterministic)
        self._ego_transform_generator.set_seed(self._ego_seed)

        self._eva_result_path = self._create_result_path(args.eva_result_path)

        # Epoch/task/attempt indices.
        self._eva_epoch_idx = 0
        self._eva_task_idx = 0
        self._eva_parking_idx = 0

        self._setup_goal_tolerances()
        self._setup_frame_limits(terminate_immediately=True)

        # Per-frame counters
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_nearby_no_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0

        # Per-slot aggregate counters
        self._target_success_nums = 0
        self._target_fail_nums = 0
        self._no_target_success_nums = 0
        self._no_target_fail_nums = 0
        self._collision_nums = 0
        self._outbound_nums = 0
        self._timeout_nums = 0
        self._position_error = []
        self._orientation_error = []
        self._parking_time = []
        self._inference_time = []

        # Per-epoch slot-rate lists
        self._target_success_rate = []
        self._target_fail_rate = []
        self._no_target_success_rate = []
        self._no_target_fail_rate = []
        self._collision_rate = []
        self._outbound_rate = []
        self._timeout_rate = []
        self._average_position_error = []
        self._average_orientation_error = []
        self._average_parking_time = []
        self._average_inference_time = []

        self._epoch_metric_info = {}

        self._metric_names = {
            "target_success_rate": "TSR",
            "target_fail_rate": "TFR",
            "no_target_success_rate": "NTSR",
            "no_target_fail_rate": "NTFR",
            "collision_rate": "CR",
            "outbound_rate": "OR",
            "timeout_rate": "TR",
            "average_position_error": "APE",
            "average_orientation_error": "AOE",
            "average_parking_time": "APT",
            "average_inference_time": "AIT",
        }

        self._ego_transform = None
        self._eva_parking_goal = None
        self._agent_need_init = True
        self._start_time = None

        self._reset_goal_diff_cache()

        self.running = True

        self.init()
        self.start_eva_epoch()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_seed_state(self, seed):
        self._seed = seed
        self._init_seed = seed
        self._ego_seed = seed + 1_000_003
        self._init_ego_seed = self._ego_seed

    @staticmethod
    def _create_result_path(root_path):
        now = datetime.now()
        result_dir = "_".join(
            map(
                lambda x: "%02d" % x,
                (now.year, now.month, now.day, now.hour, now.minute, now.second),
            )
        )
        path = pathlib.Path(root_path) / result_dir
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _setup_goal_tolerances(self):
        self._goal_reach_x_diff = 1.0
        self._goal_reach_y_diff = 0.6
        self._goal_reach_orientation_diff = 10.0

    def _setup_frame_limits(self, terminate_immediately=True):
        self._frames_per_second = 30
        self._num_frames_in_goal_needed = 1 if terminate_immediately else 2 * self._frames_per_second
        self._num_frames_nearby_goal_needed = 2 * self._frames_per_second
        self._num_frames_nearby_no_goal_needed = 2 * self._frames_per_second
        self._num_frames_outbound_needed = 10 * self._frames_per_second
        self._num_frames_total_needed = 30 * self._frames_per_second

    def _reset_goal_diff_cache(self):
        self._x_diff_to_goal = sys.float_info.max
        self._y_diff_to_goal = sys.float_info.max
        self._distance_diff_to_goal = sys.float_info.max
        self._orientation_diff_to_goal = sys.float_info.max

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self):
        logging.info("***************** Start init eva environment *****************")

        self._ego_transform = self._ego_transform_generator.get_init_ego_transform()
        self._world.init_ego_vehicle(self._ego_transform)
        logging.info("Init ego vehicle success!")

        self._world.init_sensors()
        logging.info("Init sensors success!")

        self._advance_weather()
        logging.info("Init weather success!")

        logging.info("*****************   End init eva environment *****************")

    def _advance_weather(self):
        if hasattr(self._world, "next_weather"):
            self._world.next_weather()
            return
        if hasattr(self._world, "weather_manager"):
            self._world.weather_manager.next_weather()

    def save_weather(self):
        weather = self._world._world.get_weather()
        out_file = os.path.join(self._eva_result_path, "weather_log.txt")
        with open(out_file, "a") as file_obj:
            file_obj.write(f"{weather}")

    def tick(self, clock):
        world = self._world
        self._publish_goal_diffs_to_world(world)

        self._num_frames_total += 1
        is_collision = world.tick(clock, self._parking_goal_index)

        if is_collision:
            self._collision_nums += 1
            logging.info(
                "parking collision for task %s-%d, collision_num: %d",
                parking_position.slot_id[self._eva_task_idx],
                self._eva_parking_idx + 1,
                self._collision_nums,
            )
            self.start_next_parking()
            return

        if self._render_bev:
            frame = world.sensor_data_frame
            frame["topdown"] = world.render_BEV()

        if self._num_frames_total > self._num_frames_total_needed:
            self._timeout_nums += 1
            logging.info(
                "parking timeout for task %s-%d, timeout_num: %d",
                parking_position.slot_id[self._eva_task_idx],
                self._eva_parking_idx + 1,
                self._timeout_nums,
            )
            self.start_next_parking()
            return

        ego_loc = world.ego_transform.location
        if self.is_out_of_bound(ego_loc):
            self._num_frames_outbound += 1
        else:
            self._num_frames_outbound = 0

        if self._num_frames_outbound > self._num_frames_outbound_needed:
            self._outbound_nums += 1
            logging.info(
                "parking outbound for task %s-%d, outbound_num: %d",
                parking_position.slot_id[self._eva_task_idx],
                self._eva_parking_idx + 1,
                self._outbound_nums,
            )
            self.start_next_parking()
            return

        self.eva_check_goal()

    def _publish_goal_diffs_to_world(self, world):
        world.distance_diff_to_goal = self._distance_diff_to_goal
        world.rotation_diff_to_goal = self._orientation_diff_to_goal
        world.x_diff_to_goal = self._x_diff_to_goal
        world.y_diff_to_goal = self._y_diff_to_goal

    def start_eva_epoch(self):
        logging.info("***************** Start eva epoch %d *****************", self._eva_epoch_idx + 1)

        self._start_time = datetime.now()

        self.soft_destroy()

        self._seed = self._init_seed
        self._ego_seed = self._init_ego_seed

        self._parking_goal_index = 16
        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]

        self._ego_transform_generator.set_seed(self._ego_seed)
        self._ego_transform_generator.update_eva_goal_y(self._parking_goal.y, self._eva_parking_nums)
        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform()

        self._world.player.set_transform(self._ego_transform)
        self._world.init_npc(self._seed, self._parking_goal_index)
        self._world.set_npc_attempt_seed(self._eva_parking_idx)

        self._eva_parking_goal = [self._parking_goal.x, self._parking_goal.y, 180]
        self._agent_need_init = True

        self._eva_task_idx = 0
        self.clear_metric_rate()
        self._epoch_metric_info = {}

        logging.info(
            "***************** Start eva task %s *****************",
            parking_position.slot_id[self._eva_task_idx],
        )

    def start_next_parking(self):
        self._world.init_drive_out_npc = True
        self._world.init_block_npc = True
        self._agent_need_init = True
        self._eva_parking_idx += 1

        if self.is_complete_slot(self._eva_parking_idx):
            logging.info(
                "*****************   End eva task %s *****************",
                parking_position.slot_id[self._eva_task_idx],
            )
            self.save_slot_metric()
            self.start_next_slot()
            return

        self.clear_metric_frame()

        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform()
        self._world.restart(self._ego_transform)

        self._world.clear_npcs()
        self._world.set_npc_attempt_seed(self._eva_parking_idx)

    def is_complete_slot(self, eva_parking_idx):
        return eva_parking_idx >= self._eva_parking_nums

    def start_next_slot(self):
        self._eva_task_idx += 1
        self._seed += 1
        self._ego_seed += 1

        if self.is_complete_epoch(self._eva_task_idx):
            logging.info("*****************   End eva epoch %d *****************", self._eva_epoch_idx + 1)
            self.save_epoch_metric_csv()

            self._eva_epoch_idx += 1
            if self._eva_epoch_idx >= self._eva_epochs:
                self.save_mean_std_csv()
                self.running = False
            else:
                self.start_eva_epoch()
            return

        if self._eva_task_idx < 16:
            self._parking_goal_index += 2
        else:
            self._parking_goal_index = 16

        self.soft_destroy()

        self._parking_goal = parking_position.parking_vehicle_locations_Town04[self._parking_goal_index]
        self._ego_transform_generator.set_seed(self._ego_seed)
        self._ego_transform_generator.update_eva_goal_y(self._parking_goal.y, self._eva_parking_nums)
        self._ego_transform = self._ego_transform_generator.get_eva_ego_transform()

        self._world.player.set_transform(self._ego_transform)
        self._world.init_npc(self._seed, self._parking_goal_index)
        self._world.set_npc_attempt_seed(self._eva_parking_idx)

        self._eva_parking_goal = [self._parking_goal.x, self._parking_goal.y, 180]
        self._world.restart(self._ego_transform)
        self._agent_need_init = True

        logging.info(
            "***************** Start eva task %s *****************",
            parking_position.slot_id[self._eva_task_idx],
        )

    def is_complete_epoch(self, eva_task_idx):
        return eva_task_idx >= self._eva_task_nums

    # ------------------------------------------------------------------
    # Metric reset
    # ------------------------------------------------------------------

    def clear_metric_num(self):
        self._target_success_nums = 0
        self._target_fail_nums = 0
        self._no_target_success_nums = 0
        self._no_target_fail_nums = 0
        self._collision_nums = 0
        self._outbound_nums = 0
        self._timeout_nums = 0

        self._position_error = []
        self._orientation_error = []
        self._inference_time = []
        self._parking_time = []

    def clear_metric_frame(self):
        self._num_frames_in_goal = 0
        self._num_frames_nearby_goal = 0
        self._num_frames_nearby_no_goal = 0
        self._num_frames_outbound = 0
        self._num_frames_total = 0

    def clear_metric_rate(self):
        self._target_success_rate = []
        self._target_fail_rate = []
        self._no_target_success_rate = []
        self._no_target_fail_rate = []
        self._collision_rate = []
        self._outbound_rate = []
        self._timeout_rate = []
        self._average_position_error = []
        self._average_orientation_error = []
        self._average_inference_time = []
        self._average_parking_time = []

    def soft_destroy(self):
        self._eva_parking_idx = 0
        self.clear_metric_num()
        self.clear_metric_frame()
        self._reset_goal_diff_cache()
        self._world.soft_destroy()

    def destroy(self):
        self._world.destroy()

    # ------------------------------------------------------------------
    # Goal evaluation
    # ------------------------------------------------------------------

    def eva_check_goal(self):
        world = self._world
        player = world.player

        transform = player.get_transform()
        location = transform.location
        rotation = transform.rotation
        velocity = player.get_velocity()
        control = player.get_control()

        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        closest_goal, x_diff, y_diff, dist_diff = self._closest_goal_and_diff(location, world.all_parking_goals)
        orientation_diff = min(abs(rotation.yaw), 180 - abs(rotation.yaw))

        self._distance_diff_to_goal = dist_diff
        self._x_diff_to_goal = x_diff
        self._y_diff_to_goal = y_diff
        self._orientation_diff_to_goal = orientation_diff

        is_stop = (control.throttle == 0.0) and (speed < 1e-3) and control.reverse
        if not is_stop:
            self._num_frames_in_goal = 0
            self._num_frames_nearby_goal = 0
            self._num_frames_nearby_no_goal = 0
            return

        if self.check_success_slot(closest_goal, location):
            self.start_next_parking()
            return

        if self.check_fail_slot(closest_goal, location):
            self.start_next_parking()
            return

    @staticmethod
    def _closest_goal_and_diff(current_location, goals):
        closest_goal = [0.0, 0.0]
        x_diff = sys.float_info.max
        y_diff = sys.float_info.max
        dist_diff = sys.float_info.max

        for goal in goals:
            dist = current_location.distance(goal)
            if dist < dist_diff:
                dist_diff = dist
                x_diff = abs(current_location.x - goal.x)
                y_diff = abs(current_location.y - goal.y)
                closest_goal[0] = goal.x
                closest_goal[1] = goal.y

        return closest_goal, x_diff, y_diff, dist_diff

    def check_success_slot(self, closest_goal, ego_transform):
        x_in_slot = abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff
        y_in_slot = abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff
        r_in_slot = self._orientation_diff_to_goal <= self._goal_reach_orientation_diff

        if x_in_slot and y_in_slot and r_in_slot:
            self._num_frames_in_goal += 1

        if self._num_frames_in_goal > self._num_frames_in_goal_needed:
            if (self._eva_parking_goal[0] == closest_goal[0]) and (self._eva_parking_goal[1] == closest_goal[1]):
                self._target_success_nums += 1
                self._position_error.append(self._distance_diff_to_goal)
                self._orientation_error.append(self._orientation_diff_to_goal)
                self._parking_time.append(self._num_frames_total / self._frames_per_second)
                logging.info(
                    "parking target success for task %s-%d, target_success_nums: %d",
                    parking_position.slot_id[self._eva_task_idx],
                    self._eva_parking_idx + 1,
                    self._target_success_nums,
                )
            else:
                self._no_target_success_nums += 1
                logging.info(
                    "parking no target success for task %s-%d, no_target_success_nums: %d",
                    parking_position.slot_id[self._eva_task_idx],
                    self._eva_parking_idx + 1,
                    self._no_target_success_nums,
                )
            return True
        return False

    def check_fail_slot(self, closest_goal, ego_transform):
        x_not_in_slot = self._goal_reach_x_diff < abs(ego_transform.x - closest_goal[0]) <= self._goal_reach_x_diff * 2
        y_not_in_slot = self._goal_reach_y_diff < abs(ego_transform.y - closest_goal[1]) <= self._goal_reach_y_diff * 2
        r_not_in_slot = (
            self._goal_reach_orientation_diff
            < self._orientation_diff_to_goal
            <= self._goal_reach_orientation_diff * 2
        )

        if x_not_in_slot or y_not_in_slot or r_not_in_slot:
            if (self._eva_parking_goal[0] == closest_goal[0]) and (self._eva_parking_goal[1] == closest_goal[1]):
                self._num_frames_nearby_goal += 1
            else:
                self._num_frames_nearby_no_goal += 1

        if self._num_frames_nearby_goal > self._num_frames_nearby_goal_needed:
            self._target_fail_nums += 1
            logging.info(
                "parking target fail for task %s-%d, target_fail_nums: %d",
                parking_position.slot_id[self._eva_task_idx],
                self._eva_parking_idx + 1,
                self._target_fail_nums,
            )
            return True

        if self._num_frames_nearby_no_goal > self._num_frames_nearby_no_goal_needed:
            self._no_target_fail_nums += 1
            logging.info(
                "parking no target fail for task %s-%d, no_target_fail_nums: %d",
                parking_position.slot_id[self._eva_task_idx],
                self._eva_parking_idx + 1,
                self._no_target_fail_nums,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_slot_metric(self):
        tsr = (self._target_success_nums / float(self._eva_parking_nums)) * 100.0
        tfr = (self._target_fail_nums / float(self._eva_parking_nums)) * 100.0
        ntsr = (self._no_target_success_nums / float(self._eva_parking_nums)) * 100.0
        ntfr = (self._no_target_fail_nums / float(self._eva_parking_nums)) * 100.0
        cr = (self._collision_nums / float(self._eva_parking_nums)) * 100.0
        orr = (self._outbound_nums / float(self._eva_parking_nums)) * 100.0
        tr = (self._timeout_nums / float(self._eva_parking_nums)) * 100.0

        ape = np.mean(self._position_error) if self._position_error else np.nan
        aoe = np.mean(self._orientation_error) if self._orientation_error else np.nan
        apt = np.mean(self._parking_time) if self._parking_time else np.nan
        ait = np.mean(self._inference_time) if self._inference_time else np.nan

        slot_id = parking_position.slot_id[self._eva_task_idx]
        self._epoch_metric_info[slot_id] = {
            "target_success_rate": tsr,
            "target_fail_rate": tfr,
            "no_target_success_rate": ntsr,
            "no_target_fail_rate": ntfr,
            "collision_rate": cr,
            "outbound_rate": orr,
            "timeout_rate": tr,
            "average_position_error": ape,
            "average_orientation_error": aoe,
            "average_parking_time": apt,
            "average_inference_time": ait,
        }

        self._target_success_rate.append(tsr)
        self._target_fail_rate.append(tfr)
        self._no_target_success_rate.append(ntsr)
        self._no_target_fail_rate.append(ntfr)
        self._collision_rate.append(cr)
        self._outbound_rate.append(orr)
        self._timeout_rate.append(tr)
        self._average_position_error.append(ape)
        self._average_orientation_error.append(aoe)
        self._average_parking_time.append(apt)
        self._average_inference_time.append(ait)

    def save_epoch_metric_csv(self):
        self._epoch_metric_info["Avg"] = {
            "target_success_rate": np.mean(self._target_success_rate),
            "target_fail_rate": np.mean(self._target_fail_rate),
            "no_target_success_rate": np.mean(self._no_target_success_rate),
            "no_target_fail_rate": np.mean(self._no_target_fail_rate),
            "collision_rate": np.mean(self._collision_rate),
            "outbound_rate": np.mean(self._outbound_rate),
            "timeout_rate": np.mean(self._timeout_rate),
            "average_position_error": np.nanmean(self._average_position_error),
            "average_orientation_error": np.nanmean(self._average_orientation_error),
            "average_inference_time": np.nanmean(self._average_inference_time),
            "average_parking_time": np.nanmean(self._average_parking_time),
        }

        info_df = pd.DataFrame(self._epoch_metric_info)
        csv_name = "eva_epoch_" + str(self._eva_epoch_idx + 1) + "_result.csv"
        self.save_csv(info_df, csv_name)

        logging.info("eva epoch %d total time: %s", self._eva_epoch_idx + 1, datetime.now() - self._start_time)

    def save_csv(self, info_df, csv_name):
        info_df = info_df.T
        info_df.rename(columns=self._metric_names, inplace=True)

        pd.set_option("display.max_columns", 1000)
        pd.options.display.float_format = "{:,.3f}".format

        print(info_df)
        info_df.to_csv(self._eva_result_path / csv_name)

    def save_mean_std_csv(self):
        df_mean = pd.DataFrame()
        df_std = pd.DataFrame()

        csv_files = glob.glob(f"{self._eva_result_path}/*_result.csv")
        for task_idx in range(self._eva_task_nums):
            df_row = pd.DataFrame()
            for csv in csv_files:
                df_csv = pd.read_csv(csv)
                row = df_csv.iloc[[task_idx]]
                df_row = pd.concat([df_row, row], axis=0)

            row_mean = df_row.select_dtypes(include="number").mean(axis=0).to_frame().T
            row_std = df_row.select_dtypes(include="number").std(axis=0, ddof=0).to_frame().T / math.sqrt(6)

            df_mean = pd.concat([df_mean, row_mean], axis=0)
            df_std = pd.concat([df_std, row_std], axis=0)

        row_mean = df_mean.mean(axis=0).to_frame().T
        row_std = df_std.mean(axis=0).to_frame().T
        df_mean = pd.concat([df_mean, row_mean], axis=0)
        df_std = pd.concat([df_std, row_std], axis=0)

        all_name = [
            "2-1", "2-3", "2-5", "2-7", "2-9", "2-11", "2-13", "2-15",
            "3-1", "3-3", "3-5", "3-7", "3-9", "3-11", "3-13", "3-15",
        ]
        name = all_name[: self._eva_task_nums]
        name.append("Avg")

        df_mean.index = name
        df_std.index = name

        pd.set_option("display.max_columns", 1000)
        pd.options.display.float_format = "{:,.3f}".format

        logging.info("Mean")
        print(df_mean)

        logging.info("Std")
        print(df_std)

        df_mean.to_csv(self._eva_result_path / "result_mean.csv")
        df_std.to_csv(self._eva_result_path / "result_std.csv")

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def is_out_of_bound(self, ego_loc):
        x_out_bound = (
            (ego_loc.x < parking_position.town04_bound["x_min"])
            or (ego_loc.x > parking_position.town04_bound["x_max"])
        )
        y_out_bound = (
            (ego_loc.y < parking_position.town04_bound["y_min"])
            or (ego_loc.y > parking_position.town04_bound["y_max"])
        )
        return x_out_bound or y_out_bound

    def world_tick(self):
        self._world.world_tick()

    def render(self, display):
        self._world.render(display)

    # ------------------------------------------------------------------
    # Public properties consumed by agents
    # ------------------------------------------------------------------

    @property
    def world(self):
        return self._world

    @property
    def agent_need_init(self):
        return self._agent_need_init

    @agent_need_init.setter
    def agent_need_init(self, need_init):
        self._agent_need_init = need_init

    @property
    def inference_time(self):
        return self._inference_time

    @property
    def eva_parking_goal(self):
        if 16 <= self._parking_goal_index <= 30:
            return self._eva_parking_goal

        self._eva_parking_goal[2] = 0
        return self._eva_parking_goal

    @property
    def ego_transform(self):
        return self._ego_transform

    @property
    def eva_result_path(self):
        return self._eva_result_path
