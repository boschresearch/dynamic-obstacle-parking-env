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

import carla
import logging
import numpy as np
import re
import sys
from queue import Empty, Queue
import yaml

from . import parking_layout as parking_position
from .bev_render import BevRender
from .hud import HUD
from .npc_manager import NpcManager
from .sensor_manager import CameraManager, CollisionSensor
from .ui_utils import get_actor_display_name


class WeatherManager:
    """Manages weather system and presets."""
    
    def __init__(self, carla_world):
        self.world = carla_world
        self.presets = self._load_presets()
        self.current_index = 0
    
    @staticmethod
    def _load_presets():
        """Load all available weather presets."""
        names = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        presets = [(getattr(carla.WeatherParameters, x), x) for x in names]
        if len(presets) > 5:
            presets.pop(5)  # Remove weather preset at index 5
        return presets
    
    def next_weather(self, reverse=False):
        """Cycle to next weather preset."""
        self.current_index %= len(self.presets)
        preset_params, preset_name = self.presets[self.current_index]
        self.world.set_weather(preset_params)
        self.current_index += -1 if reverse else 1
        return preset_name


class SimulationState:
    """Tracks global simulation state."""
    
    def __init__(self):
        self.step = -1
        self.collision_detected = False
        self.keyboard_restart_requested = False
        self.need_ego_state_init = True
    
    def increment_step(self):
        self.step += 1
    
    def reset(self):
        self.step = -1
        self.collision_detected = False


class SensorDataCollector:
    """Manages sensor data collection and queuing."""
    
    def __init__(self):
        self.queue = Queue()
        self.frame_data = {}
        self.sensors = []
    
    def add_sensor(self, sensor):
        self.sensors.append(sensor)
    
    def queue_data(self, data, sensor_name):
        self.queue.put((data, sensor_name))
    
    def collect_frame(self, vehicle, sensor_count):
        """Collect all sensor data for current frame."""
        self.frame_data.clear()
        
        # Vehicle state
        self.frame_data['veh_transform'] = vehicle.get_transform()
        self.frame_data['veh_velocity'] = vehicle.get_velocity()
        self.frame_data['veh_control'] = vehicle.get_control()
        
        # Sensor data
        for _ in range(sensor_count):
            try:
                data, name = self.queue.get(block=True, timeout=1.0)
                self.frame_data[name] = data
            except Empty:
                logging.error(f"Sensor {name} data missed")
    
    def cleanup(self):
        """Clean up all sensors."""
        for sensor in self.sensors:
            if sensor:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        self.queue = Queue()


class CameraCalibrationManager:
    """Manages camera intrinsics and extrinsics."""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.cam_config = {}
        self.cam_specs = {}
        self.intrinsic = None
        self.veh2cam_dict = {}
        self.cam2pixel_transform = self._create_cam2pixel_matrix()
    
    @staticmethod
    def _create_cam2pixel_matrix():
        """Create transformation matrix from camera to pixel coordinates."""
        return np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=float)
    
    def load_config(self):
        """Load camera configuration from YAML."""
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
            self.cam_specs = data.get("cam_specs", {})
    
    def compute_intrinsics(self, width, height, fov):
        """Compute camera intrinsic matrix."""
        focal = width / (2 * np.tan(fov * np.pi / 360))
        self.intrinsic = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def compute_camera_transforms(self):
        """Compute camera-to-vehicle transforms."""
        for cam_id, spec in self.cam_specs.items():
            sensor_type = spec.get('type', '')
            if 'sensor.camera.rgb' in sensor_type:
                cam2veh = carla.Transform(
                    carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                    carla.Rotation(yaw=spec['yaw'], pitch=spec['pitch'], roll=spec['roll'])
                )
                veh2cam = self.cam2pixel_transform @ np.array(cam2veh.get_inverse_matrix())
                self.veh2cam_dict[cam_id] = veh2cam


class VehicleManager:
    """Manages ego vehicle and spectator."""
    
    INITIAL_SPECTATOR_LOC = carla.Location(x=283.825165, y=-210.039487, z=35.0)
    INITIAL_SPECTATOR_ROT = carla.Rotation(pitch=-90)
    
    def __init__(self, carla_world):
        self.world = carla_world
        self.player = None
        self.spectator = None
    
    def spawn_vehicle(self, spawn_point):
        """Spawn ego vehicle at specified location."""
        bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        self.player = self.world.spawn_actor(bp, spawn_point)
        return self.player
    
    def initialize_spectator(self):
        """Initialize spectator camera."""
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(self.INITIAL_SPECTATOR_LOC, self.INITIAL_SPECTATOR_ROT)
        )
    
    def reset_player(self, transform):
        """Reset player to specified transform."""
        self.player.set_transform(transform)
        self.player.apply_control(carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=1.0, hand_brake=True
        ))
        self.player.set_target_velocity(carla.Vector3D(0, 0, 0))
    
    def update_spectator(self, ego_location):
        """Update spectator to follow ego vehicle."""
        if self.spectator:
            chase_transform = carla.Transform(
                ego_location + carla.Location(z=30),
                carla.Rotation(pitch=-90)
            )
            self.spectator.set_transform(chase_transform)


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    """Global sensor callback function."""
    sensor_queue.put((sensor_data, sensor_name))


class World:
    """Main simulation world coordinator."""
    
    def __init__(self, carla_world, args):
        self._init_world_settings(carla_world)
        self._init_map_and_parking(args)
        
        # Core subsystems
        self.hud = HUD(args.width, args.height)
        self.state = SimulationState()
        self.weather_manager = WeatherManager(self._world)
        self.vehicle_manager = VehicleManager(self._world)
        self.sensor_collector = SensorDataCollector()
        self.calibration_mgr = CameraCalibrationManager(args.sensor_config_path)

        # Actor management
        self._actor_list = []
        
        # Sensors and rendering
        self.collision_sensor = None
        self.camera_manager = None
        self.bev_render = None
        self.bev_render_device = args.bev_render_device
        
        # NPC management
        self.npc_manager = NpcManager(
            self._world,
            self._parking_spawn_points,
            self._actor_list,
            args.perfect_parking,
            args.npc_mode,
        )
        
        # Parking state
        self.all_parking_goals = []
        self.goal_metrics = {
            'x_diff': 0, 'y_diff': 0,
            'distance_diff': 0, 'rotation_diff': 0
        }
        
        # Gamma for camera processing
        self.gamma = args.gamma
        self.shuffle_weather = args.shuffle_weather
        
        self._world.on_tick(self.hud.on_world_tick)
    
    def _init_world_settings(self, carla_world):
        """Initialize CARLA world settings."""
        self._world = carla_world
        settings = self._world.get_settings()
        settings.fixed_delta_seconds = 1.0 / 30  # 30 FPS
        settings.synchronous_mode = True
        self._world.apply_settings(settings)
    
    def _init_map_and_parking(self, args):
        """Load map and parking locations."""
        if args.map != 'Town04_Opt':
            logging.error('Invalid map: %s', args.map)
            sys.exit(1)
        
        self._parking_spawn_points = parking_position.parking_vehicle_locations_Town04.copy()
        
        try:
            self._map = self._world.get_map()
        except RuntimeError as error:
            logging.error('Map load error: %s', error)
            logging.error('Ensure map file exists and is correct')
            sys.exit(1)

    def _reset_player_state(self, ego_transform):
        """Reset player vehicle to initial state."""
        self.vehicle_manager.reset_player(ego_transform)
    
    def restart(self, ego_transform):
        """Full restart of simulation."""
        self._reset_player_state(ego_transform)
        actor_name = get_actor_display_name(self.vehicle_manager.player)
        self.hud.show_message(actor_name)
        
        if self.shuffle_weather:
            preset_name = self.weather_manager.next_weather()
            self.hud.show_message(f'Weather: {preset_name}')
        
        self.camera_manager.clear_saved_images()
        self.state.need_ego_state_init = True
    
    def init_ego_vehicle(self, ego_transform):
        """Initialize ego vehicle at specified location."""
        self.vehicle_manager.spawn_vehicle(ego_transform)
        self.bev_render = BevRender(self, self.bev_render_device)
        self.vehicle_manager.initialize_spectator()
        
        actor_name = get_actor_display_name(self.vehicle_manager.player)
        self.hud.show_message(actor_name)
    
    def init_npc(self, seed, target_index):
        """Initialize NPC vehicles."""
        self.all_parking_goals = self.npc_manager.init_npc(seed, target_index)
    
    def init_sensors(self):
        """Initialize all sensors."""
        player = self.vehicle_manager.player
        self.collision_sensor = CollisionSensor(player, self.hud)
        self.camera_manager = CameraManager(player, self.hud, self.gamma)
        self.camera_manager.transform_index = 0
        self.camera_manager.set_sensor(0, notify=False)
        
        # Setup additional sensors
        self._setup_vehicle_sensors()
    
    def _setup_vehicle_sensors(self):
        """Setup GNSS, IMU, and camera sensors."""
        player = self.vehicle_manager.player
        
        # GNSS
        bp_gnss = self._world.get_blueprint_library().find('sensor.other.gnss')
        gnss = self._world.spawn_actor(bp_gnss, carla.Transform(), 
                                       attach_to=player, 
                                       attachment_type=carla.AttachmentType.Rigid)
        gnss.listen(lambda data: sensor_callback(data, self.sensor_collector.queue, "gnss"))
        self.sensor_collector.add_sensor(gnss)
        
        # IMU
        bp_imu = self._world.get_blueprint_library().find('sensor.other.imu')
        imu = self._world.spawn_actor(bp_imu, carla.Transform(),
                                     attach_to=player,
                                     attachment_type=carla.AttachmentType.Rigid)
        imu.listen(lambda data: sensor_callback(data, self.sensor_collector.queue, "imu"))
        self.sensor_collector.add_sensor(imu)
        
        # Load calibration
        self.calibration_mgr.load_config()
        self.calibration_mgr.cam_config = {
            'width': 400, 'height': 300, 'fov': 100
        }
        self.calibration_mgr.compute_intrinsics(400, 300, 100)
        
        # Spawn configured cameras
        for cam_id, spec in self.calibration_mgr.cam_specs.items():
            if spec.get('type', '').startswith('sensor.camera.'):
                self._spawn_configured_camera(cam_id, spec)
        
        self.calibration_mgr.compute_camera_transforms()
    
    def _spawn_configured_camera(self, cam_id, spec):
        """Spawn a configured camera sensor."""
        bp_lib = self._world.get_blueprint_library()
        bp = bp_lib.find(spec['type'])
        
        # Configure blueprint
        width = spec.get('width', self.calibration_mgr.cam_config['width'])
        height = spec.get('height', self.calibration_mgr.cam_config['height'])
        fov = spec.get('fov', self.calibration_mgr.cam_config['fov'])
        
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(fov))
        
        # Spawn sensor
        sensor_transform = carla.Transform(
            carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
            carla.Rotation(pitch=spec['pitch'], roll=spec['roll'], yaw=spec['yaw'])
        )
        
        camera = self._world.spawn_actor(
            bp, sensor_transform,
            attach_to=self.vehicle_manager.player,
            attachment_type=carla.AttachmentType.Rigid
        )
        camera.listen(lambda data: sensor_callback(data, self.sensor_collector.queue, cam_id))
        self.sensor_collector.add_sensor(camera)
    
    def soft_restart(self, ego_transform):
        """Soft restart preserving some state."""
        self._reset_player_state(ego_transform)
        self.camera_manager.clear_saved_images()
        self.state.need_ego_state_init = True
    
    def tick(self, clock, target_index):
        """Advance simulation by one tick."""
        # Update NPCs
        self.npc_manager.update(target_index, self.vehicle_manager.player)
        
        # Collect sensor data
        self.sensor_collector.collect_frame(
            self.vehicle_manager.player,
            len(self.sensor_collector.sensors)
        )
        
        # Update HUD
        self.hud.tick(self, clock)
        
        # Draw target parking spot
        target_goal = self._parking_spawn_points[target_index]
        self._world.debug.draw_string(target_goal, 'T', draw_shadow=True, 
                                      color=carla.Color(255, 0, 0))
        
        # Update spectator
        self.vehicle_manager.update_spectator(self.vehicle_manager.player.get_transform().location)
        
        # Increment step
        self.state.increment_step()
        
        # Check for restart conditions
        if self.collision_sensor.is_collision or self.state.keyboard_restart_requested:
            self.collision_sensor.is_collision = False
            self.state.keyboard_restart_requested = False
            return True
        
        return False
    
    def render(self, display):
        """Render world state to display."""
        self.camera_manager.render(display)
        self.hud.render(display)
    
    def destroy(self):
        """Destroy all world resources."""
        self.sensor_collector.cleanup()
        
        if self.camera_manager:
            self.camera_manager.cleanup()
        if self.collision_sensor:
            self.collision_sensor.cleanup()
        
        if self.vehicle_manager.player:
            self.vehicle_manager.player.destroy()
        
        for actor in self._actor_list:
            actor.destroy()
        self._actor_list.clear()
        
        self.npc_manager.reset_dynamic_npcs()
        self.state.reset()
    
    def soft_destroy(self):
        """Destroy simulation state while preserving world."""
        for actor in self._actor_list:
            actor.destroy()
        self._actor_list.clear()
        self.npc_manager.reset_dynamic_npcs()
        self.state.reset()
        self.all_parking_goals.clear()
    
    def clear_npcs(self):
        """Clear all NPC scenarios."""
        self.npc_manager.clear_drive_out_npc()
        self.npc_manager.clear_block_npc()
        self.npc_manager.clear_follow_npc()
    
    def set_npc_attempt_seed(self, attempt_index):
        """Set random seed for NPC behavior."""
        self.npc_manager.set_attempt_seed(attempt_index)
    
    # ============= Property accessors for backward compatibility =============
    
    @property
    def map(self):
        return self._map
    
    @property
    def step(self):
        return self.state.step
    
    @property
    def player(self):
        return self.vehicle_manager.player
    
    @property
    def world(self):
        return self._world
    
    @property
    def sensor_data_frame(self):
        return self.sensor_collector.frame_data
    
    @property
    def bev_state(self):
        return self.bev_render.get_bev_states()
    
    @property
    def ego_transform(self):
        return self.vehicle_manager.player.get_transform()
    
    @property
    def cam_config(self):
        return self.calibration_mgr.cam_config
    
    @property
    def intrinsic(self):
        return self.calibration_mgr.intrinsic
    
    @property
    def veh2cam_dict(self):
        return self.calibration_mgr.veh2cam_dict
    
    @property
    def keyboard_restart_task(self):
        return self.state.keyboard_restart_requested
    
    @keyboard_restart_task.setter
    def keyboard_restart_task(self, value):
        self.state.keyboard_restart_requested = value
    
    @property
    def need_init_ego_state(self):
        return self.state.need_ego_state_init
    
    @need_init_ego_state.setter
    def need_init_ego_state(self, value):
        self.state.need_ego_state_init = value
    
    @property
    def x_diff_to_goal(self):
        return self.goal_metrics['x_diff']
    
    @x_diff_to_goal.setter
    def x_diff_to_goal(self, diff):
        self.goal_metrics['x_diff'] = diff
    
    @property
    def y_diff_to_goal(self):
        return self.goal_metrics['y_diff']
    
    @y_diff_to_goal.setter
    def y_diff_to_goal(self, diff):
        self.goal_metrics['y_diff'] = diff
    
    @property
    def distance_diff_to_goal(self):
        return self.goal_metrics['distance_diff']
    
    @distance_diff_to_goal.setter
    def distance_diff_to_goal(self, diff):
        self.goal_metrics['distance_diff'] = diff
    
    @property
    def rotation_diff_to_goal(self):
        return self.goal_metrics['rotation_diff']
    
    @rotation_diff_to_goal.setter
    def rotation_diff_to_goal(self, diff):
        self.goal_metrics['rotation_diff'] = diff
    
    @property
    def all_parking_goals(self):
        return self._all_parking_goals
    
    @all_parking_goals.setter
    def all_parking_goals(self, goals):
        self._all_parking_goals = goals
    
    @property
    def init_drive_out_npc(self):
        return self.npc_manager.init_drive_out_npc
    
    @init_drive_out_npc.setter
    def init_drive_out_npc(self, value):
        self.npc_manager.init_drive_out_npc = value
    
    @property
    def init_block_npc(self):
        return self.npc_manager.init_block_npc
    
    @init_block_npc.setter
    def init_block_npc(self, value):
        self.npc_manager.init_block_npc = value
    
    @property
    def init_follow_npc(self):
        return self.npc_manager.init_follow_npc
    
    @init_follow_npc.setter
    def init_follow_npc(self, value):
        self.npc_manager.init_follow_npc = value
    
    def render_BEV_from_state(self, state):
        """Render bird's eye view from state."""
        return self.bev_render.render_BEV_from_state(state)
    
    def render_BEV(self):
        """Render current bird's eye view."""
        return self.bev_render.render_BEV()
    
    def save_video(self, path):
        """Save recorded video."""
        self.camera_manager.save_video(path)
    
    def world_tick(self):
        """Advance CARLA world by one tick."""
        self._world.tick()
