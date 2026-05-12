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
import cv2
import logging
import numpy as np
import pygame
import weakref
from carla import ColorConverter as cc

from .ui_utils import get_actor_display_name


class EventHandler:
    """Base class for sensor event callback handling."""
    
    def __init__(self, hud_ref):
        self.hud = hud_ref
        self.triggered = False
    
    def handle_event(self, event):
        raise NotImplementedError
    
    def reset(self):
        self.triggered = False


class CollisionEventHandler(EventHandler):
    """Manages collision detection and reporting."""
    
    def handle_event(self, event):
        self.triggered = True
        actor_name = get_actor_display_name(event.other_actor)
        logging.info(f'Collision with {actor_name}')
        if self.hud:
            self.hud.show_message(f'Collision with {actor_name}; Restart task')


class ImageProcessor:
    """Processes sensor image data and converts to renderable format."""
    
    def __init__(self, display_dims):
        self.display_dims = display_dims
        self.lidar_range = 50.0
    
    def process_lidar(self, raw_image):
        """Convert LIDAR point cloud to pygame surface."""
        points = np.frombuffer(raw_image.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))
        lidar_2d = points[:, :2]
        
        # Project to display
        scale_factor = min(self.display_dims) / (2.0 * self.lidar_range)
        lidar_2d *= scale_factor
        lidar_2d += np.array([self.display_dims[0] / 2, self.display_dims[1] / 2])
        lidar_2d = np.abs(lidar_2d).astype(np.int32)
        
        # Create image
        img = np.zeros((*self.display_dims, 3), dtype=np.uint8)
        valid_idx = np.all((lidar_2d >= 0) & (lidar_2d < self.display_dims), axis=1)
        img[lidar_2d[valid_idx, 1], lidar_2d[valid_idx, 0]] = (255, 255, 255)
        
        return pygame.surfarray.make_surface(img)
    
    def process_dvs(self, raw_image):
        """Convert DVS events to pygame surface."""
        dvs_events = np.frombuffer(raw_image.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)
        ]))
        
        img = np.zeros((raw_image.height, raw_image.width, 3), dtype=np.uint8)
        img[dvs_events['y'], dvs_events['x'], dvs_events['pol'].astype(int) * 2] = 255
        
        return pygame.surfarray.make_surface(img.swapaxes(0, 1))
    
    def process_camera(self, raw_image, converter):
        """Convert camera image to pygame surface and numpy array."""
        raw_image.convert(converter)
        array = np.frombuffer(raw_image.raw_data, dtype=np.uint8)
        array = array.reshape((raw_image.height, raw_image.width, 4))[:, :, :3]
        array = array[:, :, ::-1]  # BGR to RGB
        
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        cv_array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        return surface, cv_array


class SensorConfiguration:
    """Manages sensor type definitions and blueprint setup."""
    
    SENSOR_SPECS = {
        'RGB': {
            'type': 'sensor.camera.rgb',
            'converter': cc.Raw,
            'label': 'Camera RGB',
            'params': {}
        },
        'DEPTH_RAW': {
            'type': 'sensor.camera.depth',
            'converter': cc.Raw,
            'label': 'Camera Depth (Raw)',
            'params': {}
        },
        'DEPTH_GRAY': {
            'type': 'sensor.camera.depth',
            'converter': cc.Depth,
            'label': 'Camera Depth (Gray Scale)',
            'params': {}
        },
        'DEPTH_LOG': {
            'type': 'sensor.camera.depth',
            'converter': cc.LogarithmicDepth,
            'label': 'Camera Depth (Logarithmic)',
            'params': {}
        },
        'SEMANTIC_RAW': {
            'type': 'sensor.camera.semantic_segmentation',
            'converter': cc.Raw,
            'label': 'Semantic Segmentation (Raw)',
            'params': {}
        },
        'SEMANTIC_CITY': {
            'type': 'sensor.camera.semantic_segmentation',
            'converter': cc.CityScapesPalette,
            'label': 'Semantic Segmentation (CityScapes)',
            'params': {}
        },
        'LIDAR': {
            'type': 'sensor.lidar.ray_cast',
            'converter': None,
            'label': 'Lidar (Ray-Cast)',
            'params': {'range': '50'}
        },
        'DVS': {
            'type': 'sensor.camera.dvs',
            'converter': cc.Raw,
            'label': 'Dynamic Vision Sensor',
            'params': {}
        },
        'RGB_DISTORTED': {
            'type': 'sensor.camera.rgb',
            'converter': cc.Raw,
            'label': 'Camera RGB (Distorted)',
            'params': {
                'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'
            }
        }
    }


class CollisionSensor:
    """Collision detection system."""
    
    def __init__(self, vehicle, hud_ref):
        self.vehicle = vehicle
        self.handler = CollisionEventHandler(hud_ref)
        self.sensor = None
        self._initialize_sensor()
    
    def _initialize_sensor(self):
        world = self.vehicle.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        
        weak_handler = weakref.ref(self.handler)
        self.sensor.listen(lambda evt: self._on_collision(weak_handler, evt))
    
    @staticmethod
    def _on_collision(weak_handler, event):
        handler = weak_handler()
        if handler:
            handler.handle_event(event)
    
    @property
    def is_collision(self):
        return self.handler.triggered
    
    @is_collision.setter
    def is_collision(self, value):
        self.handler.triggered = value
    
    def cleanup(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


class CameraManager:
    """Multi-camera management and visualization."""
    
    def __init__(self, vehicle, hud_ref, gamma):
        self.vehicle = vehicle
        self.hud = hud_ref
        self.gamma = gamma
        
        self.sensor = None
        self.current_surface = None
        self.image_buffer = []
        self.active_index = None
        self.transform_index = 0
        
        self.processor = ImageProcessor(hud_ref.dim)
        self.config = SensorConfiguration()
        
        self._sensor_list = list(self.config.SENSOR_SPECS.keys())
        self._mount_points = self._setup_mount_points()
        self._blueprints = {}
        
        self._initialize_blueprints()
    
    def _setup_mount_points(self):
        """Define camera mounting positions relative to vehicle."""
        margin = 0.5 + self.vehicle.bounding_box.extent.y
        Att = carla.AttachmentType
        
        return [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Att.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Att.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Att.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Att.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-margin, z=0.5)), Att.Rigid)
        ]
    
    def _initialize_blueprints(self):
        """Configure sensor blueprints from specifications."""
        world = self.vehicle.get_world()
        bp_lib = world.get_blueprint_library()
        
        for name, spec in self.config.SENSOR_SPECS.items():
            bp = bp_lib.find(spec['type'])
            
            if 'camera' in spec['type']:
                bp.set_attribute('image_size_x', str(self.hud.dim[0]))
                bp.set_attribute('image_size_y', str(self.hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(self.gamma))
            
            for key, value in spec['params'].items():
                bp.set_attribute(key, value)
                if key == 'range':
                    self.processor.lidar_range = float(value)
            
            self._blueprints[name] = bp
    
    def set_sensor(self, index, notify=True, force_respawn=False):
        """Switch to a different sensor."""
        index = index % len(self._sensor_list)
        sensor_name = self._sensor_list[index]
        
        needs_respawn = (self.active_index is None or 
                        force_respawn or 
                        self._sensor_list[self.active_index] != sensor_name)
        
        if needs_respawn:
            if self.sensor:
                self.sensor.destroy()
                self.current_surface = None
            
            spec = self.config.SENSOR_SPECS[sensor_name]
            mount_transform, mount_type = self._mount_points[self.transform_index]
            
            self.sensor = self.vehicle.get_world().spawn_actor(
                self._blueprints[sensor_name],
                mount_transform,
                attach_to=self.vehicle,
                attachment_type=mount_type
            )
            
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda img: CameraManager._process_image(weak_self, img))
        
        if notify:
            self.hud.show_message(self.config.SENSOR_SPECS[sensor_name]['label'])
        
        self.active_index = index
    
    def toggle_camera(self):
        """Switch to next camera mount position."""
        self.transform_index = (self.transform_index + 1) % len(self._mount_points)
        self.set_sensor(self.active_index or 0, notify=False, force_respawn=True)
    
    def next_sensor(self):
        """Switch to next sensor type."""
        next_idx = (self.active_index + 1) if self.active_index is not None else 0
        self.set_sensor(next_idx)
    
    @staticmethod
    def _process_image(weak_self, image):
        """Process incoming sensor data."""
        cam = weak_self()
        if not cam or cam.active_index is None:
            return
        
        sensor_name = cam._sensor_list[cam.active_index]
        spec = cam.config.SENSOR_SPECS[sensor_name]
        sensor_type = spec['type']
        
        if 'lidar' in sensor_type:
            cam.current_surface = cam.processor.process_lidar(image)
        elif 'dvs' in sensor_type:
            cam.current_surface = cam.processor.process_dvs(image)
        else:
            surface, cv_array = cam.processor.process_camera(image, spec['converter'])
            cam.current_surface = surface
            if 'rgb' in sensor_type:
                cam.image_buffer.append(cv_array)
    
    def render(self, display):
        """Draw current camera view on display."""
        if self.current_surface:
            display.blit(self.current_surface, (0, 0))
    
    def save_video(self, output_path):
        """Save buffered images as video."""
        if not self.image_buffer:
            return
        
        height, width = self.image_buffer[0].shape[:2]
        output_file = output_path / 'task.avi'
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(output_file), fourcc, 20.0, (width, height))
        for frame in self.image_buffer:
            writer.write(frame)
        writer.release()
    
    def clear_saved_images(self):
        """Clear image buffer."""
        self.image_buffer.clear()
    
    def cleanup(self):
        """Clean up resources."""
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
        self.image_buffer.clear()
