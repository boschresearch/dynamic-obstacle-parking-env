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
import datetime
import math
import os
import pygame

from .ui_utils import get_actor_display_name


class TextOverlay:
    """Base class for text-based UI overlays with rendering capabilities."""
    
    def __init__(self, font, position=(0, 0)):
        self.font = font
        self.position = position
        self._buffer = None
        self._visible = True
    
    def _create_surface(self, width, height):
        self._buffer = pygame.Surface((width, height))
        self._buffer.fill((0, 0, 0, 0))
    
    def show(self):
        self._visible = True
    
    def hide(self):
        self._visible = False
    
    def toggle_visibility(self):
        self._visible = not self._visible
    
    def render(self, display):
        if self._visible and self._buffer is not None:
            display.blit(self._buffer, self.position)


class FadingNotification(TextOverlay):
    """Time-limited text that fades out over time."""
    
    def __init__(self, font, width, height, position=(0, 0)):
        super().__init__(font, position)
        self._create_surface(width, height)
        self.remaining_time = 0.0
        self.max_alpha = 500.0
    
    def display_message(self, text, color=(255, 255, 255), duration=2.0):
        """Queue a message to be displayed and faded out."""
        self._buffer.fill((0, 0, 0, 0))
        rendered = self.font.render(text, True, color)
        self._buffer.blit(rendered, (10, 11))
        self.remaining_time = duration
    
    def update(self, delta_time_ms):
        """Update fade animation based on elapsed time."""
        delta_s = delta_time_ms * 1e-3
        self.remaining_time = max(0.0, self.remaining_time - delta_s)
        alpha = self.max_alpha * (self.remaining_time / max(self.remaining_time, 0.01) if self.remaining_time > 0 else 0)
        if self._buffer is not None:
            self._buffer.set_alpha(int(alpha))


class HelpOverlay(TextOverlay):
    """Informational overlay with keyboard control reference."""
    
    HELP_CONTENT = """
Keyboard Controls:

Vehicle Control:
    W            : throttle
    S            : brake  
    A/D          : steer left/right
    Space        : hand-brake

Task:
    Backspace    : restart task
"""
    
    def __init__(self, font, width, height):
        super().__init__(font)
        lines = self.HELP_CONTENT.strip().split('\n')
        line_height = 18
        total_height = len(lines) * line_height + 12
        
        self._create_surface(780, total_height)
        self.position = (
            0.5 * width - 0.5 * 780,
            0.5 * height - 0.5 * total_height
        )
        
        for idx, line in enumerate(lines):
            rendered = self.font.render(line, True, (255, 255, 255))
            self._buffer.blit(rendered, (22, idx * line_height))
        
        self._buffer.set_alpha(220)
        self._visible = False


class InfoPanel:
    """Renders detailed telemetry and vehicle state information."""
    
    INFO_PANEL_WIDTH = 220
    INFO_PANEL_HEIGHT_FACTOR = 1.0  # Full height
    BAR_OFFSET = 100
    BAR_WIDTH = 106
    BAR_HEIGHT = 6
    ROW_HEIGHT = 18
    
    def __init__(self, font, display_dims):
        self.font = font
        self.display_dims = display_dims
        self._info_items = []
        self._visible = True
    
    def update_telemetry(self, world, clock):
        """Update all telemetry from world state."""
        self._info_items = []
        
        # FPS and timing info
        compass = math.degrees(world.sensor_data_frame['imu'].compass)
        
        self._info_items.append(['Server:  % 16.0f FPS' % world.hud.server_clock.get_fps()])
        self._info_items.append(['Client:  % 16.0f FPS' % clock.get_fps()])
        self._info_items.append([''])
        
        # Vehicle and map
        vehicle_name = get_actor_display_name(world.player, truncate=20)
        self._info_items.append(['Vehicle: % 20s' % vehicle_name])
        self._info_items.append(['Map:     % 20s' % world.map.name])
        
        # Simulation time
        elapsed = world.hud.simulation_time
        time_str = str(datetime.timedelta(seconds=int(elapsed)))
        self._info_items.append(['Simulation time: % 12s' % time_str])
        self._info_items.append([''])
        
        # Physics
        v = world.player.get_velocity()
        speed_kmh = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self._info_items.append(['Speed:   % 15.0f km/h' % speed_kmh])
        
        acc = world.sensor_data_frame['imu'].accelerometer
        self._info_items.append(['Accelero: (%5.1f,%5.1f,%5.1f)' % (acc.x, acc.y, acc.z)])
        
        # Position
        t = world.player.get_transform()
        self._info_items.append(['Location   x: %.6f' % t.location.x])
        self._info_items.append(['Location   y: %.6f' % t.location.y])
        self._info_items.append(['Location   z: %.6f' % t.location.z])
        self._info_items.append(['Rotation yaw: %.6f' % t.rotation.yaw])
        self._info_items.append([''])
        
        # Control state
        c = world.player.get_control()
        if isinstance(c, carla.VehicleControl):
            self._info_items.append([('Throttle:', c.throttle, 0.0, 1.0)])
            self._info_items.append([('Steer:', c.steer, -1.0, 1.0)])
            self._info_items.append([('Brake:', c.brake, 0.0, 1.0)])
            self._info_items.append([('Reverse:', c.reverse)])
            self._info_items.append([('Hand brake:', c.hand_brake)])
            self._info_items.append([('Manual:', c.manual_gear_shift)])
            gear_str = {-1: 'R', 0: 'N'}.get(c.gear, c.gear)
            self._info_items.append(['Gear:        %s' % gear_str])
        
        # Parking metrics
        self._info_items.append([''])
        self._info_items.append(['Distance x diff: % .6f' % world.x_diff_to_goal])
        self._info_items.append(['Distance y diff: % .6f' % world.y_diff_to_goal])
        self._info_items.append(['Distance   diff: % .6f' % world.distance_diff_to_goal])
        self._info_items.append(['Rotation   diff: % .6f' % world.rotation_diff_to_goal])
    
    def render(self, display):
        """Draw info panel to display."""
        if not self._visible or not self._info_items:
            return
        
        # Semi-transparent background
        bg = pygame.Surface((self.INFO_PANEL_WIDTH, self.display_dims[1]))
        bg.set_alpha(100)
        display.blit(bg, (0, 0))
        
        # Draw telemetry items
        v_offset = 4
        for item in self._info_items:
            if v_offset + self.ROW_HEIGHT > self.display_dims[1]:
                break
            
            if isinstance(item, list):
                if len(item) == 1 and isinstance(item[0], str):
                    # Text item
                    if item[0]:
                        surface = self.font.render(item[0], True, (255, 255, 255))
                        display.blit(surface, (8, v_offset))
                elif len(item) > 1:
                    # Graph item (list of values)
                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
            
            elif isinstance(item, tuple):
                # Tuple: (label, value, min, max) for bars or (label, bool) for checkboxes
                if len(item) == 2:
                    # Boolean indicator
                    is_active = item[1]
                    rect = pygame.Rect((self.BAR_OFFSET, v_offset + 8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if is_active else 1)
                    label_surf = self.font.render(item[0], True, (255, 255, 255))
                    display.blit(label_surf, (8, v_offset))
                
                elif len(item) >= 4:
                    # Progress bar
                    label = item[0]
                    value = item[1]
                    min_val = item[2]
                    max_val = item[3]
                    
                    # Draw label
                    label_surf = self.font.render(label, True, (255, 255, 255))
                    display.blit(label_surf, (8, v_offset))
                    
                    # Draw bar background
                    rect_bg = pygame.Rect((self.BAR_OFFSET, v_offset + 8), (self.BAR_WIDTH, self.BAR_HEIGHT))
                    pygame.draw.rect(display, (255, 255, 255), rect_bg, 1)
                    
                    # Draw fill
                    norm_value = (value - min_val) / (max_val - min_val)
                    if min_val < 0.0:
                        # Center bar at 0
                        x = self.BAR_OFFSET + norm_value * (self.BAR_WIDTH - 6)
                        rect = pygame.Rect((x, v_offset + 8), (6, self.BAR_HEIGHT))
                    else:
                        # Left-aligned bar
                        width = norm_value * self.BAR_WIDTH
                        rect = pygame.Rect((self.BAR_OFFSET, v_offset + 8), (width, self.BAR_HEIGHT))
                    pygame.draw.rect(display, (255, 255, 255), rect)
            
            v_offset += self.ROW_HEIGHT
    
    def toggle_visibility(self):
        self._visible = not self._visible


class HUD:
    """Main heads-up display coordinator with telemetry and overlays."""
    
    def __init__(self, width, height):
        self.dim = (width, height)
        
        # Font setup
        fonts = pygame.font.get_fonts()
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        
        # UI components
        self.notification = FadingNotification(self._font_mono, width, 40, (0, height - 40))
        self.help_panel = HelpOverlay(self._font_mono, width, height)
        self.info_panel = InfoPanel(self._font_mono, (width, height))
        
        # Timing and state
        self.server_clock = pygame.time.Clock()
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0.0
    
    def on_world_tick(self, timestamp):
        """Called by CARLA world tick event."""
        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def tick(self, world, clock):
        """Update all display state for current frame."""
        self.notification.update(clock.get_time())
        self.info_panel.update_telemetry(world, clock)
    
    def show_message(self, text, duration=2.0):
        """Display a temporary notification message."""
        self.notification.display_message(text, duration=duration)
    
    def show_error(self, text):
        """Display an error notification."""
        self.notification.display_message(f'Error: {text}', color=(255, 0, 0), duration=5.0)
    
    def toggle_info(self):
        """Toggle info panel visibility."""
        self.info_panel.toggle_visibility()
    
    def toggle_help(self):
        """Toggle help overlay visibility."""
        self.help_panel.toggle_visibility()
    
    # Backward compatibility aliases
    def notification_text(self, text, seconds=2.0):
        self.show_message(text, duration=seconds)
    
    def error(self, text):
        self.show_error(text)
    
    @property
    def help(self):
        """Backward compatibility property for help panel."""
        return self.help_panel
    
    def render(self, display):
        """Render all HUD elements to display."""
        self.info_panel.render(display)
        self.notification.render(display)
        self.help_panel.render(display)
