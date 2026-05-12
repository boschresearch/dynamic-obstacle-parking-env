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

import pygame
import numpy as np


def encode_npy_to_pil(bev_array):
    c, w, h = bev_array.shape

    img = np.zeros([3, w, h]).astype('uint8')
    bev = np.ceil(bev_array).astype('uint8')

    for i in range(c):
        if 0 <= i <= 4:
            # road, lane, light 
            img[0] = img[0] | (bev[i] << (8 - i - 1))
        elif 5 <= i <= 9:
            img[1] = img[1] | (bev[i] << (8 - (i - 5) - 1))
        elif 10 <= i <= 14:
            img[2] = img[2] | (bev[i] << (8 - (i - 10) - 1))

    return img


def get_actor_display_name(actor, truncate=250):
    """Extract human-readable actor name from type identifier."""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class ControlVisualization:
    """Renders vehicle control state indicators on the display surface."""
    
    # Visual configuration for rendering
    CONTROL_MARGIN = 30  # Right margin from display edge
    INDICATOR_MARGIN = 50  # Top margin from display edge
    HISTOGRAM_WIDTH = 15
    INDICATOR_SPACING = 30
    
    # Color definitions
    THROTTLE_COLOR = (0, 255, 0)
    BRAKE_COLOR = (255, 0, 0)
    TEXT_COLOR = (0, 0, 0)
    
    def __init__(self, font):
        self.font = font
        self._steering_wheel_asset = None
    
    def set_steering_wheel_image(self, image):
        """Set the steering wheel image asset for display."""
        self._steering_wheel_asset = image
    
    def render(self, display_surface, control_dict, display_width, display_height):
        """Draw all control indicators on the display."""
        # Calculate base positions
        base_x = display_width - self.CONTROL_MARGIN
        base_y = display_height - self.INDICATOR_MARGIN
        
        # Render indicators
        self._render_throttle_bar(display_surface, control_dict, base_x, base_y)
        self._render_brake_bar(display_surface, control_dict, base_x - self.INDICATOR_SPACING, base_y)
        self._render_steering_wheel(display_surface, control_dict, base_x - 80, base_y - 40)
        self._render_reverse_indicator(display_surface, control_dict, base_x - 140, base_y)
    
    def _render_throttle_bar(self, surface, control, x, y):
        """Draw throttle level indicator."""
        height = int((control['throttle'] * 200) * 0.8)
        rect = pygame.Rect(x, y - height, self.HISTOGRAM_WIDTH, height)
        pygame.draw.rect(surface, self.THROTTLE_COLOR, rect)
        label = self.font.render("T", True, self.THROTTLE_COLOR)
        surface.blit(label, (x + 2, y + 10))
    
    def _render_brake_bar(self, surface, control, x, y):
        """Draw brake level indicator."""
        height = int((control['brake'] * 100) * 0.8)
        rect = pygame.Rect(x, y - height, self.HISTOGRAM_WIDTH, height)
        pygame.draw.rect(surface, self.BRAKE_COLOR, rect)
        label = self.font.render("B", True, self.BRAKE_COLOR)
        surface.blit(label, (x + 2, y + 10))
    
    def _render_steering_wheel(self, surface, control, x, y):
        """Draw steering wheel rotated by steering angle."""
        if self._steering_wheel_asset is None:
            return
        
        angle = -control['steer'] * 90
        rotated = pygame.transform.rotate(self._steering_wheel_asset, angle)
        rect = rotated.get_rect(center=(x, y))
        surface.blit(rotated, rect)
        
        label = self.font.render("S", True, self.TEXT_COLOR)
        surface.blit(label, (x - 4, y + 50))
    
    def _render_reverse_indicator(self, surface, control, x, y):
        """Draw reverse gear indicator (filled square when active)."""
        is_active = bool(control['reverse'])
        rect = pygame.Rect(x, y - 10, 10, 10)
        
        # Draw filled if reverse is active, outline otherwise
        line_width = 0 if is_active else 1
        pygame.draw.rect(surface, self.TEXT_COLOR, rect, line_width)
        
        label = self.font.render("R", True, self.TEXT_COLOR)
        surface.blit(label, (x, y + 10))


def show_control_info(window, control, steering_wheel_image, width, height, font):
    """Legacy function interface for backward compatibility."""
    visualizer = ControlVisualization(font)
    visualizer.set_steering_wheel_image(steering_wheel_image)
    visualizer.render(window, control, width, height)


