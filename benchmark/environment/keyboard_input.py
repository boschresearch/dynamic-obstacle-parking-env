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

try:
    import pygame
    from pygame.locals import (
        K_BACKSPACE, K_ESCAPE, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT,
        K_w, K_a, K_s, K_d
    )
except ImportError:
    raise RuntimeError('pygame not installed')


class VehicleInputState:
    """Maintains vehicle input state across frames."""
    
    def __init__(self):
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.hand_brake = False
    
    def apply_to_vehicle(self, vehicle):
        """Apply stored state to vehicle."""
        control = carla.VehicleControl(
            throttle=self.throttle,
            brake=self.brake,
            steer=self.steer,
            hand_brake=self.hand_brake
        )
        vehicle.apply_control(control)


class ContinuousInputProcessor:
    """Processes continuously held keyboard inputs for vehicle control."""
    
    THROTTLE_INCREMENT = 0.05
    MAX_THROTTLE = 0.5
    BRAKE_INCREMENT = 0.2
    MAX_BRAKE = 1.0
    STEER_INCREMENT_FACTOR = 5e-4
    MAX_STEER = 0.7
    
    def __init__(self, input_state):
        self.state = input_state
    
    def update(self, pressed_keys, delta_time_ms):
        """Update input state based on held keys."""
        # Throttle: W or UP arrow
        if pressed_keys[K_UP] or pressed_keys[K_w]:
            self.state.throttle = min(self.state.throttle + self.THROTTLE_INCREMENT, self.MAX_THROTTLE)
        else:
            self.state.throttle = 0.0
        
        # Brake: S, SPACE, or DOWN arrow
        if pressed_keys[K_DOWN] or pressed_keys[K_SPACE]:
            self.state.brake = min(self.state.brake + self.BRAKE_INCREMENT, self.MAX_BRAKE)
        else:
            self.state.brake = 0.0
        
        # Steering with smooth accumulation
        steer_delta = self.STEER_INCREMENT_FACTOR * delta_time_ms
        
        if pressed_keys[K_LEFT] or pressed_keys[K_a]:
            if self.state.steer > 0:
                self.state.steer = 0
            self.state.steer -= steer_delta
        elif pressed_keys[K_RIGHT] or pressed_keys[K_d]:
            if self.state.steer < 0:
                self.state.steer = 0
            self.state.steer += steer_delta
        else:
            self.state.steer = 0.0
        
        self.state.steer = min(self.MAX_STEER, max(-self.MAX_STEER, self.state.steer))
        self.state.steer = round(self.state.steer, 1)
        
        # Hand brake: S key
        self.state.hand_brake = pressed_keys[K_s]


class KeyboardInput:
    """Simplified keyboard input handler for vehicle control and restart task."""
    
    def __init__(self, world):
        if not isinstance(world.player, carla.Vehicle):
            raise NotImplementedError("Only vehicle control supported")
        
        self.world = world
        self.vehicle_state = VehicleInputState()
        self.continuous_processor = ContinuousInputProcessor(self.vehicle_state)
    
    def process(self, client, world, clock):
        """Process all input for one frame."""
        # Handle discrete events (key up events and quit)
        should_quit = self._process_events()
        
        # Handle continuous inputs (held keys)
        self._process_continuous_input(clock.get_time())
        
        # Apply state to vehicle
        self.vehicle_state.apply_to_vehicle(self.world.player)
        
        return should_quit
    
    def _process_events(self):
        """Process discrete keyboard events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            
            if event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    return True
                elif event.key == K_BACKSPACE:
                    self.world.keyboard_restart_task = True
        
        return False
    
    def _process_continuous_input(self, delta_time_ms):
        """Update continuous inputs from held keys."""
        keys = pygame.key.get_pressed()
        self.continuous_processor.update(keys, delta_time_ms)


# Legacy function interface for backward compatibility
def KeyboardControl(world):
    """Factory function for backward compatibility."""
    return KeyboardInput(world)
