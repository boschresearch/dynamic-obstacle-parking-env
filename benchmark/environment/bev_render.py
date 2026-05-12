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

import os
import carla
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from PIL import Image

# Global Flags
PIXELS_PER_METER = 5

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

COLOR_TRAFFIC_RED = pygame.Color(255, 0, 0)
COLOR_TRAFFIC_YELLOW = pygame.Color(0, 255, 0)
COLOR_TRAFFIC_GREEN = pygame.Color(0, 0, 255)


class BevRender:
    """Bird's-eye-view coordinator for static map and nearby actors."""

    def __init__(self, world, device):
        self._device = device
        self._world = world.world
        self._vehicle = world.player
        self._actors = None

        hd_map = self._world.get_map().to_opendrive()
        self._map = hd_map
        self.world_map = carla.Map("RouteMap", hd_map)

        self.vehicle_template = torch.ones(1, 1, 22, 9, device=self._device)
        self.global_map, world_offset = self._build_global_map()
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = Renderer(
            world_offset,
            self.map_dims,
            data_generation=True,
            device=self._device,
        )
        self.detection_radius = 50.0

    def _build_global_map(self):
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)

        def _surface_to_grayscale(surface):
            arr = pygame.surfarray.array3d(surface)
            return np.swapaxes(arr, 0, 1).mean(axis=-1)

        road = _surface_to_grayscale(map_image.map_surface)
        lane = _surface_to_grayscale(map_image.lane_surface)

        grid = np.zeros((1, 15) + road.shape, dtype=np.float32)
        grid[:, 0, ...] = road / 255.0
        grid[:, 1, ...] = lane / 255.0

        global_map = torch.tensor(grid, device=self._device, dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device=self._device, dtype=torch.float32)
        return global_map, world_offset

    def _transform_to_pose_tensors(self, transform):
        position = torch.tensor(
            [transform.location.x, transform.location.y],
            device=self._device,
            dtype=torch.float32,
        )
        yaw = torch.tensor(
            [transform.rotation.yaw / 180.0 * np.pi],
            device=self._device,
            dtype=torch.float32,
        )
        return position, yaw

    @staticmethod
    def _vehicle_template_for(vehicle, device):
        veh_x_extent = int(max(vehicle.bounding_box.extent.x * 2, 1) * PIXELS_PER_METER)
        veh_y_extent = int(max(vehicle.bounding_box.extent.y * 2, 1) * PIXELS_PER_METER)
        return torch.ones(1, 1, veh_x_extent, veh_y_extent, device=device)

    def _render_other_vehicle(self, birdview, ego_pos, ego_yaw, other_transform, actor_for_bbox):
        pos, yaw = self._transform_to_pose_tensors(other_transform)
        self.vehicle_template = self._vehicle_template_for(actor_for_bbox, self._device)
        self.renderer.render_agent_bv(
            birdview,
            ego_pos,
            ego_yaw,
            self.vehicle_template,
            pos,
            yaw,
            channel=5,
        )

    def set_player(self, player):
        self._vehicle = player

    def render_BEV(self):
        ego_t = self._vehicle.get_transform()
        ego_pos, ego_yaw = self._transform_to_pose_tensors(ego_t)

        birdview = self.renderer.get_local_birdview(self.global_map, ego_pos, ego_yaw)

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter("*vehicle*")
        for vehicle in vehicles:
            if vehicle.id == self._vehicle.id:
                continue
            if vehicle.get_location().distance(ego_t.location) >= self.detection_radius:
                continue
            self._render_other_vehicle(birdview, ego_pos, ego_yaw, vehicle.get_transform(), vehicle)

        return birdview

    def get_bev_states(self):
        def _get_element_transforms(keyword):
            elements = self._world.get_actors().filter(keyword)
            return [
                carla.Transform(element.get_transform().location, element.get_transform().rotation)
                for element in elements
            ]

        return {
            "ego_t": carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation),
            "vehicle_ts": _get_element_transforms("*vehicle*"),
        }

    def render_BEV_from_state(self, state):
        ego_t = state["ego_t"]
        ego_pos, ego_yaw = self._transform_to_pose_tensors(ego_t)
        birdview = self.renderer.get_local_birdview(self.global_map, ego_pos, ego_yaw)

        # Keep the actor iteration order aligned with CARLA filter order.
        for vehicle_t, vehicle in zip(state["vehicle_ts"], self._world.get_actors().filter("*vehicle*")):
            if vehicle.id == self._vehicle.id:
                continue
            if vehicle_t.location.distance(ego_t.location) >= self.detection_radius:
                continue
            self._render_other_vehicle(birdview, ego_pos, ego_yaw, vehicle_t, vehicle)

        return birdview


class Renderer:
    """Map-space / crop-space transforms and rasterization helpers."""

    def __init__(self, map_offset, map_dims, data_generation=True, device="cpu"):
        self.args = {"device": device}

        if data_generation:
            self.PIXELS_AHEAD_VEHICLE = 0
            self.local_view_dims = (500, 500)
            self.crop_dims = (500, 500)
        else:
            self.PIXELS_AHEAD_VEHICLE = 110
            self.local_view_dims = (320, 320)
            self.crop_dims = (192, 192)

        self.map_offset = map_offset
        self.map_dims = map_dims
        self.local_view_scale = (
            self.local_view_dims[1] / self.map_dims[1],
            self.local_view_dims[0] / self.map_dims[0],
        )
        self.crop_scale = (
            self.crop_dims[1] / self.map_dims[1],
            self.crop_dims[0] / self.map_dims[0],
        )

    def _stn_scale_transform(self, grid, vehicle, batch_size=1):
        scale_h = torch.tensor([grid.size(2) / vehicle.size(2)], device=self.args["device"])
        scale_w = torch.tensor([grid.size(3) / vehicle.size(3)], device=self.args["device"])

        base = torch.tensor(
            [[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]],
            device=self.args["device"],
        ).view(1, 3, 3)

        if batch_size == 1:
            return base
        return base.expand(batch_size, -1, -1)

    def _sample_vehicle(self, vehicle, affine_transform, out_shape):
        affine_grid = F.affine_grid(
            affine_transform[:, 0:2, :],
            out_shape,
            align_corners=True,
        )
        return F.grid_sample(vehicle, affine_grid, align_corners=True)

    @staticmethod
    def _channel_for_light_state(channel, state):
        if state == "Green":
            return 4
        if state == "Yellow":
            return 3
        if state == "Red":
            return 2
        return channel

    def world_to_pix(self, pos):
        return (pos - self.map_offset) * PIXELS_PER_METER

    def world_to_pix_crop_batched(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        del offset
        crop_yaw = crop_yaw + np.pi / 2
        batch_size = crop_pos.shape[0]

        rotation = torch.stack(
            [
                torch.cos(crop_yaw),
                -torch.sin(crop_yaw),
                torch.sin(crop_yaw),
                torch.cos(crop_yaw),
            ],
            dim=-1,
        ).view(batch_size, 2, 2)

        crop_pos_px = self.world_to_pix(crop_pos)
        query_pos_px_map = self.world_to_pix(query_pos)

        shift = torch.tensor([0.0, -self.PIXELS_AHEAD_VEHICLE], device=self.args["device"])

        query_pos_px = torch.transpose(rotation, -2, -1).unsqueeze(1) @ (
            query_pos_px_map - crop_pos_px
        ).unsqueeze(-1)
        query_pos_px = query_pos_px.squeeze(-1) - shift

        return query_pos_px + torch.tensor(
            [self.crop_dims[1] / 2, self.crop_dims[0] / 2],
            device=self.args["device"],
        )

    def world_to_pix_crop(self, query_pos, crop_pos, crop_yaw, offset=(0, 0)):
        del offset
        crop_yaw = crop_yaw + np.pi / 2

        rotation = torch.tensor(
            [
                [torch.cos(crop_yaw), -torch.sin(crop_yaw)],
                [torch.sin(crop_yaw), torch.cos(crop_yaw)],
            ],
            device=self.args["device"],
        )

        crop_pos_px = self.world_to_pix(crop_pos)
        query_pos_px_map = self.world_to_pix(query_pos)

        shift = torch.tensor([0.0, -self.PIXELS_AHEAD_VEHICLE], device=self.args["device"])
        query_pos_px = rotation.T @ (query_pos_px_map - crop_pos_px) - shift

        return query_pos_px + torch.tensor(
            [self.crop_dims[1] / 2, self.crop_dims[0] / 2],
            device=self.args["device"],
        )

    def world_to_rel(self, pos):
        pos_px = self.world_to_pix(pos)
        pos_rel = pos_px / torch.tensor([self.map_dims[1], self.map_dims[0]], device=self.args["device"])
        return pos_rel * 2 - 1

    def render_agent(self, grid, vehicle, position, orientation):
        orientation = orientation - np.pi / 2
        position = self.world_to_rel(position) * -1

        scale_transform = self._stn_scale_transform(grid, vehicle)
        rotation_transform = torch.tensor(
            [
                [torch.cos(orientation), torch.sin(orientation), 0],
                [-torch.sin(orientation), torch.cos(orientation), 0],
                [0, 0, 1],
            ],
            device=self.args["device"],
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, position[0]], [0, 1, position[1]], [0, 0, 1]],
            device=self.args["device"],
        ).view(1, 3, 3)

        affine_transform = scale_transform @ rotation_transform @ translation_transform
        rendering = self._sample_vehicle(vehicle, affine_transform, (1, 1, grid.shape[2], grid.shape[3]))

        grid[:, 5, ...] += rendering.squeeze()
        return grid

    def render_agent_bv(
        self,
        grid,
        grid_pos,
        grid_orientation,
        vehicle,
        position,
        orientation,
        channel=5,
        state=None,
    ):
        orientation = orientation + np.pi / 2

        pos_pix_bv = self.world_to_pix_crop(position, grid_pos, grid_orientation)

        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args["device"])
        pos_rel_bv = (pos_rel_bv * 2 - 1) * -1

        scale_transform = self._stn_scale_transform(grid, vehicle)

        grid_orientation = grid_orientation + np.pi / 2
        rotation_transform = torch.tensor(
            [
                [torch.cos(orientation - grid_orientation), torch.sin(orientation - grid_orientation), 0],
                [-torch.sin(orientation - grid_orientation), torch.cos(orientation - grid_orientation), 0],
                [0, 0, 1],
            ],
            device=self.args["device"],
        ).view(1, 3, 3)

        translation_transform = torch.tensor(
            [[1, 0, pos_rel_bv[0]], [0, 1, pos_rel_bv[1]], [0, 0, 1]],
            device=self.args["device"],
        ).view(1, 3, 3)

        affine_transform = scale_transform @ rotation_transform @ translation_transform
        rendering = self._sample_vehicle(vehicle, affine_transform, (1, 1, grid.shape[2], grid.shape[3]))

        target_channel = self._channel_for_light_state(channel, state)
        grid[:, target_channel, ...] += rendering.squeeze()

    def render_agent_bv_batched(
        self,
        grid,
        grid_pos,
        grid_orientation,
        vehicle,
        position,
        orientation,
        channel=5,
    ):
        orientation = orientation + np.pi / 2
        batch_size = position.shape[0]

        pos_pix_bv = self.world_to_pix_crop_batched(position, grid_pos, grid_orientation)

        h, w = (grid.size(-2), grid.size(-1))
        pos_rel_bv = pos_pix_bv / torch.tensor([h, w], device=self.args["device"])
        pos_rel_bv = (pos_rel_bv * 2 - 1) * -1

        scale_transform = self._stn_scale_transform(grid, vehicle, batch_size=batch_size)

        grid_orientation = grid_orientation + np.pi / 2
        angle_delta = orientation - grid_orientation
        zeros = torch.zeros_like(angle_delta)
        ones = torch.ones_like(angle_delta)

        rotation_transform = torch.stack(
            [
                torch.cos(angle_delta),
                torch.sin(angle_delta),
                zeros,
                -torch.sin(angle_delta),
                torch.cos(angle_delta),
                zeros,
                zeros,
                zeros,
                ones,
            ],
            dim=-1,
        ).view(batch_size, 3, 3)

        translation_transform = torch.stack(
            [
                ones,
                zeros,
                pos_rel_bv[..., 0:1],
                zeros,
                ones,
                pos_rel_bv[..., 1:2],
                zeros,
                zeros,
                ones,
            ],
            dim=-1,
        ).view(batch_size, 3, 3)

        affine_transform = scale_transform @ rotation_transform @ translation_transform
        rendering = self._sample_vehicle(
            vehicle,
            affine_transform,
            (batch_size, 1, grid.shape[2], grid.shape[3]),
        )

        for idx in range(batch_size):
            grid[:, int(channel[idx].item()), ...] += rendering[idx].squeeze()

    def get_local_birdview(self, grid, position, orientation):
        position = self.world_to_rel(position)
        orientation = orientation + np.pi / 2

        scale_transform = torch.tensor(
            [[self.crop_scale[1], 0, 0], [0, self.crop_scale[0], 0], [0, 0, 1]],
            device=self.args["device"],
        ).view(1, 3, 3)

        rotation_transform = torch.tensor(
            [[torch.cos(orientation), -torch.sin(orientation), 0],
             [torch.sin(orientation), torch.cos(orientation), 0],
             [0, 0, 1]],
            device=self.args["device"],
        ).view(1, 3, 3)

        shift = torch.tensor(
            [0.0, -2 * self.PIXELS_AHEAD_VEHICLE / self.map_dims[0]],
            device=self.args["device"],
        )
        position = position + rotation_transform[0, 0:2, 0:2] @ shift

        translation_transform = torch.tensor(
            [[1, 0, position[0] / self.crop_scale[0]],
             [0, 1, position[1] / self.crop_scale[1]],
             [0, 0, 1]],
            device=self.args["device"],
        ).view(1, 3, 3)

        local_view_transform = scale_transform @ translation_transform @ rotation_transform

        affine_grid = F.affine_grid(
            local_view_transform[:, 0:2, :],
            (1, 1, self.crop_dims[0], self.crop_dims[0]),
            align_corners=True,
        )

        return F.grid_sample(grid, affine_grid, align_corners=True)

    def step(self, actions):
        # Preserved for compatibility with external code paths.
        print(self.ego.state, actions)
        self.ego.set_state(self.ego.motion_model(self.ego.state, actions=actions))
        self.adv.set_state(self.adv.motion_model(self.adv.state))
        self.timestep += 1

    def visualize_grid(self, grid, type="LTS_Reduced"):
        if type == "LTS_Reduced":
            colors = [
                (102, 102, 102),
                (253, 253, 17),
                (0, 0, 142),
                (220, 20, 60),
            ]
        elif type == "Trajectory_planner":
            colors = [
                (102, 102, 102),
                (253, 253, 17),
            ]
        elif type == "LTS_Full":
            colors = [
                (102, 102, 102),
                (253, 253, 17),
                (204, 6, 5),
                (250, 210, 1),
                (39, 232, 51),
                (0, 0, 142),
                (220, 20, 60),
            ]
        elif type == "LTS_FullFuture":
            colors = [
                (102, 102, 102),
                (253, 253, 17),
                (204, 6, 5),
                (250, 210, 1),
                (39, 232, 51),
                (0, 0, 142),
                (220, 20, 60),
                *[(0, 0, 142 + (11 * i)) for i in range(grid.shape[1] - 7)],
            ]
        elif type == "LTS_ReducedFuture":
            colors = [
                (102, 102, 102),
                (253, 253, 17),
                (0, 0, 142),
                (220, 20, 60),
                *[(0, 0, 142 + (11 * i)) for i in range(grid.shape[1] - 7)],
            ]
        else:
            colors = []

        grid = grid.detach().cpu()
        grid_img = np.zeros((grid.shape[2:4] + (3,)), dtype=np.uint8)
        grid_img[...] = [0, 47, 0]

        for idx, color in enumerate(colors):
            grid_img[grid[0, idx, ...] > 0] = color

        return Image.fromarray(grid_img)

    def bev_to_gray_img(self, grid):
        levels = [1, 2, 3, 4, 5, 6, 7]

        grid = grid.detach().cpu()
        grid_img = np.zeros(grid.shape[2:4], dtype=np.uint8)

        for idx, level in enumerate(levels):
            grid_img[grid[0, idx, ...] > 0] = level

        return Image.fromarray(grid_img)


class ModuleManager(object):
    def __init__(self):
        self.modules = []

    def register_module(self, module):
        self.modules.append(module)

    def clear_modules(self):
        del self.modules[:]

    def tick(self, clock):
        for module in self.modules:
            module.tick(clock)

    def render(self, display, snapshot=None):
        display.fill(COLOR_ALUMINIUM_4)
        for module in self.modules:
            module.render(display, snapshot=snapshot)

    def get_module(self, name):
        for module in self.modules:
            if module.name == name:
                return module
        return None

    def start_modules(self):
        for module in self.modules:
            module.start()


module_manager = ModuleManager()


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter=10):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        module_manager.clear_modules()

        pygame.init()
        _display = pygame.display.set_mode((320, 320), 0, 32)
        del _display

        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0

        waypoints = carla_map.generate_waypoints(2)
        margin = 50

        max_x = max(waypoints, key=lambda wp: wp.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda wp: wp.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda wp: wp.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda wp: wp.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)
        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.big_lane_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()

        self.draw_road_map(
            self.big_map_surface,
            self.big_lane_surface,
            carla_world,
            carla_map,
            self.world_to_pixel,
            self.world_to_pixel_width,
        )

        self.map_surface = self.big_map_surface
        self.lane_surface = self.big_lane_surface

    def draw_road_map(self, map_surface, lane_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        map_surface.fill(COLOR_BLACK)
        precision = 0.05

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def does_cross_solid_line(waypoint, shift):
            shifted_wp = carla_map.get_waypoint(
                lateral_shift(waypoint.transform, shift),
                project_to_road=False,
            )
            if shifted_wp is None or shifted_wp.road_id != waypoint.road_id:
                return True
            return (shifted_wp.lane_id * waypoint.lane_id < 0) or shifted_wp.lane_id == waypoint.lane_id

        def draw_lane_marking(surface, points, solid=True):
            if solid and len(points) > 1:
                pygame.draw.lines(surface, COLOR_WHITE, False, points, 2)
                return

            broken_lines = [segment for idx, segment in enumerate(zip(*(iter(points),) * 20)) if idx % 3 == 0]
            for line in broken_lines:
                pygame.draw.lines(surface, COLOR_WHITE, False, line, 2)

        topology = [entry[0] for entry in carla_map.get_topology()]
        topology = sorted(topology, key=lambda waypoint: waypoint.transform.location.z)

        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)[0]

            left_marking = [lateral_shift(wp.transform, -wp.lane_width * 0.5) for wp in waypoints]
            right_marking = [lateral_shift(wp.transform, wp.lane_width * 0.5) for wp in waypoints]

            polygon = left_marking + [point for point in reversed(right_marking)]
            polygon = [world_to_pixel(point) for point in polygon]

            if len(polygon) > 2:
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon, 10)
                pygame.draw.polygon(map_surface, COLOR_WHITE, polygon)

            if waypoint.is_intersection:
                continue

            sample = waypoints[int(len(waypoints) / 2)]
            draw_lane_marking(
                lane_surface,
                [world_to_pixel(point) for point in left_marking],
                does_cross_solid_line(sample, -sample.lane_width * 1.1),
            )
            draw_lane_marking(
                lane_surface,
                [world_to_pixel(point) for point in right_marking],
                does_cross_solid_line(sample, sample.lane_width * 1.1),
            )

        actors = carla_world.get_actors()
        _stops_transform = [actor.get_transform() for actor in actors if "stop" in actor.type_id]
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont("Arial", font_size, True)
        font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        _font_surface = pygame.transform.scale(
            font_surface,
            (font_surface.get_width(), font_surface.get_height() * 2),
        )
        del _stops_transform, _font_surface

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale == self.scale:
            return
        self.scale = scale
        width = int(self.big_map_surface.get_width() * self.scale)
        self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
