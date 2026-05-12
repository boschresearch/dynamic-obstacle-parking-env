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

import argparse
import time
import logging
import carla
import pygame
import importlib
import os

from benchmark.config.config import Config
from benchmark.environment.network_evaluator import NetworkEvaluator
from benchmark.environment.keyboard_input import KeyboardControl
from benchmark.environment.ui_utils import show_control_info

BENCHMARK_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_ROOT = os.path.join(BENCHMARK_ROOT, '../models')


# TODO: better way to import parking agents with an adaptor
def get_parking_agent(agent_type):
    if agent_type == "e2e_parking_carla":
        module = importlib.import_module("models.e2e_parking_carla.agent.parking_agent")
    elif agent_type == "caa_policy":
        module = importlib.import_module("models.caa_policy.agent.parking_agent")
    elif agent_type == "dino_diffusion_parking":
        module = importlib.import_module("models.dino_diffusion_parking.agent.parking_agent")
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    return module.ParkingAgent


def wait_for_carla(host: str, port: int, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            client = carla.Client(host, port)
            client.set_timeout(2.0)
            client.get_world()
            return True
        except Exception:
            time.sleep(1)
    return False


def handle_controller_events(controller, client, world, clock):
    """Compatibility wrapper for old/new keyboard controller interfaces."""
    if hasattr(controller, "parse_events"):
        return controller.parse_events(client, world, clock)
    if hasattr(controller, "process"):
        return controller.process(client, world, clock)
    raise AttributeError("Controller does not provide parse_events or process")


def game_loop(args):
    pygame.init()
    pygame.font.init()
    network_evaluator = None
    client = None

    try:
        if not wait_for_carla(args.host, args.port):
            raise RuntimeError("CARLA did not become ready")
        
        client = carla.Client(args.host, args.port)
        client.set_timeout(120.0)

        logging.info('Load Map %s', args.map)
        carla_world = client.load_world(args.map)
        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        network_evaluator = NetworkEvaluator(carla_world, args)
        ParkingAgent = get_parking_agent(args.agent_type)
        parking_agent = ParkingAgent(network_evaluator, args)
        controller = KeyboardControl(network_evaluator.world)

        display = pygame.display.set_mode((args.width, args.height),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

        steer_wheel_img = pygame.image.load(os.path.join(BENCHMARK_ROOT, "resource/steer_wheel.png"))
        steer_wheel_img = pygame.transform.scale(steer_wheel_img, (100, 100))
        font = pygame.font.Font(None, 25)

        clock = pygame.time.Clock()

        logging.info("Starting main simulation loop...")

        while True:
            if not network_evaluator.running:
                break

            network_evaluator.world_tick()
            clock.tick_busy_loop(60)
            if handle_controller_events(controller, client, network_evaluator.world, clock):
                return
            parking_agent.tick()
            network_evaluator.tick(clock)
            network_evaluator.render(display)
            show_control_info(
                display, 
                parking_agent.get_eva_control(), 
                steer_wheel_img,
                args.width, 
                args.height, 
                font
            )
            pygame.display.flip()

    finally:
        if network_evaluator and client is not None:
            try:
                client.stop_recorder()
            except RuntimeError:
                pass

        if network_evaluator is not None:
            network_evaluator.destroy()

        pygame.quit()
        logging.info("Simulation ended.")


def main():
    p = argparse.ArgumentParser(description="E2E Parking Agent benchmarks in a dynamic CARLA environment")
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information"
    )
    p.add_argument(
        "-c", "--config",
        dest="config_file",
        default=os.path.join(BENCHMARK_ROOT, 'config/config.yaml'),
        help="path to the configuration file (default: config/config.yaml)"
    )
    cli_args = p.parse_args()
    args = Config(config_file=cli_args.config_file)
    args.update(cli_args)

    args.debug_print()

    args.width, args.height = [int(x) for x in args.resolution.split('x')]
    # TODO: support various maps
    args.map = "Town04_Opt"
    args.sensor_config_path = os.path.join(MODELS_ROOT, args.agent_type, 'sensor_specs.yaml')
    args.eva_result_path = os.path.join(BENCHMARK_ROOT, 'results', args.agent_type)
    args.log_path = os.path.join(BENCHMARK_ROOT, 'logs', args.agent_type)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()
