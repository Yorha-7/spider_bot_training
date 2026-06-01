# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import sys
import threading
import time

log = logging.getLogger(__name__)

if sys.platform == "linux":
    import select


class KeyboardInput:
    def __init__(self):
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.ang_z = 0.0
        self._running = False
        self._thread = None
        self._key_map = {
            "w": ("vel_x", 0.3),
            "s": ("vel_x", -0.3),
            "a": ("vel_y", -0.15),
            "d": ("vel_y", 0.15),
            "q": ("ang_z", 0.5),
            "e": ("ang_z", -0.5),
        }

    def _read_key_linux(self):
        if sys.platform == "linux":
            dr, _, _ = select.select([sys.stdin], [], [], 0.1)
            if dr:
                return sys.stdin.read(1)
        return None

    def _keyboard_loop(self):
        while self._running:
            key = self._read_key_linux()
            if key:
                key = key.lower()
                if key == " ":
                    self.vel_x = 0.0
                    self.vel_y = 0.0
                    self.ang_z = 0.0
                elif key in self._key_map:
                    attr, value = self._key_map[key]
                    setattr(self, attr, value)
            time.sleep(0.01)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            if sys.platform == "linux":
                self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
                self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def is_active(self):
        return self._running

    def print_controls(self):
        log.info(
            "\n--- Keyboard Controls ---\n"
            "W/S: Forward/Backward\n"
            "A/D: Strafe Left/Right\n"
            "Q/E: Turn Left/Right\n"
            "SPACE: Stop\n"
            "-------------------------"
        )