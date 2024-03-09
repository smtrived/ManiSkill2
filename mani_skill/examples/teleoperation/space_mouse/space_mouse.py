"""SpaceMouse device interface.

The coordinate frame I assume in this file:

       z
        ^    x
        |  /
        | /
        |/
y <-----

References:
    - https://trezor.github.io/cython-hidapi/examples.html
    - https://github.com/ARISE-Initiative/robosuite/blob/48c1b8a6c077d04399a00db05694d7f9f876ffc9/robosuite/devices/spacemouse.py
    - https://github.com/JakubAndrysek/PySpaceMouse/blob/f2ff80317c75671af841d8a666a0c17d5d4a2c68/pyspacemouse/pyspacemouse.py
"""

import threading
import time
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import hid
except ModuleNotFoundError as e:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "`pip install hidapi` to install the additional requirements."
    ) from e


# axis mappings are specified as:
# [channel, byte1, byte2, scale]; scale is usually just -1 or 1 and multiplies the result by this value
# (but per-axis scaling can also be achieved by setting this value)
# byte1 and byte2 are indices into the HID array indicating the two bytes to read to form the value for this axis
# For the SpaceNavigator, these are consecutive bytes following the channel number.
AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

# button states are specified as:
# [channel, data byte,  bit of byte, index to write to]
# If a message is received on the specified channel, the value of the data byte is set in the button bit array
ButtonSpec = namedtuple("ButtonSpec", ["channel", "byte", "bit"])


# convert two 8 bit bytes to a signed 16 bit integer
def to_int16(y1, y2):
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


@dataclass
class DeviceSpec:
    vendor_id: int
    product_id: int
    axis_mapping: dict[str, AxisSpec]  # axis specifications
    button_mapping: list[ButtonSpec]  # button specifications
    axis_scale: float
    max_bytes: int  # maximum number of bytes to read


class SpaceMouse:
    """SpaceMouse device interface."""

    def __init__(self):
        self.device = None
        self.device_spec = None

    def open(self, device_spec: DeviceSpec):
        if self.device is not None:
            self.close()
        self.device = hid.device()
        self.device.open(device_spec.vendor_id, device_spec.product_id)
        print("Manufacturer: {:s}".format(self.device.get_manufacturer_string()))
        self.device_spec = device_spec

    def close(self):
        if self.device is not None:
            self.device.close()
            print(
                "Closed device: ({}, {})".format(
                    self.device_spec.vendor_id, self.device_spec.product_id
                )
            )
        self.device = None

    def __del__(self):
        self.close()

    def read(self) -> list[int]:
        """Read raw bytes from the SpaceMouse."""
        if self.device is None:
            raise RuntimeError("Device not open")
        return self.device.read(self.device_spec.max_bytes)

    def decode(self, data):
        """Decode the data received from the SpaceMouse."""
        ret = {}

        for name, axis_spec in self.device_spec.axis_mapping.items():
            channel, byte1, byte2, scale = axis_spec
            if data[0] == channel:
                axis_value = to_int16(data[byte1], data[byte2])
                ret[name] = axis_value * scale / float(self.device_spec.axis_scale)

        for name, button_spec in self.device_spec.button_mapping.items():
            channel, byte, bit = button_spec
            if data[0] == channel:
                mask = 1 << bit
                ret[name] = (data[byte] & mask) != 0

        return ret


DEVICE_SPECS = {
    "SpaceMouse Compact": DeviceSpec(
        vendor_id=0x256F,
        product_id=0xC635,
        axis_mapping={
            "x": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
            "y": AxisSpec(channel=1, byte1=1, byte2=2, scale=-1),
            "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
            "roll": AxisSpec(channel=2, byte1=1, byte2=2, scale=-1),
            "pitch": AxisSpec(channel=2, byte1=3, byte2=4, scale=-1),
            "yaw": AxisSpec(channel=2, byte1=5, byte2=6, scale=1),
        },
        button_mapping={
            "left": ButtonSpec(channel=3, byte=1, bit=0),
            "right": ButtonSpec(channel=3, byte=1, bit=1),
        },
        axis_scale=350.0,
        max_bytes=7,
    )
}


def get_available_devices():
    available_devices = []
    for info in hid.enumerate():
        for spec in DEVICE_SPECS.values():
            if (
                info["vendor_id"] == spec.vendor_id
                and info["product_id"] == spec.product_id
            ):
                print(f"Found device: {info}")
    for info in hid.enumerate():
        for spec in DEVICE_SPECS.values():
            if (
                info["vendor_id"] == spec.vendor_id
                and info["product_id"] == spec.product_id
            ):
                print(f"Found device: {info}")
                available_devices.append(spec)
                break
    return available_devices


@dataclass
class SpaceMouseState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    left: bool = False
    right: bool = False


class SpaceMouseThread:
    """A daemon thread to listen to SpaceMouse.

    It keeps reading data from the SpaceMouse and updates the state.
    Some behaviors of SpaceMouse (blocking mode):
    1. After you release the mouse, it will send multiple messages with 0s before fully settling down.
    2. When you press the button, it will send one message. And when you release, it will send another.
    """

    def __init__(self, device_spec: DeviceSpec = None):
        if device_spec is None:
            device_specs = get_available_devices()
            if len(device_specs) == 0:
                raise RuntimeError("SpaceMouse not found")
            device_spec = device_specs[0]

        self.device = SpaceMouse()
        self.device.open(device_spec)
        self.state = SpaceMouseState()

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        """A daemon thread to listen to SpaceMouse."""
        try:
            while True:
                data = self.device.read()
                state = self.device.decode(data)
                self.state.__dict__.update(state)
        except OSError:
            print("Fail to read from SpaceMouse")

    def close(self):
        self.device.close()

    def get_xyz(self):
        return np.array([self.state.x, self.state.y, self.state.z])

    def get_rpy(self):
        return np.array([self.state.roll, self.state.pitch, self.state.yaw])

    def get_rotation(self):
        return Rotation.from_euler("XYZ", self.get_rpy())

    def get_rotvec(self):
        return self.get_rotation().as_rotvec()

    def get_left_button(self):
        return self.state.left

    def get_right_button(self):
        return self.state.right


def main():
    device_specs = get_available_devices()
    if len(device_specs) == 0:
        print("No devices found")
        return
    device_spec = device_specs[0]

    space_mouse = SpaceMouse()
    space_mouse.open(device_spec)
    # space_mouse.device.set_nonblocking(1)

    print("Press the right button to exit.")
    while True:
        bytes = space_mouse.read()
        if len(bytes) != 0:  # no data for non-blocking read
            data = space_mouse.decode(bytes)
            print(data)
            if data.get("right", False):
                break
        time.sleep(1 / 50)

    space_mouse.close()


if __name__ == "__main__":
    main()
