"""
Ultrasonic Sensor Driver for HC-SR04 / JSN-SR04T (Waterproof)
Features:
- Microsecond GPIO pulse timing for Raspberry Pi
- Medium-specific speed of sound calibration (Seawater, Freshwater, Air)
- Rolling median filter to eliminate false echo reflections and acoustic bubble noise
- Non-blocking hardware timeout
- Automated simulation fallback for testing on non-Raspberry Pi systems (Windows/Mac/Linux)
"""
import time
import collections
from typing import Optional

try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except (ImportError, RuntimeError):
    HAS_RPI_GPIO = False

import config


class UltrasonicSensor:
    def __init__(self,
                 trigger_pin: int = config.PIN_TRIGGER,
                 echo_pin: int = config.PIN_ECHO,
                 medium: str = config.MEDIUM,
                 window_size: int = config.MEDIAN_FILTER_WINDOW,
                 simulate: bool = False):
        """
        :param trigger_pin: BCM pin for trigger (Default: GPIO 23)
        :param echo_pin: BCM pin for echo (Default: GPIO 24)
        :param medium: 'seawater', 'freshwater', or 'air'
        :param window_size: rolling filter window size for outlier rejection
        :param simulate: Set True ONLY if you want software simulation for testing without hardware
        """
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.medium = medium.lower()
        self.speed_of_sound = config.SPEED_OF_SOUND.get(self.medium, 34300.0)
        self.history = collections.deque(maxlen=window_size)
        self.simulated = simulate
        self.connected = False
        self._sim_distance = 120.0
        self._sim_direction = -1.0

        if self.simulated:
            print("[i] Ultrasonic Sensor: Software SIMULATION mode active (--simulate).")
            self.connected = True
        elif HAS_RPI_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.trigger_pin, GPIO.OUT)
                GPIO.setup(self.echo_pin, GPIO.IN)
                GPIO.output(self.trigger_pin, False)
                time.sleep(0.05)  # Let sensor settle
                self.connected = True
                print(f"[+] Hardware Ultrasonic Sensor initialized on BCM Trigger={trigger_pin}, Echo={echo_pin}")
                print(f"[+] Medium: {self.medium.upper()} (Speed: {self.speed_of_sound} cm/s)")
            except Exception as e:
                print(f"[!] Warning initializing GPIO: {e}")
                self.connected = False
        else:
            self.connected = False
            print(f"[!] Ultrasonic Sensor: Hardware GPIO not available. Sensor status: DISCONNECTED.")

    def read_raw_distance(self) -> float:
        """Measures single pulse roundtrip time and computes distance in cm. Returns -1.0 if disconnected or timeout."""
        if self.simulated:
            # Generate realistic fluctuating distance ONLY when explicitly asked via --simulate
            self._sim_distance += self._sim_direction * 1.5
            if self._sim_distance <= 15.0:
                self._sim_direction = 1.0
            elif self._sim_distance >= 150.0:
                self._sim_direction = -1.0
            return round(self._sim_distance, 1)

        if not self.connected or not HAS_RPI_GPIO:
            return -1.0  # Hardware not connected

        try:
            # Send 10-microsecond trigger pulse
            GPIO.output(self.trigger_pin, True)
            time.sleep(0.00001)
            GPIO.output(self.trigger_pin, False)

            pulse_start = time.time()
            timeout_start = pulse_start

            # Wait for echo to go HIGH
            while GPIO.input(self.echo_pin) == 0:
                pulse_start = time.time()
                if pulse_start - timeout_start > config.SENSOR_TIMEOUT_SEC:
                    return -1.0  # Timeout, no signal received

            # Wait for echo to go LOW
            pulse_end = pulse_start
            timeout_echo = pulse_start
            while GPIO.input(self.echo_pin) == 1:
                pulse_end = time.time()
                if pulse_end - timeout_echo > config.SENSOR_TIMEOUT_SEC:
                    return -1.0

            pulse_duration = pulse_end - pulse_start

            # Distance = (time * speed) / 2
            distance = (pulse_duration * self.speed_of_sound) / 2.0
            return round(distance, 2)
        except Exception:
            return -1.0

    def get_distance(self) -> float:
        """
        Reads raw distance and applies a rolling median filter.
        Returns distance in cm (or -1.0 if sensor is disconnected or out of range).
        """
        raw = self.read_raw_distance()

        if raw < 0:
            # Sensor is disconnected or no echo received
            return -1.0

        # Reject out of physical range readings
        if raw < 5.0 or raw > 600.0:
            if len(self.history) > 0:
                return sorted(self.history)[len(self.history) // 2]
            return -1.0

        self.history.append(raw)
        median_val = sorted(self.history)[len(self.history) // 2]
        return round(median_val, 1)

    def cleanup(self):
        """Release GPIO resources."""
        if not self.simulated and HAS_RPI_GPIO:
            try:
                GPIO.cleanup()
                print("[+] GPIO cleaned up successfully.")
            except Exception:
                pass


if __name__ == "__main__":
    sensor = UltrasonicSensor()
    print("Testing Ultrasonic Sensor reading (Press Ctrl+C to stop)...")
    try:
        while True:
            dist = sensor.get_distance()
            status = "DANGER" if dist < config.PROXIMITY_DANGER_CM else "SAFE"
            print(f"\rDistance: {dist:5.1f} cm [{status}]", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sensor.cleanup()
