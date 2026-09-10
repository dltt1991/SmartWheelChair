import importlib.util
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch


ROS_AVAILABLE = importlib.util.find_spec("rclpy") is not None


@unittest.skipUnless(ROS_AVAILABLE, "requires ROS")
class WebModeStateTest(unittest.TestCase):
    def setUp(self):
        import rclpy
        from smart_wheelchair_safety.web_joystick_node import WebJoystickNode
        rclpy.init()
        self.server_patch = patch.object(WebJoystickNode, "_start_http_server")
        self.server_patch.start()
        self.node = WebJoystickNode()
        self.commands = []
        self.modes = []
        self.node._pub = SimpleNamespace(publish=self.commands.append)
        self.node._mode_pub = SimpleNamespace(publish=self.modes.append)

    def command(self, x, y, client_id="a", revision=None, session=None):
        payload = {"x": x, "y": y, "client_id": client_id}
        if revision is not None:
            payload["mode_revision"] = revision
        if session is not None:
            payload["mode_session"] = session
        self.node._handle_message(json.dumps(payload))

    def tearDown(self):
        import rclpy
        self.node.destroy_node()
        self.server_patch.stop()
        rclpy.shutdown()

    def test_mode_change_stops_and_requires_neutral(self):
        self.command(0.3, 0.5, revision=0,
                     session=self.node._mode_session)

        result = self.node._set_mode('{"assist_enabled": false}')

        self.assertEqual(result, {
            "assist_enabled": False,
            "revision": 1,
            "session": self.node._mode_session,
        })
        self.assertEqual(self.node._last_input, (0.0, 0.0))
        self.assertEqual((self.commands[-1].linear.x,
                          self.commands[-1].angular.z), (0.0, 0.0))
        self.assertFalse(self.modes[-1].data)
        self.command(0.3, 0.5, revision=1,
                     session=self.node._mode_session)
        self.assertEqual(self.node._last_input, (0.0, 0.0))
        self.command(0, 0, revision=1, session=self.node._mode_session)
        self.command(0.3, 0.5, revision=1,
                     session=self.node._mode_session)
        self.assertEqual(self.node._last_input, (0.3, 0.5))

    def test_mode_payload_defaults_to_assist_and_rejects_non_boolean(self):
        payload = self.node._mode_payload()
        self.assertEqual(payload["assist_enabled"], True)
        self.assertEqual(payload["revision"], 0)
        self.assertEqual(payload["session"], self.node._mode_session)

        self.assertIsNone(self.node._set_mode('{"assist_enabled": "false"}'))

        self.assertEqual(self.node._mode_payload(), payload)

    def test_stale_tab_command_cannot_unlock_or_drive_after_mode_change(self):
        self.node._set_mode('{"assist_enabled": false}')

        self.command(0, 0, "new", 1, self.node._mode_session)
        self.command(0.4, 0.5, "old", 0, self.node._mode_session)

        self.assertEqual(self.node._last_input, (0., 0.))
        self.command(0.4, 0.5, "new", 1, self.node._mode_session)
        self.assertEqual(self.node._last_input, (.4, .5))

    def test_previous_process_session_cannot_drive_after_restart(self):
        self.command(0.4, 0.5, revision=0, session="old-process")

        self.assertEqual(self.node._last_input, (0., 0.))

        self.command(0.4, 0.5)
        self.assertEqual(self.node._last_input, (.4, .5))

    def test_non_finite_web_command_is_rejected(self):
        self.command(.2, .3, revision=0, session=self.node._mode_session)

        self.command(float("nan"), .4, revision=0,
                     session=self.node._mode_session)

        self.assertEqual(self.node._last_input, (.2, .3))

    def test_command_timer_publishes_current_mode(self):
        self.node._publish_command()

        self.assertTrue(self.modes[-1].data)

    def test_mode_switch_cannot_be_overwritten_by_stale_timer_heartbeat(self):
        publish_started = threading.Event()
        release_publish = threading.Event()

        def publish_command(message):
            self.commands.append(message)
            if abs(message.linear.x) > 0.001:
                publish_started.set()
                release_publish.wait(timeout=1.)

        self.node._pub = SimpleNamespace(publish=publish_command)
        self.node._last_input = (0., -0.5)
        self.node._last_input_time = time.monotonic()
        timer = threading.Thread(target=self.node._publish_command)
        timer.start()
        self.assertTrue(publish_started.wait(timeout=1.))
        switch = threading.Thread(
            target=self.node._set_mode,
            args=('{"assist_enabled": false}',),
        )
        switch.start()
        time.sleep(.02)

        release_publish.set()
        timer.join(timeout=1.)
        switch.join(timeout=1.)

        self.assertFalse(timer.is_alive())
        self.assertFalse(switch.is_alive())
        self.assertFalse(self.modes[-1].data)


@unittest.skipUnless(ROS_AVAILABLE, "requires ROS")
class WebModeHttpTest(unittest.TestCase):
    def setUp(self):
        from smart_wheelchair_safety.web_joystick_node import _handler_class
        self.assist_enabled = True
        self.revision = 0
        self.session = "test-session"

        def mode_payload():
            return {
                "assist_enabled": self.assist_enabled,
                "revision": self.revision,
                "session": self.session,
            }

        def set_mode(message):
            try:
                enabled = json.loads(message)["assist_enabled"]
            except (KeyError, TypeError, json.JSONDecodeError):
                return None
            if type(enabled) is not bool:
                return None
            if enabled != self.assist_enabled:
                self.assist_enabled = enabled
                self.revision += 1
            return mode_payload()

        handler = _handler_class(lambda _: None, lambda: None,
                                 mode_payload, set_mode)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.)

    def test_get_and_post_mode(self):
        with urllib.request.urlopen(self.base + "/mode") as response:
            self.assertEqual(
                json.load(response),
                {"assist_enabled": True, "revision": 0,
                 "session": "test-session"},
            )

        request = urllib.request.Request(
            self.base + "/mode",
            data=b'{"assist_enabled": false}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(
                json.load(response),
                {"assist_enabled": False, "revision": 1,
                 "session": "test-session"},
            )

    def test_post_mode_rejects_missing_boolean(self):
        request = urllib.request.Request(
            self.base + "/mode", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")

        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)

        self.assertEqual(error.exception.code, 400)

    def test_post_requires_json_and_rejects_large_body(self):
        plain = urllib.request.Request(
            self.base + "/mode", data=b'{"assist_enabled": false}',
            headers={"Content-Type": "text/plain"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(plain)
        self.assertEqual(error.exception.code, 415)

        large = urllib.request.Request(
            self.base + "/mode", data=b" " * 4097,
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(large)
        self.assertEqual(error.exception.code, 413)


@unittest.skipUnless(ROS_AVAILABLE, "requires ROS")
class WebModePageTest(unittest.TestCase):
    def test_page_has_accessible_mode_switch_and_server_sync(self):
        from smart_wheelchair_safety.web_joystick_node import PAGE

        self.assertIn('id="mode"', PAGE)
        self.assertIn('role="switch"', PAGE)
        self.assertIn('fetch("/mode"', PAGE)
        self.assertIn('mode_revision: modeRevision', PAGE)
        self.assertIn('mode_session: modeSession', PAGE)
        self.assertIn('responseSequence < appliedModeSequence', PAGE)
        self.assertIn('辅助模式', PAGE)
        self.assertIn('手动模式', PAGE)

    def test_manual_mode_uses_gray_knob_and_assist_keeps_teal(self):
        from smart_wheelchair_safety.web_joystick_node import PAGE

        self.assertIn('body[data-assist="false"] .knob', PAGE)
        self.assertIn('#9ca3af', PAGE)
        self.assertIn('#2dd4bf', PAGE)

    def test_camera_intrinsic_width_cannot_overflow_mobile_layout(self):
        from smart_wheelchair_safety.web_joystick_node import PAGE

        self.assertIn('min-width: 0;', PAGE)
        self.assertIn('overflow-wrap: anywhere;', PAGE)
