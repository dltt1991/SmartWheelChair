from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import threading
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from smart_wheelchair_safety.joystick import camera_frame_payload, joystick_to_velocity


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartWheelChair Joystick</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111827;
      color: #f9fafb;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        linear-gradient(135deg, rgba(20, 184, 166, .18), transparent 35%),
        linear-gradient(315deg, rgba(244, 114, 182, .14), transparent 40%),
        #111827;
    }
    main {
      width: min(94vw, 760px);
      display: grid;
      gap: 18px;
      align-items: center;
      justify-content: center;
    }
    h1 {
      margin: 0;
      max-width: 100%;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
      overflow-wrap: anywhere;
      text-align: center;
    }
    .status {
      min-height: 24px;
      color: #cbd5e1;
      font-size: 15px;
    }
    .pad {
      width: min(76vw, 320px);
      aspect-ratio: 1;
      border-radius: 50%;
      position: relative;
      touch-action: none;
      background: radial-gradient(circle at 50% 50%, #1f2937 0 26%, #0f172a 27% 100%);
      border: 1px solid rgba(255,255,255,.16);
      box-shadow: inset 0 0 0 2px rgba(255,255,255,.04), 0 24px 80px rgba(0,0,0,.36);
    }
    .axis {
      position: absolute;
      inset: 12%;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 50%;
    }
    .axis::before, .axis::after {
      content: "";
      position: absolute;
      background: rgba(255,255,255,.13);
    }
    .axis::before {
      width: 1px;
      top: 0;
      bottom: 0;
      left: 50%;
    }
    .axis::after {
      height: 1px;
      left: 0;
      right: 0;
      top: 50%;
    }
    .knob {
      width: 30%;
      aspect-ratio: 1;
      border-radius: 50%;
      position: absolute;
      left: 35%;
      top: 35%;
      background: #2dd4bf;
      box-shadow: 0 12px 32px rgba(45,212,191,.35), inset 0 -10px 18px rgba(15,23,42,.25);
      transform: translate(0, 0);
      transition: background .18s ease, box-shadow .18s ease;
    }
    body[data-assist="false"] .knob {
      background: #9ca3af;
      box-shadow: 0 12px 32px rgba(15,23,42,.28), inset 0 -10px 18px rgba(15,23,42,.20);
    }
    .readout {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-variant-numeric: tabular-nums;
    }
    .readout div {
      padding: 12px;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 8px;
      background: rgba(15,23,42,.72);
      text-align: center;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 11px 18px;
      color: #052e2b;
      background: #5eead4;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      opacity: .65;
    }
    .mode-toggle {
      min-width: 112px;
      min-height: 40px;
    }
    body[data-assist="false"] .mode-toggle {
      color: #f9fafb;
      background: #4b5563;
    }
    .control, .camera {
      display: grid;
      gap: 18px;
      justify-items: center;
      min-width: 0;
      width: 100%;
    }
    .control {
      width: min(100%, 440px);
      justify-self: center;
    }
    .camera canvas {
      display: block;
      width: 100%;
      max-width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 8px;
      background: #020617;
      border: 1px solid rgba(255,255,255,.16);
    }
    .camera-label {
      color: #cbd5e1;
      font-size: 15px;
    }
    @media (max-width: 820px) {
      main {
        padding: 18px 0;
      }
    }
  </style>
</head>
<body data-assist="true">
  <main>
    <div class="camera">
      <canvas id="camera" width="640" height="360"></canvas>
      <div class="camera-label" id="cameraStatus">后摄画面等待中...</div>
    </div>
    <div class="control">
      <h1>SmartWheelChair Joystick</h1>
      <div class="status" id="status">连接中...</div>
      <button id="mode" class="mode-toggle" type="button"
              role="switch" aria-checked="true">辅助模式</button>
      <div class="pad" id="pad" aria-label="virtual joystick">
        <div class="axis"></div>
        <div class="knob" id="knob"></div>
      </div>
      <div class="readout">
        <div>前后 <span id="linear">0.00</span></div>
        <div>转向 <span id="angular">0.00</span></div>
      </div>
      <button id="stop">停止</button>
    </div>
  </main>
  <script>
    const pad = document.getElementById("pad");
    const knob = document.getElementById("knob");
    const statusEl = document.getElementById("status");
    const linearEl = document.getElementById("linear");
    const angularEl = document.getElementById("angular");
    const stopButton = document.getElementById("stop");
    const modeButton = document.getElementById("mode");
    const cameraCanvas = document.getElementById("camera");
    const cameraStatus = document.getElementById("cameraStatus");
    const cameraContext = cameraCanvas.getContext("2d");
    const MAX_FORWARD_LINEAR_MPS = 1.6666667;
    const MAX_REVERSE_LINEAR_MPS = 0.8333333;
    const MAX_ANGULAR_RPS = 1.4;
    const WHEEL_WIDTH_M = 0.72;
    const TRAJECTORY_LENGTH_M = 2.0;
    const CAMERA_X_M = -0.58;
    const CAMERA_HEIGHT_M = 0.72;
    const CAMERA_PITCH_RAD = 0.60;
    const CAMERA_HFOV_RAD = 2.094;
    const clientId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    let dragging = false;
    let current = {x: 0, y: 0};
    let assistEnabled = true;
    let modeRevision = null;
    let modeSession = null;
    let nextModeSequence = 0;
    let appliedModeSequence = 0;

    function send(x, y) {
      current = {x, y};
      linearEl.textContent = y.toFixed(2);
      angularEl.textContent = x.toFixed(2);
      fetch("/cmd", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...current, client_id: clientId,
                              mode_revision: modeRevision,
                              mode_session: modeSession}),
        keepalive: true
      }).then(() => {
        statusEl.textContent = "已连接";
      }).catch(() => {
        statusEl.textContent = "连接错误";
      });
    }

    function center() {
      dragging = false;
      knob.style.transform = "translate(0px, 0px)";
      send(0, 0);
    }

    function renderMode(enabled) {
      assistEnabled = enabled;
      document.body.dataset.assist = String(enabled);
      modeButton.textContent = enabled ? "辅助模式" : "手动模式";
      modeButton.setAttribute("aria-checked", String(enabled));
    }

    function acceptMode(payload, responseSequence) {
      if (typeof payload.assist_enabled !== "boolean" ||
          !Number.isInteger(payload.revision) ||
          typeof payload.session !== "string") throw new Error("bad mode");
      if (responseSequence < appliedModeSequence) return;
      appliedModeSequence = responseSequence;
      const changed = payload.assist_enabled !== assistEnabled ||
                      payload.revision !== modeRevision ||
                      payload.session !== modeSession;
      modeRevision = payload.revision;
      modeSession = payload.session;
      renderMode(payload.assist_enabled);
      if (changed) center();
    }

    function refreshMode() {
      if (modeButton.disabled) return;
      const responseSequence = ++nextModeSequence;
      fetch("/mode").then(response => {
        if (!response.ok) throw new Error("mode unavailable");
        return response.json();
      }).then(payload => acceptMode(payload, responseSequence)).catch(() => {
        statusEl.textContent = "连接错误";
      });
    }

    function hasCommand() {
      return Math.hypot(current.x, current.y) > 0.001;
    }

    function move(event) {
      const rect = pad.getBoundingClientRect();
      const radius = rect.width / 2;
      const dx = event.clientX - (rect.left + radius);
      const dy = event.clientY - (rect.top + radius);
      const limit = radius * 0.7;
      const distance = Math.hypot(dx, dy);
      const scale = distance > limit ? limit / distance : 1;
      const x = (dx * scale) / limit;
      const y = -(dy * scale) / limit;
      knob.style.transform = `translate(${x * limit}px, ${-y * limit}px)`;
      send(x, y);
    }

    pad.addEventListener("pointerdown", event => {
      dragging = true;
      pad.setPointerCapture(event.pointerId);
      move(event);
    });
    pad.addEventListener("pointermove", event => {
      if (dragging) move(event);
    });
    pad.addEventListener("pointerup", () => {
      dragging = false;
      center();
    });
    pad.addEventListener("pointercancel", () => {
      dragging = false;
      center();
    });
    stopButton.addEventListener("click", center);
    modeButton.addEventListener("click", () => {
      modeButton.disabled = true;
      const responseSequence = ++nextModeSequence;
      fetch("/mode", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({assist_enabled: !assistEnabled})
      }).then(response => {
        if (!response.ok) throw new Error("mode rejected");
        return response.json();
      }).then(payload => acceptMode(payload, responseSequence)).catch(() => {
        statusEl.textContent = "连接错误";
      }).finally(() => {
        modeButton.disabled = false;
      });
    });
    setInterval(() => {
      if (dragging || hasCommand()) send(current.x, current.y);
    }, 100);
    setInterval(updateCamera, 200);
    setInterval(refreshMode, 1000);
    center();
    refreshMode();

    function updateCamera() {
      fetch("/camera/rear/frame").then(response => {
        if (!response.ok) throw new Error("no frame");
        return response.json();
      }).then(frame => {
        if (!frame.width || !frame.height || !frame.data) return;
        if (cameraCanvas.width !== frame.width || cameraCanvas.height !== frame.height) {
          cameraCanvas.width = frame.width;
          cameraCanvas.height = frame.height;
        }
        const raw = atob(frame.data);
        const image = cameraContext.createImageData(frame.width, frame.height);
        for (let src = 0, dst = 0; src < raw.length && dst < image.data.length; src += 3, dst += 4) {
          image.data[dst] = raw.charCodeAt(src);
          image.data[dst + 1] = raw.charCodeAt(src + 1);
          image.data[dst + 2] = raw.charCodeAt(src + 2);
          image.data[dst + 3] = 255;
        }
        cameraContext.putImageData(image, 0, 0);
        drawReverseTrajectory(frame.width, frame.height);
        cameraStatus.textContent = "后摄画面";
      }).catch(() => {
        cameraStatus.textContent = "后摄画面等待中...";
      });
    }

    function drawReverseTrajectory(width, height) {
      const linearLimit = current.y >= 0 ? MAX_FORWARD_LINEAR_MPS : MAX_REVERSE_LINEAR_MPS;
      const linear = current.y * linearLimit;
      const angular = current.x * MAX_ANGULAR_RPS;
      if (linear >= -0.01) return;

      const rearAxlePath = reverseWheelPath(linear, angular, 0);
      const leftPath = reverseWheelPath(linear, angular, WHEEL_WIDTH_M / 2);
      const rightPath = reverseWheelPath(linear, angular, -WHEEL_WIDTH_M / 2);
      drawPath(rearAxlePath, "#00ffff", width, height);
      drawPath(leftPath, "#ffff00", width, height);
      drawPath(rightPath, "#ff0000", width, height);
    }

    function reverseWheelPath(linear, angular, sideOffset) {
      const curvature = Math.abs(linear) > 1e-6 ? angular / linear : 0;
      const points = [];
      for (let distance = 0; distance <= TRAJECTORY_LENGTH_M + 1e-6; distance += 0.1) {
        const s = -Math.min(distance, TRAJECTORY_LENGTH_M);
        let x;
        let y;
        let theta;
        if (Math.abs(curvature) < 1e-6) {
          x = s;
          y = 0;
          theta = 0;
        } else {
          theta = curvature * s;
          x = Math.sin(theta) / curvature;
          y = (1 - Math.cos(theta)) / curvature;
        }
        points.push({x: x - Math.sin(theta) * sideOffset, y: y + Math.cos(theta) * sideOffset});
      }
      return points;
    }

    function projectGroundPoint(point, width, height) {
      const rearForward = -(point.x - CAMERA_X_M);
      const cameraDown = CAMERA_HEIGHT_M;
      const pitchCos = Math.cos(CAMERA_PITCH_RAD);
      const pitchSin = Math.sin(CAMERA_PITCH_RAD);
      const zForward = rearForward * pitchCos + cameraDown * pitchSin;
      const yDown = cameraDown * pitchCos - rearForward * pitchSin;
      if (zForward <= 0.05) return null;

      const aspect = width / height;
      const vFov = 2 * Math.atan(Math.tan(CAMERA_HFOV_RAD / 2) / aspect);
      const u = width / 2 + ((-point.y) / (zForward * Math.tan(CAMERA_HFOV_RAD / 2))) * width / 2;
      const v = height / 2 + (yDown / (zForward * Math.tan(vFov / 2))) * height / 2;
      if (u < -width || u > width * 2 || v < -height || v > height * 2) return null;
      return {x: u, y: v};
    }

    function drawPath(points, color, width, height) {
      const projected = points.map(point => projectGroundPoint(point, width, height)).filter(Boolean);
      if (projected.length < 2) return;

      cameraContext.save();
      cameraContext.lineWidth = 5;
      cameraContext.lineCap = "round";
      cameraContext.lineJoin = "round";
      cameraContext.shadowColor = "rgba(0, 0, 0, 0.55)";
      cameraContext.shadowBlur = 3;
      cameraContext.strokeStyle = color;
      cameraContext.beginPath();
      cameraContext.moveTo(projected[0].x, projected[0].y);
      for (const point of projected.slice(1)) {
        cameraContext.lineTo(point.x, point.y);
      }
      cameraContext.stroke();

      cameraContext.setLineDash([14, 12]);
      cameraContext.lineWidth = 2;
      cameraContext.strokeStyle = "rgba(255, 255, 255, 0.62)";
      cameraContext.stroke();
      cameraContext.restore();
    }
  </script>
</body>
</html>
"""


class WebJoystickNode(Node):
    def __init__(self):
        super().__init__("web_joystick_node")
        self.declare_parameter("http_port", 8090)
        self.declare_parameter("max_forward_linear_mps", 1.6666667)
        self.declare_parameter("max_reverse_linear_mps", 0.8333333)
        self.declare_parameter("max_angular_rps", 1.4)
        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("command_timeout_s", 1.0)

        self._lock = threading.Lock()
        self._last_input = (0.0, 0.0)
        self._last_input_time = 0.0
        self._active_client_id = None
        self._camera_frame = None
        self._assist_enabled = True
        self._mode_session = secrets.token_hex(8)
        self._mode_revision = 0
        self._mode_revision_required = False
        self._mode_requires_neutral = False
        self._pub = self.create_publisher(Twist, "cmd_vel_raw", 10)
        self._mode_pub = self.create_publisher(Bool, "assist_enabled", 10)
        self.create_subscription(Image, "camera/rear/image", self._on_camera_image, 10)
        self.create_timer(0.05, self._publish_command)

        http_port = int(self.get_parameter("http_port").value)
        self._start_http_server(http_port)
        self.get_logger().info(f"Web joystick: http://localhost:{http_port}")

    def _start_http_server(self, port):
        server = ThreadingHTTPServer(
            ("0.0.0.0", port),
            _handler_class(self._handle_message, self._camera_payload,
                           self._mode_payload, self._set_mode),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

    def _on_camera_image(self, msg):
        payload = camera_frame_payload(msg.width, msg.height, msg.encoding, msg.data)
        if payload["data"]:
            with self._lock:
                self._camera_frame = payload

    def _camera_payload(self):
        with self._lock:
            return self._camera_frame

    def _mode_payload(self):
        with self._lock:
            return {
                "assist_enabled": self._assist_enabled,
                "revision": self._mode_revision,
                "session": self._mode_session,
            }

    def _set_mode(self, message):
        try:
            enabled = json.loads(message)["assist_enabled"]
            if type(enabled) is not bool:
                return None
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        with self._lock:
            changed = enabled != self._assist_enabled
            self._assist_enabled = enabled
            if changed:
                self._mode_revision += 1
                self._mode_revision_required = True
                self._last_input = (0.0, 0.0)
                self._last_input_time = time.monotonic()
                self._active_client_id = None
                self._mode_requires_neutral = True
                self._pub.publish(Twist())
                self._mode_pub.publish(Bool(data=enabled))
            return {
                "assist_enabled": enabled,
                "revision": self._mode_revision,
                "session": self._mode_session,
            }

    def _handle_message(self, message):
        valid = True
        try:
            data = json.loads(message)
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            client_id = data.get("client_id")
            mode_revision = data.get("mode_revision")
            mode_session = data.get("mode_session")
            valid = math.isfinite(x) and math.isfinite(y)
        except (TypeError, ValueError, json.JSONDecodeError):
            x = 0.0
            y = 0.0
            client_id = None
            mode_revision = None
            mode_session = None
            valid = False
        with self._lock:
            if not valid:
                return
            versioned = mode_revision is not None or mode_session is not None
            if (versioned and (mode_revision != self._mode_revision
                               or mode_session != self._mode_session)):
                return
            if self._mode_revision_required and not versioned:
                return
            if self._mode_requires_neutral:
                if math.hypot(x, y) > 0.001:
                    return
                self._mode_requires_neutral = False
            now = time.monotonic()
            has_active_command = (
                math.hypot(*self._last_input) > 0.001
                and now - self._last_input_time
                <= float(self.get_parameter("command_timeout_s").value)
            )
            if math.hypot(x, y) <= 0.001 and has_active_command:
                if client_id is None or client_id != self._active_client_id:
                    return
            if math.hypot(x, y) > 0.001:
                self._active_client_id = client_id
            elif client_id == self._active_client_id:
                self._active_client_id = None
            self._last_input = (x, y)
            self._last_input_time = now

    def _publish_command(self):
        timeout = float(self.get_parameter("command_timeout_s").value)
        max_forward_linear = float(self.get_parameter("max_forward_linear_mps").value)
        max_reverse_linear = float(self.get_parameter("max_reverse_linear_mps").value)
        max_angular = float(self.get_parameter("max_angular_rps").value)
        deadzone = float(self.get_parameter("deadzone").value)
        with self._lock:
            x, y = self._last_input
            if time.monotonic() - self._last_input_time > timeout:
                x = 0.0
                y = 0.0
            linear, angular = joystick_to_velocity(
                x,
                y,
                max_forward_linear,
                max_angular,
                deadzone,
                max_reverse_linear,
            )

            msg = Twist()
            msg.linear.x = linear
            msg.angular.z = angular
            self._pub.publish(msg)
            self._mode_pub.publish(Bool(data=self._assist_enabled))


def _handler_class(on_message, camera_payload, mode_payload, set_mode):
    class JoystickHandler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self):
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                self._send_json(415, {"error": "application/json required"})
                return None
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length < 0:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length > 4096:
                self._send_json(413, {"error": "request body too large"})
                return None
            try:
                return self.rfile.read(length).decode("utf-8")
            except UnicodeDecodeError:
                self._send_json(400, {"error": "request body must be UTF-8"})
                return None

        def do_GET(self):
            if self.path == "/mode":
                self._send_json(200, mode_payload())
                return
            if self.path == "/camera/rear/frame":
                payload = camera_payload()
                if payload is None:
                    self.send_error(404)
                    return
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ("/cmd", "/mode"):
                self.send_error(404)
                return
            message = self._read_json_body()
            if message is None:
                return
            if self.path == "/mode":
                result = set_mode(message)
                if result is None:
                    self._send_json(400, {
                        "error": "assist_enabled must be boolean",
                    })
                else:
                    self._send_json(200, result)
                return
            on_message(message)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    return JoystickHandler


def main(args=None):
    rclpy.init(args=args)
    node = WebJoystickNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
