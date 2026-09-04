from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

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
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
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
    .control, .camera {
      display: grid;
      gap: 18px;
      justify-items: center;
      width: 100%;
    }
    .control {
      width: min(100%, 440px);
      justify-self: center;
    }
    .camera canvas {
      width: 100%;
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
<body>
  <main>
    <div class="camera">
      <canvas id="camera" width="640" height="360"></canvas>
      <div class="camera-label" id="cameraStatus">后摄画面等待中...</div>
    </div>
    <div class="control">
      <h1>SmartWheelChair Joystick</h1>
      <div class="status" id="status">连接中...</div>
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
    const cameraCanvas = document.getElementById("camera");
    const cameraStatus = document.getElementById("cameraStatus");
    const cameraContext = cameraCanvas.getContext("2d");
    const MAX_LINEAR_MPS = 0.8;
    const MAX_ANGULAR_RPS = 1.4;
    const WHEEL_WIDTH_M = 0.72;
    const TRAJECTORY_LENGTH_M = 2.0;
    const CAMERA_X_M = -0.58;
    const CAMERA_HEIGHT_M = 0.72;
    const CAMERA_HFOV_RAD = 2.094;
    let dragging = false;
    let current = {x: 0, y: 0};

    function send(x, y) {
      current = {x, y};
      linearEl.textContent = y.toFixed(2);
      angularEl.textContent = x.toFixed(2);
      fetch("/cmd", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(current),
        keepalive: true
      }).then(() => {
        statusEl.textContent = "已连接";
      }).catch(() => {
        statusEl.textContent = "连接错误";
      });
    }

    function center() {
      knob.style.transform = "translate(0px, 0px)";
      send(0, 0);
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
    setInterval(() => send(current.x, current.y), 100);
    setInterval(updateCamera, 200);
    center();

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
      const linear = current.y * MAX_LINEAR_MPS;
      const angular = -current.x * MAX_ANGULAR_RPS;
      if (linear >= -0.01) return;

      const leftPath = reverseWheelPath(linear, angular, WHEEL_WIDTH_M / 2);
      const rightPath = reverseWheelPath(linear, angular, -WHEEL_WIDTH_M / 2);
      drawPath(leftPath, "#22d3ee", width, height);
      drawPath(rightPath, "#fb923c", width, height);
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
      const zForward = -(point.x - CAMERA_X_M);
      if (zForward <= 0.05) return null;

      const aspect = width / height;
      const vFov = 2 * Math.atan(Math.tan(CAMERA_HFOV_RAD / 2) / aspect);
      const u = width / 2 + ((-point.y) / (zForward * Math.tan(CAMERA_HFOV_RAD / 2))) * width / 2;
      const v = height / 2 + (CAMERA_HEIGHT_M / (zForward * Math.tan(vFov / 2))) * height / 2;
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
        self.declare_parameter("max_linear_mps", 0.8)
        self.declare_parameter("max_angular_rps", 1.4)
        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("command_timeout_s", 0.35)

        self._lock = threading.Lock()
        self._last_input = (0.0, 0.0)
        self._last_input_time = 0.0
        self._camera_frame = None
        self._pub = self.create_publisher(Twist, "cmd_vel_raw", 10)
        self.create_subscription(Image, "camera/rear/image", self._on_camera_image, 10)
        self.create_timer(0.05, self._publish_command)

        http_port = int(self.get_parameter("http_port").value)
        self._start_http_server(http_port)
        self.get_logger().info(f"Web joystick: http://localhost:{http_port}")

    def _start_http_server(self, port):
        server = ThreadingHTTPServer(
            ("0.0.0.0", port),
            _handler_class(self._handle_message, self._camera_payload),
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

    def _handle_message(self, message):
        try:
            data = json.loads(message)
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            x = 0.0
            y = 0.0
        with self._lock:
            self._last_input = (x, y)
            self._last_input_time = time.monotonic()

    def _publish_command(self):
        timeout = float(self.get_parameter("command_timeout_s").value)
        with self._lock:
            x, y = self._last_input
            fresh = time.monotonic() - self._last_input_time <= timeout
        if not fresh:
            x = 0.0
            y = 0.0

        max_linear = float(self.get_parameter("max_linear_mps").value)
        max_angular = float(self.get_parameter("max_angular_rps").value)
        deadzone = float(self.get_parameter("deadzone").value)
        linear, angular = joystick_to_velocity(x, y, max_linear, max_angular, deadzone)

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._pub.publish(msg)


def _handler_class(on_message, camera_payload):
    class JoystickHandler(BaseHTTPRequestHandler):
        def do_GET(self):
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
            if self.path != "/cmd":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            on_message(self.rfile.read(length).decode("utf-8"))
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
