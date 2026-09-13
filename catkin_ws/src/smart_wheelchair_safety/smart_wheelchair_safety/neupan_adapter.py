"""Dependency-optional adapter around NeuPAN inference."""
import os
import time
import numpy as np


class NeuPANAdapter:
    def __init__(self, planner=None, config_file=None, checkpoint=None, dune_checkpoint=None,
                 max_linear=0.8, max_angular=0.65, action_timeout=0.25,
                 max_range=5.0, body_radius=0.0, planner_kwargs=None):
        self.max_linear = float(max_linear)
        self.reference_speed = self.max_linear
        self.max_angular = float(max_angular)
        self.action_timeout = float(action_timeout)
        self.max_range = float(max_range)
        self.body_radius = float(body_radius)
        self.obstacles = np.empty((0, 2), dtype=float)
        self.initial_path = np.empty((0, 2), dtype=float)
        self._last_update = None
        self.reason = ""
        if planner is not None:
            self.planner = planner
        elif config_file:
            try:
                import yaml
                with open(config_file) as source:
                    config = yaml.safe_load(source)
                checkpoint = dune_checkpoint or checkpoint or config.get("pan", {}).get("dune_checkpoint")
                if not checkpoint or not os.path.isfile(checkpoint):
                    raise ValueError("DUNE checkpoint missing; train it explicitly before startup")
                from neupan import neupan
                kwargs = dict(planner_kwargs or {})
                if checkpoint:
                    pan = dict(config.get("pan", {}))
                    pan.update(kwargs.get("pan", {}))
                    pan["dune_checkpoint"] = checkpoint
                    kwargs["pan"] = pan
                self.planner = neupan.init_from_yaml(config_file, **kwargs)
            except Exception as exc:
                self.planner = None
                self.reason = "neupan unavailable: %s" % exc
        else:
            self.planner = None
            self.reason = "neupan configuration missing"

    @property
    def available(self):
        return self.planner is not None

    @staticmethod
    def _points(points):
        a = np.asarray(points, dtype=float)
        if a.size == 0:
            return np.empty((0, 2), dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError("points must be Nx2")
        return a[np.isfinite(a).all(axis=1)]

    def set_obstacles(self, points, now=None):
        p = self._points(points)
        # Points share the state/path frame (usually odom). Range and footprint
        # filtering belongs to scan conversion, before the world transform.
        self.obstacles = p
        self._last_update = time.monotonic() if now is None else float(now)

    @staticmethod
    def merge_scans(left, right):
        """Merge two scan point arrays into a finite Nx2 array."""
        arrays = [NeuPANAdapter._points(left), NeuPANAdapter._points(right)]
        return np.vstack([a for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.empty((0, 2))

    def set_initial_path(self, path):
        p = np.asarray(path, dtype=float)
        if p.size:
            if p.ndim != 2 or p.shape[1] not in (2, 3):
                raise ValueError("path must be Nx2 or Nx3")
            if not np.isfinite(p).all():
                raise ValueError("path must be finite")
        else:
            p = np.empty((0, 2))
        if len(p):
            # Preserve explicit headings, including turns at the same position.
            p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
        if (p.shape == self.initial_path.shape
                and np.allclose(p, self.initial_path, rtol=0., atol=1e-8)):
            return
        self.initial_path = p.copy()
        if self.planner is not None and hasattr(self.planner, "set_initial_path") and len(p):
            # NeuPAN expects [x, y, heading, gear] columns.
            if p.shape[1] == 2:
                delta = np.vstack((np.diff(p, axis=0), p[-1] - p[-2] if len(p) > 1 else [1., 0.]))
                heading = np.arctan2(delta[:, 1], delta[:, 0])
                p = np.column_stack((p, heading))
            path4 = np.column_stack((p, np.ones(len(p)))).T
            self.planner.set_initial_path([column.reshape(4, 1) for column in path4.T])
            # Preserve the optimizer velocity warm start across reference
            # updates; only clear the endpoint latch from the previous path.
            self.planner.ipath.arrive_flag = False
            self.planner.info["arrive"] = False

    def set_reference_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or not 0. <= speed <= self.max_linear:
            raise ValueError("reference speed outside configured linear limit")
        if self.planner is not None:
            self.planner.set_reference_speed(speed)
        self.reference_speed = speed

    def _clip(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2 or not np.isfinite(a).all():
            raise ValueError("invalid action")
        v = float(a[0]) if len(a) else 0.0
        w = float(a[1]) if len(a) > 1 else 0.0
        return np.array([np.clip(v, -self.reference_speed, self.reference_speed),
                         np.clip(w, -self.max_angular, self.max_angular)])

    def step(self, state, now=None):
        now = time.monotonic() if now is None else float(now)
        if self._last_update is None or not 0 <= now - self._last_update <= self.action_timeout:
            self.reason = "stale input"
            return np.zeros(2), np.empty((0, 3))
        if not self.available:
            return np.zeros(2), np.empty((0, 3))
        try:
            state3 = np.asarray(state, dtype=float).reshape(3, 1)
            if not np.isfinite(state3).all():
                raise ValueError("state must be finite")
            ipath = getattr(self.planner, "ipath", None)
            if len(self.initial_path) and hasattr(ipath, "cur_curve"):
                # Upstream resets to index 0 for each new path and searches
                # only ten points by XY. Locate the remaining path globally,
                # using heading too so co-located pivot poses can progress.
                start = ipath.point_index
                remaining = np.asarray(ipath.cur_curve[start:])[:, :3, 0]
                delta = remaining-state3[:, 0]
                delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
                ipath.point_index = start+int(np.argmin(
                    np.sum(delta[:, :2]**2, axis=1)+(.2*delta[:, 2])**2))
            # Upstream NeuPAN is callable and returns (action, info).
            action, info = self.planner(state3, self.obstacles.T if len(self.obstacles) else None)
            if not isinstance(info, dict):
                raise ValueError("NeuPAN returned no info dictionary")
            if any(info.get(key, False) for key in ("stop", "arrive", "collision")):
                self.reason = "planner stopped"
                return np.zeros(2), np.empty((0, 3))
            trajectory = info.get("opt_state_list", [])
            self.reason = "ok"
            trajectory = np.asarray(trajectory, dtype=float)
            if trajectory.ndim != 3 or trajectory.shape[1:] != (3, 1) or not len(trajectory):
                raise ValueError("invalid trajectory shape")
            trajectory = trajectory[:, :, 0]
            if trajectory.size and not np.isfinite(trajectory).all():
                raise ValueError("invalid trajectory")
            return self._clip(action), trajectory.reshape((-1, 3))
        except Exception as exc:
            self.reason = "inference failed: %s" % exc
            return np.zeros(2), np.empty((0, 3))
