"""Dependency-optional adapter around NeuPAN inference."""
import os
import time
import numpy as np


class NeuPANAdapter:
    def __init__(self, planner=None, config_file=None, checkpoint=None, dune_checkpoint=None,
                 max_linear=0.8, max_angular=0.65, action_timeout=0.25,
                 max_range=5.0, body_radius=0.0, planner_kwargs=None):
        self.max_linear = float(max_linear)
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
        a = a.reshape((-1, 2))
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
            p = p[np.isfinite(p).all(axis=1), :2]
        else:
            p = np.empty((0, 2))
        if len(p):
            # Normalize to a finite, ordered Nx2 path and remove duplicate points.
            p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
        self.initial_path = p
        if self.planner is not None and hasattr(self.planner, "set_initial_path") and len(p):
            # NeuPAN expects [x, y, heading, gear] columns.
            delta = np.vstack((np.diff(p, axis=0), p[-1] - p[-2] if len(p) > 1 else [1., 0.]))
            heading = np.arctan2(delta[:, 1], delta[:, 0])
            path4 = np.column_stack((p, heading, np.ones(len(p)))).T
            self.planner.set_initial_path([column.reshape(4, 1) for column in path4.T])
            # Preserve the optimizer velocity warm start across reference
            # updates; only clear the endpoint latch from the previous path.
            self.planner.ipath.arrive_flag = False
            self.planner.info["arrive"] = False

    def _clip(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2 or not np.isfinite(a).all():
            raise ValueError("invalid action")
        v = float(a[0]) if len(a) else 0.0
        w = float(a[1]) if len(a) > 1 else 0.0
        return np.array([np.clip(v, -self.max_linear, self.max_linear),
                         np.clip(w, -self.max_angular, self.max_angular)])

    def step(self, state, now=None):
        now = time.monotonic() if now is None else float(now)
        if self._last_update is None or not 0 <= now - self._last_update <= self.action_timeout:
            self.reason = "stale input"
            return np.zeros(2), np.empty((0, 2))
        if not self.available:
            return np.zeros(2), np.empty((0, 2))
        try:
            state3 = np.asarray(state, dtype=float).reshape(-1)[:3].reshape(3, 1)
            # Upstream NeuPAN is callable and takes state (3,1), points (2,N),
            # and optional point velocities; test doubles may expose step().
            if hasattr(self.planner, "step"):
                result = self.planner.step(self.obstacles, self.initial_path, state3)
            else:
                result = self.planner(state3, self.obstacles.T if len(self.obstacles) else None)
            if isinstance(result, dict):
                action = result.get("action", (0, 0))
                trajectory = result.get("trajectory", result.get("opt_state_list", []))
            else:
                action, info = result
                if isinstance(info, dict) and any(info.get(key, False) for key in ("stop", "arrive", "collision")):
                    self.reason = "planner stopped"
                    return np.zeros(2), np.empty((0, 3))
                trajectory = info.get("opt_state_list", []) if isinstance(info, dict) else []
            self.reason = "ok"
            trajectory = np.asarray(trajectory, dtype=float)
            if trajectory.ndim == 3:
                trajectory = np.concatenate([x.T[:, :3] for x in trajectory], axis=0)
            elif trajectory.ndim == 2 and trajectory.shape[0] >= 3 and trajectory.shape[1] != 3:
                trajectory = trajectory[:3].T
            elif trajectory.ndim == 2 and trajectory.shape[1] == 2:
                trajectory = np.column_stack((trajectory, np.zeros(len(trajectory))))
            elif trajectory.size == 0:
                trajectory = np.empty((0, 3))
            else:
                raise ValueError("invalid trajectory shape")
            if trajectory.size and not np.isfinite(trajectory).all():
                raise ValueError("invalid trajectory")
            return self._clip(action), trajectory.reshape((-1, 3))
        except Exception as exc:
            self.reason = "inference failed: %s" % exc
            return np.zeros(2), np.empty((0, 2))
