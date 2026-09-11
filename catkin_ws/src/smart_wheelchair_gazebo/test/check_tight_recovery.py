#!/usr/bin/env python3
"""Repeat jamb recovery from a 2 mm rather than 2 cm initial clearance."""
from pathlib import Path
from check_recovery_matrix import cases, main


def tight_cases(angles):
    for case in cases(angles):
        x, y, yaw = case['pose']
        dx, dy = .018*case['normal']
        yield dict(case, name='tight-'+case['name'], pose=(x+dx, y+dy, yaw))


if __name__ == '__main__':
    main(case_factory=tight_cases, extra_sources=(Path(__file__),))
