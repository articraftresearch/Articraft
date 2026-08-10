from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from articraft.simulate import _tracked_body_separations


def test_prismatic_motion_is_not_counted_as_part_separation() -> None:
    slide = 7
    hinge = 8
    model = SimpleNamespace(
        nbody=4,
        njnt=2,
        jnt_type=np.array([slide, hinge]),
        jnt_bodyid=np.array([2, 3]),
        body_parentid=np.array([0, 0, 1, 2]),
    )
    start = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.2],
        ]
    )

    separations = _tracked_body_separations(model, start, slide_joint_type=slide)

    assert set(separations) == {(2, 3)}
    end = start.copy()
    end[2:, 2] += 0.08
    drift = max(
        abs(float(np.linalg.norm(end[a] - end[b])) - distance)
        for (a, b), distance in separations.items()
    )
    assert drift < 1e-12
