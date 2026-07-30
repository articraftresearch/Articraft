"""Minimal stub for the MuJoCo bindings.

MuJoCo ships no type information, and it is an optional dependency, so the type
checker cannot see it on a machine that has not installed the ``sim`` group.
This declares the handful of names ``mini_articraft.simulate`` uses.
"""

from typing import Any

__version__: str

class MjModel:
    nbody: int
    njnt: int
    nv: int
    nq: int
    body_mass: Any
    jnt_type: Any
    jnt_range: Any
    jnt_qposadr: Any
    jnt_dofadr: Any
    opt: Any
    @staticmethod
    def from_xml_path(path: str) -> MjModel: ...
    def __getattr__(self, name: str) -> Any: ...

class MjData:
    ncon: int
    qpos: Any
    qvel: Any
    xpos: Any
    contact: Any
    def __init__(self, model: MjModel) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class mjtObj:
    mjOBJ_BODY: Any
    mjOBJ_JOINT: Any

class mjtJoint:
    mjJNT_HINGE: Any
    mjJNT_SLIDE: Any

def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mj_id2name(model: MjModel, type_: Any, id_: int) -> str: ...
