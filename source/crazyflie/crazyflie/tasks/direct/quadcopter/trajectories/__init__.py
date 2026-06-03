from .hover import HoverTrajectory
from .static import StaticTrajectory
from .circle import CircleTrajectory
from .lemniscate import LemniscateTrajectory
from .lissajous import LissajousTrajectory
from .land import LandTrajectory

TRAJECTORY_REGISTRY = {
    "hover": HoverTrajectory,
    "static": StaticTrajectory,
    "circle": CircleTrajectory,
    "land": LandTrajectory,
    "lemniscate": LemniscateTrajectory,
    "lissajous": LissajousTrajectory,
}

# from .static import StaticTrajectory

# TRAJECTORY_REGISTRY = {
#     "static": StaticTrajectory,
# }