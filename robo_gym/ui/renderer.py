"""PyGame-based simulation renderer for MazeEnv (§8.1)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robo_gym.maze.cell import TileType
from robo_gym.sim_core.robot import RobotConfig, RobotState
from robo_gym.sim_core.sensor import sensor_world_pose

if TYPE_CHECKING:
    from robo_gym.maze.maze import Maze
    from robo_gym.sim_core.maze_world import MazeWorld
    from robo_gym.sim_core.sensor import Sensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLOUR_NORMAL: tuple[int, int, int] = (240, 240, 240)
_COLOUR_START: tuple[int, int, int] = (180, 230, 180)
_COLOUR_CHECKPOINT: tuple[int, int, int] = (180, 210, 255)
_COLOUR_BLACK_TILE: tuple[int, int, int] = (30, 30, 30)
_COLOUR_WALL: tuple[int, int, int] = (60, 60, 60)
_COLOUR_ROBOT_BODY: tuple[int, int, int] = (220, 80, 80)
_COLOUR_ROBOT_OUTLINE: tuple[int, int, int] = (0, 0, 0)
_COLOUR_HEADING_ARROW: tuple[int, int, int] = (255, 255, 255)
_COLOUR_RAY_NORMAL: tuple[int, int, int] = (100, 200, 100)
_COLOUR_RAY_CLOSE: tuple[int, int, int] = (220, 50, 50)
_COLOUR_HIT_POINT: tuple[int, int, int] = (255, 200, 0)
_COLOUR_TRAJECTORY: tuple[int, int, int] = (100, 140, 220)
_COLOUR_VISITED_TINT: tuple[int, int, int] = (255, 200, 120)

_TILE_COLOURS: dict[TileType, tuple[int, int, int]] = {
    TileType.NORMAL: _COLOUR_NORMAL,
    TileType.START: _COLOUR_START,
    TileType.CHECKPOINT: _COLOUR_CHECKPOINT,
    TileType.BLACK_TILE: _COLOUR_BLACK_TILE,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RendererConfig:
    """Visual and timing configuration for :class:`SimRenderer`.

    Attributes:
        pixels_per_metre:       Display scale factor (px/m).
        margin_px:              Border around the maze in pixels.
        speed_multiplier:       Simulation-to-wall-clock ratio (0.1 to 10.0).
                                1.0 = real-time; 2.0 = twice as fast as real-time.
        trajectory_length:      Maximum number of past positions drawn as a polyline.
        wall_thickness_px:      Wall line width in pixels.
        close_wall_threshold_m: Sensor ray turns red below this true distance (metres).
        window_title:           PyGame window caption.
        fps_cap:                Hard ceiling on rendered frames per second.
    """

    pixels_per_metre: float = 300.0
    margin_px: int = 40
    speed_multiplier: float = 1.0
    trajectory_length: int = 200
    wall_thickness_px: int = 3
    close_wall_threshold_m: float = 0.15
    window_title: str = "MazeBot Simulator"
    fps_cap: int = 60


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WorldToScreen:
    """Stateless world-frame to screen-pixel coordinate transform.

    World origin is at the bottom-left; screen origin is at the top-left.
    The y-axis is flipped: increasing world-y maps to decreasing screen-y.

    Args:
        ppm:      Pixels per metre (scale factor).
        margin:   Pixel border around the maze drawing area.
        window_h: Total window height in pixels.
    """

    ppm: float
    margin: int
    window_h: int

    def point(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world coordinates (metres) to screen pixels."""
        return (
            int(self.margin + wx * self.ppm),
            int(self.window_h - self.margin - wy * self.ppm),
        )

    def length(self, metres: float) -> int:
        """Convert a world-space length to pixels; minimum 1."""
        return max(1, int(metres * self.ppm))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class SimRenderer:
    """PyGame-based visual renderer for MazeEnv.

    Owns the PyGame display surface, clock, and step-mode state.
    Construct once per environment; call :meth:`render` each tick and
    :meth:`close` when done.

    The renderer is intentionally separate from ``MazeEnv`` so that
    environments used headlessly (RL training) never import pygame.
    """

    def __init__(
        self,
        maze: Maze,
        cell_size: float,
        robot_config: RobotConfig,
        sensors: list[Sensor],
        dt: float,
        config: RendererConfig,
        render_mode: str,
    ) -> None:
        """Initialise the PyGame display and layout constants.

        Args:
            maze:         Maze grid (cells, dimensions, start).
            cell_size:    Physical size of one cell in metres.
            robot_config: Robot configuration (chassis, sensors).
            sensors:      Instantiated sensor list (used for draw ordering only).
            dt:           Simulation tick duration in seconds (for timing).
            config:       Visual and timing configuration.
            render_mode:  ``"human"`` or ``"rgb_array"``.
        """
        import pygame  # deferred — only when renderer is actually constructed
        self._pygame = pygame
        self._config = config
        self._maze = maze
        self._cell_size = cell_size
        self._robot_config = robot_config
        self._dt = dt
        self._render_mode = render_mode

        # Compute window size from maze physical dimensions.
        maze_w_px = int(maze.width * cell_size * config.pixels_per_metre)
        maze_h_px = int(maze.height * cell_size * config.pixels_per_metre)
        win_w = maze_w_px + 2 * config.margin_px
        win_h = maze_h_px + 2 * config.margin_px

        self._transform = _WorldToScreen(
            ppm=config.pixels_per_metre,
            margin=config.margin_px,
            window_h=win_h,
        )
        self._win_w = win_w
        self._win_h = win_h

        pygame.init()
        if render_mode == "human":
            self._screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption(config.window_title)
        else:  # rgb_array — off-screen surface
            self._screen = pygame.Surface((win_w, win_h))

        self._clock = pygame.time.Clock()

        # Step-mode state (§8.3).
        self._paused: bool = False
        self._advance_one: bool = False

        # Per-cell visit alpha accumulator for the visited overlay.
        self._visited_alpha: dict[tuple[int, int], int] = {}

        logger.debug(
            "SimRenderer: window %dx%d px, %.0f px/m, mode=%s",
            win_w, win_h, config.pixels_per_metre, render_mode,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        state: RobotState,
        world: MazeWorld,
        trajectory: list[tuple[float, float]],
    ) -> np.ndarray | None:
        """Draw one frame and enforce timing.

        In ``"human"`` mode handles step-mode: blocks while paused until the
        user presses Space (resume) or Enter (advance one tick).

        Args:
            state:      Current robot kinematic state.
            world:      MazeWorld providing wall geometry and ray-cast.
            trajectory: Ordered list of (x, y) positions since last reset.

        Returns:
            ``(H, W, 3)`` uint8 RGB array for ``"rgb_array"`` mode; ``None``
            for ``"human"`` mode.
        """
        pygame = self._pygame

        # --- Step-mode event handling (human mode only) ---
        if self._render_mode == "human":
            self._handle_events()
            # Block while paused until user presses Space or Enter.
            while self._paused and not self._advance_one:
                pygame.display.flip()
                self._clock.tick(30)
                self._handle_events()
            if self._advance_one:
                self._print_state(state)
                self._advance_one = False

        # --- Update visited-cell alpha ---
        cx = int(state.x / self._cell_size)
        cy = int(state.y / self._cell_size)
        self._visited_alpha[(cx, cy)] = min(
            200, self._visited_alpha.get((cx, cy), 0) + 25
        )

        # --- Draw layers ---
        surface = self._screen
        self._draw_tiles(surface)
        self._draw_visited_overlay(surface)
        tail = trajectory[-self._config.trajectory_length:]
        self._draw_trajectory(surface, tail)
        self._draw_walls(surface, world)
        self._draw_sensor_rays(surface, state, world)
        self._draw_robot(surface, state)

        # --- Present / return ---
        if self._render_mode == "human":
            pygame.display.flip()
            target_fps = min(
                self._config.fps_cap,
                self._config.speed_multiplier / self._dt,
            )
            self._clock.tick(target_fps)
            return None

        # rgb_array: pygame surfarray uses (width, height, 3); transpose to (H, W, 3).
        return np.transpose(
            pygame.surfarray.array3d(surface), axes=(1, 0, 2)
        )

    def close(self) -> None:
        """Tear down PyGame and release all resources."""
        self._pygame.quit()
        logger.debug("SimRenderer: closed")

    # ------------------------------------------------------------------
    # Draw methods
    # ------------------------------------------------------------------

    def _draw_tiles(self, surface: Any) -> None:
        """Fill each cell rectangle with its TileType colour."""
        pygame = self._pygame
        cs = self._cell_size
        t = self._transform
        cs_px = t.length(cs)

        for (cx, cy), cell in self._maze.cells.items():
            colour = _TILE_COLOURS[cell.tile_type]
            # World top-left of the cell → screen top-left (y is flipped).
            sx, sy = t.point(cx * cs, (cy + 1) * cs)
            surface.fill(colour, pygame.Rect(sx, sy, cs_px, cs_px))

    def _draw_visited_overlay(self, surface: Any) -> None:
        """Draw an amber tint over visited cells; alpha increases with visits."""
        pygame = self._pygame
        cs = self._cell_size
        t = self._transform
        cs_px = t.length(cs)

        overlay = pygame.Surface((cs_px, cs_px), pygame.SRCALPHA)
        for (cx, cy), alpha in self._visited_alpha.items():
            overlay.fill((*_COLOUR_VISITED_TINT, alpha))
            sx, sy = t.point(cx * cs, (cy + 1) * cs)
            surface.blit(overlay, (sx, sy))

    def _draw_trajectory(
        self,
        surface: Any,
        trajectory: list[tuple[float, float]],
    ) -> None:
        """Draw a fading polyline of the robot's recent positions."""
        if len(trajectory) < 2:
            return
        pygame = self._pygame
        t = self._transform
        n = len(trajectory)
        screen_pts = [t.point(x, y) for x, y in trajectory]
        for i in range(1, n):
            # Older segments are more transparent; approximate by blending
            # with the background by darkening the colour.
            alpha_factor = i / n  # 0 (oldest) → 1 (newest)
            r = int(_COLOUR_TRAJECTORY[0] * alpha_factor)
            g = int(_COLOUR_TRAJECTORY[1] * alpha_factor)
            b = int(_COLOUR_TRAJECTORY[2] * alpha_factor)
            pygame.draw.line(surface, (r, g, b), screen_pts[i - 1], screen_pts[i], 2)

    def _draw_walls(self, surface: Any, world: MazeWorld) -> None:
        """Draw all fused wall segments as thick lines."""
        pygame = self._pygame
        t = self._transform
        thickness = self._config.wall_thickness_px

        for wall in world.walls:
            if wall.is_horizontal:
                p1 = t.point(wall.range_min, wall.fixed_coord)
                p2 = t.point(wall.range_max, wall.fixed_coord)
            else:
                p1 = t.point(wall.fixed_coord, wall.range_min)
                p2 = t.point(wall.fixed_coord, wall.range_max)
            pygame.draw.line(surface, _COLOUR_WALL, p1, p2, thickness)

    def _draw_sensor_rays(
        self,
        surface: Any,
        state: RobotState,
        world: MazeWorld,
    ) -> None:
        """Draw each sensor's ray overlay and hit-point dot."""
        pygame = self._pygame
        t = self._transform
        threshold = self._config.close_wall_threshold_m

        for sensor_cfg in self._robot_config.sensors:
            max_range = getattr(sensor_cfg, "max_range", None)
            if max_range is None:
                continue  # non-ray-casting sensor type

            sx, sy, heading = sensor_world_pose(state, sensor_cfg)
            dx = math.cos(heading)
            dy = math.sin(heading)
            hit = world.ray_cast(origin=(sx, sy), direction=(dx, dy), max_range=max_range)

            hit_x = sx + dx * hit.distance
            hit_y = sy + dy * hit.distance

            is_close = hit.wall_normal is not None and hit.distance < threshold
            colour = _COLOUR_RAY_CLOSE if is_close else _COLOUR_RAY_NORMAL

            pygame.draw.line(
                surface,
                colour,
                t.point(sx, sy),
                t.point(hit_x, hit_y),
                2,
            )
            if hit.wall_normal is not None:
                pygame.draw.circle(surface, _COLOUR_HIT_POINT, t.point(hit_x, hit_y), 4)

    def _draw_robot(self, surface: Any, state: RobotState) -> None:
        """Draw the robot as a rotated rectangle with a heading arrow."""
        pygame = self._pygame
        t = self._transform
        cfg = self._robot_config.chassis

        hw = cfg.body_width / 2.0
        hl = cfg.body_length / 2.0
        cos_t = math.cos(state.theta)
        sin_t = math.sin(state.theta)

        # Body-frame corners: (±hl along heading, ±hw lateral).
        body_corners = [( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)]
        screen_corners = []
        for bx, by in body_corners:
            # Rotate body frame → world frame.
            wx = state.x + cos_t * bx - sin_t * by
            wy = state.y + sin_t * bx + cos_t * by
            screen_corners.append(t.point(wx, wy))

        pygame.draw.polygon(surface, _COLOUR_ROBOT_BODY, screen_corners)
        pygame.draw.polygon(surface, _COLOUR_ROBOT_OUTLINE, screen_corners, 1)

        # Heading arrow: CoM → front midpoint.
        arrow_end_x = state.x + cos_t * hl * 0.8
        arrow_end_y = state.y + sin_t * hl * 0.8
        pygame.draw.line(
            surface,
            _COLOUR_HEADING_ARROW,
            t.point(state.x, state.y),
            t.point(arrow_end_x, arrow_end_y),
            3,
        )

    # ------------------------------------------------------------------
    # Step-mode helpers
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        """Process PyGame events; update paused and advance-one state."""
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("Renderer window closed by user")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                    logger.info(
                        "Step mode: %s", "PAUSED" if self._paused else "RUNNING"
                    )
                elif event.key == pygame.K_RETURN and self._paused:
                    self._advance_one = True

    def _print_state(self, state: RobotState) -> None:
        """Log the full robot state to the terminal when advancing in step mode."""
        logger.info(
            "Step | x=%.4f m  y=%.4f m  theta=%.4f rad"
            "  vx=%.4f m/s  vy=%.4f m/s  omega=%.4f rad/s",
            state.x, state.y, state.theta,
            state.vx, state.vy, state.omega,
        )
