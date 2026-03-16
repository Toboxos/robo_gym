"""Gym environment smoke test: two-sensor wheeled robot, trajectory + sensor plot.

Creates a MazeEnv with a wheeled robot carrying a front-facing and a
right-facing ultrasonic sensor, runs a fixed number of steps, then plots:
  - top panel: maze walls + robot trajectory coloured by time
  - bottom panel: both sensor readings over time

Edit the CONFIG / STEPS / ACTION sections, then run:
    uv run python .user/test_gym.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from robo_gym.env import MazeEnv, SubStepWrapper, RenderWrapper, RealtimeWrapper
from robo_gym.maze import Maze
from robo_gym.maze import generate_prims
from robo_gym.sim_core import (
    ChassisConfig,
    DrivetrainConfig,
    DriveType,
    RobotConfig,
)
from robo_gym.sim_core.ultrasonic import UltrasonicSensorConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ROBOT_CONFIG = RobotConfig(
    chassis=ChassisConfig(
        wheel_base=0.13,
        axle_offset=0.0,
        body_width=0.10,
        body_length=0.12,
    ),
    drivetrain=DrivetrainConfig(
        wheel_radius=0.027,
        max_speed=0.2,
        drive_type=DriveType.WHEEL,
        turn_drag=1.0,
        lateral_slip=None,
    ),
    sensors=(
        UltrasonicSensorConfig(
            name="front",
            position_offset=(0.06, 0.0),   # metres forward from CoM
            angle_offset=0.0,              # facing straight ahead
            max_range=2.0,
        ),
        UltrasonicSensorConfig(
            name="right",
            position_offset=(0.04, -0.05),  # metres to the right of CoM
            angle_offset=-math.pi / 2,     # facing 90° right of heading
            max_range=2.0,
        ),
    ),
)

CELL_SIZE = 0.3     # metres per maze cell
PHYSICS_DT = 0.001   # seconds per phyiscal simulation tick
DT = 0.1            # seconds per observable simulation tick
STEPS = 800000        # total steps to simulate
RNG_SEED = 0

# ---------------------------------------------------------------------------
# PID CONFIG
# ---------------------------------------------------------------------------
P = 0.65
I = 0.001
D = 0.002

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging
print("+-------------------- LOGGING")
loggers = logging.root.manager.loggerDict
print(f"Avaialble channels ({len(loggers.items())})")
for name, logger_obj in loggers.items():
    print(f"- {name}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run() -> None:
    maze = Maze.load_json("mazes/maze_6x6_dfs_seed42.maze.json")
    # maze = generate_prims(6, 6)

    maze_env = MazeEnv(
        robot_config=ROBOT_CONFIG,
        maze=maze,
        cell_size=CELL_SIZE,
        dt=PHYSICS_DT,
        max_steps=None,
        rng_seed=RNG_SEED,
        render_mode="human"
    )

    env = RenderWrapper(maze_env, render_fps=30)
    env = SubStepWrapper(env, control_dt=DT)
    # env = RealtimeWrapper(env, sim_dt=DT)

    obs, _ = env.reset()

    xs      = [maze_env._state.x]
    ys      = [maze_env._state.y]
    thetas  = [maze_env._state.theta]
    ts      = [0.0]
    sensor_front: list[float] = []
    sensor_right: list[float] = []

    # A fixed action that is played until time has passed the limit 
    fixed_actions = [
        (0, np.array([0, 0], np.float32))  # tuple[t, action]
    ]


    err_prev = 0.
    err_sum = 0.

    for i in range(STEPS):
        current_time = (i * DT)
        us_front, us_right = float(obs[0]), float(obs[1])

        if not fixed_actions:
            if us_front < 0.05:
                end_time = current_time + 1.2
                fixed_actions.append((end_time, np.array([-0.5, -0], np.float32)))

                end_time = end_time + 0.5
                fixed_actions.append((end_time, np.array([0.3, 0.3], np.float32)))


        while fixed_actions:
            fixed_action = fixed_actions[0]
            if current_time >= fixed_action[0]:
                fixed_actions.pop(0)
                continue
            
            # Reset error values -> new situation begins
            err_sum = 0
            err_prev = 0

            action = fixed_action[1]
            break
           
        if not fixed_actions:
            err = us_right - 0.12
            if abs(err) < 0.02:
                err = 0

            err_sum += err

            err_diff = err - err_prev if err_prev != 0 else 0.

            turn_rate = P * err + I * err_sum + D * err_diff
            action = np.array([0.3 + turn_rate, 0.3 - turn_rate], dtype=np.float32)
            err_prev = err
        
        print(obs, action, fixed_actions)
        obs, _reward, _terminated, truncated, _info = env.step(action)

        xs.append(maze_env._state.x)
        ys.append(maze_env._state.y)
        thetas.append(maze_env._state.theta)
        ts.append((i + 1) * DT)
        sensor_front.append(us_front)
        sensor_right.append(us_right)

        if truncated:
            print("TRUNCATED")
            break

    _plot(maze, maze_env, xs, ys, thetas, ts, sensor_front, sensor_right)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

ARROW_INTERVAL = 2.0   # seconds between heading arrows
ARROW_LEN = 0.04       # metres


def _plot(
    maze,
    env: MazeEnv,
    xs: list[float],
    ys: list[float],
    thetas: list[float],
    ts: list[float],
    sensor_front: list[float],
    sensor_right: list[float],
) -> None:
    fig, (ax_maze, ax_sensors) = plt.subplots(
        2, 1,
        figsize=(9, 12),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # --- Top panel: maze + trajectory ---
    ax = ax_maze
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    dt_cfg = ROBOT_CONFIG.drivetrain
    ax.set_title(
        f"MazeEnv  —  {maze.width}×{maze.height} cells  |  "
        f"{dt_cfg.drive_type.value}  max_speed={dt_cfg.max_speed} m/s  |  "
        f"front + right ultrasonic sensors"
    )

    # Maze walls
    for wall in env._world._walls:
        if wall.is_horizontal:
            ax.plot(
                [wall.range_min, wall.range_max],
                [wall.fixed_coord, wall.fixed_coord],
                color="saddlebrown", linewidth=2, zorder=3,
            )
        else:
            ax.plot(
                [wall.fixed_coord, wall.fixed_coord],
                [wall.range_min, wall.range_max],
                color="saddlebrown", linewidth=2, zorder=3,
            )

    # Trajectory coloured by time
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(ts[0], ts[-1])
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=2, zorder=4)
    lc.set_array(np.array(ts[:-1]))
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="time (s)")

    # Robot body at start and end
    for pose_x, pose_y, pose_theta, color, label in [
        (xs[0],  ys[0],  thetas[0],  "green", "start body"),
        (xs[-1], ys[-1], thetas[-1], "red",   "end body"),
    ]:
        _draw_robot_body(ax, pose_x, pose_y, pose_theta, color, label)

    # Heading arrows at regular intervals
    last_arrow_t = -ARROW_INTERVAL
    for x, y, theta, t in zip(xs, ys, thetas, ts):
        if t - last_arrow_t >= ARROW_INTERVAL:
            ax.annotate(
                "",
                xy=(x + ARROW_LEN * math.cos(theta), y + ARROW_LEN * math.sin(theta)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
                zorder=6,
            )
            last_arrow_t = t

    ax.plot(xs[0],  ys[0],  "go", markersize=10, label="start", zorder=8)
    ax.plot(xs[-1], ys[-1], "rs", markersize=10, label="end",   zorder=8)
    ax.autoscale_view()
    ax.legend(loc="upper left")

    # --- Bottom panel: sensor readings ---
    step_ts = ts[1:]   # one reading per step, aligned to post-step time
    ax_sensors.plot(step_ts, sensor_front, label="front", color="steelblue", linewidth=1.2)
    ax_sensors.plot(step_ts, sensor_right, label="right", color="darkorange", linewidth=1.2)
    ax_sensors.axhline(
        ROBOT_CONFIG.sensors[0].max_range,
        linestyle="--", color="grey", alpha=0.5, label="max_range",
    )
    ax_sensors.set_xlabel("time (s)")
    ax_sensors.set_ylabel("distance (m)")
    ax_sensors.set_title("Sensor readings over time")
    ax_sensors.legend()
    ax_sensors.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def _draw_robot_body(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    color: str,
    label: str,
) -> None:
    """Draw the oriented robot body rectangle on the axes."""
    chassis = ROBOT_CONFIG.chassis
    w = chassis.body_width
    l = chassis.body_length
    rect = Rectangle(
        (-l / 2, -w / 2), l, w,
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=0.15,
        label=label,
        zorder=5,
    )
    transform = (
        plt.matplotlib.transforms.Affine2D()
        .rotate(theta)
        .translate(x, y)
        + ax.transData
    )
    rect.set_transform(transform)
    ax.add_patch(rect)


if __name__ == "__main__":
    run()
