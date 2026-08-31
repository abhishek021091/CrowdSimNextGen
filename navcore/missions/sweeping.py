class SweepingMission:
    def __init__(self): ...

    def reach_nearest_corner(self, px, py):
        gx = np.sign(px) * (self.arena_width - self.robot_sweep_margin)
        gy = np.sign(py) * (self.arena_height - self.robot_sweep_margin)
        sweep_start = (gx, gy)
        if self.robot_sweep_axes == "random":
            sweep_axes = np.random.randint(0, 2)
        else:
            sweep_axes = self.robot_sweep_axes
        if sweep_axes == 0:
            sweep_dir = -1 if gx > 0 else 1
            total_lanes_required = (self.arena_width * 2) / (self.robot_radius * 2)
            if total_lanes_required % 2 == 0:
                sweep_stop = (gx, -gy)
            else:
                sweep_stop = (-gx, -gy)
        else:
            sweep_dir = -1 if gy > 0 else 1
            total_lanes_required = (self.arena_height * 2) / (self.robot_radius * 2)
            if total_lanes_required % 2 == 0:
                sweep_stop = (-gx, gy)
            else:
                sweep_stop = (-gx, -gy)
        self.robot.set(
            px,
            py,
            gx,
            gy,
            0,
            0,
            np.random.uniform(0, 2 * np.pi),
            agent="robot",
            sweep_stop=sweep_stop,
            sweep_start=sweep_start,
            sweep_dir=sweep_dir,
            sweep_axes=sweep_axes,
        )
