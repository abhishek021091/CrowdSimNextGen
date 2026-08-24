from pathlib import Path

import cv2
import numpy as np


class Recorder:
    def __init__(self):
        self.recording = False
        self.video_writer: cv2.VideoWriter | None = None

    def start_recording(
        self,
        filename: str = "simulation.mp4",
        fps: int = 60,
        width: int = 1200,
        height: int = 800,
    ) -> None:
        output = Path(filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.video_writer = cv2.VideoWriter(
            str(output),
            fourcc,
            fps,
            (width, height),
        )

        if not self.video_writer.isOpened():
            raise RuntimeError("Failed to create video writer.")

        self.recording = True

    def write_frame(self, frame: np.ndarray) -> None:
        if not self.recording or self.video_writer is None:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

    def stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
