import numpy as np
from multiprocessing import Process, Queue, Event
from ace_teleop.server.dynamixel_agent import DynamixelAgent
from ace_teleop.configs.server.ace_const import *
from geort.mocap.mediapipe_mocap import MediaPipeMocap


class AgentProcess(Process):
    def __init__(
        self, mode: str, cfg: dict, name: str, process_event: Event, res_queue: Queue
    ) -> None:
        super(AgentProcess, self).__init__()
        self.dynamixel_cfg = cfg["dynamixel_cfg"]

        hand_cfg = cfg["hand_cfg"]
        self.cam_num = hand_cfg["cam_num"]
        self.hand_type = hand_cfg["hand_type"]
        self.mode = mode

        self.name = name

        self.process_event = process_event
        self.res_queue = res_queue

    def init(self) -> None:
        self.agent = DynamixelAgent(**self.dynamixel_cfg)
        self.mocap = MediaPipeMocap(cam_id=self.cam_num)

    def compute(self) -> None:
        wrist = self.agent.get_ee()
        fingers = self.mocap.get()["result"]
        self.res_queue.put([wrist, fingers])

    def run(self) -> None:
        self.init()
        while True:
            if self.process_event.is_set():
                self.compute()
                self.process_event.clear()

    def terminate(self) -> None:
        return super().terminate()
