import numpy as np


class Mesh:
    def __init__(self, conditions: dict):

        if not self.check_conditions(conditions):
            raise ValueError("Not enough data to build a mesh!")

        self.interval = conditions["interval"]
        self.steps = conditions["steps"]

        if conditions["step-size"]:
            self.step_size = conditions["step-size"]
            self.partition = np.arange(start=self.interval[0], stop=self.interval[1], step=conditions["step-size"])
        else:
            self.step_size = abs(self.interval[1] - self.interval[0]) / self.steps
            self.partition = np.linspace(start=self.interval[0], stop=self.interval[1], num=conditions["steps"])

    @staticmethod
    def check_conditions(conditions: dict):
        if "interval" not in conditions or "steps" not in conditions or "step-size" not in conditions:
            return False

        elif not conditions["interval"] or not conditions["steps"]:
            return False

        return True

    def __getitem__(self, item: int) -> float:
        return self.partition[item]
