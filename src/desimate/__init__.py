from abc import ABC, abstractmethod

from rich.console import Group
from rich.live import Live

from desimate.config import config
from desimate.utils.console import TaskSpeed, get_progress


class Step(ABC):
    """
    Generic step class representing a step in the detection process.
    """

    def __init__(self) -> None:
        pass

    def enter(self) -> None:
        pass

    def execute(self, *args, **kwargs):
        self.enter()
        r = self.run(*args, **kwargs)
        self.exit()
        return r

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        """
        Create and add every needed table to the database.
        Run the step, and flush every table.
        """

    def exit(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def get_config(self):
        return config


class ProgressStep(Step, ABC):
    def __init__(self, show_progress: bool = True):
        super().__init__()
        self.show_progress = show_progress
        self.progress = get_progress()

    def enter(self) -> None:
        super().enter()
        if self.show_progress:
            self.progress.start()

    def exit(self) -> None:
        super().exit()
        if self.show_progress:
            self.progress.stop()


class StatusProgressStep(ProgressStep, ABC):
    def __init__(self, show_progress: bool = True):
        super().__init__(show_progress)
        self.live = Live(Group(self.progress), refresh_per_second=10)

    def enter(self) -> None:
        # super().enter()
        if self.show_progress:
            self.live.start()

    def update_status(self, status: str):
        self.live.update(Group(status, self.progress))

    def exit(self) -> None:
        # super().exit()
        if self.show_progress:
            self.live.stop()
