from rich.progress import (
    ProgressColumn,
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)


class TaskSpeed(ProgressColumn):
    def render(self, task):
        return f"{(task.speed or 0):4.1f}it/s"


def get_progress():
    return Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TaskSpeed(),
    )
