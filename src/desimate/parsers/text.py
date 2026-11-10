from pathlib import Path
from typing import NamedTuple


from desimate.parsers import Parser, ReadFile, MultiParser, FileValue


class TextParser(Parser):
    def __init__(self):
        super().__init__()

    def run(self, file: str, granularity: str) -> list[NamedTuple]:
        rf = ReadFile()
        data = rf(file)
        if granularity == "line":
            res = []
            current_offset = 0
            for _i, _t in enumerate(data.split("\n")):
                if _t.strip() != "":
                    res.append(
                        FileValue(
                            ind=_i,
                            path=["line-" + str(_i)],
                            abstract_path=["line"],
                            value=_t,
                            offset_start=current_offset,
                            offset_end=current_offset + len(_t) + 1,
                        )
                    )
                current_offset += len(_t) + 1
            return res
        elif granularity == "file":
            return [
                FileValue(
                    ind=1,
                    path=["file-1"],
                    abstract_path=["file"],
                    value=data,
                    offset_start=0,
                    offset_end=len(data),
                )
            ]


class MultiTextParser(MultiParser):
    def __init__(self, db: Path):
        super().__init__(db, TextParser())
