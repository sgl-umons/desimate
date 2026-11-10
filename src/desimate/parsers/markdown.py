import logging
from enum import Enum
from pathlib import Path

from joblib import Parallel, delayed
from wenmode import Wenmode
from wenmode.plugins import frontmatter
from wenmode.presets import github as wenmode_github

from desimate import ProgressStep
from desimate.parsers import ReadFile, FileValue, Parser, MultiParserDB

logger = logging.getLogger(__name__)


class Types(Enum):
    ROOT = "root"
    HEADER = "heading"
    TEXT = "text"
    INLINE_CODE = "inlineCode"
    CODE = "code"
    QUOTE = "blockquote"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "listItem"
    THEMATIC_BREAK = "thematicBreak"
    STRONG = "strong"
    LINK = "link"
    HTML = "html"
    IMAGE = "image"
    TABLE = "table"
    TABLE_ROW = "tableRow"
    TABLE_CELL = "tableCell"
    BREAK = "break"
    DELETE = "delete"
    FOOTNOTE = "footnoteDefinition"
    FOOTNOTE_REF = "footnoteReference"

    BLANK_LINE = "blank_line"
    BLOCK_TEXT = "block_text"
    SOFT_BREAK = "softbreak"
    EMPHASIS = "emphasis"
    LINE_BREAK = "linebreak"
    INLINE_HTML = "inline_html"


class MarkdownToAst(Parser):
    def __init__(self):
        super().__init__()

    def run(self, data: str) -> list[dict]:
        md = Wenmode(wenmode_github, positions=True, plugins=[frontmatter])
        return [md.parse(data).to_ast()]


class ParseAST(Parser):
    IGNORED_TYPES = (
        Types.BLANK_LINE,
        Types.THEMATIC_BREAK,
        Types.SOFT_BREAK,
        Types.LINE_BREAK,
        Types.BREAK,
    )

    def __init__(self):
        super().__init__()

    def get_tree_list(
        self,
        file_content: str,
        ast: list[FileValue],
        stops_at: list[Types] | Types | None,
    ) -> list[FileValue]:
        res = []
        self._get_tree_list_rec(file_content, ast, [], res, stops_at)
        return res

    def _generate_path(self, p: list[str | Types]) -> list[str]:
        res = []
        for _e_id, _type, _level in p:
            _r = _type.value if isinstance(_type, Types) else str(_type)
            if _level is not None:
                _r += "-" + str(_level)
            _r += "#" + str(_e_id)
            res.append(_r)
        return res

    def _generate_no_id_path(self, p: list[str | Types]) -> list[str]:
        res = []
        for _e_id, _type, _level in p:
            _r = _type.value if isinstance(_type, Types) else str(_type)
            if _level is not None:
                _r += "-" + str(_level)
            res.append(_r)
        return res

    def _get_tree_list_rec(
        self,
        file_content: str,
        ast: list[FileValue],
        path: list,
        res: list,
        stops_at: list[Types] | Types | None,
    ) -> None:
        times = {}
        starting_path = path.copy()

        def add_to_res(_tpl, value, _pos, c):
            _path = tuple(path + [(times[_tpl], *_tpl)])
            res.append(
                FileValue(
                    len(res),
                    self._generate_path(_path),
                    self._generate_no_id_path(_path),
                    value,
                    _pos[0],
                    _pos[1],
                    # c.get("attrs", {}),
                )
            )

        for i, c in enumerate(ast):
            try:
                t = Types(c["type"])
            except ValueError as e:
                logger.warning(e)
                continue
            if t in self.IGNORED_TYPES:
                continue

            _tpl = (t, None)

            if "position" in c:
                _pos = c["position"]
                _pos = (_pos["start"]["offset"], _pos["end"]["offset"])
            else:
                _pos = (-1, -1)

            if t != Types.HEADER:
                times.setdefault(_tpl, 0)
                times[_tpl] += 1

            if t in stops_at:
                add_to_res(_tpl, file_content[_pos[0] : _pos[1]], _pos, c)
                return

            if t == Types.ROOT:
                path.append((0, t, None))
                self._get_tree_list_rec(
                    file_content, c["children"], path, res, stops_at
                )
                path.pop()
            if t == Types.HEADER:
                level = c["depth"]
                times.setdefault((t, level), 0)
                times[(t, level)] += 1
                for i in range(len(starting_path), len(path)):
                    p = path[i]
                    if p[1] == Types.HEADER and p[2] >= level:
                        # we need to remove everything after i (included)
                        while len(path) > i:
                            path.pop()
                        break
                path.append((times[(t, level)], t, level))
                self._get_tree_list_rec(
                    file_content, c["children"], path, res, stops_at
                )
            elif t in [
                Types.TEXT,
                Types.INLINE_CODE,
                Types.CODE,
                Types.HTML,
                Types.INLINE_HTML,
            ]:
                add_to_res(_tpl, c["value"], _pos, c)
            elif t in [
                Types.PARAGRAPH,
                Types.BLOCK_TEXT,
                Types.QUOTE,
                Types.TABLE,
                Types.TABLE_ROW,
                Types.TABLE_CELL,
                Types.FOOTNOTE,
                Types.STRONG,
                Types.LINK,
                Types.EMPHASIS,
                Types.DELETE,
            ]:
                path.append(
                    (times[_tpl], t, None)
                )  # TODO: Should all treat them as "text"?
                self._get_tree_list_rec(
                    file_content, c["children"], path, res, stops_at
                )
                path.pop()
            elif t == Types.LIST:
                path.append((times[(t, None)], t, None))
                for i, li in enumerate(c["children"]):
                    path.append((i + 1, Types.LIST_ITEM, None))
                    self._get_tree_list_rec(
                        file_content, li["children"], path, res, stops_at
                    )
                    path.pop()
                path.pop()
            elif t in [Types.IMAGE]:
                add_to_res(_tpl, c["url"], _pos, c)  # or c["alt"] TODO
            elif t == Types.FOOTNOTE_REF:
                add_to_res(_tpl, c["identifier"], _pos, c)  # or c["label"] TODO

    def run(
        self, content: str, ast: list[FileValue], stops_at: list[Types] | None = None
    ) -> list[FileValue] | None:
        """
        Transform the intermediary AST representation into the final one.

        If `stops_at` is given, the AST will be truncated once
        a node of type `stops_at` is reached. If `stops_at` is a list,
        the AST will be truncated once a node corresponding to one of
        the type in the list is reached.
        Example: if `stops_at` = Types.PARAGRAPH, then
        no child of nodes of type PARAGRAPH will be computed.
        They will be contained inside the value of PARAGRAPH in
        their Markdown representation.
        """
        return self.get_tree_list(content, ast, stops_at)


class ReadParseAST(Parser):
    def __init__(self):
        super().__init__()

    def run(self, file: Path, stops_at: list[Types]) -> None:
        try:
            logger.debug("Processing %s", str(file))
            read_file_step = ReadFile()
            ast_step = MarkdownToAst()
            rfstep = ParseAST()
            content = read_file_step(file)
            ast = ast_step(content)
            res = rfstep(content, ast, stops_at=stops_at)
            return res
        except Exception as e:  # noqa: BLE001
            logger.error(e)
            return None


class MultiAST(Parser, ProgressStep):
    def __init__(self, db: Path):
        super().__init__()
        self.db = db
        self.ps = MultiParserDB(self.db, ReadParseAST())
        self.granularities = (
            self.get_config().get("markdown", {}).get("granularities", {})
        )

    def run(self, files: list[Path], granularity: str, batch_size=1000) -> None:
        def safe_types(_type):
            try:
                return Types(_type)
            except ValueError:
                logger.warning("Unknown type '%s', ignoring...", _type)
                return None

        if granularity == "section":
            raise NotImplementedError(
                "This granularity was not yet implemented, but will come in a further version."
            )

        stops_at = self.granularities.get(granularity, [])
        stops_at = [safe_types(_i) for _i in stops_at]
        stops_at = [_i for _i in stops_at if _i is not None]
        batches = [
            (i, min(len(files), i + batch_size))
            for i in range(0, len(files), batch_size)
        ]

        task = self.progress.add_task("Reading and storing...", total=len(files))
        parallel = Parallel(return_as="generator_unordered", n_jobs=-1)
        output = parallel(
            delayed(self.ps)(files[b[0] : b[1]], stops_at) for b in batches
        )
        for _n in output:
            self.progress.update(task, advance=_n)
