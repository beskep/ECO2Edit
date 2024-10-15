from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import rich
from loguru import logger
from rich import progress
from rich.logging import RichHandler
from rich.theme import Theme

if TYPE_CHECKING:
    from logging import LogRecord


class _Handler(RichHandler):
    LVLS: ClassVar[dict[str, int]] = {
        'TRACE': 5,
        'DEBUG': 10,
        'INFO': 20,
        'SUCCESS': 25,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50,
    }
    BLANK_NO = 21
    _NEW_LVLS: ClassVar[dict[int, str]] = {5: 'TRACE', 25: 'SUCCESS', BLANK_NO: ''}

    def emit(self, record: LogRecord) -> None:
        if name := self._NEW_LVLS.get(record.levelno, None):
            record.levelname = name

        return super().emit(record)


console = rich.get_console()
console.push_theme(Theme({'logging.level.success': 'bold blue'}))
_handler = _Handler(console=console, markup=True, log_time_format='[%X]')


def set_logger(level: int | str = 20):
    if isinstance(level, str):
        try:
            level = _Handler.LVLS[level.upper()]
        except KeyError as e:
            msg = f'`{level}` not in {list(_Handler.LVLS.keys())}'
            raise KeyError(msg) from e

    logger.remove()
    logger.add(_handler, level=level, format='{message}', backtrace=False)
    logger.add(
        'eco2.log',
        level='DEBUG',
        rotation='1 month',
        retention='1 year',
        encoding='UTF-8-SIG',
    )

    try:
        logger.level('BLANK')
    except ValueError:
        # 빈 칸 표시하는 'BLANK' level 새로 등록
        logger.level(name='BLANK', no=_Handler.BLANK_NO)


class Progress(progress.Progress):
    @classmethod
    def get_default_columns(cls) -> tuple[progress.ProgressColumn, ...]:
        return (
            progress.TextColumn('[progress.description]{task.description}'),
            progress.BarColumn(bar_width=60),
            progress.MofNCompleteColumn(),
            progress.TaskProgressColumn(show_speed=True),
            progress.TimeRemainingColumn(elapsed_when_finished=True),
        )
