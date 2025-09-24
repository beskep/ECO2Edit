"""
`rich.progress` decorator for iterators.

from https://github.com/tqdm/tqdm/pull/1596
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Self

import rich
from rich.progress import BarColumn, Progress, ProgressColumn, TimeRemainingColumn
from rich.text import Text
from tqdm.std import TqdmWarning, tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from rich.progress import Task
    from rich.table import Table


class UnitScaleColumn(ProgressColumn):
    def __init__(self, *, unit_scale: bool = False, unit_divisor: float = 1000) -> None:
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def unit_format(self, task: Task, num: float, fmt: str = '') -> str:  # noqa: PLR6301
        if task.fields['unit_scale']:
            return tqdm.format_sizeof(num, divisor=task.fields['unit_divisor'])
        return f'{num:{fmt}}'


class FractionColumn(UnitScaleColumn):
    def render(self, task: Task) -> Text:
        if task.total is not None:
            n_fmt = self.unit_format(task, task.completed)
            total_fmt = self.unit_format(task, task.total)
            return Text(f'{n_fmt}/{total_fmt}', style='progress.download')
        return Text('')


class RateColumn(UnitScaleColumn):
    """Renders human readable transfer speed."""

    def __init__(
        self,
        *,
        unit: str = 'it',
        unit_scale: bool = False,
        unit_divisor: float = 1000,
    ) -> None:
        super().__init__(unit_scale=unit_scale, unit_divisor=unit_divisor)
        self.unit = unit

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""  # noqa: DOC201
        speed = task.fields['rate']
        if task.fields['elapsed'] and speed is None:
            speed = task.completed / task.fields['elapsed']
        if speed is not None:
            inv_speed = 1 / speed if speed != 0 else None
            if inv_speed and inv_speed > 1:
                unit_fmt = f's/{task.fields["unit"]}'
                speed_ = inv_speed
            else:
                unit_fmt = f'{task.fields["unit"]}/s'
                speed_ = speed

            speed_fmt = self.unit_format(task, speed_, fmt='5.2f')
        else:
            speed_fmt = '?'
            unit_fmt = f'{task.fields["unit"]}/s'

        return Text(f'{speed_fmt}{unit_fmt}', style='progress.data.speed')


class UnitCompletedColumn(UnitScaleColumn):
    def render(self, task: Task) -> Text:
        if task.total is None:
            completed = self.unit_format(task, task.completed)
            return Text(
                f'{completed:>3}{task.fields["unit"]}', style='progress.percentage'
            )
        return Text(f'{task.percentage:>3.0f}%', style='progress.percentage')


class CompactTimeElapsedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:  # noqa: PLR6301
        elapsed = task.fields['elapsed']
        formatted = tqdm.format_interval(int(elapsed)) if elapsed else '--:--'
        return Text(formatted, style='progress.elapsed')


class PrefixTimeRemainingColumn(TimeRemainingColumn):
    def __init__(self, prefix_str: str = '<', *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prefix_txt = Text(prefix_str)

    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text('')
        return self.prefix_txt + super().render(task)


class PostFixColumn(ProgressColumn):
    def render(self, task: Task) -> Text:  # noqa: PLR6301
        postfix = task.fields.get('postfix')
        return Text(f', {postfix}' if postfix else '', style='progress.percentage')


class NoPaddingProgress(Progress):
    def make_tasks_table(self, tasks: Iterable[Task]) -> Table:
        table = super().make_tasks_table(tasks)
        table.padding = 0  # type: ignore[assignment]
        return table


class tqdmr(tqdm):  # noqa: N801
    _progress: Progress

    def __new__(cls, *_: Any, **__: Any) -> Self:
        return object.__new__(cls)

    @staticmethod
    def _get_free_pos(*_: Any, **__: Any) -> None:
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs = kwargs.copy()
        kwargs['gui'] = True

        progress_columns = kwargs.pop('progress', None)
        options: dict = kwargs.pop('options', {}).copy()

        for k in ('position', 'bar_format'):
            if kwargs.pop(k, None) is not None:
                warnings.warn(
                    f'tqdm.rich does not support the `{k}` option. ',
                    TqdmWarning,
                    stacklevel=2,
                )

        # NOTE: temporary dummy_lock to reuse std_tqdm's __init__
        self._lock = nullcontext()
        self._instances = [self]
        super().__init__(*args, **kwargs)
        del self._lock
        del self._instances

        if self.disable:
            return

        d = self.format_dict
        if progress_columns is None:
            progress_columns = self._columns(
                format_dict=d, bar_options=kwargs.pop('bar_options', {}).copy()
            )

        cls = self.__class__
        if not hasattr(cls, '_progress') or cls._progress is None:
            options.setdefault('transient', not self.leave)

            console = options.get('console', rich.get_console())
            if options.get('update_console'):
                console.file = self.fp
                console.width = d['ncols']
                console.height = d['nrows']

            options['console'] = console
            cls._progress = NoPaddingProgress(*progress_columns, **options)
            cls._progress.__enter__()
        elif 'console' not in options:
            warnings.warn(
                'ignoring passed `console` since tqdm_rich._progress exists',
                TqdmWarning,
                stacklevel=2,
            )

        with cls._progress._lock:  # noqa: SLF001
            # workaround to not refresh on task addition
            disable, cls._progress.disable = cls._progress.disable, True
            task_id = cls._progress.add_task(self.desc or '', **d, start=False)
            self._task = next(t for t in cls._progress.tasks if t.id == task_id)
            cls._progress.disable = disable

    def _columns(
        self,
        format_dict: Mapping,
        bar_options: dict,
    ) -> tuple[str | ProgressColumn, ...]:
        description = '[progress.description]{task.description}: ' if self.desc else ''
        completed = UnitCompletedColumn(
            unit_scale=format_dict['unit_scale'],
            unit_divisor=format_dict['unit_divisor'],
        )
        fraction = FractionColumn(
            unit_scale=format_dict['unit_scale'],
            unit_divisor=format_dict['unit_divisor'],
        )
        rate = RateColumn(
            unit=format_dict['unit'],
            unit_scale=format_dict['unit_scale'],
            unit_divisor=format_dict['unit_divisor'],
        )

        bar_options.setdefault('bar_width', None)
        if format_dict['colour'] is not None:
            bar_options.setdefault('complete_style', format_dict['colour'])
            bar_options.setdefault('finished_style', format_dict['colour'])
            bar_options.setdefault('pulse_style', format_dict['colour'])

        return (
            description,
            completed,
            ' ',
            BarColumn(**bar_options),
            ' ',
            fraction,
            ' [',
            CompactTimeElapsedColumn(),
            PrefixTimeRemainingColumn(compact=True),
            ', ',
            rate,
            PostFixColumn(),
            ']',
        )

    def close(self) -> None:
        if self.disable:
            return
        cls = self.__class__

        if cls._progress is None or self._task is None:
            return
        with cls._progress._lock:  # noqa: SLF001
            if not self._task.finished:
                cls._progress.stop_task(self._task.id)
                self._task.finished_time = self._task.stop_time
            if not self.leave:
                self._task.visible = False

            # print 100%, vis #1306
            self.display(refresh=cls._progress.console.is_jupyter)

            if all(t.finished for t in cls._progress.tasks):
                cls._progress.__exit__(None, None, None)
                cls._progress = None  # type: ignore[assignment]

    def clear(self, *_: Any, **__: Any) -> None:
        pass

    def display(self, *_: Any, refresh: bool = False, **__: Any) -> None:
        cls = self.__class__
        if not hasattr(cls, '_progress') or cls._progress is None or self._task is None:
            return
        if not self._task.started and self.n > 0:
            cls._progress.start_task(self._task.id)

        d = self.format_dict
        self._task.fields['rate'] = d['rate']
        self._task.fields['postfix'] = d['postfix']

        cls._progress.update(
            self._task.id,
            completed=self.n,
            description=self.desc,
            refresh=refresh,
            **d,
        )

        cls._progress.console.width = d['ncols']
        cls._progress.console.height = d['nrows']
        cls._progress.console.file = self.fp

    def refresh(self, *_: Any, **__: Any) -> None:
        self.display()

    def reset(self, total: float | None = None) -> None:
        cls = self.__class__
        if cls._progress is not None:
            cls._progress.reset(task_id=self._task.id)  # see #1378
        super().reset(total=total)


if __name__ == '__main__':
    import time

    for _ in tqdmr(range(100)):
        time.sleep(0.01)
