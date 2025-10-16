from __future__ import annotations

import dataclasses as dc
import functools
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import more_itertools as mi

import eco2edit.editor as ee

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from lxml.etree import _Element

Kind = Literal['individual', 'central']


@dc.dataclass
class RenewableConfig:
    pv: tuple[dict[str, str | float], ...]


@dc.dataclass
class DesignConfig:
    light_density: float
    shgc: float
    uvalue: dict[str, float]

    boiler_efficiency: float
    heat_recovery: dict[str, dict[str, float]]
    heat_pump: dict[str, tuple[float, float]]


@dc.dataclass
class Config:
    design: DesignConfig
    renewable: RenewableConfig

    @classmethod
    def read(cls, path: str | Path = 'config/pim.toml'):
        data = tomllib.loads(Path(path).read_text(encoding='utf-8'))
        return cls(
            design=DesignConfig(**data['design']),
            renewable=RenewableConfig(**data['renewable']),
        )


@dc.dataclass(frozen=True)
class LookupTable:
    xml: ee.Eco2Xml

    # 빈 칸 대신 '0'으로 표시된 element 변환용
    ZERO_NULL: ClassVar[dict[str, str]] = {'0': ''}

    @staticmethod
    def _findtext(e: _Element, path: str) -> str:
        if (text := e.findtext(path)) is None:
            raise KeyError(e, path)

        return text

    def _iter_element(self, path: str | Iterable[str]) -> Generator[_Element]:
        for p in mi.always_iterable(path):
            yield from self.xml.ds.iterfind(p)

    def _table(
        self,
        path: str | Iterable[str],
        tag: str = '설명',
        condition: tuple[str, str] | None = None,
    ) -> Generator[tuple[str, str | None]]:
        for element in self._iter_element(path):
            if condition and element.findtext(condition[0]) != condition[1]:
                continue

            code = self._findtext(element, 'code')
            value = element.findtext(tag)

            yield code, value

    def table(
        self,
        path: str | Iterable[str],
        tag: str = '설명',
        condition: tuple[str, str] | None = None,
    ) -> dict[str, str | None]:
        return dict(self._table(path=path, tag=tag, condition=condition))

    @functools.cached_property
    def profile(self) -> dict[str, str | None]:
        return self.table(path='tbl_profile')


class BaseEco2Editor(ee.Eco2Editor):
    def __init__(
        self,
        src: str | Path | ee.Eco2,
        conf: str | Path | Config = 'config/pim.toml',
        kind: Kind | None = None,
    ):
        super().__init__(src)
        self.conf = conf if isinstance(conf, Config) else Config.read(conf)
        self.kind: Kind = kind or self.guess_kind(src)

    @staticmethod
    def guess_kind(src) -> Kind:
        if not isinstance(src, str | Path):
            raise TypeError(src)

        stem = Path(src).stem
        if re.findall(r'(?=.*개별)(?=.*중앙)', stem):
            raise ValueError(stem)

        if '개별' in stem:
            return 'individual'
        if '중앙' in stem:
            return 'central'

        raise ValueError(stem)

    @functools.cached_property
    def lookup_table(self):
        return LookupTable(self.xml)

    def face_profile(self, face: _Element):
        """입력면 -> 존 프로파일."""
        if (code := face.findtext('존분류')) is None:
            raise ValueError(face)

        zone = mi.one(
            x for x in self.xml.ds.iterfind('tbl_zone') if x.findtext('code') == code
        )
        return self.lookup_table.profile[zone.findtext('프로필')]


if __name__ == '__main__':
    import rich

    rich.print(Config.read())
