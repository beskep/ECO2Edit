from __future__ import annotations

import dataclasses as dc
import functools
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import more_itertools as mi
from eco2 import Eco2
from eco2 import Eco2Xml as _Eco2Xml
from loguru import logger
from lxml import etree

if TYPE_CHECKING:
    from pathlib import Path

    from lxml.etree import _Element

Level = Literal['debug', 'info', 'warning', 'error', 'raise']

CUSTOM_LAYER = """
<tbl_ykdetail>
  <pcode>0000</pcode>
  <code>0001</code>
  <설명>CustomMaterial</설명>
  <열전도율>0</열전도율>
  <두께>1000</두께>
  <구분>5</구분>
  <열저항>0</열저항>
  <전경색>-16776961</전경색>
  <후경색>-1</후경색>
  <커스텀>Y</커스텀>
</tbl_ykdetail>
"""


class EditorError(ValueError):
    pass


class ElementNotFoundError(EditorError):
    pass


def set_child_text(element: _Element, child: str, value: Any):
    c = mi.one(
        element.iterfind(child),
        too_short=ValueError(f'Child "{child}" not found in {element}'),
        too_long=ValueError(f'Child "{child}" found more than once in {element}'),
    )
    c.text = str(value)


@dc.dataclass(frozen=True)
class Area:
    site: float | None
    """대지면적"""

    building: float | None
    """건축면적"""

    floor: float | None
    """연면적"""

    @staticmethod
    def _value(element: _Element, path: str):
        if (text := element.findtext(path)) is None:
            raise ElementNotFoundError(element, path)

        try:
            return float(text.replace(',', ''))
        except ValueError:
            return None

    @classmethod
    def create(cls, desc: _Element):
        return cls(
            site=cls._value(desc, 'buildm21'),
            building=cls._value(desc, 'buildm22'),
            floor=cls._value(desc, 'buildm23'),
        )


# NOTE 개별 element 수정 함수 등은 Eco2Xml에,
# 전체 케이스 수정 함수는 Eco2Editor에 구현


@dc.dataclass
class Eco2Xml(_Eco2Xml):
    insulation_u_threshold: float = 0.1

    SURFACE_TYPE: ClassVar[tuple[str, ...]] = (
        '외벽(벽체)',
        '외벽(지붕)',
        '외벽(바닥)',
        '내벽(벽체)',
        '내벽(지붕)',
        '내벽(바닥)',
        '간벽',
        '외부창',
        '내부창',
        '지중벽',
    )

    @functools.cached_property
    def area(self):
        if (desc := self.ds.find('tbl_Desc')) is None:
            msg = 'tbl_Desc'
            raise ElementNotFoundError(msg)

        return Area.create(desc)

    def set_elements(self, path: str, value: str | None, *, skip_none: bool = True):
        """
        XML 내 모든 path 수정.

        e.g. 침기율 수정 시 path로 `tbl_zone/침기율` 지정.

        Parameters
        ----------
        path : str
        value : str | None
        set_none : bool, optional
        """
        for e in self.iterfind(path):
            if skip_none and e.text is None:
                continue

            e.text = value

    def surfaces_by_type(self, t: int | str, /):
        if isinstance(t, str):
            t = self.SURFACE_TYPE.index(t)

        for e in self.iterfind('tbl_yk'):
            if int(e.findtext('면형태', -1)) == t:
                yield e

    def layers(self, pcode: str | None):
        for e in self.iterfind('tbl_ykdetail'):
            if e.findtext('pcode') == pcode:
                yield e

    def set_wall_uvalue(self, wall: _Element, uvalue: float):
        code = wall.findtext('code')
        assert code is not None

        # 기존 레이어 삭제
        for layer in self.iterfind('tbl_ykdetail'):
            if layer.findtext('pcode') == code:
                self.ds.remove(layer)

        # 새 레이어 추가
        layer = etree.fromstring(CUSTOM_LAYER)
        set_child_text(layer, 'pcode', code)
        set_child_text(layer, '열전도율', str(uvalue))
        set_child_text(layer, '열저항', f'{1 / uvalue:.4f}')

        last_layer = mi.last(self.iterfind('tbl_ykdetail'))
        index = self.ds.index(last_layer) + 1
        self.ds.insert(index, layer)

        # 벽체 열관류율 수정
        set_child_text(wall, '열관류율', uvalue)

    @staticmethod
    def set_window_uvalue(window: _Element, uvalue: float):
        # 창호열관류율 수정
        set_child_text(window, '창호열관류율', uvalue)

        # 전체 열관류율 수정
        if balcony := float(window.findtext('발코니창호열관류율') or 0):
            t = 1.0 / (1.0 / uvalue + 1.0 / (2.0 * balcony))
            logger.debug('창호 전체 열관류율: {} (glazing={})', t, window)
            total = f'{t:.3f}'
        else:
            total = str(uvalue)

        set_child_text(window, '열관류율', total)

    def set_window_shgc(
        self,
        window: _Element,
        shgc: float,
        *,
        update_zero: bool = False,
    ):
        path = '일사에너지투과율'

        if not update_zero and (float(window.findtext(path) or 0) == 0):
            # '외부창'의 원래 SHGC가 0인 경우 (투과율 없는 문) 값을 수정하지 않음.
            return

        # 일사에너지투과율 수정
        set_child_text(window, path, shgc)

        # 전체 투과율 수정
        if balcony := float(window.findtext('발코니투과율') or 0):
            total = f'{balcony * shgc:.4f}' if balcony else str(shgc)
            set_child_text(window, '투과율', total)

            # tbl_myoun 투과율 수정
            pcode = window.findtext('code')
            assert pcode is not None
            for e in self.iterfind('tbl_myoun'):
                if e.findtext('열관류율2') == pcode:
                    set_child_text(e, '투과율', total)


class Eco2Editor:
    def __init__(self, src: str | Path | Eco2):
        self.eco2 = src if isinstance(src, Eco2) else Eco2.read(src)
        self.xml = Eco2Xml.create(self.eco2)

    def write(self, path: str | Path, *, dsr: bool | None = False):
        """`.eco`, `.tpl` 파일 저장."""
        eco2 = Eco2(
            header=self.eco2.header,
            ds=self.xml.tostring('DS'),
            dsr=self.xml.tostring('DSR'),
        )
        eco2.write(path, dsr=dsr)

    def set_walls(
        self,
        uvalue: float,
        surface_type: str = '외벽(벽체)',
        *,
        if_empty: Level = 'debug',
    ):
        if not (walls := list(self.xml.surfaces_by_type(surface_type))):
            if if_empty == 'raise':
                raise ElementNotFoundError(surface_type)

            logger.log(if_empty.upper(), '`{}`이 존재하지 않음.', surface_type)

        for w in walls:
            if w.findtext('code') == '0':
                continue

            self.xml.set_wall_uvalue(wall=w, uvalue=uvalue)

        return self

    def set_windows(
        self,
        uvalue: float | None = None,
        shgc: float | None = None,
        surface_type: str = '외부창',
        *,
        update_zero_shgc: bool = False,
        if_empty: Level = 'debug',
    ):
        if uvalue is None and shgc is None:
            msg = f'{uvalue=}, {shgc=}'
            raise EditorError(msg)

        if not (windows := list(self.xml.surfaces_by_type(surface_type))):
            if if_empty == 'raise':
                raise ElementNotFoundError(surface_type)

            logger.log(if_empty.upper(), '`{}`이 존재하지 않음.', surface_type)

        for w in windows:
            if w.findtext('code') == '0' and w.findtext('열관류율') == 0:
                continue

            if uvalue is not None:
                self.xml.set_window_uvalue(window=w, uvalue=uvalue)
            if shgc is not None:
                self.xml.set_window_shgc(
                    window=w, shgc=shgc, update_zero=update_zero_shgc
                )

        return self
