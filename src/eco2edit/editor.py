from __future__ import annotations

import dataclasses as dc
import functools
from math import inf
from typing import TYPE_CHECKING, ClassVar

from eco2 import Eco2
from eco2 import Eco2Xml as _Eco2Xml
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from lxml.etree import _Element


NOT_FOUND = '__NOT_FOUND__'  # _Element.findtext 기본 값


class EditorError(ValueError):
    pass


class ElementNotFoundError(EditorError):
    pass


@dc.dataclass(frozen=True)
class Area:
    site: float
    """대지면적"""

    building: float
    """건축면적"""

    floor: float
    """연면적"""

    @classmethod
    def create(cls, desc: _Element):
        def f(path):
            return float(desc.findtext(path, NOT_FOUND).replace(',', ''))

        return cls(site=f('buildm21'), building=f('buildm22'), floor=f('buildm23'))


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
        desc = next(self.iterfind('tbl_Desc'))
        return Area.create(desc)

    def surfaces_by_type(self, t: int | str, /):
        if isinstance(t, str):
            t = self.SURFACE_TYPE.index(t)

        for e in self.iterfind('tbl_yk'):
            if int(e.findtext('면형태', NOT_FOUND)) == t:
                yield e

    def layers(self, pcode: str | None):
        for e in self.iterfind('tbl_ykdetail'):
            if e.findtext('pcode') == pcode:
                yield e

    def find_insulation(self, wall: _Element):
        pcode = wall.findtext('code')

        if not (layers := list(self.layers(pcode))):
            msg = (
                f'벽체 "{wall.findtext("code")}({wall.findtext("설명")})"에 속한 '
                '`ykdetail`이 존재하지 않음.'
            )
            raise ElementNotFoundError(msg)

        # u-value를 맞추기 위해 열저항이 큰 단열재 선택
        # NOTE 에러 발생 시 열전도율순으로 trial and error 방식으로 변경
        possible_insulations = [
            e
            for e in layers
            if float(e.findtext('열전도율') or inf) < self.insulation_u_threshold
        ]
        if possible_insulations:
            insulation = max(
                possible_insulations,
                key=lambda e: (
                    float(e.findtext('열저항') or -inf),
                    float(e.findtext('두께') or -inf),
                    -float(e.findtext('열전도율') or inf),
                ),
            )
        else:
            logger.debug(
                '단열재 미발견: code={}, 설명={}',
                wall.findtext('code'),
                wall.findtext('설명'),
            )
            insulation = min(
                layers,
                key=lambda e: (
                    float(e.findtext('열전도율') or inf),
                    -float(e.findtext('두께') or -inf),
                ),
            )

        layers.remove(insulation)

        return insulation, layers

    def set_wall_uvalue(self, wall: _Element, uvalue: float):
        insulation, layers = self.find_insulation(wall)

        # 변경할 두께 계산, 수정
        r = [float(e.findtext('열저항', NOT_FOUND)) for e in layers]
        u0 = insulation.findtext('열전도율')
        assert u0 is not None
        if (d := (1.0 / uvalue - sum(r)) * float(u0)) <= 0:
            msg = (
                '대상 재료의 두께 계산 결과가 양수가 아닙니다.'
                f'(벽={wall.findtext("설명")}, '
                f'재료={insulation.findtext("설명")}, 두께={d})'
            )
            raise EditorError(msg, insulation)

        logger.debug('target={}, d={:.6f}', insulation.findtext('설명'), d)
        next(insulation.iterfind('두께')).text = f'{1000 * d:.2f}'

        # tbl_yk 열관류율 수정
        next(wall.iterfind('열관류율')).text = str(uvalue)

    @staticmethod
    def set_window_uvalue(window: _Element, uvalue: float):
        # 창호열관류율 수정
        next(window.iterfind('창호열관류율')).text = str(uvalue)

        # 전체 열관류율 수정
        if balcony := float(window.findtext('발코니창호열관류율', NOT_FOUND)):
            # XXX 수식 체크
            t = 1.0 / (1.0 / uvalue + 1.0 / (2.0 * balcony))
            logger.info('창호 전체 열관류율: {} (glazing={})', t, window)
            total = f'{t:.3f}'
        else:
            total = str(uvalue)

        next(window.iterfind('열관류율')).text = total

    def set_window_shgc(self, window: _Element, shgc: float):
        # 일사에너지투과율 수정
        next(window.iterfind('일사에너지투과율')).text = str(shgc)

        # 전체 투과율 수정
        balcony = float(window.findtext('발코니투과율', NOT_FOUND))
        total = f'{balcony * shgc:.4f}' if balcony else str(shgc)
        next(window.iterfind('투과율')).text = total

        # tbl_myoun 투과율 수정
        # XXX 테스트 필요
        pcode = window.findtext('code')
        assert pcode is not None
        for e in self.iterfind('tbl_myoun'):
            if e.findtext('열관류율2') == pcode:
                e.find('투과율').text = total  # type: ignore[union-attr]


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

    def set_walls(self, uvalue: float, surface_type: str = '외벽(벽체)'):
        if not (walls := list(self.xml.surfaces_by_type(surface_type))):
            if surface_type == '외벽(바닥)':
                logger.warning('`외벽(바닥)`이 존재하지 않음.')
            else:
                raise ElementNotFoundError(surface_type)

        for w in walls:
            if w.findtext('code') == '0':
                continue

            logger.trace('{} {}', surface_type, w.findtext('설명'))
            self.xml.set_wall_uvalue(wall=w, uvalue=uvalue)

        return self

    def set_windows(
        self,
        uvalue: float | None = None,
        shgc: float | None = None,
        surface_type: str = '외부창',
    ):
        if uvalue is None and shgc is None:
            msg = f'{uvalue=}, {shgc=}'
            raise EditorError(msg)

        if not (windows := list(self.xml.surfaces_by_type(surface_type))):
            raise ElementNotFoundError(surface_type)

        for w in windows:
            if w.findtext('code') == '0' and w.findtext('열관류율') == 0:
                continue

            if uvalue is not None:
                self.xml.set_window_uvalue(window=w, uvalue=uvalue)
            if shgc is not None:
                self.xml.set_window_shgc(window=w, shgc=shgc)

        return self

    def set_infiltration_rate(self, value: float | str):
        value = str(value)
        for e in self.xml.iterfind('tbl_zone/침기율'):
            if e.text is not None:
                e.text = value

        return self

    def set_light_density(self, value: float | str):
        value = str(value)
        for e in self.xml.iterfind('tbl_zone/조명에너지부하율입력치'):
            if e.text is not None:
                e.text = value

        return self
