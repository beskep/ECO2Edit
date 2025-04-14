from __future__ import annotations

import functools
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING

from eco.eco2xml import Eco2Xml
from loguru import logger

if TYPE_CHECKING:
    from lxml.etree._element import _Element


NOT_FOUND = '__NOT_FOUND__'  # _Element.findtext 기본 값


class ElementNotFoundError(ValueError):
    pass


@dataclass
class Area:
    site: float
    building: float
    floor: float

    @classmethod
    def create(cls, desc: _Element):
        def f(path):
            return float(desc.findtext(path, NOT_FOUND).replace(',', ''))

        return cls(site=f('buildm21'), building=f('buildm22'), floor=f('buildm23'))


class Eco2Editor(Eco2Xml):
    SURFACE_TYPE = (
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
    INSULATION_U_THRESHOLD = 0.1

    @functools.cached_property
    def area(self):
        desc = next(self.iterfind('tbl_Desc'))
        return Area.create(desc)

    def _surfaces_by_type(self, t: int | str, /):
        if isinstance(t, str):
            t = self.SURFACE_TYPE.index(t)

        for e in self.iterfind('tbl_yk'):
            if int(e.findtext('면형태', NOT_FOUND)) == t:
                yield e

    def _layers(self, pcode: str | None):
        for e in self.iterfind('tbl_ykdetail'):
            if e.findtext('pcode') == pcode:
                yield e

    def set_wall_uvalue(self, wall: _Element, uvalue: float):
        pcode = wall.findtext('code')

        if not (layers := list(self._layers(pcode))):
            msg = (
                f'벽체 "{wall.findtext("code")}({wall.findtext("설명")})"에 속한 '
                '`ykdetail`이 존재하지 않음.'
            )
            raise ElementNotFoundError(msg)

        # u-value를 맞추기 위해 열전도율이 낮고 두꺼운 단열재 선택
        # TODO 열전도율순으로 trial and error
        possible_insulations = [
            e
            for e in layers
            if float(e.findtext('열전도율') or inf) < self.INSULATION_U_THRESHOLD
        ]
        if possible_insulations:
            insulation = max(
                possible_insulations,
                key=lambda e: (
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
            raise ValueError(msg)

        logger.debug('target={}, d={:.6f}', insulation.findtext('설명'), d)
        next(insulation.iterfind('두께')).text = f'{1000 * d:.2f}'

        # tbl_yk 열관류율 수정
        next(wall.iterfind('열관류율')).text = str(uvalue)

    def set_walls(self, uvalue: float, surface_type: str = '외벽(벽체)'):
        if not (walls := list(self._surfaces_by_type(surface_type))):
            if surface_type == '외벽(바닥)':
                logger.warning('`외벽(바닥)`이 존재하지 않음.')
                return

            raise ElementNotFoundError(surface_type)

        for w in walls:
            if w.findtext('code') == '0':
                continue

            logger.trace('{} {}', surface_type, w.findtext('설명'))
            self.set_wall_uvalue(wall=w, uvalue=uvalue)

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

    def set_windows(
        self,
        uvalue: float | None = None,
        shgc: float | None = None,
        surface_type: str = '외부창',
    ):
        if uvalue is None and shgc is None:
            msg = f'{uvalue=}, {shgc=}'
            raise ValueError(msg)

        if not (windows := list(self._surfaces_by_type(surface_type))):
            raise ElementNotFoundError(surface_type)

        for w in windows:
            if w.findtext('code') == '0' and w.findtext('열관류율') == 0:
                continue

            if uvalue is not None:
                self.set_window_uvalue(window=w, uvalue=uvalue)
            if shgc is not None:
                self.set_window_shgc(window=w, shgc=shgc)

    def set_infiltration_rate(self, value: float | str):
        value = str(value)
        for e in self.iterfind('tbl_zone/침기율'):
            if e.text is not None:
                e.text = value

    def set_light_density(self, value: float | str):
        value = str(value)
        for e in self.iterfind('tbl_zone/조명에너지부하율입력치'):
            if e.text is not None:
                e.text = value
