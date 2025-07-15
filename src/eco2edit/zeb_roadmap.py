from __future__ import annotations

import contextlib
import dataclasses as dc
import io
import json
import re
from functools import cached_property
from operator import length_hint
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar, Literal

import more_itertools as mi
import pydash as pyd
from eco2 import Eco2
from loguru import logger
from scipy.stats.qmc import LatinHypercube

from eco2edit.console import Progress
from eco2edit.editor import Eco2Editor

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from lxml.etree import _Element


@dc.dataclass(frozen=True)
class Sample:
    u_wall: float  # 외벽, 지붕, 바닥 열관류율
    u_window: float  # 창호 열관류율
    shgc: float  # SHGC

    heating: float  # 난방 효율
    cooling: float  # 냉방 효율
    hrv: float  # 열회수형 환기장치 / 전열교환기 효율
    cooling_control: float  # 냉방설비 제어방식

    light_density: float  # 조명밀도
    pv_area: float  # PV 모듈면적
    pv_efficiency: float  # PV 모율효율

    _: dc.KW_ONLY

    precision: int = 3
    ranges: dict[str, list[float]]
    COOLING_CONTROL_TYPES: ClassVar[tuple[str, str]] = ('on/off제어', '회전수제어')

    @classmethod
    def n_vars(cls):
        return sum(1 for f in dc.fields(cls) if not f.kw_only)

    @classmethod
    def cooling_control_type(cls, value: float):
        idx = 0 if value < 0.5 else 1  # noqa: PLR2004
        return cls.COOLING_CONTROL_TYPES[idx]

    def _value(self, norm: float, path: int | str | list[str]):
        ranges = pyd.get(self.ranges, path)  # type: ignore[arg-type]
        vmin = min(ranges)
        vmax = max(ranges)
        return float(round(vmin + norm * (vmax - vmin), self.precision))

    def _design_vars(self):
        for norm, path in [
            [self.u_wall, '외벽'],
            [self.u_wall, '지붕'],
            [self.u_wall, '바닥'],
            [self.u_window, '창호'],
            [self.shgc, 'SHGC'],
            [self.heating, '난방설비.EHP'],
            [self.heating, '난방설비.GHP(LNG)'],
            [self.heating, '난방설비.보일러(LNG)'],
            [self.heating, '난방설비.보일러(LPG)'],
            [self.cooling, '난방설비.보일러(난방유)'],
            [self.cooling, '냉방설비.압축식(LNG)'],
            [self.cooling, '냉방설비.압축식(전기)'],
            [self.cooling, '냉방설비.흡수식(LNG)'],
            [self.cooling, '냉방설비.흡수식(지역난방)'],
            [self.hrv, '열회수형환기장치.난방'],
            [self.hrv, '열회수형환기장치.냉방'],
            [self.cooling_control, '냉방설비제어'],
            [self.light_density, '조명밀도'],
            [self.pv_area, 'PV.모듈면적'],
            [self.pv_efficiency, 'PV.모듈효율'],
        ]:
            yield path, self._value(norm=norm, path=path)  # type: ignore[arg-type]

    @cached_property
    def design_vars(self) -> dict[str, float]:
        return dict(self._design_vars())

    def vars(self):
        fields = [f.name for f in dc.fields(self) if not f.kw_only]
        lhs = {f'LHS.{f}': getattr(self, f) for f in fields}
        eco = {
            f'ECO.{k}': (self.cooling_control_type(v) if k == '냉방설비제어' else v)
            for k, v in self.design_vars.items()
        }
        return lhs | eco


def _read_ranges(path: str | Path = 'config/ZebRoadmapVarRange.json'):
    return json.loads(Path(path).read_text('UTF-8'))


@dc.dataclass
class Sampler:
    lhs: LatinHypercube = dc.field(
        default_factory=lambda: LatinHypercube(
            d=Sample.n_vars(),
            scramble=True,
            strength=1,
            optimization='random-cd',
            rng=42,
        )
    )
    ranges: dict = dc.field(default_factory=_read_ranges)

    def _ranges(
        self,
        region: Literal['중부2', '남부'],
        scale: Literal['소규모', '중소규모'],
    ) -> Iterator[tuple[str, list[float]]]:
        for k, v in self.ranges.items():
            if k in {'외벽', '지붕', '바닥'}:
                yield k, v[region]
            elif k == 'SHGC':
                yield k, v[scale]
            else:
                yield k, v

    def sample(
        self,
        n: int,
        region: Literal['중부2', '남부'],
        scale: Literal['소규모', '중소규모'],
        *,
        precision: int = 4,
        reset: bool = False,
    ):
        if reset:
            self.lhs.reset()

        ranges = dict(self._ranges(region=region, scale=scale))
        sample = self.lhs.random(n=n).round(precision)

        for s in sample:
            yield Sample(*s, ranges=ranges)


class Editor(Eco2Editor):
    FLOOR_AREA_THRESHOLD = 3000
    COP_TEMPERATURE_COEF = 0.42

    def __init__(
        self,
        source: str | Path | IO[str],
        name: str,
        *,
        log_excluded: int = 10,
    ) -> None:
        super().__init__(source)

        if (m := re.match(r'^(교육|근린|업무)_(중부2|남부).*', name)) is None:
            raise ValueError(name)

        self.usage = m.group(1)
        self.region = m.group(2)
        self.scale = (
            '소규모' if self.area.floor < self.FLOOR_AREA_THRESHOLD else '중대규모'
        )
        self.log_excluded = log_excluded

    def _heating_type(self, e: _Element):
        path = [
            '설명',
            '열생산기기방식',
            '사용연료',
            '정격보일러효율',
            '히트연료',
            '신재생연결여부',
        ]
        system = {p: e.findtext(p, '') for p in path}
        system_type = None

        match system:
            case {'열생산기기방식': '지역난방'} | {'신재생연결여부': '시스템연결'}:
                logger.log(
                    self.log_excluded,
                    '난방기기 제외: {}',
                    pyd.pick(system, '설명', '열생산기기방식', '신재생연결여부'),
                )
                return None
            case {'정격보일러효율': eff} if float(eff) == 100:  # noqa: PLR2004
                # 기존 효율 100% -> 변경대상 제외 (e.g. 대구대표도서관 CDU)
                logger.log(
                    self.log_excluded,
                    '난방기기 제외: {}',
                    pyd.pick(system, '설명', '정격보일러효율'),
                )
                return None
            case {'열생산기기방식': '보일러', '사용연료': fuel}:
                f = {'천연가스': 'LNG', '액화가스': 'LPG'}.get(fuel, fuel)
                system_type = f'보일러({f})'
            case {'열생산기기방식': '히트펌프', '히트연료': '전기'}:
                system_type = 'EHP'
            case {'열생산기기방식': '히트펌프', '히트연료': '천연가스'}:
                system_type = 'GHP(LNG)'

        if system_type is None:
            logger.warning('미분류 난방기기: {}', system)

        return system_type

    def iter_heating_system(self):
        for e in self.iterfind('tbl_nanbangkiki'):
            if e.findtext('code') != '0':
                yield e, self._heating_type(e)

    def _cooling_type(self, e: _Element):
        path = ['설명', '냉동기방식', '열생산연결방식', '신재생연결여부']
        system = {p: e.findtext(p, '') for p in path}

        match system:
            case {'신재생연결여부': '시스템연결'}:
                logger.log(
                    self.log_excluded,
                    '냉방기기 제외: {}',
                    pyd.pick(system, '설명', '신재생연결여부'),
                )
                return None
            case {'냉동기방식': '압축식'}:
                return '압축식(전기)'
            case {'냉동기방식': '압축식(LNG)'}:
                return '압축식(LNG)'
            case {'냉동기방식': '흡수식', '열생산연결방식': c}:
                c = {'직화식': 'LNG', '외부연결': '지역난방'}[c]
                return f'흡수식({c})'

        logger.warning('미분류 냉방기기: {}', system)
        return None

    def iter_cooling_system(self):
        for e in self.iterfind('tbl_nangbangkiki'):
            if e.findtext('code') != '0':
                yield e, self._cooling_type(e)

    def iter_heat_recovery_ventilator(self):
        for e in self.iterfind('tbl_kongjo'):
            if e.findtext('열교환기유형') == '전열교환':
                yield e

    def set_pv(
        self,
        area: float | str | None = None,
        efficiency: float | str | None = None,
    ):
        # PV 하나만 설치 가정
        pv = mi.one(
            e for e in self.iterfind('tbl_new') if e.findtext('기기종류') == '태양광'
        )

        if area is not None:
            next(pv.iterfind('태양광모듈면적')).text = str(area)
        if efficiency is not None:
            next(pv.iterfind('태양광모듈효율')).text = str(efficiency)

    @classmethod
    def _edit_heating_system(cls, e: _Element, system: str, data: dict[str, float]):
        cop = data[f'난방설비.{system}']

        # 보일러
        if system.startswith('보일러'):
            next(e.iterfind('정격보일러효율')).text = f'{100 * cop:.3f}'
            return

        # HP
        cop_neg15 = (  # -15도 COP
            cop
            if e.findtext('히트난방정격7') == e.findtext('히트난방정격15')
            else cls.COP_TEMPERATURE_COEF * cop
        )
        next(e.iterfind('히트난방정격7')).text = f'{cop:.3f}'
        next(e.iterfind('히트난방정격10')).text = f'{cop_neg15:.3f}'

    def _edit_equipment_efficiency(self, data: dict[str, float]):
        for e, system in self.iter_heating_system():
            if system is None:
                continue

            self._edit_heating_system(e=e, system=system, data=data)

        for e, system in self.iter_cooling_system():
            if system is None:
                continue

            next(e.iterfind('열성능비')).text = str(data[f'냉방설비.{system}'])

        for e in self.iter_heat_recovery_ventilator():
            next(e.iterfind('열회수율')).text = str(data['열회수형환기장치.난방'])
            next(e.iterfind('열회수율냉')).text = str(data['열회수형환기장치.냉방'])

    def _edit_cooling_control(self, value: float):
        control = Sample.cooling_control_type(value)
        for e, system in self.iter_cooling_system():
            if system is None:
                continue

            if (
                e.findtext('냉동기종류') != '(없음)'
                and e.findtext('제어방식') not in Sample.COOLING_CONTROL_TYPES
            ):
                raise ValueError({p: e.findtext(p) for p in ['설명', '제어방식']})

            next(e.iterfind('제어방식')).text = control

    def _edit(self, variable: str, value: float):
        match variable:
            case '외벽' | '지붕' | '바닥':
                surface = {
                    '외벽': '외벽(벽체)',
                    '지붕': '외벽(지붕)',
                    '바닥': '외벽(바닥)',
                }[variable]
                self.set_walls(uvalue=value, surface_type=surface)
            case '창호':
                self.set_windows(uvalue=value, shgc=None)
            case 'SHGC':
                self.set_windows(uvalue=None, shgc=value)
            case '조명밀도':
                self.set_light_density(value)
            case '냉방설비제어':
                self._edit_cooling_control(value)
            case 'PV.모듈면적':
                self.set_pv(area=round(self.area.building * value, 3))
            case 'PV.모듈효율':
                self.set_pv(efficiency=value)
            case _:
                raise ValueError(variable)

    def edit(self, sample: Sample):
        equipment = {
            k: v
            for k, v in sample.design_vars.items()
            if k.startswith(('난방설비.', '냉방설비.', '열회수형환기장치.'))
        }
        others = {k: v for k, v in sample.design_vars.items() if k not in equipment}

        self._edit_equipment_efficiency(equipment)
        for variable, value in others.items():
            self._edit(variable=variable, value=value)


@dc.dataclass
class BatchEditor:
    input_: Iterable[Path]
    output: Path

    _: dc.KW_ONLY

    n: int
    subdir: bool = True

    # TODO range test 기능
    sampler: Sampler = dc.field(default_factory=Sampler)

    def iter_case(self):
        for src in self.input_:
            xml = src if src.suffix == '.xml' else io.StringIO(Eco2.read(src).xml)
            editor = Editor(xml, name=src.name)

            it = self.sampler.sample(
                n=self.n,
                region=editor.region,  # type: ignore[arg-type]
                scale=editor.scale,  # type: ignore[arg-type]
            )
            for is_first, _, sample in mi.mark_ends(it):
                editor = Editor(xml, name=src.name, log_excluded=20 if is_first else 10)
                editor.edit(sample)

                dst = self.output / src.stem if self.subdir else self.output
                if is_first:
                    dst.mkdir(exist_ok=True)

                yield src, dst, editor, sample

    def execute(self):
        total = length_hint(self.input_) * self.n
        t = len(str(total))
        it = Progress.iter(self.iter_case(), total=total)

        variables = []
        for idx, (src, dst, editor, sample) in enumerate(it):
            xml = dst / f'{idx:0{t}d}_{src.stem}.xml'
            variables.append(
                {'source': src.name, 'destination': xml.name} | sample.vars()
            )
            editor.write(xml)

            with contextlib.suppress(OSError):
                data = (
                    Eco2(
                        header=src.with_suffix('.header').read_bytes(),
                        xml=xml.read_text(),
                    )
                    .replace_sftype('10')
                    .drop_dsr()
                    .encrypt()
                )
                xml.with_suffix('.eco').write_bytes(data)

        self.output.joinpath('[variable].json').write_text(
            json.dumps(variables, ensure_ascii=False, indent=4), encoding='UTF-8'
        )


if __name__ == '__main__':
    from cyclopts import App

    from eco2edit.console import LogHandler

    LogHandler.set()
    app = App()

    @app.command
    def edit(
        n: int,
        input_: Path,
        output: Path | None = None,
        ext: tuple[str, ...] = ('eco', 'ecox', 'tpl', 'tplx'),
    ):
        output = output or input_ / 'samples'
        output.mkdir(exist_ok=True)

        dotext = {f'.{x}' for x in ext}
        source = tuple(x for x in input_.glob('*') if x.suffix.lower() in dotext)
        BatchEditor(input_=source, output=output, n=n).execute()

    app()
