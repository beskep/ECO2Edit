"""(2025) 제로에너지건축물 탄소절감 성과 분석 및 수용성 강화를 위한 제도 개선 지원 연구."""  # noqa: E501

from __future__ import annotations

import dataclasses as dc
import functools
import itertools
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import more_itertools as mi
import polars as pl
from loguru import logger
from lxml import etree

from eco2edit.editor import Eco2Editor, EditorError, set_child_text

if TYPE_CHECKING:
    from lxml.etree import _Element


Region = Literal['중부1', '중부2', '남부', '제주']
Grade = Literal[5, 4, 3, 2, 1, 0] | None  # None -> base, 0 -> ZEB+

HeatingType = Literal['EHP', 'GHP', '보일러', '지역난방']
CoolingType = Literal['EHP', 'GHP', '흡수식']
EnergySource = Literal['전기', 'LNG', 'LPG', '난방유', '지역난방']


PV = """
<tbl_new>
    <code>0</code>
    <설명>PV</설명>
    <기기종류>태양광</기기종류>
    <가동연료>(없음)</가동연료>
    <태양열종류>(없음)</태양열종류>
    <집열기유형>(없음)</집열기유형>
    <집열판면적 />
    <집열판방위>(없음)</집열판방위>
    <솔라펌프의정격출력 />
    <태양열시스템의성능>(없음)</태양열시스템의성능>
    <무손실효율계수 />
    <열손실계수1차 />
    <열손실계수2차 />
    <축열탱크체적급탕 />
    <축열탱크체적난방 />
    <축열탱크설치장소>(없음)</축열탱크설치장소>
    <태양광모듈면적>9999</태양광모듈면적>
    <태양광모듈기울기>수평</태양광모듈기울기>
    <태양광모듈방위>(없음)</태양광모듈방위>
    <태양광모듈종류>성능치 입력</태양광모듈종류>
    <태양광모듈적용타입>후면통풍형</태양광모듈적용타입>
    <지열히트펌프용량 />
    <열성능비난방 />
    <열성능비냉방 />
    <펌프용량1차 />
    <펌프용량2차 />
    <열교환기설치여부>아니오</열교환기설치여부>
    <팽창탱크설치여부>아니오</팽창탱크설치여부>
    <팽창탱크체적 />
    <열생산능력 />
    <열생산효율 />
    <발전효율 />
    <태양광모듈효율>0.2</태양광모듈효율>
    <지열비고 />
    <열병합신재생여부>false</열병합신재생여부>
    <태양광용량>0</태양광용량>
</tbl_new>
"""


def _dict(obj: _Element, /) -> dict[str, str | None]:
    return {str(e.tag): e.text for e in obj.iterchildren()}


def _filter(variable: str, /, value):
    expr = pl.col(variable)
    return expr.is_null() | (expr == value)


def _grade(value: Grade, /):
    match value:
        case None:
            return 'Base'
        case 0:
            return 'ZEB+'
        case _:
            return f'ZEB{value}'


@dc.dataclass(frozen=True)
class HeatingSystem:
    type: HeatingType
    source: EnergySource

    @classmethod
    def create(cls, element: _Element):
        sources: dict[str | None, EnergySource] = {
            '전기': '전기',
            '천연가스': 'LNG',
            '액화가스': 'LPG',
            '난방유': '난방유',
        }
        data = tuple(element.findtext(x) for x in ['열생산기기방식', '히트연료'])
        args: tuple[HeatingType, EnergySource]

        match data:
            case '히트펌프', '전기':
                args = ('EHP', '전기')
            case '히트펌프', s:
                args = ('GHP', sources[s])
            case '전기보일러', _:
                args = ('보일러', '전기')
            case '보일러', _:
                args = ('보일러', sources[element.findtext('사용연료')])
            case '지역난방', _:
                args = ('지역난방', '지역난방')
            case _:
                msg = 'Unknown heating system'
                raise EditorError(msg, _dict(element))

        return cls(*args)


@dc.dataclass(frozen=True)
class CoolingSystem:
    type: CoolingType
    source: EnergySource

    @classmethod
    def create(cls, element: _Element):
        source: dict[str | None, EnergySource] = {
            '전기': '전기',
            '천연가스': 'LNG',
            '액화가스': 'LPG',
        }
        data = tuple(element.findtext(x) for x in ['냉동기방식', '열생산연결방식'])
        args: tuple[CoolingType, EnergySource]

        match data:
            case '압축식', _:
                args = ('EHP', '전기')
            case '압축식(LNG)', _:
                args = ('GHP', 'LNG')
            case '흡수식', '외부연결':
                args = ('흡수식', '지역난방')
            case '흡수식', '직화식':
                args = ('흡수식', source[element.findtext('사용연료')])
            case _:
                msg = 'Unknown cooling system'
                raise EditorError(msg, _dict(element))

        return cls(*args)


@dc.dataclass
class Case:
    src: Path
    scale: str
    region: str
    grade: Grade

    def __str__(self):
        return f'{self.src.stem}-{self.region}-{_grade(self.grade)}'


class Editor(Eco2Editor):
    REGION: ClassVar[dict[str, str]] = {
        '중부1': '춘천',
        '중부2': '서울',
        '남부': '부산',
        '제주': '제주',
    }
    REGION_CODE: ClassVar[dict[str, str]] = {
        '춘천': '101300',
        '서울': '010100',
        '부산': '020100',
        '제주': '170100',
    }

    def __init__(self, case: Case, setting: pl.DataFrame):
        super().__init__(case.src)
        self.case = case
        self.setting = setting

        self.category = pl.col('category')
        self.part = pl.col('part')
        self.source = pl.col('source')

    @functools.cached_property
    def weather_group(self):
        """{name: code}."""
        return {
            e.findtext('name'): e.findtext('code')
            for e in self.xml.iterfind('weather_group')
        }

    def _value(self, expr: pl.Expr):
        return self.setting.row(by_predicate=expr)[-1]

    def connected_renewable(self, equipment: _Element):
        code = equipment.findtext('연결된시스템')
        assert code is not None

        if code == '0':
            return None

        return mi.one(
            e for e in self.xml.ds.iterfind('tbl_new') if e.findtext('code') == code
        )

    def edit(self):
        self.edit_region()
        self.edit_wall_and_window()
        self.edit_heating_equipment()
        self.edit_cooling_equipment()
        self.edit_heat_recovery_rate()
        self.edit_lighting_load()
        self.edit_pv()

        return self

    def edit_region(self):
        region = self.REGION[self.case.region]
        if (code := self.weather_group[region]) is None:
            raise EditorError(region)

        if code != self.REGION_CODE[region]:
            logger.warning(
                '{} code 불일치 {} != {}', region, code, self.REGION_CODE[region]
            )

        desc = mi.one(self.xml.iterfind('tbl_Desc'))
        set_child_text(desc, 'buildarea', code)

    def edit_wall_and_window(self):
        uvalue = self.category == '열관류율'

        # 벽
        for surface in [
            '외벽(벽체)',
            '외벽(지붕)',
            '외벽(바닥)',
            '내벽(벽체)',
            '내벽(지붕)',
            '내벽(바닥)',
        ]:
            value = float(self._value(uvalue & (self.part == surface)))
            self.set_walls(uvalue=value, surface_type=surface)

        # 창
        window_uvalue = float(self._value(uvalue & (self.part == '외부창')))
        shgc = float(self._value((self.category == 'SHGC') & (self.part == '외부창')))
        self.set_windows(uvalue=window_uvalue, shgc=shgc)

    def _edit_heating_equipment(
        self,
        element: _Element,
        boiler_control_threshold: float = 100,
    ):
        heating = HeatingSystem.create(element)

        # 효율
        part = (
            '전기보일러'
            if (heating.type == '보일러' and heating.source == '전기')
            else heating.type
        )

        try:
            value = float(
                self._value(
                    (self.category == '난방효율')
                    & (self.part == part)
                    & (self.source.is_null() | (self.source == heating.source))
                )
            )
        except pl.exceptions.RowsError as e:
            raise EditorError(heating) from e

        match heating.type:
            case 'EHP' | 'GHP':
                set_child_text(element, '히트난방정격7', str(value))
                set_child_text(element, '히트난방정격10', str(round(value * 0.42, 3)))
            case '보일러' | '지역난방':
                set_child_text(element, '정격보일러효율', str(value * 100))
            case _:
                raise EditorError(heating)

        # 펌프제어
        if (
            heating.type == '지역난방'  # fmt
            or (
                heating.type == '보일러'
                and float(element.findtext('보일러정격출력', 0))  # kW
                >= boiler_control_threshold
            )
        ):
            control = self._value(
                (self.category == '난방제어') & (self.part == '펌프제어')
            )
            set_child_text(element, '펌프제어유형', control)

    def edit_heating_equipment(self):
        for element in self.xml.iterfind('tbl_nanbangkiki'):
            if element.findtext('code') == '0' and element.findtext('설명') == '(없음)':
                continue

            renewable = self.connected_renewable(element)
            if renewable is not None and renewable.findtext('기기종류') == '지열':
                continue

            self._edit_heating_equipment(element)

    def _edit_cooling_equipment(self, element: _Element):
        cooling = CoolingSystem.create(element)

        # 효율
        try:
            value = self._value(
                (self.category == '냉방효율')
                & (self.part == cooling.type)
                & (self.source.is_null() | (self.source == cooling.source))
            )
        except pl.exceptions.RowsError as e:
            raise EditorError(cooling) from e

        set_child_text(element, '열성능비', value)

        # 제어
        match cooling.type:
            case 'EHP' | 'GHP':
                control = self._value(
                    (self.category == '냉방제어') & (self.part == 'HP')
                )

                if (kind := element.findtext('냉동기종류')) != '실내공조시스템':
                    logger.warning('냉동기종류="{}"', kind)

                set_child_text(element, '제어방식', control)
            case '흡수식':
                control = self._value(
                    (self.category == '냉방제어') & (self.part == '흡수식')
                )

                code = element.findtext('code')
                assert code is not None

                for dist in self.xml.ds.iterfind('tbl_bunbae'):
                    if dist.findtext('냉동기') == code:
                        set_child_text(dist, '펌프운전제어유무', control)

    def edit_cooling_equipment(self):
        for element in self.xml.iterfind('tbl_nangbangkiki'):
            if element.findtext('code') == '0' and element.findtext('설명') == '(없음)':
                continue

            renewable = self.connected_renewable(element)
            if renewable is not None and renewable.findtext('기기종류') == '지열':
                continue

            self._edit_cooling_equipment(element)

    def edit_heat_recovery_rate(self):
        recovery = self.category == '열회수율'
        heating = self._value(recovery & (self.part == '난방'))
        cooling = self._value(recovery & (self.part == '냉방'))

        for element in self.xml.iterfind('tbl_kongjo'):
            if element.findtext('code') == '0' and element.findtext('설명') == '(없음)':
                continue
            if element.findtext('열교환기유형') != '전열교환':
                continue

            set_child_text(element, '열회수율', heating)
            set_child_text(element, '열회수율냉', cooling)

    def edit_lighting_load(self):
        value = self._value(
            (self.category == '전기') & (self.part == '평균조명에너지부하율')
        )
        self.set_lighting_load(value)

    def _pv_area(self):
        ratio = float(self._value(self.part == '최대 태양광 모듈면적비'))
        return round(self.xml.area.building * ratio, 2)

    def _sort_renewable(self):
        for idx, element in enumerate(self.xml.ds.iterfind('tbl_new')):
            prev = element.findtext('code')
            assert prev is not None

            next_ = '0' if idx == 0 else f'{idx:04d}'
            set_child_text(element, 'code', next_)

            yield prev, next_

    def edit_pv(self):
        # 기존 PV 삭제
        for element in self.xml.iterfind('tbl_new'):
            if element.findtext('기기종류') == '태양광':
                self.xml.ds.remove(element)

        # 새 PV 적용
        pv = etree.fromstring(PV)
        set_child_text(pv, '태양광모듈면적', self._pv_area())  # TODO 면적 케이스 세팅

        last_renewable = mi.last(self.xml.ds.iterfind('tbl_new'))
        index = self.xml.ds.index(last_renewable) + 1

        self.xml.ds.insert(index, pv)

        # code 순서에 따라 정렬
        codes = dict(self._sort_renewable())

        # 냉난방기기 '연결된시스템' 새 code로 변경
        for equipment in mi.flatten([
            self.xml.iterfind('tbl_nanbangkiki'),
            self.xml.iterfind('tbl_nangbangkiki'),
        ]):
            prev = equipment.findtext('연결된시스템')
            assert prev is not None

            if prev == '0':
                continue

            set_child_text(equipment, '연결된시스템', codes[prev])


app = cyclopts.App(
    config=cyclopts.config.Toml('config.toml', root_keys='carbon-reduction')
)


@app.command
def filename(src: Path, dst: Path, *, name: bool = False):
    """각 인증 폴더 정리.

    `공-A1-교-99` 형식 각 폴더 안에 tpl, tplx 파일이 하나만 있다면 형식에 맞춰 복사.
    둘 이상 있다면 로깅.

    Parameters
    ----------
    src : Path
    dst : Path
    name : bool

    Raises
    ------
    ValueError
    """
    pattern = re.compile(r'^(\w(\-C\d)?\-A\d\-\w\-\d+).*$')
    errors: list[str] = []

    for path in src.glob('*'):
        if not path.is_dir():
            continue

        if (m := pattern.match(path.name)) is None:
            raise ValueError(path.name)

        code = m.group(1)
        logger.info('{}: {}', code, path)

        try:
            s = mi.one(
                x for x in path.rglob('*') if x.suffix.lower() in {'.tpl', '.tplx'}
            )
        except ValueError as e:
            errors.append(f'{path.name}: {e}')
            continue

        n = f'-{s.name}' if name else ''
        d = dst / f'{code}{n}{s.suffix}'
        shutil.copy2(s, d)

    dst.joinpath('errors.txt').write_text('\n'.join(errors), encoding='UTF-8')


@cyclopts.Parameter(name='*')
@dc.dataclass
class BatchEditor:
    src: Path
    dst: Path

    xml: bool = False

    @functools.cached_property
    def settings(self):
        root = Path(__file__).parents[1]
        index = ['category', 'part', 'region', 'source', 'scale']
        return (
            # NOTE 주거 추가 시 수정
            pl.scan_csv(root / 'config/CarbonReduction-NonResidential.csv')
            .unpivot(index=index, variable_name='grade')
            .collect()
        )

    def filter_setting(self, scale: str, region: Region, grade: Grade):
        return self.settings.filter(
            _filter('region', region),
            _filter('scale', scale),
            _filter('grade', _grade(grade)),
        )

    def sources(self):
        for path in self.src.glob('*'):
            if path.is_file() and path.suffix.lower() in {'.tpl', '.tplx'}:
                yield path

    def cases(self):
        sources = tuple(self.sources())
        regions: tuple[Region, ...] = ('중부1', '중부2', '남부', '제주')
        grades: tuple[Grade, ...] = (None, 0, 1, 2, 3, 4, 5)

        pattern = re.compile(r'^\w(\-C\d)?\-(A\d)\-\w\-\d+')

        # TODO track
        for src, region, grade in itertools.product(sources, regions, grades):
            if (m := pattern.search(src.stem)) is None:
                raise ValueError(src.name)

            scale = m.group(2)
            yield Case(src=src, scale=scale, region=region, grade=grade)

    def execute(self):
        for case in self.cases():
            logger.info(case)

            setting = self.filter_setting(
                scale=case.scale, region=case.region, grade=case.grade
            )

            if not (setting).height:
                raise ValueError(case)

            editor = Editor(case, setting)

            try:
                editor.edit()
            except EditorError as e:
                logger.error(repr(e))
                raise

            editor.write(self.dst / f'{case}.tpl')
            if self.xml:
                editor.xml.write(self.dst / f'{case}.xml')


@app.command
def edit(editor: BatchEditor):
    editor.execute()


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level='INFO')

    app()
