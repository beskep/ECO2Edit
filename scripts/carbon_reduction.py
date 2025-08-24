"""(2025) 제로에너지건축물 탄소절감 성과 분석 및 수용성 강화를 위한 제도 개선 지원 연구."""  # noqa: E501

from __future__ import annotations

import dataclasses as dc
import enum
import functools
import itertools
import math
import re
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import more_itertools as mi
import polars as pl
from loguru import logger
from lxml import etree
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from eco2edit import report
from eco2edit.editor import Eco2Editor, Eco2Xml, EditorError, set_child_text

if TYPE_CHECKING:
    from lxml.etree import _Element


Region = Literal['중부1', '중부2', '남부', '제주']
Grade = Literal[5, 4, 3, 2, 1, 0] | None  # None -> base, 0 -> ZEB+
PVArea = Literal['zero', 'required']

HeatingType = Literal['EHP', 'GHP', '보일러', '지역난방']
CoolingType = Literal['EHP', 'GHP', '흡수식']
EnergySource = Literal['전기', 'LNG', 'LPG', '난방유', '지역난방']

EIR: dict[Grade, float | None] = {
    None: 0.13,
    5: 0.2,
    4: 0.4,
    3: 0.6,
    2: 0.8,
    1: 1.0,
    0: 1.2,
}
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
PV_GEN_PER_AREA: dict[str, dict[Region, float]] = {
    # kWh/m²
    'PV': {
        '중부1': 201.983736,
        '중부2': 197.503394,
        '남부': 224.25875,
        '제주': 209.032917,
    },
    'BIPV': {
        '중부1': 89.046987,
        '중부2': 97.839892,
        '남부': 115.092109,
        '제주': 80.210067,
    },
}


class Building(enum.StrEnum):
    RESIDENTIAL = 'residential'
    NON_RESIDENTIAL = 'non-residential'

    R = RESIDENTIAL
    NR = NON_RESIDENTIAL


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
    boiler_capacity: float

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

        boiler_capacity = (
            float(element.findtext('보일러정격출력', 0)) if args[0] == '보일러' else 0
        )

        return cls(*args, boiler_capacity=boiler_capacity)


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
    region: Region
    grade: Grade

    def __str__(self):
        return f'{self.src.stem}-{self.region}-{_grade(self.grade)}'


class Editor(Eco2Editor):
    REGION: ClassVar[dict[Region, str]] = {
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

    def __init__(
        self,
        case: Case,
        setting: pl.DataFrame,
        bldg: Building,
        pv: float = 0,
        bipv: float = 0,
    ):
        if pv == 'required':
            msg = f'{pv=}'
            raise EditorError(msg)

        super().__init__(case.src)
        self.case = case
        self.setting = setting
        self.bldg = bldg
        self.pv: float = pv
        self.bipv: float = bipv

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
        try:
            return self.setting.row(by_predicate=expr)[-1]
        except pl.exceptions.TooManyRowsReturnedError as e:
            raise EditorError(self.setting) from e

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
        boiler_control_threshold: float | None = None,
    ):
        heating = HeatingSystem.create(element)
        boiler_control_threshold = (
            boiler_control_threshold  # fmt
            or (100 if self.bldg == Building.NON_RESIDENTIAL else math.inf)
        )

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
            or (heating.boiler_capacity >= boiler_control_threshold)
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
                if (kind := element.findtext('냉동기종류')) != '실내공조시스템':
                    logger.warning('냉동기종류="{}"', kind)
                    return

                control = self._value(
                    (self.category == '냉방제어') & (self.part == 'HP')
                )
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

    @functools.cached_property
    def max_pv_area(self):
        if self.xml.area.building is None:
            msg = '건축면적 해석 불가'
            raise EditorError(msg)

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

        if not (self.pv or self.bipv):
            return

        # PV
        if self.pv:
            pv = etree.fromstring(PV)
            set_child_text(pv, '태양광모듈면적', f'{self.pv:.3f}')
            set_child_text(pv, '태양광모듈효율', '0.2')

            last_renewable = mi.last(self.xml.ds.iterfind('tbl_new'))
            index = self.xml.ds.index(last_renewable) + 1

            self.xml.ds.insert(index, pv)

        # BIPV
        if self.bipv:
            bipv = etree.fromstring(PV)
            set_child_text(bipv, '설명', 'BIPV')
            set_child_text(bipv, '태양광모듈면적', f'{self.bipv:.3f}')
            set_child_text(bipv, '태양광모듈기울기', '수직')
            set_child_text(bipv, '태양광모듈방위', '남')
            set_child_text(bipv, '태양광모듈적용타입', '밀착형')
            set_child_text(bipv, '태양광모듈효율', '0.16')

            last_renewable = mi.last(self.xml.ds.iterfind('tbl_new'))
            index = self.xml.ds.index(last_renewable) + 1

            self.xml.ds.insert(index, bipv)

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
    bldg: Building
    src: Path
    dst: Path | None = None

    _: dc.KW_ONLY

    pv: PVArea | float = 'zero'
    xml: bool = False
    required_pv: str | Path | None = 'Required-PV.parquet'
    precision: int = 2

    _BASE_INDEX: ClassVar[int] = 6

    @functools.cached_property
    def settings(self):
        root = Path(__file__).parents[1]
        index = ['category', 'part', 'region', 'source', 'scale']
        return (
            pl.scan_csv(root / f'config/carbon-reduction/{self.bldg}.csv')
            .unpivot(index=index, variable_name='grade')
            .collect()
        )

    @property
    def output(self):
        return self.dst or self.src.parent / f'output-pv-{self.pv}'

    def filter_setting(self, scale: str, region: Region, grade: Grade):
        return self.settings.filter(
            _filter('region', region),
            _filter('scale', scale),
            _filter('grade', _grade(grade)),
        )

    @functools.cached_property
    def _required_pv(self):
        if not self.required_pv:
            return None

        path = self.src.parent / self.required_pv
        if not path.exists():
            return None

        k = 10**self.precision
        return pl.read_parquet(path).select(
            'case',
            pl.col('grade').fill_null(self._BASE_INDEX),
            (pl.col('PV면적') * k).ceil().truediv(k).alias('PV'),
            (pl.col('BIPV면적') * k).ceil().truediv(k).alias('BIPV'),
        )

    def _pv_area(self, case: Case) -> tuple[float, float]:
        if isinstance(self.pv, float | int):
            return self.pv, 0

        match self.pv:
            case 'zero':
                return 0, 0
            case 'required':
                if self._required_pv is None:
                    msg = '요구 PV 면적이 입력되지 않음.'
                    raise EditorError(msg)

                grade = self._BASE_INDEX if case.grade is None else case.grade
                row = self._required_pv.row(
                    by_predicate=(pl.col('case').str.starts_with(str(case)))
                    & (pl.col('grade') == grade),
                    named=True,
                )
                return float(row['PV']), float(row['BIPV'])
            case _:
                raise EditorError(self.pv)

    @functools.cached_property
    def sources(self):
        return tuple(
            x
            for x in self.src.glob('*')
            if x.is_file() and x.suffix.lower() in {'.tpl', '.tplx'}
        )

    def _cases(self):
        return (
            self.sources,
            ('중부1', '중부2', '남부', '제주'),
            (None, 0, 1, 2, 3, 4, 5),
        )

    def cases(self):
        pattern = re.compile(r'^\w(-(?P<s1>C\d))?\-(?P<s2>A\d)\-\w\-\d+')

        for src, region, grade in itertools.product(*self._cases()):
            if (m := pattern.search(src.stem)) is None:
                raise EditorError(src.name)

            if (scale := m.group('s1') or m.group('s2')) is None:
                raise EditorError(src.name)

            yield Case(src=src, scale=scale, region=region, grade=grade)

    def edit(self, case: Case):
        setting = self.filter_setting(
            scale=case.scale, region=case.region, grade=case.grade
        )

        if not setting.height:
            raise EditorError(case)

        pv, bipv = self._pv_area(case)

        if pv < 0:
            suffix = 'PvNotRequired'
        elif bipv:
            suffix = 'BIPV'
        else:
            suffix = 'PV'

        path = self.output / f'{case}-PV-{self.pv}-{suffix}.tpl'
        if path.exists():
            return

        editor = Editor(
            case=case,
            setting=setting,
            bldg=self.bldg,
            pv=max(0, pv),
            bipv=bipv,
        )

        try:
            editor.edit()
        except EditorError as e:
            logger.error('{} {}', case, repr(e))
            return

        editor.write(path)
        if self.xml:
            editor.xml.write(path.with_suffix('.xml'))

    def execute(self):
        self.output.mkdir(exist_ok=True)

        total = math.prod(len(x) for x in self._cases())
        warnings.simplefilter('ignore', TqdmExperimentalWarning)
        for case in tqdm(self.cases(), total=total, miniters=1, smoothing=0.9):
            self.edit(case)


@app.command
def edit(editor: BatchEditor):
    """ECO2 파일 수정."""
    editor.execute()


@app.command
def gen_per_area(editor: BatchEditor, *, area: tuple[float, ...] = (0, 10, 100)):
    """PV 면적 테스트 케이스 생성."""
    cases = tuple(x for x in editor.cases() if x.grade is None)
    for case, a in itertools.product(cases, area):
        logger.info('{} {}', case, a)
        setting = editor.filter_setting(
            scale=case.scale, region=case.region, grade=case.grade
        )
        kwargs = {'case': case, 'setting': setting, 'bldg': editor.bldg}

        pv = editor.output / f'{case}-PV-{a}.tpl'
        Editor(**kwargs, pv=a).edit().write(pv)

        bipv = editor.output / f'{case}-BIPV-{a}.tpl'
        Editor(**kwargs, pv=0, bipv=a).edit().write(bipv)


@cyclopts.Parameter(name='*')
@dc.dataclass(frozen=True)
class RequiredPV:
    src: Path  # PV Zero Reports
    dst: Path | None = None
    setting: Path = Path('config/carbon-reduction/non-residential.csv')
    safety: float = 0.001  # 안전률 (요구 자립률에 더함)
    xls_suffix: str = ' 계산결과'

    @functools.cached_property
    def output(self):
        return self.dst or self.src.parent

    def _read_raw(self):
        tpls = tuple(x for x in self.src.glob('*') if x.suffix in {'.tpl', '.tplx'})
        if (
            sorted(x.stem for x in tpls)  # fmt
            != sorted(
                x.stem.removesuffix(self.xls_suffix)
                for x in self.src.glob('*.xls')
                if '결과그래프' not in x.name
            )
        ):
            raise AssertionError

        for tpl in tqdm(tpls):
            area = Eco2Xml.read(tpl).area.building
            xls = tpl.parent / f'{tpl.stem}{self.xls_suffix}.xls'
            yield report.CalculationsReport(xls).data.select(
                pl.lit(tpl.stem).alias('case'),
                pl.lit(area).alias('건축면적'),
                pl.all(),
            )

    @functools.cached_property
    def raw(self):
        cache = self.output / f'{self.src.name}.parquet'

        if cache.exists():
            return pl.read_parquet(cache)

        raw = pl.concat(self._read_raw()).rename({'변수': 'variable', '합계': 'value'})
        raw.write_parquet(cache)
        raw.write_excel(cache.with_suffix('.xlsx'), column_widths=100)

        return raw

    def max_pv_ratio(self):
        data = (
            pl.read_csv(self.setting)
            .filter(pl.col('part') == '최대 태양광 모듈면적비')
            .to_dict()
        )

        for key, values in data.items():
            if key == 'Base' or key.startswith('ZEB'):
                yield key, values[0]

    def consumption(self):
        # 보정면적 계산 (단위면적당 전력 생산량 계산용)
        index = {49: '난방', 57: '냉방', 64: '급탕', 67: '조명', 70: '환기'}
        data = self.raw.filter(pl.col('index').is_in(tuple(index.keys())))
        if not (data['variable'] == '전력 소요량').all():
            raise ValueError(data['variable'])

        return data.select(
            'case',
            pl.lit('소요량').alias('variable'),
            pl.col('index').replace_strict(index).alias('use'),
            'value',
        )

    def area_by_use(self):
        # 냉난방급탕조명환기 면적
        pattern = r'^(?<variable>\w+)\((?<use>\w+)\)$'
        return (
            self.raw.filter(pl.col('variable').str.starts_with('사용면적'))
            .select('case', pl.col('variable').str.extract_groups(pattern), 'value')
            .unnest('variable')
        )

    def elec_area(self):
        # 전력 생산량 보정 면적
        return (
            pl.concat([self.consumption(), self.area_by_use()])
            .pivot('variable', index=['case', 'use'], values='value')
            .with_columns(
                w=pl.when(pl.col('소요량') == 0)
                .then(pl.lit(0))
                .otherwise(pl.col('소요량') / pl.col('사용면적'))
            )
            .group_by('case')
            .agg(pl.sum('소요량', 'w'))
            .select('case', pl.col('소요량').truediv('w').alias('보정면적'))
        )

    def __call__(self):
        # 면적당 생산, 소요, 자립률, 필요 PV 면적 계산
        variables = {
            '42단위면적당 1차에너지 소요량': '면적당1차소요량',
            '전기에너지 생산량(태양광)': '태양광전력생산량',
            '전기에너지 생산량(풍력)': '기타전력생산량',
            '전기에너지 생산량(열병합)': '기타전력생산량',
            '단위면적당 생산량(태양열)': '면적당기타생산량',
            '단위면적당 생산량(지열)': '면적당기타생산량',
            '단위면적당 생산량(수열)': '면적당기타생산량',
            '단위면적당 생산량(열병합)': '면적당기타생산량',
        }

        generation = (
            self.raw.with_columns(
                pl.when(pl.col('variable') == '단위면적당 1차에너지 소요량')
                .then(pl.format('{}{}', 'index', 'variable'))
                .otherwise('variable')
                .alias('variable')
            )
            .select(
                'case',
                pl.col('case').str.extract('(중부[12]|남부|제주)').alias('region'),
                pl.col('variable')
                .replace_strict(variables, default=None)
                .alias('variable'),
                'value',
            )
            .drop_nulls('variable')
            .group_by('case', 'region', 'variable')
            .agg(pl.sum('value'))
            .pivot(
                'variable', index=['case', 'region'], values='value', sort_columns=True
            )
        )

        max_ratio = dict(self.max_pv_ratio())
        return (
            generation.join(self.elec_area(), on='case', how='left')
            .with_columns(ZEB=pl.col('case').str.extract(r'\-(Base|ZEB.)\-'))
            .with_columns(
                grade=pl.col('ZEB')
                .replace({'Base': None, 'ZEB+': '0'})
                .str.strip_prefix('ZEB')
                .cast(pl.UInt8)
            )
            .with_columns(
                EIR=pl.col('grade').replace_strict(EIR, return_dtype=pl.Float64)
                + self.safety
            )
            .with_columns(
                (
                    pl.col('면적당1차소요량')
                    + 2.75
                    * (pl.col('태양광전력생산량') + pl.col('기타전력생산량'))
                    / pl.col('보정면적')
                    + pl.col('면적당기타생산량')
                ).alias('면적당1차소요량')  # 제외했던 생산량 다시 더하기
            )
            .with_columns(
                (
                    pl.col('보정면적')
                    * (
                        pl.col('EIR') * pl.col('면적당1차소요량')
                        - pl.col('면적당기타생산량')
                    )
                    / 2.75
                    - pl.col('기타전력생산량')
                ).alias('요구PV생산량'),
                pl.col('region')
                .replace_strict(PV_GEN_PER_AREA['PV'], return_dtype=pl.Float64)
                .alias('면적당PV발전량'),
                pl.col('region')
                .replace_strict(PV_GEN_PER_AREA['BIPV'], return_dtype=pl.Float64)
                .alias('면적당BIPV발전량'),
            )
            .with_columns(
                pl.col('요구PV생산량')
                .truediv(pl.col('면적당PV발전량'))
                .alias('요구PV면적'),
                pl.col('ZEB')
                .replace_strict(max_ratio, return_dtype=pl.Float64)
                .alias('최대면적비'),
            )
            .join(
                self.raw.select('case', '건축면적').unique(),
                on='case',
                how='left',
            )
            .with_columns((pl.col('건축면적') * pl.col('최대면적비')).alias('최대면적'))
            .with_columns(
                pl.min_horizontal('요구PV면적', '최대면적').alias('PV면적'),
                pl.max_horizontal(
                    pl.lit(0),
                    (pl.col('요구PV면적') - pl.col('최대면적'))
                    .mul(pl.col('면적당PV발전량'))
                    .truediv(pl.col('면적당BIPV발전량')),
                ).alias('BIPV면적'),
            )
            .sort('case')
        )


@app.command
def required_pv(required_pv: RequiredPV):
    pv = required_pv()

    pv.write_parquet(required_pv.output / 'Required-PV.parquet')
    pv.write_excel(required_pv.output / 'Required-PV.xlsx', column_widths=100)


@app.command
def filter_chp(root: Path):
    """열병합 케이스 필터링."""
    chp = root / 'CHP'
    non_chp = root / 'NON-CHP'

    chp.mkdir(exist_ok=True)
    non_chp.mkdir(exist_ok=True)

    for path in tqdm(tuple(root.glob('*.tpl'))):
        eco = Eco2Xml.read(path)
        elements = eco.ds.xpath('//tbl_new[기기종류="열병합"]')
        dst = chp if elements else non_chp
        path.rename(dst / path.name)


@app.command
def pv_bldg_area(root: Path, threshold: float = 0.901):
    """PV면적, 건축면적 비율 검토."""
    for path in tqdm(tuple(root.glob('*.tplx'))):
        eco = Eco2Xml.read(path)
        area = eco.area.building
        assert area is not None

        pv = mi.only(
            x for x in eco.ds.iterfind('tbl_new') if x.findtext('설명') == 'PV'
        )
        if pv is None:
            logger.info('No PV in {}', path.name)
            continue

        pv_area = float(pv.findtext('태양광모듈면적'))
        r = pv_area / area
        if r >= threshold:
            logger.warning('r={} | case={}', r, path.name)


if __name__ == '__main__':
    logger.remove()
    logger.add(RichHandler(log_time_format='[%X]'), level='INFO', format='{message}')
    logger.add('carbon-reduction.log', level='WARNING')

    app()
