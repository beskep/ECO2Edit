from __future__ import annotations

import dataclasses as dc
import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import cyclopts
import more_itertools as mi
from logly import logger

import eco2edit.editor as ee
from eco2edit.tqdmr import tqdmr
from scripts.pim.base import BaseEco2Editor, Config

if TYPE_CHECKING:
    from lxml.etree import _Element

    from scripts.pim.base import Kind


class NotDefinedCaseError(ee.EditorError):
    pass


class Eco2EditorIndividual(BaseEco2Editor):  # noqa: PLR0904
    """Case 1-23."""

    MAX_CASE = 23

    @staticmethod
    def not_defined():
        caller = inspect.stack()[1].function
        raise NotDefinedCaseError(caller)

    def set_uvalue(
        self,
        uvalue: float = 0,
        surface_type: str = '외벽(벽체)',
        *,
        if_empty: ee.Level = 'debug',
    ):
        uvalue = self.conf.design.uvalue[surface_type]

        if uvalue <= 0:
            # 해당하는 면이 없는지 확인
            try:
                next(self.xml.surfaces_by_type(surface_type))
            except StopIteration:
                pass
            else:
                msg = f'{surface_type}이 존재.'
                raise ee.EditorError(msg)

        return super().set_walls(uvalue, surface_type, if_empty=if_empty)

    def set_hvac(
        self,
        control: Literal['정풍량', '변풍량'],
        heat_recovery_rate: Literal['0.6', '1.0'] | None = None,
    ):
        self.xml.set_elements(path='tbl_kongjo/공조방식', value=control)

        if heat_recovery_rate is None:
            return

        hr = {
            k: str(v)
            for k, v in self.conf.design.heat_recovery[heat_recovery_rate].items()
        }

        for e in self.xml.ds.iterfind('tbl_kongjo/열회수율'):
            if e.text and e.text != '0':
                e.text = hr['heat']

        for e in self.xml.ds.iterfind('tbl_kongjo/열회수율냉'):
            if e.text and e.text != '0':
                e.text = hr['cool']

    def _is_target_face(self, face: _Element):
        if face.findtext('code') == '0':
            return False
        if face.findtext('방위') in {'북', '일사없음'}:
            return False

        if (profile := self.face_profile(face)) is None:
            raise ValueError(face)

        return not any(x in profile for x in ['화장실', '부속공간', '창고/설비/문서실'])

    # ==== case ================================================================

    def case1(self):
        # 블라인드
        for element in self.xml.ds.iterfind('tbl_myoun'):
            if not self._is_target_face(element):
                continue

            ee.set_child_text(element, '블라인드유무', '유')
            ee.set_child_text(element, '블라인드위치', '내부')
            ee.set_child_text(element, '블라인드각도', '45도')
            ee.set_child_text(element, '블라인드빛종류', '약투과(t=0.2)')
            ee.set_child_text(element, '블라인드색상', '밝은색')

    def case2(self):
        # 수직 차양
        for element in self.xml.ds.iterfind('tbl_myoun'):
            if not self._is_target_face(element):
                continue

            ee.set_child_text(element, '차양각선택', '0')
            ee.set_child_text(element, '수직입력각', '60')
            ee.set_child_text(element, '수직차양각', '60')

    def case3(self):
        # 수평 차양
        for element in self.xml.ds.iterfind('tbl_myoun'):
            if not self._is_target_face(element):
                continue

            ee.set_child_text(element, '차양각선택', '0')
            ee.set_child_text(element, '수평입력각', '60')
            ee.set_child_text(element, '수평차양각', '60')

    def case4(self):
        self.set_uvalue(surface_type='외벽(벽체)')
        self.set_uvalue(surface_type='내벽(벽체)')

    def case5(self):
        self.set_uvalue(surface_type='외벽(지붕)')
        self.set_uvalue(surface_type='내벽(지붕)')

    def case6(self):
        self.set_uvalue(surface_type='외벽(바닥)')
        self.set_uvalue(surface_type='내벽(바닥)')

    def case7(self):
        self.set_windows(
            uvalue=self.conf.design.uvalue['외부창'],
            shgc=None,
            surface_type='외부창',
        )

    def case8(self):
        self.set_windows(
            uvalue=None,
            shgc=self.conf.design.shgc,
            surface_type='외부창',
        )

    def case9(self):
        # 외단열
        for element in self.xml.ds.iterfind('tbl_zone'):
            ee.set_child_text(element, '열교가산치', '외단열')

    def case10(self):
        # 조명밀도
        self.xml.set_elements(
            path='tbl_zone/조명에너지부하율입력치',
            value=str(self.conf.design.light_density),
        )

    def case11(self):
        self.set_hvac(control='정풍량', heat_recovery_rate='0.6')

    def case12(self):
        self.set_hvac(control='정풍량', heat_recovery_rate='1.0')

    def case13(self):
        self.set_hvac(control='변풍량', heat_recovery_rate='0.6')

    def case14(self):
        self.set_hvac(control='변풍량', heat_recovery_rate='1.0')

    def case15(self):
        # 난방 히트펌프 COP수치
        cop = self.conf.design.heat_pump['individual']
        for element in self.xml.ds.iterfind('tbl_nanbangkiki'):
            prev = element.findtext('히트난방정격7')
            if not prev or prev == '0':
                continue

            ee.set_child_text(element, '히트난방정격7', cop[0])
            ee.set_child_text(element, '히트난방정격10', cop[1])

    def case16(self):
        # 냉방 히트펌프 COP수치
        cop = self.conf.design.heat_pump['individual'][0]
        for element in self.xml.ds.iterfind('tbl_nangbangkiki'):
            # 전체 냉방기기에 적용
            ee.set_child_text(element, '열성능비', cop)

    def case17(self):
        # 급탕펌프 효율
        for element in self.xml.ds.iterfind('tbl_nanbangkiki'):
            if '(급탕용)' not in element.findtext('설명', ''):  # XXX
                continue

            ee.set_child_text(
                element, '정격보일러효율', self.conf.design.boiler_efficiency
            )

    def case18(self):
        #  급탕펌프 제어
        for element in self.xml.ds.iterfind('tbl_nanbangkiki'):
            if '(급탕용)' not in element.findtext('설명', ''):  # XXX
                continue

            ee.set_child_text(element, '펌프제어', '제어')

    def case19(self):
        # 난방 공급 실내기 PI 제어
        self.xml.set_elements('tbl_kongkub/전기난방제어', 'PI제어')

    def case20(self):
        self.not_defined()

    def case21(self):
        self.not_defined()

    def case22(self):
        # 냉방 회전수 제어
        for system in self.xml.ds.iterfind('tbl_nangbangkiki'):
            ee.set_child_text(system, '제어방식', '회전수제어')

    def case23(self):
        self.not_defined()

    def case24(self):
        for idx in range(1, self.MAX_CASE + 1):
            fn = getattr(self, f'case{idx}')

            try:
                fn()
            except NotDefinedCaseError as e:
                logger.info(f'{e!r}')


class Eco2EditorCentral(Eco2EditorIndividual):
    def case11(self):
        self.set_hvac(control='정풍량')

    def case12(self):
        self.not_defined()

    def case13(self):
        self.set_hvac(control='변풍량')

    def case14(self):
        self.not_defined()

    def case15(self):
        # 난방 보일러 효율
        for element in self.xml.ds.iterfind('tbl_nanbangkiki'):
            if '(난방용)' not in element.findtext('설명', ''):  # XXX
                continue

            ee.set_child_text(
                element, '정격보일러효율', self.conf.design.boiler_efficiency
            )

    def case16(self):
        # 냉방 COP
        cop = self.conf.design.heat_pump['central'][0]
        for element in self.xml.ds.iterfind('tbl_nangbangkiki'):
            # 전체 냉방기기에 적용
            ee.set_child_text(element, '열성능비', cop)

    def case19(self):
        self.not_defined()

    def case20(self):
        # 난방 순환 펌프 정압
        for element in self.xml.ds.iterfind('tbl_nanbangkiki'):
            if '(난방용)' not in element.findtext('설명', ''):  # XXX
                continue

            ee.set_child_text(element, '펌프제어유형', '정압')

    def case21(self):
        # 난방 순환 펌프 변압
        for element in self.xml.ds.iterfind('tbl_nanbangkiki'):
            if '(난방용)' not in element.findtext('설명', ''):  # XXX
                continue

            ee.set_child_text(element, '펌프제어유형', '변압')

    def case22(self):
        # 냉방 펌프 대수 제어
        for element in self.xml.ds.iterfind('tbl_bunbae'):
            ee.set_child_text(element, '펌프운전제어유무', '대수제어')

    def case23(self):
        for element in self.xml.ds.iterfind('tbl_bunbae'):
            ee.set_child_text(element, '펌프운전제어유무', '제어')


@dc.dataclass
class BatchEditor:
    root: Path
    kind: Kind

    MAX_CASE: ClassVar[int] = 24

    @functools.cached_property
    def src(self):
        s = '개별' if self.kind == 'individual' else '중앙'
        return mi.one(self.root.glob(f'*{s}*.ecox'))

    @functools.cached_property
    def conf(self):
        return Config.read()

    @functools.cached_property
    def kind_kor(self):
        match self.kind:
            case 'individual':
                return '개별'
            case 'central':
                return '중앙'

    def __call__(self):
        cls = Eco2EditorIndividual if self.kind == 'individual' else Eco2EditorCentral
        output = self.root / 'output'
        output.mkdir(exist_ok=True)

        for idx in tqdmr(list(range(1, self.MAX_CASE + 1)), desc=self.kind):
            editor = cls(self.src, self.conf, self.kind)

            fn = getattr(editor, f'case{idx}')

            try:
                fn()
            except NotDefinedCaseError as e:
                logger.info(f'{e!r}')
                continue

            editor.write(output / f'{self.kind_kor}-case{idx:03d}.eco')


app = cyclopts.App(
    config=cyclopts.config.Toml(
        'config/pim.toml', root_keys=['cli', 'condition1', 'basic']
    )
)


@app.default
def main(root: Path):
    root = Path(root)
    kinds: list[Kind] = ['individual', 'central']

    for kind in kinds:
        BatchEditor(root=root, kind=kind)()


if __name__ == '__main__':
    app()
