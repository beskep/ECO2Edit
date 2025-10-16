from __future__ import annotations

import dataclasses as dc
import functools
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cyclopts
import more_itertools as mi
from lxml import etree

import eco2edit.editor as ee
from eco2edit.tqdmr import tqdmr
from scripts.pim.base import BaseEco2Editor, Config

if TYPE_CHECKING:
    from scripts.pim.base import Kind

Slope = Literal['45도', '수직', '수평']
Azimuth = Literal['남', '동', '서', '']
Efficient = float | Literal['단결정']
Mount = Literal['후면통풍', '밀착형']

PV_TEMPLATE = """
  <tbl_new>
    <code>0001</code>
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
    <태양광모듈면적>0</태양광모듈면적>
    <태양광모듈기울기>수직</태양광모듈기울기>
    <태양광모듈방위>동</태양광모듈방위>
    <태양광모듈종류>성능치 입력</태양광모듈종류>
    <태양광모듈적용타입>밀착형</태양광모듈적용타입>
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
    <태양광모듈효율 />
    <지열비고 />
    <열병합신재생여부>false</열병합신재생여부>
    <태양광용량>0</태양광용량>
    <수열_수열원>하천수</수열_수열원>
    <수열_열교환기설치여부>예</수열_열교환기설치여부>
    <수열_수열팽창탱크설치여부>예</수열_수열팽창탱크설치여부>
    <sortkey>1</sortkey>
  </tbl_new>
"""


@dc.dataclass
class PV:
    slope: Slope
    azimuth: Azimuth
    efficient: Efficient
    mount: Mount

    @classmethod
    def iter(cls, conf: Config):
        for case in conf.renewable.pv:
            yield cls(**case)  # type: ignore[arg-type]

    def __str__(self):
        efficient = (
            f'{self.efficient:.1%}'
            if isinstance(self.efficient, float | int)
            else self.efficient
        )
        s = f'{self.slope}-{self.azimuth}-{efficient}-{self.mount}'
        return s.replace('--', '-')

    def element(self, area: float | str):
        pv = etree.fromstring(PV_TEMPLATE)

        ee.set_child_text(pv, '태양광모듈면적', str(area))
        ee.set_child_text(pv, '태양광모듈기울기', self.slope)
        ee.set_child_text(pv, '태양광모듈방위', self.azimuth)
        ee.set_child_text(pv, '태양광모듈적용타입', self.mount)

        if self.efficient == '단결정':
            ee.set_child_text(pv, '태양광모듈종류', '단결정')
        else:
            ee.set_child_text(pv, '태양광모듈종류', '성능치 입력')
            ee.set_child_text(pv, '태양광모듈효율', str(self.efficient))

        return pv


class Eco2Editor(BaseEco2Editor):
    def clear_pv(self):
        for element in self.xml.iterfind('tbl_new'):
            if element.findtext('기기종류') == '태양광':
                self.xml.ds.remove(element)

    def add_pv(self, pv: PV, area: float):
        element = pv.element(area=area)

        last_renewable = mi.last(self.xml.iterfind('tbl_new'))
        index = self.xml.ds.index(last_renewable) + 1

        self.xml.ds.insert(index, element)


@dc.dataclass
class BatchEditor:
    root: Path
    kind: Kind

    # PV 면적 설정
    # XXX 추후 50 m² 간격으로 변경
    multiplier: float = 1000  # [m²]
    max_index: int = 25

    @functools.cached_property
    def src(self):
        return mi.one(self.root.glob(f'*{self.kind_kor}*.eco'))

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
        output = self.root / 'output'
        output.mkdir(exist_ok=True)

        editor = Eco2Editor(self.src, self.conf, self.kind)

        it = itertools.product(PV.iter(self.conf), range(self.max_index + 1))
        for pv, index in tqdmr(
            it,
            total=len(self.conf.renewable.pv) * (self.max_index + 1),
            desc=self.kind,
        ):
            editor.clear_pv()

            area = index * self.multiplier
            editor.add_pv(pv, area=area)
            editor.write(output / f'{self.kind_kor}-pv-{pv}-{area=}.eco')


app = cyclopts.App(
    config=cyclopts.config.Toml(
        'config/pim.toml', root_keys=['cli', 'condition1', 'renewable']
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
