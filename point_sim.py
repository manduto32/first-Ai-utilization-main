"""Update the walkable-map mask in the virtual fly experiment."""

import base64
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "지도.png"
HTML_PATH = ROOT / "point_simulation.html"
GRID_STEP = 4


def create_walkable_mask(image: Image.Image) -> tuple[int, int, bytes]:
    width, height = image.size
    grid_width = (width + GRID_STEP - 1) // GRID_STEP
    grid_height = (height + GRID_STEP - 1) // GRID_STEP
    sampled = image.resize(
        (grid_width, grid_height),
        Image.Resampling.BILINEAR,
    ).convert("RGB")
    packed = bytearray((grid_width * grid_height + 7) // 8)
    walkable_count = 0

    index = 0
    for y in range(grid_height):
        for x in range(grid_width):
            red, green, blue = sampled.getpixel((x, y))
            brightness = (red + green + blue) // 3
            saturation = max(red, green, blue) - min(red, green, blue)
            if 35 <= brightness <= 125 and saturation <= 35:
                packed[index >> 3] |= 1 << (index & 7)
                walkable_count += 1
            index += 1

    if walkable_count == 0:
        raise ValueError("회색 통로 후보를 찾지 못했습니다. 지도 색상 기준을 확인하세요.")
    return grid_width, grid_height, bytes(packed)


def replace_once(content: str, pattern: str, value: str) -> str:
    updated, count = re.subn(
        pattern,
        lambda match: f"{match.group(1)}{value}{match.group(2)}",
        content,
        count=1,
    )
    if count != 1:
        raise RuntimeError(
            "point_simulation.html의 지도 데이터 형식이 예상과 다릅니다. "
            "기존 실험 UI는 변경하지 않았습니다."
        )
    return updated


def main() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"지도 이미지를 찾을 수 없습니다: {IMAGE_PATH}")
    if not HTML_PATH.is_file():
        raise FileNotFoundError(
            f"HTML 앱을 찾을 수 없습니다: {HTML_PATH}. 기존 UI를 보존하기 위해 "
            "point_simulation.html을 먼저 준비해야 합니다."
        )

    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")
    mask_width, mask_height, packed_mask = create_walkable_mask(image)
    mask_data = base64.b64encode(packed_mask).decode("ascii")

    html = HTML_PATH.read_text(encoding="utf-8")
    html = replace_once(html, r"(?m)^(\s*const maskWidth = )\d+(;)$", str(mask_width))
    html = replace_once(
        html, r"(?m)^(\s*const maskHeight = )\d+(;)$", str(mask_height)
    )
    html = replace_once(html, r"(?m)^(\s*const gridStep = )\d+(;)$", str(GRID_STEP))
    html = replace_once(
        html,
        r'(?m)^(\s*const walkableBits = Uint8Array\.from\(atob\(")[^"]+("\), c => c\.charCodeAt\(0\)\);)$',
        mask_data,
    )
    HTML_PATH.write_text(html, encoding="utf-8", newline="\n")
    print(
        f"완료: {HTML_PATH} ({image.width}x{image.height}, 이동 가능 영역 "
        f"{mask_width}x{mask_height}; 초파리 실험 UI 유지)"
    )


if __name__ == "__main__":
    main()
