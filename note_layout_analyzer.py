"""
노트 레이아웃 분석 시스템
OpenCV를 사용한 노트 내부 레이아웃 구조 분석 및 영역 분리
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json
from datetime import datetime


class LayoutType(Enum):
    """레이아웃 타입 정의"""
    SINGLE = "single"                  # 단일 영역
    VERTICAL_SPLIT = "vertical_split"  # 세로 2분할
    GRID_4 = "grid_4"                  # 4분할 격자
    HORIZONTAL_LINES = "horizontal_lines"  # 수평선 분할


class NoteLayoutAnalyzer:
    """노트 레이아웃 분석 클래스"""

    def __init__(self):
        """초기화"""
        # 선 검출 파라미터
        self.min_line_length_ratio = 0.3  # 이미지 크기 대비 최소 선 길이 비율 (30% 이상)
        self.max_line_gap = 50            # 선 연결 최대 간격
        self.angle_tolerance = 10         # 수직/수평 판단 각도 허용 범위

        # 중심선 검출 오차 범위 (이미지 크기의 %)
        self.center_tolerance_ratio = 0.2  # 중심에서 ±20% 범위

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        선 검출을 위한 이미지 전처리

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            전처리된 이미지 (그레이스케일, 엣지)
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 적응형 이진화로 선을 더 잘 감지
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 엣지 검출
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        return edges

    def detect_lines_with_boundary(self, image: np.ndarray, boundary: Optional[np.ndarray] = None) -> Dict[str, List]:
        """
        경계 영역 내에서만 선 검출

        Args:
            image: 입력 이미지 (BGR)
            boundary: 경계 좌표 (4개의 꼭짓점) - 없으면 전체 이미지 사용

        Returns:
            검출된 수직선과 수평선 딕셔너리
        """
        h, w = image.shape[:2]

        # 경계가 있으면 마스크 생성
        if boundary is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [boundary.astype(np.int32)], 255)
            # 이미지를 마스킹
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = image

        # 이미지 전처리
        edges = self.preprocess_image(masked_image)

        # 최소 선 길이 계산 (더 짧은 선도 검출)
        min_line_length = int(min(h, w) * self.min_line_length_ratio)

        # Hough Line Transform (파라미터 조정)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,  # 임계값 낮춤
            minLineLength=min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return {"vertical": [], "horizontal": []}

        # 수직선과 수평선 분류
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 선의 각도 계산
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # 수직선 (90도 근처)
            if abs(angle - 90) < self.angle_tolerance:
                x_avg = (x1 + x2) / 2
                vertical_lines.append({
                    'x': x_avg,
                    'y1': min(y1, y2),
                    'y2': max(y1, y2),
                    'x1': x1, 'x2': x2
                })

            # 수평선 (0도 또는 180도 근처)
            elif angle < self.angle_tolerance or abs(angle - 180) < self.angle_tolerance:
                y_avg = (y1 + y2) / 2
                horizontal_lines.append({
                    'y': y_avg,
                    'x1': min(x1, x2),
                    'x2': max(x1, x2),
                    'y1': y1, 'y2': y2
                })

        # 길이 필터링 (짧은 선 제거)
        min_vertical_length = h * self.min_line_length_ratio
        min_horizontal_length = w * self.min_line_length_ratio

        filtered_vertical = []
        for vline in vertical_lines:
            line_length = abs(vline['y2'] - vline['y1'])
            if line_length >= min_vertical_length:
                filtered_vertical.append(vline)

        filtered_horizontal = []
        for hline in horizontal_lines:
            line_length = abs(hline['x2'] - hline['x1'])
            if line_length >= min_horizontal_length:
                filtered_horizontal.append(hline)

        return {
            "vertical": filtered_vertical,
            "horizontal": filtered_horizontal
        }

    def extend_line_to_boundary(self, line: Dict, direction: str, image_shape: Tuple,
                               boundary: Optional[np.ndarray] = None) -> Dict:
        """
        선을 경계까지 연장

        Args:
            line: 선 정보
            direction: 'vertical' 또는 'horizontal'
            image_shape: 이미지 크기
            boundary: 경계 좌표 (옵션)

        Returns:
            연장된 선 정보
        """
        h, w = image_shape[:2]
        extended_line = line.copy()

        if boundary is not None and len(boundary) == 4:
            # 경계가 있으면 경계까지 연장
            if direction == 'vertical':
                # 세로선을 위아래 경계까지 연장
                extended_line['y1'] = 0
                extended_line['y2'] = h
            else:  # horizontal
                # 가로선을 좌우 경계까지 연장
                extended_line['x1'] = 0
                extended_line['x2'] = w
        else:
            # 경계가 없으면 이미지 끝까지 연장
            if direction == 'vertical':
                extended_line['y1'] = 0
                extended_line['y2'] = h
            else:  # horizontal
                extended_line['x1'] = 0
                extended_line['x2'] = w

        return extended_line

    def classify_layout_with_extension(self, lines: Dict, image_shape: Tuple,
                                      boundary: Optional[np.ndarray] = None) -> Tuple[LayoutType, Dict]:
        """
        레이아웃 분류 및 선 연장

        Args:
            lines: 검출된 선 정보
            image_shape: 이미지 크기
            boundary: 경계 좌표

        Returns:
            (레이아웃 타입, 연장된 중심선 정보)
        """
        h, w = image_shape[:2]
        vertical_lines = lines.get("vertical", [])
        horizontal_lines = lines.get("horizontal", [])

        # 중심점과 오차 범위 계산
        center_x = w / 2
        center_y = h / 2
        tolerance_x = w * self.center_tolerance_ratio
        tolerance_y = h * self.center_tolerance_ratio

        # 중심 근처의 세로선들 찾기
        center_vertical_candidates = []
        for vline in vertical_lines:
            if abs(vline['x'] - center_x) < tolerance_x:
                center_vertical_candidates.append(vline)

        # 가장 중앙에 가까운 세로선 선택
        center_vertical_line = None
        if center_vertical_candidates:
            center_vertical_line = min(center_vertical_candidates,
                                      key=lambda v: abs(v['x'] - center_x))
            # 경계까지 연장
            center_vertical_line = self.extend_line_to_boundary(
                center_vertical_line, 'vertical', image_shape, boundary
            )

        # 중심 근처의 가로선들 찾기
        center_horizontal_candidates = []
        for hline in horizontal_lines:
            if abs(hline['y'] - center_y) < tolerance_y:
                center_horizontal_candidates.append(hline)

        # 가장 중앙에 가까운 가로선 선택
        center_horizontal_line = None
        if center_horizontal_candidates:
            center_horizontal_line = min(center_horizontal_candidates,
                                        key=lambda h: abs(h['y'] - center_y))
            # 경계까지 연장
            center_horizontal_line = self.extend_line_to_boundary(
                center_horizontal_line, 'horizontal', image_shape, boundary
            )

        has_center_vertical = center_vertical_line is not None
        has_center_horizontal = center_horizontal_line is not None

        print(f"  중심 세로선: {'있음' if has_center_vertical else '없음'}")
        if has_center_vertical:
            print(f"    위치: x={center_vertical_line['x']:.0f}")
        print(f"  중심 수평선: {'있음' if has_center_horizontal else '없음'}")
        if has_center_horizontal:
            print(f"    위치: y={center_horizontal_line['y']:.0f}")

        # 레이아웃 타입 결정
        layout_type = LayoutType.SINGLE
        if has_center_vertical and has_center_horizontal:
            layout_type = LayoutType.GRID_4
        elif has_center_vertical:
            layout_type = LayoutType.VERTICAL_SPLIT

        # 연장된 선 정보 반환
        extended_lines = {
            'vertical': center_vertical_line,
            'horizontal': center_horizontal_line
        }

        return layout_type, extended_lines

    def extract_regions_with_extended_lines(self, image: np.ndarray, layout_type: LayoutType,
                                           extended_lines: Dict) -> List[Dict]:
        """
        연장된 선을 기준으로 영역 추출

        Args:
            image: 입력 이미지
            layout_type: 레이아웃 타입
            extended_lines: 연장된 중심선 정보

        Returns:
            각 영역의 좌표 및 이미지 리스트
        """
        h, w = image.shape[:2]
        regions = []

        if layout_type == LayoutType.SINGLE:
            # 단일 영역 - 전체 이미지
            regions.append({
                "x": 0, "y": 0,
                "width": w, "height": h,
                "image": image,
                "label": "전체 영역"
            })

        elif layout_type == LayoutType.VERTICAL_SPLIT:
            # 세로 2분할
            v_line = extended_lines.get('vertical')
            if v_line:
                x_split = int(v_line['x'])

                # 좌측 영역
                regions.append({
                    "x": 0, "y": 0,
                    "width": x_split, "height": h,
                    "image": image[:, :x_split],
                    "label": "좌측 영역"
                })

                # 우측 영역
                regions.append({
                    "x": x_split, "y": 0,
                    "width": w - x_split, "height": h,
                    "image": image[:, x_split:],
                    "label": "우측 영역"
                })

        elif layout_type == LayoutType.GRID_4:
            # 4분할 격자
            v_line = extended_lines.get('vertical')
            h_line = extended_lines.get('horizontal')

            if v_line and h_line:
                x_split = int(v_line['x'])
                y_split = int(h_line['y'])

                # 4개 영역 추출
                regions.extend([
                    {
                        "x": 0, "y": 0,
                        "width": x_split, "height": y_split,
                        "image": image[:y_split, :x_split],
                        "label": "좌상단"
                    },
                    {
                        "x": x_split, "y": 0,
                        "width": w - x_split, "height": y_split,
                        "image": image[:y_split, x_split:],
                        "label": "우상단"
                    },
                    {
                        "x": 0, "y": y_split,
                        "width": x_split, "height": h - y_split,
                        "image": image[y_split:, :x_split],
                        "label": "좌하단"
                    },
                    {
                        "x": x_split, "y": y_split,
                        "width": w - x_split, "height": h - y_split,
                        "image": image[y_split:, x_split:],
                        "label": "우하단"
                    }
                ])

        return regions

    def save_regions(self, regions: List[Dict], base_name: str,
                    output_dir: str = "output/regions") -> List[str]:
        """
        각 영역을 개별 이미지로 저장

        Args:
            regions: 추출된 영역 리스트
            base_name: 기본 파일명
            output_dir: 출력 디렉토리

        Returns:
            저장된 파일 경로 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, region in enumerate(regions):
            # 파일명 생성
            label = region.get("label", f"region_{i+1}")
            filename = f"{base_name}_{label.replace(' ', '_')}.jpg"
            filepath = os.path.join(output_dir, filename)

            # 이미지 저장
            cv2.imwrite(filepath, region["image"])
            saved_paths.append(filepath)

            print(f"  - {label}: {filepath}")

        return saved_paths

    def visualize_layout(self, image: np.ndarray, regions: List[Dict],
                        layout_type: LayoutType, extended_lines: Dict,
                        boundary: Optional[np.ndarray] = None) -> np.ndarray:
        """
        레이아웃 시각화

        Args:
            image: 원본 이미지
            regions: 추출된 영역 리스트
            layout_type: 레이아웃 타입
            extended_lines: 연장된 중심선 정보
            boundary: 경계 좌표 (옵션)

        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()
        h, w = image.shape[:2]

        # 경계 내부 마스크 생성
        boundary_mask = None
        if boundary is not None and len(boundary) == 4:
            boundary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(boundary_mask, [boundary.astype(np.int32)], 255)

        # 각 영역에 다른 색상 오버레이 및 레이블
        colors = [(255, 100, 100), (100, 255, 100),
                 (100, 100, 255), (255, 255, 100)]

        # 오버레이용 이미지 생성
        overlay = np.zeros_like(vis_image)

        for i, region in enumerate(regions):
            x, y = region["x"], region["y"]
            width, height = region["width"], region["height"]

            # 각 영역에 색상 채우기
            color = colors[i % len(colors)]
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)

        # 경계 마스크가 있으면 마스크 내부만 적용
        if boundary_mask is not None:
            overlay = cv2.bitwise_and(overlay, overlay, mask=boundary_mask)

        # 반투명 효과 적용
        vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)

        # 레이블 텍스트 추가
        for i, region in enumerate(regions):
            x, y = region["x"], region["y"]
            width, height = region["width"], region["height"]

            label = region.get("label", f"Region {i+1}")
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.7, 2)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y + (height + text_size[1]) // 2

            # 텍스트 배경
            cv2.rectangle(vis_image,
                        (text_x - 5, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 0, 0), -1)

            # 텍스트
            cv2.putText(vis_image, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 경계선 그리기 (노트 외곽선)
        if boundary is not None and len(boundary) == 4:
            cv2.polylines(vis_image, [boundary.astype(np.int32)], True, (0, 255, 0), 3)

        # 연장된 중심선 그리기 (분할선)
        v_line = extended_lines.get('vertical')
        h_line = extended_lines.get('horizontal')

        if v_line:
            # 세로 중심선 - 빨간색
            cv2.line(vis_image,
                    (int(v_line['x']), int(v_line['y1'])),
                    (int(v_line['x']), int(v_line['y2'])),
                    (0, 0, 255), 3)

        if h_line:
            # 가로 중심선 - 빨간색
            cv2.line(vis_image,
                    (int(h_line['x1']), int(h_line['y'])),
                    (int(h_line['x2']), int(h_line['y'])),
                    (0, 0, 255), 3)

        # 레이아웃 타입 정보 추가
        info_text = f"Layout: {layout_type.value}"
        cv2.putText(vis_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 영역 개수 정보
        count_text = f"Regions: {len(regions)}"
        cv2.putText(vis_image, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return vis_image

    def analyze_layout(self, image_path: str, boundary_corners: Optional[np.ndarray] = None,
                      output_dir: str = "output") -> Dict:
        """
        전체 레이아웃 분석 파이프라인

        Args:
            image_path: 입력 이미지 경로
            boundary_corners: 경계 검출 결과 (4개의 꼭짓점) - 옵션
            output_dir: 출력 디렉토리

        Returns:
            분석 결과 딕셔너리
        """
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        print(f"\n노트 레이아웃 분석 시작: {image_path}")
        print(f"이미지 크기: {image.shape}")

        # 경계 정보가 있으면 표시
        if boundary_corners is not None:
            print("경계 검출 결과를 사용합니다.")

        # 1. 선 검출 (경계 내에서만)
        print("\n1. 선 검출 중...")
        lines = self.detect_lines_with_boundary(image, boundary_corners)
        print(f"  - 수직선: {len(lines['vertical'])}개")
        print(f"  - 수평선: {len(lines['horizontal'])}개")

        # 2. 레이아웃 분류 및 선 연장
        print("\n2. 레이아웃 분류 및 선 연장 중...")
        layout_type, extended_lines = self.classify_layout_with_extension(lines, image.shape, boundary_corners)
        print(f"  - 레이아웃 타입: {layout_type.value}")

        # 3. 영역 추출 (연장된 선 사용)
        print("\n3. 영역 추출 중...")
        regions = self.extract_regions_with_extended_lines(image, layout_type, extended_lines)
        print(f"  - 추출된 영역: {len(regions)}개")

        # 4. 결과 저장
        print("\n4. 결과 저장 중...")
        base_name = Path(image_path).stem

        # 현재 시간 추가
        now = datetime.now()
        time_str = now.strftime("%H_%M")

        # 영역별 이미지 저장
        region_paths = self.save_regions(regions, f"{base_name}_{time_str}",
                                        os.path.join(output_dir, "regions"))

        # 시각화 이미지 저장 (경계 정보 전달)
        vis_image = self.visualize_layout(image, regions, layout_type, extended_lines, boundary_corners)
        vis_path = os.path.join(output_dir, f"{base_name}_{time_str}_layout_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"  - 시각화 이미지: {vis_path}")

        # 분석 결과 JSON 저장
        result = {
            "image_path": image_path,
            "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
            "image_shape": image.shape[:2],
            "layout_type": layout_type.value,
            "num_regions": len(regions),
            "regions": [
                {
                    "label": r.get("label", f"region_{i}"),
                    "x": r["x"],
                    "y": r["y"],
                    "width": r["width"],
                    "height": r["height"]
                }
                for i, r in enumerate(regions)
            ],
            "lines": {
                "vertical": len(lines["vertical"]),
                "horizontal": len(lines["horizontal"])
            }
        }

        json_path = os.path.join(output_dir, f"{base_name}_{time_str}_layout_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  - 분석 결과: {json_path}")

        print(f"\n분석 완료!")

        return result


def main():
    """메인 실행 함수"""
    import os
    from dotenv import load_dotenv

    # .env 파일 로드
    load_dotenv()

    # .env에서 기본값 읽기
    image_dir = os.getenv('IMAGE_DIRECTORY', './test_images')
    image_file = os.getenv('IMAGE_FILENAME', 'IMG_5588.JPG')
    output_dir = os.getenv('OUTPUT_DIRECTORY', 'output')

    print("="*50)
    print("노트 레이아웃 분석")
    print("="*50)

    # 입력 이미지 경로 설정
    image_path = os.path.join(image_dir, image_file)

    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        print(f"\n.env 파일을 확인하세요:")
        print(f"  IMAGE_DIRECTORY={image_dir}")
        print(f"  IMAGE_FILENAME={image_file}")
        return

    # 자동으로 가장 최근 경계 파일 찾기
    boundary_corners = None
    corners_files = list(Path(output_dir).glob("*_corners.txt"))

    if corners_files:
        # 가장 최근 파일 자동 선택
        latest_corners = max(corners_files, key=os.path.getmtime)
        print(f"최신 경계 파일 자동 감지: {latest_corners.name}")

        # corners.txt 파일 파싱
        corners = []
        with open(latest_corners, 'r') as f:
            for line in f:
                if "Corner" in line and ":" in line:
                    # "Corner 1: (290.55, 316.79)" 형식 파싱
                    try:
                        coords = line.split("(")[1].split(")")[0]
                        x_str, y_str = coords.split(",")
                        x = float(x_str.strip())
                        y = float(y_str.strip())
                        corners.append([x, y])
                    except:
                        continue

        if len(corners) == 4:
            boundary_corners = np.array(corners)
            print("경계 정보를 자동으로 로드했습니다.")
        else:
            print("경계 정보 파싱 실패. 경계 없이 진행합니다.")
    else:
        print("경계 파일이 없습니다. 먼저 note_boundary_detector.py를 실행하세요.")
        print("경계 없이 레이아웃 분석을 진행합니다.")

    # 분석기 생성 및 실행
    analyzer = NoteLayoutAnalyzer()
    result = analyzer.analyze_layout(image_path, boundary_corners, output_dir)

    # 결과 출력
    print("\n" + "="*50)
    print("분석 결과 요약:")
    print(f"  - 레이아웃: {result['layout_type']}")
    print(f"  - 영역 개수: {result['num_regions']}")
    for region in result['regions']:
        print(f"    * {region['label']}: ({region['x']}, {region['y']}) - "
              f"{region['width']}x{region['height']}")
    print("="*50)


if __name__ == "__main__":
    main()