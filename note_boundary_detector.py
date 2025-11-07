"""
노트 경계 검출 시스템
DocAligner를 사용한 정확한 노트/문서 경계 검출
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# DocAligner import
from docaligner import DocAligner


class NoteBoundaryDetector:
    """노트 경계 검출 클래스"""

    def __init__(self):
        """
        초기화
        """
        # DocAligner 모델 초기화
        self.model = DocAligner(
            model_type='heatmap',
            model_cfg='fastvit_sa24',
            backend='cpu'  # GPU 있으면 'gpu'로 변경
        )
        print("DocAligner model loaded")

    def detect_corners_docaligner(self, image: np.ndarray) -> np.ndarray:
        """
        DocAligner를 사용한 코너 검출

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            4개의 코너 포인트 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # DocAligner는 RGB 이미지를 기대함
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 모델 실행
        corners = self.model(image_rgb)

        if corners is None or len(corners) != 4:
            print("Warning: DocAligner couldn't detect document corners")
            return None

        return corners


    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        코너 포인트를 표준 순서로 정렬
        순서: [top-left, top-right, bottom-right, bottom-left]

        Args:
            corners: 4개의 코너 포인트

        Returns:
            정렬된 코너 포인트
        """
        # 중심점 계산
        center = np.mean(corners, axis=0)

        # 각 포인트의 각도 계산
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])

        # 각도 기준으로 정렬
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]

        # top-left부터 시작하도록 조정
        # top-left는 x+y가 가장 작은 점
        sums = sorted_corners[:, 0] + sorted_corners[:, 1]
        min_idx = np.argmin(sums)

        # 순서 조정
        ordered = np.roll(sorted_corners, -min_idx, axis=0)

        return ordered

    def detect_boundary(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지에서 노트 경계 검출

        Args:
            image_path: 입력 이미지 경로

        Returns:
            (원본 이미지, 코너 포인트)
        """
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")

        # 코너 검출
        corners = self.detect_corners_docaligner(image)

        if corners is None:
            print("Failed to detect document corners")
            return image, None

        print(f"Detected corners:\n{corners}")
        return image, corners

    def draw_boundary(self, image: np.ndarray, corners: np.ndarray,
                     color=(0, 255, 0), thickness=3) -> np.ndarray:
        """
        검출된 경계를 이미지에 그리기

        Args:
            image: 원본 이미지
            corners: 4개의 코너 포인트
            color: 경계선 색상 (BGR)
            thickness: 선 두께

        Returns:
            경계가 그려진 이미지
        """
        if corners is None:
            return image

        result = image.copy()

        # 코너 포인트를 정수로 변환
        corners = np.int32(corners)

        # 사각형 그리기
        cv2.drawContours(result, [corners], -1, color, thickness)

        # 각 코너에 원 그리기
        for i, corner in enumerate(corners):
            cv2.circle(result, tuple(corner), 8, (0, 0, 255), -1)
            # 코너 번호 표시
            cv2.putText(result, str(i+1), tuple(corner + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 정보 텍스트 추가
        h, w = image.shape[:2]
        info_text = "Note boundary detected - DocAligner"
        cv2.putText(result, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return result

    def process_and_save(self, image_path: str, output_dir: str = "output") -> str:
        """
        이미지 처리하고 결과 저장

        Args:
            image_path: 입력 이미지 경로
            output_dir: 출력 디렉토리

        Returns:
            저장된 파일 경로
        """
        # 현재 시간 가져오기 (시_분 형식)
        now = datetime.now()
        time_str = now.strftime("%H_%M")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 경계 검출
        image, corners = self.detect_boundary(image_path)

        if corners is None:
            print(f"Failed to detect boundary for {image_path}")
            return None

        # 경계 그리기
        result = self.draw_boundary(image, corners)

        # 결과 저장 
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_{time_str}_boundary.jpg")
        cv2.imwrite(output_path, result)

        print(f"Result saved to: {output_path}")

        # 좌표 정보도 텍스트 파일로 저장
        coords_path = os.path.join(output_dir, f"{base_name}_{time_str}_corners.txt")
        with open(coords_path, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: DocAligner\n")
            f.write("Corners (x, y):\n")
            for i, corner in enumerate(corners):
                f.write(f"  Corner {i+1}: ({corner[0]:.2f}, {corner[1]:.2f})\n")

        print(f"Coordinates saved to: {coords_path}")

        return output_path


def main():
    """메인 실행 함수"""
    # .env 파일에서 설정 읽기
    image_dir = os.getenv('IMAGE_DIRECTORY', './test_images')
    image_file = os.getenv('IMAGE_FILENAME', 'IMG_5588.JPG')
    output_dir = os.getenv('OUTPUT_DIRECTORY', 'output')

    # 이미지 경로 생성
    image_path = os.path.join(image_dir, image_file)

    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        print(f"\n.env 파일을 확인하세요:")
        print(f"  IMAGE_DIRECTORY={image_dir}")
        print(f"  IMAGE_FILENAME={image_file}")
        return

    print("="*50)
    print("노트 경계 검출 시작")
    print("="*50)
    print(f"입력 이미지: {image_path}")
    print(f"출력 폴더: {output_dir}")
    print("-"*50)

    # 검출기 생성
    detector = NoteBoundaryDetector()

    # 처리 및 저장
    result_path = detector.process_and_save(image_path, output_dir)

    if result_path:
        print("-"*50)
        print(f"처리 완료!")
        print(f"결과 파일: {result_path}")


if __name__ == "__main__":
    main()