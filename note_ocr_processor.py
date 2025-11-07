"""
노트 OCR 처리 시스템
Google Cloud Vision API를 사용한 영역별 텍스트 추출
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Google Cloud Vision API
from google.cloud import vision


class NoteOCRProcessor:
    """노트 OCR 처리 클래스"""

    def __init__(self):
        """
        초기화
        Google Cloud Vision API 클라이언트 생성
        """
        # 환경 변수 로드
        load_dotenv()

        # Vision API 클라이언트 초기화
        self.client = vision.ImageAnnotatorClient()
        print("Google Cloud Vision API client initialized")

    def perform_ocr(self, image_path: str) -> Dict:
        """
        단일 이미지에 대한 OCR 수행

        Args:
            image_path: 이미지 파일 경로

        Returns:
            OCR 결과 딕셔너리
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        print(f"  Processing: {Path(image_path).name}")

        # 이미지 읽기
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Google Vision API 호출 - DOCUMENT_TEXT_DETECTION 사용
        # (TEXT_DETECTION보다 문서/손글씨에 최적화됨)
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")

        # 전체 텍스트 추출
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""

        # 블록별 상세 정보 추출
        blocks = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    # 블록의 텍스트 추출
                    block_text = ""
                    block_confidence = 0
                    paragraph_count = 0

                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            paragraph_text += word_text + " "

                        block_text += paragraph_text.strip() + "\n"
                        block_confidence += paragraph.confidence
                        paragraph_count += 1

                    # 평균 신뢰도 계산
                    avg_confidence = block_confidence / paragraph_count if paragraph_count > 0 else 0

                    # 블록 좌표 (bounding box)
                    vertices = [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]

                    blocks.append({
                        "text": block_text.strip(),
                        "confidence": round(avg_confidence, 4),
                        "bounding_box": vertices
                    })

        # 전체 신뢰도 계산 (페이지 레벨)
        overall_confidence = 0
        if response.full_text_annotation and response.full_text_annotation.pages:
            page_confidences = [page.confidence for page in response.full_text_annotation.pages if hasattr(page, 'confidence')]
            overall_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0

        return {
            "full_text": full_text,
            "confidence": round(overall_confidence, 4),
            "block_count": len(blocks),
            "blocks": blocks,
            "character_count": len(full_text),
            "line_count": len(full_text.split('\n')) if full_text else 0
        }

    def process_layout_analysis(self, layout_json_path: str, regions_dir: str = None) -> Dict:
        """
        레이아웃 분석 결과를 기반으로 OCR 처리

        Args:
            layout_json_path: 레이아웃 분석 JSON 파일 경로
            regions_dir: 영역 이미지들이 저장된 디렉토리 (없으면 자동 탐색)

        Returns:
            통합 OCR 결과 딕셔너리
        """
        # 레이아웃 분석 결과 읽기
        with open(layout_json_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)

        print(f"\n레이아웃 분석 결과 로드: {layout_json_path}")
        print(f"  - 레이아웃 타입: {layout_data['layout_type']}")
        print(f"  - 영역 개수: {layout_data['num_regions']}")

        # regions 디렉토리 자동 탐색
        if regions_dir is None:
            json_dir = Path(layout_json_path).parent
            regions_dir = json_dir / "regions"

        if not os.path.exists(regions_dir):
            raise ValueError(f"Regions directory not found: {regions_dir}")

        print(f"\n영역 이미지 디렉토리: {regions_dir}")
        print("="*60)

        # 영역별 OCR 처리
        ocr_results = []

        # JSON에서 타임스탬프 추출 (파일명 매칭용)
        # 예: IMG_5588_14_30_layout_analysis.json -> 14_30
        json_filename = Path(layout_json_path).stem
        parts = json_filename.split('_')

        # 타임스탬프 찾기 (HH_MM 형식)
        time_str = None
        for i in range(len(parts) - 1):
            if parts[i].isdigit() and len(parts[i]) <= 2 and parts[i+1].isdigit() and len(parts[i+1]) <= 2:
                time_str = f"{parts[i]}_{parts[i+1]}"
                break

        for region_info in layout_data['regions']:
            region_label = region_info['label']

            # 영역 이미지 파일명 생성
            # 예: IMG_5588_14_30_좌측_영역.jpg
            base_name = Path(layout_data['image_path']).stem

            # 파일명 패턴 찾기
            region_filename = f"{base_name}_{time_str}_{region_label}.jpg" if time_str else f"{base_name}_{region_label}.jpg"
            region_image_path = os.path.join(regions_dir, region_filename)

            # 파일이 없으면 글로브 패턴으로 찾기
            if not os.path.exists(region_image_path):
                # 패턴: *_{label}.jpg
                matching_files = list(Path(regions_dir).glob(f"*{region_label}.jpg"))
                if matching_files:
                    region_image_path = str(matching_files[0])
                else:
                    print(f"  ⚠️  영역 이미지를 찾을 수 없음: {region_label}")
                    continue

            print(f"\n영역: {region_label}")
            print(f"  이미지: {Path(region_image_path).name}")

            # OCR 수행
            try:
                ocr_result = self.perform_ocr(region_image_path)

                # 결과 조합
                ocr_results.append({
                    "region_label": region_label,
                    "region_info": region_info,
                    "image_path": region_image_path,
                    "ocr": ocr_result
                })

                print(f"  ✓ OCR 완료")
                print(f"    - 추출된 텍스트 길이: {ocr_result['character_count']}자")
                print(f"    - 줄 수: {ocr_result['line_count']}")
                print(f"    - 신뢰도: {ocr_result['confidence']:.2%}")

            except Exception as e:
                print(f"  ✗ OCR 실패: {str(e)}")
                ocr_results.append({
                    "region_label": region_label,
                    "region_info": region_info,
                    "image_path": region_image_path,
                    "ocr": None,
                    "error": str(e)
                })

        print("="*60)

        # 통합 결과 생성
        result = {
            "source_image": layout_data['image_path'],
            "layout_json": layout_json_path,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "layout_type": layout_data['layout_type'],
            "num_regions": layout_data['num_regions'],
            "regions": ocr_results,
            "summary": {
                "total_characters": sum(r['ocr']['character_count'] for r in ocr_results if r.get('ocr')),
                "total_lines": sum(r['ocr']['line_count'] for r in ocr_results if r.get('ocr')),
                "average_confidence": round(
                    sum(r['ocr']['confidence'] for r in ocr_results if r.get('ocr')) / len([r for r in ocr_results if r.get('ocr')]),
                    4
                ) if any(r.get('ocr') for r in ocr_results) else 0,
                "successful_regions": len([r for r in ocr_results if r.get('ocr')]),
                "failed_regions": len([r for r in ocr_results if not r.get('ocr')])
            }
        }

        return result

    def save_results(self, result: Dict, output_path: str) -> str:
        """
        OCR 결과를 JSON 파일로 저장

        Args:
            result: OCR 결과 딕셔너리
            output_path: 출력 파일 경로

        Returns:
            저장된 파일 경로
        """
        # 출력 디렉토리 생성
        os.makedirs(Path(output_path).parent, exist_ok=True)

        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장 완료: {output_path}")

        # 요약 정보 출력
        print("\n" + "="*60)
        print("OCR 처리 요약")
        print("="*60)
        print(f"처리된 영역: {result['summary']['successful_regions']}개")
        print(f"실패한 영역: {result['summary']['failed_regions']}개")
        print(f"총 추출 문자 수: {result['summary']['total_characters']}자")
        print(f"총 줄 수: {result['summary']['total_lines']}줄")
        print(f"평균 신뢰도: {result['summary']['average_confidence']:.2%}")
        print("="*60)

        return output_path


def find_latest_layout_json(output_dir: str = "output") -> Optional[str]:
    """
    가장 최근 레이아웃 분석 JSON 파일 찾기

    Args:
        output_dir: 출력 디렉토리

    Returns:
        JSON 파일 경로 (없으면 None)
    """
    json_files = list(Path(output_dir).glob("*_layout_analysis.json"))

    if not json_files:
        return None

    # 가장 최근 파일 반환
    latest_file = max(json_files, key=os.path.getmtime)
    return str(latest_file)


def main():
    """메인 실행 함수"""
    load_dotenv()

    output_dir = os.getenv('OUTPUT_DIRECTORY', 'output')

    print("="*60)
    print("노트 OCR 처리 시스템")
    print("="*60)

    # 가장 최근 레이아웃 분석 결과 찾기
    layout_json = find_latest_layout_json(output_dir)

    if not layout_json:
        print(f"\n❌ 레이아웃 분석 결과를 찾을 수 없습니다: {output_dir}")
        print("먼저 note_layout_analyzer.py를 실행하세요.")
        return

    print(f"\n✓ 최신 레이아웃 분석 결과 감지: {Path(layout_json).name}")

    # OCR 프로세서 생성
    processor = NoteOCRProcessor()

    # OCR 처리
    result = processor.process_layout_analysis(layout_json)

    # 결과 저장
    # 파일명: IMG_5588_14_30_ocr_result.json
    base_name = Path(layout_json).stem.replace('_layout_analysis', '')
    output_path = os.path.join(output_dir, f"{base_name}_ocr_result.json")

    processor.save_results(result, output_path)


if __name__ == "__main__":
    main()
