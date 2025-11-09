#!/usr/bin/env python
"""
단순 Google Vision OCR 처리
레이아웃 분석 없이 원본 이미지를 직접 OCR 처리하여 결과 비교용
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Google Cloud Vision API
from google.cloud import vision


class SimpleVisionOCR:
    """레이아웃 분석 없이 Google Vision만 사용하는 단순 OCR"""

    def __init__(self):
        """Google Cloud Vision API 클라이언트 초기화"""
        load_dotenv()
        self.client = vision.ImageAnnotatorClient()
        print("Google Cloud Vision API client initialized")

    def perform_ocr(self, image_path: str) -> dict:
        """
        이미지에 대한 단순 OCR 수행

        Args:
            image_path: 이미지 파일 경로

        Returns:
            OCR 결과 딕셔너리
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        print(f"\nProcessing: {Path(image_path).name}")

        # 이미지 읽기
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Google Vision API 호출
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(f"Google Vision API Error: {response.error.message}")

        # 전체 텍스트 추출
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""

        # 블록별 정보 추출 (선택적)
        blocks = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    # 블록의 텍스트 추출
                    block_text = ""
                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            paragraph_text += word_text + " "
                        block_text += paragraph_text.strip() + "\n"

                    blocks.append({
                        "text": block_text.strip(),
                        "bounding_box": [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]
                    })

        # 전체 신뢰도 계산
        overall_confidence = 0
        if response.full_text_annotation and response.full_text_annotation.pages:
            page_confidences = [page.confidence for page in response.full_text_annotation.pages if hasattr(page, 'confidence')]
            overall_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0

        return {
            "image_path": image_path,
            "full_text": full_text,
            "confidence": round(overall_confidence, 4),
            "block_count": len(blocks),
            "blocks": blocks,
            "character_count": len(full_text),
            "line_count": len(full_text.split('\n')) if full_text else 0,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def save_text_result(self, result: dict, output_path: str):
        """
        OCR 결과를 텍스트 파일로 저장

        Args:
            result: OCR 결과 딕셔너리
            output_path: 출력 파일 경로
        """
        os.makedirs(Path(output_path).parent, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # 순수 텍스트만 저장
            f.write(result['full_text'])

        print(f"\n결과 저장 완료: {output_path}")

        # 콘솔에도 출력
        print("\n" + "="*70)
        print("OCR 결과 요약")
        print("="*70)
        print(f"신뢰도: {result['confidence']:.2%}")
        print(f"추출된 문자 수: {result['character_count']}자")
        print(f"줄 수: {result['line_count']}줄")
        print(f"블록 수: {result['block_count']}개")
        print("\n추출된 텍스트 미리보기:")
        print("-"*70)
        preview = result['full_text'][:500] + "..." if len(result['full_text']) > 500 else result['full_text']
        print(preview)
        print("="*70)


def main():
    """메인 실행 함수"""
    load_dotenv()

    # 이미지 경로 설정
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 기본값: test_images의 첫 번째 이미지
        test_images_dir = Path("test_images")
        if not test_images_dir.exists():
            print("❌ test_images 디렉토리를 찾을 수 없습니다.")
            print("사용법: python simple_vision_ocr.py <이미지_경로>")
            return 1

        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.JPG")) + list(test_images_dir.glob("*.PNG"))
        if not image_files:
            print("❌ test_images 디렉토리에 이미지가 없습니다.")
            return 1

        # IMG_5588.jpg 우선 선택 (있으면)
        target_image = None
        for img in image_files:
            if "5612" in img.name:
                target_image = img
                break

        if not target_image:
            target_image = image_files[0]

        image_path = str(target_image)

    print("="*70)
    print("단순 Google Vision OCR (레이아웃 분석 없음)")
    print("="*70)
    print(f"처리할 이미지: {image_path}\n")

    # OCR 처리
    ocr = SimpleVisionOCR()
    result = ocr.perform_ocr(image_path)

    # 결과 저장
    output_dir = Path(os.getenv('OUTPUT_DIRECTORY', 'output'))
    output_dir.mkdir(exist_ok=True)

    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime('%H_%M')
    output_path = output_dir / f"{image_name}_{timestamp}_simple_ocr.txt"

    ocr.save_text_result(result, str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
