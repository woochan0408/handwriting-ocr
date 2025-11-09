"""
노트 LLM 후처리 시스템
OpenAI API를 사용한 OCR 텍스트 정제 및 구조화
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# OpenAI API
from openai import OpenAI


class NoteLLMPostprocessor:
    """노트 LLM 후처리 클래스"""

    def __init__(self, model: str = "gpt-4o"):
        """
        초기화

        Args:
            model: 사용할 OpenAI 모델 (기본값: gpt-4o-mini - 가장 저렴)
        """
        # 환경 변수 로드
        load_dotenv()

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)
        self.model = model

        print(f"OpenAI API client initialized (model: {model})")

    def postprocess_text(self, raw_text: str, region_label: str) -> Dict[str, str]:
        """
        OCR 텍스트를 LLM으로 후처리

        Args:
            raw_text: OCR로 추출된 원본 텍스트
            region_label: 영역 라벨 (컨텍스트 제공용)

        Returns:
            {"title": "제목", "content": "내용"}
        """
        if not raw_text or not raw_text.strip():
            return {
                "title": f"{region_label} (빈 영역)",
                "content": ""
            }

        # 프롬프트 구성
        prompt = f"""
당신은 학습 노트의 손글씨 OCR 결과를 정제하는 전문가입니다.
아래는 노트의 "{region_label}" 영역에서 OCR로 추출한 텍스트입니다.
이 텍스트를 분석하여 다음 작업을 수행해주세요:

**핵심 원칙:**
- 정확성 최우선: 불확실한 내용은 추측하지 말고 원본 표기 유지
- 학술적 정확성: 수식, 공식, 전문 용어는 검증 가능한 형태로만 수정
- 맥락 기반 해석: 주제와 과목을 파악하여 적절히 처리

**작업 지침:**

1. **OCR 오류 수정**
   - 일반적 오타: 오/요, 되/돼, 띄어�기 등
   - 유사 문자 혼동: 0/O, 1/l/I, 2/Z, 5/S 등
   - 특수문자 복원: 兀→π, ㅠ→π 등

2. **과목별 표기법 정리**
   - 수학: 수식 기호(±, ×, ≤, ≥, ∞, π), 역삼각함수(sin⁻¹, cos⁻¹, tan⁻¹)
   - 과학: 화학식(H₂O, CO₂), 단위(m/s², kg, J), 원소기호
   - 언어: 문법 용어, 철자, 발음 기호
   - 기타: 인명, 지명, 연도 정확히

3. **문장 구조 정리**
   - 불완전한 문장 복원
   - 리스트나 번호는 명확하게 구조화
   - 가독성 향상 (줄바꿈, 들여쓰기)

4. **제목 생성** (10자 이내, 해당 영역의 핵심 주제나 개념)

**주의사항:**
- 수학/과학 공식이 불확실하면 추측하지 말 것
- 중요한 수치나 용어가 애매하면 [?] 표시 고려
- 학생이 오해할 수 있는 잘못된 정보는 절대 생성하지 말 것

**OCR 원본 텍스트:**
```
{raw_text}
```

**출력 형식 (JSON):**
{{
  "title": "핵심 주제를 나타내는 간단한 제목",
  "content": "정제되고 구조화된 내용"
}}

JSON 형식으로만 응답해주세요."""

        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that corrects OCR text and structures it with title and content in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 낮은 온도로 일관성 있는 결과
                response_format={"type": "json_object"}
            )

            # 응답 파싱
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # 제목과 내용 검증
            title = result.get("title", f"{region_label}").strip()
            content = result.get("content", raw_text).strip()

            return {
                "title": title,
                "content": content
            }

        except Exception as e:
            print(f"  ⚠️  LLM 후처리 실패: {str(e)}")
            # 실패 시 원본 텍스트 반환
            return {
                "title": f"{region_label}",
                "content": raw_text
            }

    def process_ocr_results(self, ocr_json_path: str) -> Dict:
        """
        OCR 결과 JSON을 읽어 각 영역별로 후처리

        Args:
            ocr_json_path: OCR 결과 JSON 파일 경로

        Returns:
            후처리된 결과 딕셔너리
        """
        # OCR 결과 읽기
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        print(f"\nOCR 결과 로드: {ocr_json_path}")
        print(f"  - 영역 개수: {ocr_data['num_regions']}")
        print(f"  - 레이아웃: {ocr_data['layout_type']}")
        print("="*60)

        # 각 영역별로 후처리
        processed_regions = []

        for region in ocr_data['regions']:
            region_label = region['region_label']
            ocr_result = region.get('ocr')

            if not ocr_result:
                print(f"\n영역: {region_label}")
                print("  ⚠️  OCR 결과 없음 - 건너뜀")
                continue

            print(f"\n영역: {region_label}")
            print(f"  원본 텍스트 길이: {ocr_result['character_count']}자")
            print(f"  신뢰도: {ocr_result['confidence']:.2%}")
            print(f"  LLM 후처리 중...")

            # LLM 후처리
            processed = self.postprocess_text(
                ocr_result['full_text'],
                region_label
            )

            # 결과 저장
            processed_regions.append({
                "region_label": region_label,
                "region_info": region['region_info'],
                "original_ocr": {
                    "full_text": ocr_result['full_text'],
                    "confidence": ocr_result['confidence'],
                    "character_count": ocr_result['character_count']
                },
                "processed": {
                    "title": processed['title'],
                    "content": processed['content']
                }
            })

            print(f"  ✓ 완료")
            print(f"    - 제목: {processed['title']}")
            print(f"    - 내용 길이: {len(processed['content'])}자")

        print("="*60)

        # 통합 결과 생성
        result = {
            "source_ocr": ocr_json_path,
            "source_image": ocr_data['source_image'],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "layout_type": ocr_data['layout_type'],
            "model_used": self.model,
            "num_regions": len(processed_regions),
            "regions": processed_regions,
            "summary": {
                "total_regions_processed": len(processed_regions),
                "average_confidence": round(
                    sum(r['original_ocr']['confidence'] for r in processed_regions) / len(processed_regions),
                    4
                ) if processed_regions else 0
            }
        }

        return result

    def save_results(self, result: Dict, output_path: str) -> str:
        """
        후처리 결과를 JSON 파일로 저장

        Args:
            result: 후처리 결과 딕셔너리
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
        print("LLM 후처리 요약")
        print("="*60)
        print(f"처리된 영역: {result['summary']['total_regions_processed']}개")
        print(f"평균 OCR 신뢰도: {result['summary']['average_confidence']:.2%}")
        print(f"사용 모델: {result['model_used']}")
        print("\n각 영역별 제목:")
        for region in result['regions']:
            print(f"  - {region['region_label']}: {region['processed']['title']}")
        print("="*60)

        return output_path


def find_latest_ocr_json(output_dir: str = "output") -> Optional[str]:
    """
    가장 최근 OCR 결과 JSON 파일 찾기

    Args:
        output_dir: 출력 디렉토리

    Returns:
        JSON 파일 경로 (없으면 None)
    """
    json_files = list(Path(output_dir).glob("*_ocr_result.json"))

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
    print("노트 LLM 후처리 시스템")
    print("="*60)

    # 가장 최근 OCR 결과 찾기
    ocr_json = find_latest_ocr_json(output_dir)

    if not ocr_json:
        print(f"\n❌ OCR 결과를 찾을 수 없습니다: {output_dir}")
        print("먼저 note_ocr_processor.py를 실행하세요.")
        return

    print(f"\n✓ 최신 OCR 결과 감지: {Path(ocr_json).name}")

    # LLM 후처리 프로세서 생성 (gpt-4o-mini 사용 - 가장 저렴)
    processor = NoteLLMPostprocessor(model="gpt-4o")

    # 후처리 실행
    result = processor.process_ocr_results(ocr_json)

    # 결과 저장
    # 파일명: IMG_5588_15_06_final_result.json
    base_name = Path(ocr_json).stem.replace('_ocr_result', '')
    output_path = os.path.join(output_dir, f"{base_name}_final_result.json")

    processor.save_results(result, output_path)


if __name__ == "__main__":
    main()
