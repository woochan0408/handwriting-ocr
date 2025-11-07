# handwrite-ocr 프로젝트

## 개요
손글씨 노트 사진을 디지털 텍스트로 변환하는 OCR 컴포넌트입니다. 레이아웃 영역 분리, Google Vision OCR, LLM 후처리 파이프라인을 통해 높은 정확도의 손글씨 인식을 제공합니다.

## 주요 기능
- 노트 레이아웃 자동 분석 (분할 없음 / 2분할 / 4분할 등)
- 각 영역별 이미지 분리 & OCR 적용
- 복습용 콘텐츠(Q&A, 요약, 카드 등) 자동 생성 (LLM 활용 가능)
- 사용자 UI에서 부분 편집, 오타 추천, 영역 재설정 지원

## 프로젝트 구조

```
handwrite-ocr/
├── note_boundary_detector.py   # DocAligner 기반 경계 검출 (94% 정확도)
├── requirements.txt            # pip 패키지 목록
├── .env                       # API 키 설정 (Google Vision, OpenAI)
├── test_images/               # 테스트 이미지 폴더
└── output/                    # 처리 결과 저장
```

## 설치 및 사용법

### 요구사항
- Python 3.10 이상 (3.11.9 권장)
- pip 패키지 관리자

### 설치
```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 2. 패키지 설치
pip install -r requirements.txt
```

### 사용법
```bash
# 기본 실행 (test_images 폴더에서 자동으로 이미지 찾음)
python note_boundary_detector.py

# 특정 이미지 지정
python note_boundary_detector.py --input path/to/image.jpg

# 출력 디렉토리 지정
python note_boundary_detector.py --input image.jpg --output results/
```

## 구현 방식 요약
### 1. 이미지 처리 및 레이아웃 분석 (완료)
- 노트 경계 검출: DocAligner 딥러닝 모델 사용
- 4개 꼭짓점 자동 검출: 94% 정확도 달성
- 투시 변환: 기울어진 노트를 정면으로 자동 변환
- 전처리(명암, 샤프닝 등)로 인식률 보강

### 2. OCR 및 영역별 텍스트 추출
- 추출된 각 영역 이미지를 OCR 엔진(구글 Vision, Tesseract 등)으로 텍스트 변환
- 단순 왼→오, 위→아래로 추출하지 않고, 노트 구조에 맞춰 텍스트 분리

### 3. LLM 활용
- 각 영역 OCR 결과를 LLM(GPT 등)에 전달
  - 전문 교정(수식, 단어, 오타, 문장 정렬 등)
  - 복습용 구조(플래시카드, Q&A, 요약, 카드화 등) 자동 생성
- 프롬프트 설계와 사용자의 학습 목적에 따라 다양한 콘텐츠 포맷으로 후처리

## 기술 스택 및 고려 사항
- 파이썬 기반 OpenCV, OCR 라이브러리
- LLM API(예: GPT) 연계 자동화
- 무료 오픈소스(BSD 라이선스 사용 가능), 커뮤니티 자료 활용
- 사용자 개인정보 보안(이미지/텍스트 처리 시 주의)

---

## 출력 파일
- `*_boundary.jpg`: 검출된 경계가 표시된 이미지
- `*_corners.txt`: 4개 꼭짓점 좌표 정보

## 설치 참고사항

### 패키지 설치 방식
- 가상환경 설정
- DocAligner: `pip install docaligner-docsaid`로 자동 설치
- 가상환경 사용 권장 (시스템 Python과 분리)

### Python 버전
- 최소: Python 3.10 이상
- 권장: Python 3.11.9

## 앞으로 추가할 기능/과제
- 노트 경계 자동 검출 (완료)
- 다양한 노트 패턴 샘플 DB 구축 및 인식률 실험
- LLM 활용 복습 콘텐츠 생성 예시 및 최적 프롬프트 설계
- 사용자 친화적 보정 UI 반영
- 전체 처리 파이프라인 튜닝 및 테스트 마크다운 기록
