# 노트 분석 시스템 사용법

## 빠른 시작

### 1. 전체 파이프라인 실행 (추천)
```bash
python run_analysis.py
```
- 경계 검출 + 레이아웃 분석을 한 번에 실행

### 2. 개별 실행
```bash
# 경계 검출만
python note_boundary_detector.py

# 레이아웃 분석만 (자동으로 최신 경계 파일 사용)
python note_layout_analyzer.py
```

## 파일 설정 (.env)

`.env` 파일에서 이미지 경로 설정:

```env
IMAGE_DIRECTORY=./test_images
IMAGE_FILENAME=IMG_5588.JPG
OUTPUT_DIRECTORY=output
```

## 주요 기능

### 경계 검출 (note_boundary_detector.py)
- DocAligner 모델로 노트의 4개 꼭짓점 검출
- 기울어진 노트를 정면으로 자동 보정
- 출력: `*_boundary.jpg`, `*_corners.txt`

### 레이아웃 분석 (note_layout_analyzer.py)
- **자동으로 최신 경계 파일 사용**
- 30% 이상 길이의 선만 검출 (노이즈 제거)
- 중심 ±20% 범위에서 가장 중앙 선 선택
- 선택된 선을 경계까지 연장하여 영역 분할

#### 지원 레이아웃
- **0분할**: 단일 영역
- **2분할**: 세로선으로 좌/우 분할
- **4분할**: 십자선으로 4개 영역

## 출력 파일 구조

```
output/
├── IMG_5588_10_35_boundary.jpg         # 경계 시각화
├── IMG_5588_10_35_corners.txt          # 경계 좌표
├── IMG_5588_10_35_layout_visualization.jpg  # 레이아웃 시각화
├── IMG_5588_10_35_layout_analysis.json     # 분석 데이터
└── regions/
    ├── IMG_5588_10_35_좌상단.jpg
    ├── IMG_5588_10_35_우상단.jpg
    ├── IMG_5588_10_35_좌하단.jpg
    └── IMG_5588_10_35_우하단.jpg
```

## 특징
- 파일명에 시간 자동 포함 (시_분)
- 터미널 옵션 불필요 (자동화)
- 경계 검출 결과 자동 활용
- 끊어진 선 연장으로 깔끔한 분할