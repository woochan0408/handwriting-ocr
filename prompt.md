# 노트 레이아웃 분석 기능 구현 요청

## 현재까지 완료된 작업

### 1. 노트 경계 검출 (완료)
- **구현 파일**: `note_boundary_detector.py`
- **기술**: DocAligner 딥러닝 모델 사용
- **성능**: 94% 정확도로 노트의 4개 꼭짓점 검출
- **기능**:
  - 노트 사진에서 자동으로 노트 영역 찾기
  - 기울어진 노트를 정면으로 투시 변환
  - 결과를 이미지와 좌표 파일로 저장

### 2. 프로젝트 구조
```
handwrite-ocr/
├── note_boundary_detector.py   # DocAligner 경계 검출 (완료)
├── requirements.txt            # 패키지 목록
├── README.md                   # 프로젝트 문서
├── test_images/               # 테스트 이미지
└── output/                    # 결과 저장
```

## 다음 작업: 노트 레이아웃 분석

### 목표
노트 내부의 레이아웃 구조를 분석하여 영역을 분리하는 기능 구현

### 구체적 요구사항

1. **입력**:
   - 경계 검출이 완료되어 정면으로 변환된 노트 이미지
   - (note_boundary_detector.py의 출력 이미지 사용)

2. **분석할 레이아웃 패턴**:
   - 분할 없음 (단일 영역)
   - 2분할 (세로선으로 좌/우 분할)
   - 4분할 (십자선으로 4개 영역 분할)
   - 수평선 분할 (여러 개의 수평선)

3. **구현 방법**: OpenCV 사용 (가볍고 빠른 구현)
   - Hough Line Transform으로 직선 검출
   - 수직/수평선 구분
   - 선의 위치와 길이로 레이아웃 패턴 판단
   - 검출된 선을 기준으로 영역 분리

4. **출력**:
   - 각 영역의 좌표 (x, y, width, height)
   - 레이아웃 타입 (single, vertical_split, grid_4, horizontal_lines 등)
   - 시각화 이미지 (각 영역에 다른 색상 표시)
   - 각 영역을 개별 이미지로 저장

5. **파일명**: `note_layout_analyzer.py`

### 구현 예시 코드 구조

```python
class NoteLayoutAnalyzer:
    def __init__(self):
        # 초기화
        pass

    def detect_lines(self, image):
        # Hough 변환으로 선 검출
        pass

    def classify_layout(self, lines):
        # 검출된 선으로 레이아웃 패턴 분류
        # single, 2-split, 4-grid 등 판단
        pass

    def extract_regions(self, image, layout_type, lines):
        # 레이아웃에 따라 영역 추출
        # 각 영역의 좌표 반환
        pass

    def save_regions(self, image, regions, output_dir):
        # 각 영역을 개별 이미지로 저장
        pass

    def visualize_layout(self, image, regions, layout_type):
        # 레이아웃 시각화
        pass
```

### 테스트 시나리오
1. 단일 영역 노트 이미지
2. 세로선 하나로 2분할된 노트
3. 십자선으로 4분할된 노트
4. 여러 수평선이 있는 노트 (단일 영역 노트로 맵핑)

### 주의사항
- 선 검출 시 노이즈와 실제 구분선을 구별해야 함
- 선의 최소 길이 임계값 설정 필요
- 완벽히 직선이 아닌 손으로 그은 선도 검출되어야 함

### 환경 정보
- Python 3.11.9
- 이미 설치된 패키지: opencv-python, numpy, pillow
- 추가 패키지는 최소화

이 기능을 구현해주세요. 코드는 간결하고 이해하기 쉽게 작성하고, 주석은 한글로 작성해주세요.