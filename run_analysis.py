#!/usr/bin/env python
"""
노트 분석 전체 파이프라인 실행 스크립트
경계 검출 → 레이아웃 분석을 순차적으로 실행
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()


def run_command(cmd):
    """명령 실행 및 출력"""
    print(f"실행: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0


def main():
    """메인 실행"""
    print("="*60)
    print("노트 분석 파이프라인")
    print("="*60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)

    # 가상환경 활성화 명령 준비
    venv_activate = "source venv/bin/activate && " if os.path.exists("venv") else ""

    # Step 1: 경계 검출
    print("\n[1/2] 노트 경계 검출 중...")
    print("-"*40)
    if not run_command(f"{venv_activate}python note_boundary_detector.py"):
        print("❌ 경계 검출 실패")
        return 1

    print("-"*40)
    print("✓ 경계 검출 완료")

    # Step 2: 레이아웃 분석
    print("\n[2/2] 레이아웃 분석 중...")
    print("-"*40)
    if not run_command(f"{venv_activate}python note_layout_analyzer.py"):
        print("❌ 레이아웃 분석 실패")
        return 1

    print("-"*40)
    print("✓ 레이아웃 분석 완료")

    # 결과 요약
    print("\n" + "="*60)
    print("분석 완료!")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 생성된 파일 목록
    output_dir = Path("output")
    if output_dir.exists():
        recent_files = sorted(output_dir.glob("IMG_5588_*"), key=os.path.getmtime, reverse=True)[:6]
        if recent_files:
            print("\n최근 생성된 파일:")
            for f in recent_files:
                print(f"  - {f.name}")

    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())