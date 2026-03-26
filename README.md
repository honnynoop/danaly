# danaly
파일 목록 요약
세트MD 파일이미지 접두사✈️ Flightsflights_classification.mdfl_📉 Dowjonesdowjones_classification.mddj_🧊 Seaiceseaice_classification.mdsi_

3개 시계열 종합 비교
항목✈️ Flights📉 Dow Jones🧊 Sea Ice분석 유형시계열 회귀시계열 회귀시계열 회귀관측 기간12년(1949~60)55년(1914~68)45년(1979~23)타겟월 승객 수다음달 로그 지수월 해빙 면적최고 R²0.970 (GBM)0.931 (LR만 양수)0.988 (LR)최적 모델GBM / LRLinear만 성공LR계절성강함 (여름 성수기)없음강함 (3월↑ 9월↓)예측 난이도쉬움어려움 (금융)쉬움나무 모델잘 작동실패 (과적합)잘 작동

핵심 교훈: 물리적 법칙(계절·추세)이 있는 데이터(Flights·Sea Ice)는 간단한 피처로도 R² > 0.97 달성. 반면 금융 시계열(Dow Jones)은 랜덤워크 특성으로 나무 모델이 실패하고 선형 AR(1)만 유효 — 시계열 특성 이해가 모델 선택보다 중요합니다.
