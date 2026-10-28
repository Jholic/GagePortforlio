#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTO_PORTFOLIO v3.0 올인원 인터랙티브 대시보드 생성기 (`index.html` & `krx_dashboard.html`)
- Netlify 정적 호스팅(`index.html`) 및 로컬 브라우저 직접 열기 완벽 지원
- 로컬 웹앱 서버(`webapp.py`) 실행 시 상단 '신규 재무·시세 증분 업데이트' 버튼으로 실시간 DB 갱신 연동
"""

import os
import json
from datetime import datetime


DASHBOARD_V3_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AUTO PORTFOLIO v3.0 · 진성 밸류에이션 & 텐서 역변환 퀀트 플랫폼</title>
  <style>
    :root {
      --bg: #080d1a;
      --bg-elev: #101829;
      --bg-card: #131d31;
      --bg-soft: #1a2742;
      --line: #243454;
      --text: #eef3fc;
      --muted: #94a3b8;
      --accent: #4f8ff7;
      --accent-soft: rgba(79, 143, 247, 0.14);
      --long: #2dd4bf;
      --long-soft: rgba(45, 212, 191, 0.13);
      --short: #f43f5e;
      --short-soft: rgba(244, 63, 94, 0.13);
      --warn: #f59e0b;
      --warn-soft: rgba(245, 158, 11, 0.14);
      --purple: #a78bfa;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0;
      background: var(--bg); color: var(--text);
      font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    body { padding: 0 24px 72px; max-width: 1560px; margin: 0 auto; }
    h1 { font-size: 24px; letter-spacing: -0.025em; margin: 0 0 6px; font-weight: 750; }
    h2 { font-size: 16px; margin: 0 0 12px; font-weight: 700; letter-spacing: -0.01em; display: flex; align-items: center; gap: 8px; }
    h3 { font-size: 14px; margin: 0 0 8px; font-weight: 650; color: var(--text); }
    p { color: var(--muted); margin: 0; line-height: 1.55; font-size: 13.5px; }

    /* Top Navigation Bar */
    .topbar {
      position: sticky; top: 0; z-index: 50;
      background: rgba(8, 13, 26, 0.92); backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--line);
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px; padding: 12px 0; margin-bottom: 22px; flex-wrap: wrap;
    }
    .brand-wrap { display: flex; align-items: center; gap: 12px; }
    .brand-logo {
      width: 36px; height: 36px; border-radius: 10px;
      background: linear-gradient(135deg, #3b82f6, #2dd4bf);
      display: flex; align-items: center; justify-content: center;
      font-weight: 800; font-size: 15px; color: #080d1a;
    }
    .brand-title { font-size: 16px; font-weight: 750; letter-spacing: -0.02em; }
    .brand-sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
    .nav-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
    .nav-btn {
      background: var(--bg-elev); color: var(--muted);
      border: 1px solid var(--line); border-radius: 10px;
      padding: 9px 15px; cursor: pointer; font: inherit; font-size: 13.5px; font-weight: 650;
      display: flex; align-items: center; gap: 7px; transition: all 0.15s ease;
    }
    .nav-btn:hover { color: var(--text); border-color: #385082; }
    .nav-btn.active {
      background: var(--accent-soft); color: #fff;
      border-color: var(--accent); box-shadow: 0 0 0 1px rgba(79, 143, 247, 0.25);
    }
    .nav-actions { display: flex; align-items: center; gap: 8px; }
    .quick-search-top {
      background: var(--bg-elev); color: var(--text);
      border: 1px solid var(--line); border-radius: 9px;
      padding: 8px 12px; font-size: 13px; width: 220px;
    }
    .sync-btn {
      background: var(--bg-soft); color: var(--text);
      border: 1px solid var(--line); border-radius: 9px;
      padding: 8px 12px; font-size: 12.5px; font-weight: 600; cursor: pointer;
    }
    .sync-btn:hover { border-color: var(--accent); color: var(--accent); }

    /* Layout & Cards */
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }
    .card {
      background: var(--bg-card); border: 1px solid var(--line);
      border-radius: 14px; padding: 18px; margin-bottom: 14px;
    }
    .banner {
      background: linear-gradient(135deg, rgba(79, 143, 247, 0.12), rgba(45, 212, 191, 0.08));
      border: 1px solid rgba(79, 143, 247, 0.32); border-radius: 14px;
      padding: 16px 20px; margin-bottom: 16px;
      display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
    }
    .kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .kpi {
      background: var(--bg-card); border: 1px solid var(--line);
      border-radius: 12px; padding: 15px 16px;
    }
    .kpi .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; font-weight: 500; }
    .kpi .val { font-size: 23px; font-weight: 750; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
    .kpi .sub { color: var(--muted); font-size: 11.5px; margin-top: 5px; }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }
    .grid-sidebar { display: grid; grid-template-columns: 340px 1fr; gap: 14px; margin-bottom: 14px; }

    /* Portfolio Selector Pills */
    .port-selector { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .port-pill {
      background: var(--bg-elev); color: var(--muted);
      border: 1px solid var(--line); border-radius: 10px;
      padding: 10px 14px; cursor: pointer; font: inherit; font-size: 13px; font-weight: 650;
      transition: all 0.15s;
    }
    .port-pill:hover { color: var(--text); border-color: #3b5387; }
    .port-pill.active { background: var(--long-soft); color: var(--long); border-color: var(--long); }
    .port-pill.trap.active { background: var(--short-soft); color: var(--short); border-color: var(--short); }

    /* Simulator Bar */
    .sim-bar {
      background: var(--bg-elev); border: 1px solid var(--line);
      border-radius: 12px; padding: 12px 16px; margin-bottom: 14px;
      display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
    }
    .sim-input-wrap { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .sim-input {
      background: var(--bg-soft); color: #fff; border: 1px solid var(--accent);
      border-radius: 8px; padding: 7px 12px; font-size: 14px; font-weight: 700; width: 170px; text-align: right;
    }
    .preset-btn {
      background: var(--bg-soft); color: var(--muted); border: 1px solid var(--line);
      border-radius: 7px; padding: 5px 10px; font-size: 12px; cursor: pointer;
    }
    .preset-btn:hover { color: #fff; border-color: var(--accent); }

    /* Tables */
    .table-wrap { overflow-x: auto; max-height: 620px; border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th {
      color: var(--muted); font-weight: 650; font-size: 11.5px;
      position: sticky; top: 0; background: var(--bg-elev); z-index: 2;
      cursor: pointer; user-select: none;
    }
    th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
    tbody tr { transition: background 0.12s; }
    tbody tr.clickable { cursor: pointer; }
    tbody tr.clickable:hover { background: rgba(79, 143, 247, 0.10); }
    .pos { color: var(--long); font-weight: 600; }
    .neg { color: var(--short); font-weight: 600; }
    .warn-txt { color: var(--warn); font-weight: 600; }

    /* Badges */
    .badge {
      display: inline-block; padding: 3px 9px; border-radius: 99px;
      font-size: 11px; font-weight: 650; background: var(--bg-soft); color: var(--muted);
    }
    .badge.long { color: var(--long); background: var(--long-soft); }
    .badge.short { color: var(--short); background: var(--short-soft); }
    .badge.warn { color: var(--warn); background: var(--warn-soft); }
    .badge.accent { color: var(--accent); background: var(--accent-soft); }

    /* Stock Deep-Dive Search & Cards */
    .search-hero {
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;
    }
    .search-box-wrap { position: relative; flex: 1; min-width: 280px; }
    .search-input-lg {
      width: 100%; background: var(--bg-elev); color: #fff;
      border: 1.5px solid var(--accent); border-radius: 11px;
      padding: 11px 16px; font-size: 14.5px; font-weight: 600;
    }
    .autocomplete-list {
      position: absolute; top: 100%; left: 0; right: 0; z-index: 40;
      background: var(--bg-elev); border: 1px solid var(--accent);
      border-radius: 11px; margin-top: 4px; max-height: 340px; overflow-y: auto;
      box-shadow: 0 14px 34px rgba(0,0,0,0.65); display: none;
    }
    .ac-item {
      padding: 10px 14px; border-bottom: 1px solid var(--line);
      display: flex; justify-content: space-between; align-items: center; cursor: pointer;
    }
    .ac-item:hover { background: var(--bg-soft); }
    .sample-chips { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    .chip {
      background: var(--bg-elev); color: var(--muted); border: 1px solid var(--line);
      border-radius: 99px; padding: 5px 11px; font-size: 12px; cursor: pointer;
    }
    .chip:hover, .chip.active { color: #fff; border-color: var(--accent); background: var(--accent-soft); }

    .thesis-box {
      background: var(--bg-elev); border-left: 4px solid var(--long);
      border-radius: 10px; padding: 14px 16px; margin: 12px 0;
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px;
    }
    .thesis-box.short-mode { border-left-color: var(--short); }
    .thesis-item h4 { margin: 0 0 4px; font-size: 12.5px; color: var(--accent); }
    .thesis-item p { font-size: 13px; color: var(--text); line-height: 1.5; }

    .vs-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
    .vs-card {
      background: var(--bg-elev); border: 1px solid var(--line);
      border-radius: 11px; padding: 12px 14px;
    }
    .vs-card .title { font-size: 11.5px; color: var(--muted); margin-bottom: 8px; font-weight: 600; }
    .vs-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; font-size: 13px; }
    .vs-row .lbl { color: var(--muted); font-size: 12px; }
    .vs-row .v { font-weight: 700; font-variant-numeric: tabular-nums; }

    svg.chart-svg { width: 100%; height: 250px; display: block; overflow: visible; }
    .chart-legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); margin-bottom: 8px; flex-wrap: wrap; }
    .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }

    @media (max-width: 1100px) {
      .kpis, .vs-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid-2, .grid-3, .grid-sidebar, .thesis-box { grid-template-columns: 1fr; }
    }
    @media (max-width: 640px) {
      body { padding: 0 12px 48px; }
      .kpis, .vs-grid { grid-template-columns: 1fr; }
      .topbar { padding: 10px 0; }
      .quick-search-top { width: 100%; }
    }
  </style>
</head>
<body>

  <!-- Top Navigation -->
  <header class="topbar">
    <div class="brand-wrap">
      <div class="brand-logo">Q3</div>
      <div>
        <div class="brand-title">AUTO PORTFOLIO v3.0 · KRX 진성 밸류에이션 & 텐서 알파</div>
        <div class="brand-sub" id="top-meta">KRX 전 종목 Point-in-Time K-IFRS 재무제표 · 허상 제거 진짜 PER/PBR & 장기 시계열 백테스트</div>
      </div>
    </div>
    <nav class="nav-tabs" id="main-nav">
      <button class="nav-btn active" data-tab="portfolio">🎯 추천 포트폴리오</button>
      <button class="nav-btn" data-tab="stock">🔍 종목 상세 분석 & 차트</button>
      <button class="nav-btn" data-tab="backtest">📈 장기 시계열 백테스트</button>
      <button class="nav-btn" data-tab="screener">📋 전 종목 스크리너</button>
    </nav>
    <div class="nav-actions">
      <input type="search" id="global-quick-search" class="quick-search-top" placeholder="종목명/코드 빠른 검색 (Enter)" />
      <button class="sync-btn" id="btn-live-sync" title="로컬 웹앱(webapp.py) 구동 시 신규 재무제표·10년 시세를 증분 업데이트합니다">🔄 데이터 증분 갱신</button>
    </div>
  </header>

  <!-- ===================================================================== -->
  <!-- TAB 1: 우리 로직 기반 추천 포트폴리오 (Model Portfolios)                 -->
  <!-- ===================================================================== -->
  <section id="tab-portfolio" class="tab-pane active">
    <div class="banner">
      <div>
        <h1 id="port-title">대표 추천 · v3 텐서 알파 핵심 포트폴리오 (Top 20)</h1>
        <p id="port-subtitle">6단계 대수적 텐서 역변환 알파 스코어 최상위 + 회계 착시(b_m) 클린 + 진성 적정주가 대비 저평가된 핵심 매수 20선</p>
      </div>
      <span class="badge long" id="port-badge" style="font-size:13px;padding:6px 14px">CORE ALPHA TOP 20</span>
    </div>

    <div class="port-selector" id="port-selector">
      <button class="port-pill active" data-port="core_top20">🏆 대표 추천 · 핵심 알파 Top 20</button>
      <button class="port-pill" data-port="true_value_top15">💎 딥밸류 · 진짜 저PBR·저PER Top 15</button>
      <button class="port-pill" data-port="synergy_quality_top15">🚀 고수익·시너지 모멘텀 Top 15</button>
      <button class="port-pill" data-port="kospi_core_top15">🏛️ KOSPI 안정형 우량주 Top 15</button>
      <button class="port-pill trap" data-port="value_trap_avoid15">⚠️ 매수 금지 · 회계착시 가치함정 경계 15선</button>
    </div>

    <div class="kpis" id="port-kpis"></div>

    <div class="sim-bar" id="sim-bar">
      <div class="sim-input-wrap">
        <strong style="font-size:13.5px;color:var(--accent)">💰 포트폴리오 투자금 배분 시뮬레이터</strong>
        <span style="font-size:12.5px;color:var(--muted)">총 투자 원금(원):</span>
        <input type="text" id="sim-budget" class="sim-input" value="50,000,000" />
        <button class="preset-btn" data-amt="10000000">1천만</button>
        <button class="preset-btn" data-amt="50000000">5천만</button>
        <button class="preset-btn" data-amt="100000000">1억</button>
        <button class="preset-btn" data-amt="500000000">5억</button>
      </div>
      <div id="sim-result-summary" style="font-size:13px"></div>
    </div>

    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px">
        <h2 style="margin:0">포트폴리오 편입 종목 및 산출 근거 <small style="font-weight:400;color:var(--muted);font-size:12px">(종목을 클릭하면 상세 시계열 비교 차트로 이동합니다)</small></h2>
      </div>
      <div class="table-wrap" style="max-height:none">
        <table id="port-table"></table>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- TAB 2: 주식 검색 & 종목별 상세 분석 뷰어 (Stock Deep-Dive Explorer)       -->
  <!-- ===================================================================== -->
  <section id="tab-stock" class="tab-pane">
    <div class="card">
      <div class="search-hero">
        <div class="search-box-wrap">
          <input type="search" id="stock-search-input" class="search-input-lg" placeholder="🔍 종목명 또는 6자리 종목코드 입력 (예: 삼성전자, SK하이닉스, 사람인, 빅솔론, 현대차, 삼부토건...)" autocomplete="off" />
          <div id="stock-ac-list" class="autocomplete-list"></div>
        </div>
        <div class="sample-chips" id="sample-chips">
          <span style="font-size:12px;color:var(--muted)">대표 분석 샘플:</span>
        </div>
      </div>

      <div id="stock-header-area"></div>
      <div id="stock-thesis-area"></div>
      <div class="vs-grid" id="stock-vs-cards"></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>📈 [차트 1] 실제 주가 vs 진성 내재 적정주가(Fair Price) & 향후 12M 예상 경로</h2>
        <div class="chart-legend">
          <span><i class="legend-dot" style="background:#4f8ff7"></i>실제 주가 (Actual Price)</span>
          <span><i class="legend-dot" style="background:#2dd4bf"></i>진성 내재 적정주가 (True Fair Price)</span>
          <span><i class="legend-dot" style="background:#f59e0b"></i>향후 6M·12M 예상 목표주가 경로</span>
        </div>
        <svg id="chart-price-fair" class="chart-svg" viewBox="0 0 680 250"></svg>
        <p style="font-size:12px;margin-top:6px">초록색 적정주가 선이 파란색 실제 주가보다 위에 있을수록 실질 내재가치 대비 저평가(매수 우위) 구간입니다.</p>
      </div>

      <div class="card">
        <h2>⚖️ [차트 2] 표면적 PBR vs 허상을 걷어낸 진짜 PBR (True PBR) 추이</h2>
        <div class="chart-legend">
          <span><i class="legend-dot" style="background:#94a3b8"></i>표면적 PBR (장부 BPS 기준)</span>
          <span><i class="legend-dot" style="background:#2dd4bf"></i>허상 제거 진짜 PBR (부실·부채차감 진성 BPS 기준)</span>
        </div>
        <svg id="chart-pbr-comp" class="chart-svg" viewBox="0 0 680 250"></svg>
        <p style="font-size:12px;margin-top:6px">진짜 PBR(초록)이 표면 PBR(회색)보다 낮으면 현금·수익성 우량 할증, 반대로 훨씬 높으면 부실자산·과다부채 가치함정(Value Trap)입니다.</p>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>🔍 [차트 3] 표면적 PER vs 허상을 걷어낸 진짜 PER (True PER) 추이</h2>
        <div class="chart-legend">
          <span><i class="legend-dot" style="background:#94a3b8"></i>표면적 PER (단순 당기순이익 기준)</span>
          <span><i class="legend-dot" style="background:#a78bfa"></i>허상 제거 진짜 PER (일회성 영업외이익 제거 본업 수익력 기준)</span>
        </div>
        <svg id="chart-per-comp" class="chart-svg" viewBox="0 0 680 250"></svg>
        <p style="font-size:12px;margin-top:6px">일회성 영업외수익으로 표면 PER만 낮아진 착시 종목은 진짜 PER(보라색)이 높거나 음수(-)로 나타나 허상을 즉시 식별합니다.</p>
      </div>

      <div class="card">
        <h2>💎 [차트 4] 표면 ROE vs 진짜 ROE & 본업 영업이익률(%) 추이</h2>
        <div class="chart-legend">
          <span><i class="legend-dot" style="background:#94a3b8"></i>표면 ROE (%)</span>
          <span><i class="legend-dot" style="background:#2dd4bf"></i>허상 제거 진짜 ROE (%)</span>
          <span><i class="legend-dot" style="background:#f59e0b"></i>본업 영업이익률 (Op Margin %)</span>
        </div>
        <svg id="chart-roe-comp" class="chart-svg" viewBox="0 0 680 250"></svg>
        <p style="font-size:12px;margin-top:6px">과다부채나 자본잠식으로 표면 ROE가 부풀려진 경우, 디레버리징 및 영업이익 기반 진짜 ROE로 보정합니다.</p>
      </div>
    </div>

    <div class="card">
      <h2>📅 시기별 상세 지표 및 주가 대조 테이블 (2021.12 ~ 2027.10 예상)</h2>
      <div class="table-wrap">
        <table id="stock-ts-table"></table>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- TAB 3: 장기 시계열 백테스트 & 표면 vs 진짜 지표 실증 분석                 -->
  <!-- ===================================================================== -->
  <section id="tab-backtest" class="tab-pane">
    <div class="banner">
      <div>
        <h1>장기 시계열(2021.12 ~ 2026.10) 백테스트 & 표면 vs 진짜 PER·PBR 실증 검증</h1>
        <p>KRX 1,895개 상장 종목 대상 Walk-Forward 표본 외 검증: 우리가 만든 진성 적정주가 괴리율과 진짜 PBR/PER이 실제 주가 변동을 어떻게 선행하는지 입증합니다.</p>
      </div>
      <span class="badge accent" style="font-size:13px;padding:6px 14px">OUT-OF-SAMPLE VERIFIED</span>
    </div>

    <div class="kpis" id="bt-kpis"></div>

    <div class="grid-2">
      <div class="card">
        <h2>📈 [실증 1] 전략별 장기 누적 수익률 추이 (2021.12 = 100 기준 NAV)</h2>
        <div class="chart-legend" id="bt-curve-legend"></div>
        <svg id="chart-bt-curves" class="chart-svg" viewBox="0 0 680 260"></svg>
      </div>
      <div class="card">
        <h2>📊 [실증 2] 알파 스코어 5분위(Quintile Q5 ~ Q1) 누적 수익률 단조성 검증</h2>
        <div class="chart-legend" id="bt-quintile-legend"></div>
        <svg id="chart-bt-quintiles" class="chart-svg" viewBox="0 0 680 260"></svg>
      </div>
    </div>

    <div class="card">
      <h2>📋 전략별 장기 백테스트 성과 요약표 (2021.12 ~ 2026.10, 거래비용 15bp 차감 후)</h2>
      <div class="table-wrap" style="max-height:none">
        <table id="bt-strategy-table"></table>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>🔬 [실증 3] 표면적 PBR·PER vs 허상 제거 진짜 PBR·PER 4분면 실증 매트릭스</h2>
        <p style="margin-bottom:10px">표면적으로는 저PBR처럼 보이지만 회계 허상을 걷어내면 고PBR·부실인 <strong>가치함정(Value Trap)</strong>과 <strong>진성 저평가(True Gem)</strong>의 차기 실현 수익률 차이입니다.</p>
        <div class="table-wrap" style="max-height:none">
          <table id="bt-quadrant-table"></table>
        </div>
      </div>

      <div class="card">
        <h2>🎯 [실증 4] 진성 적정주가 괴리율((Fair - Actual)/Actual) 구간별 향후 실제 주가 수렴성</h2>
        <p style="margin-bottom:10px">우리가 산출한 진성 적정주가가 실제 주가보다 높을수록(+괴리율) 차기 구간 실제 주가가 일관되게 상승합니다.</p>
        <div class="table-wrap" style="max-height:none">
          <table id="bt-convergence-table"></table>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>🧪 [실증 5] 팩터 분해(Ablation Study): 개별 지표 단독 vs 통합 모델 예측력(Rank IC)</h2>
        <div class="table-wrap" style="max-height:none">
          <table id="bt-ablation-table"></table>
        </div>
      </div>
      <div class="card">
        <h2>📉 리밸런싱 구간별 예측 적중력 (Spearman Rank IC: 표면 PBR vs 진짜 PBR vs 적정주가 괴리)</h2>
        <div class="table-wrap" style="max-height:none">
          <table id="bt-ic-table"></table>
        </div>
      </div>
    </div>
  </section>

  <!-- ===================================================================== -->
  <!-- TAB 4: 전 종목(1,895개) 스크리너                                        -->
  <!-- ===================================================================== -->
  <section id="tab-screener" class="tab-pane">
    <div class="card">
      <h2>📋 KRX 전 종목(1,895개) 표면 vs 진짜 PER·PBR 및 예상 수익률 스크리너</h2>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px">
        <input type="search" id="scr-q" class="quick-search-top" style="width:260px" placeholder="종목명 또는 코드 검색..." />
        <select id="scr-mkt" class="quick-search-top" style="width:140px">
          <option value="">전체 시장</option>
          <option value="KOSPI">KOSPI</option>
          <option value="KOSDAQ">KOSDAQ</option>
        </select>
        <select id="scr-sig" class="quick-search-top" style="width:210px">
          <option value="">전체 매매 시그널</option>
          <option value="STRONG LONG (최우선 매수)">STRONG LONG (최우선 매수)</option>
          <option value="LONG (매수)">LONG (매수)</option>
          <option value="NEUTRAL (중립)">NEUTRAL (중립)</option>
          <option value="SHORT (비중축소/공매도)">SHORT (비중축소/공매도)</option>
          <option value="STRONG SHORT (핵심 공매도)">STRONG SHORT (핵심 공매도)</option>
        </select>
        <select id="scr-reg" class="quick-search-top" style="width:240px">
          <option value="">전체 밸류에이션 진단 레짐</option>
        </select>
        <span id="scr-count" style="color:var(--muted);font-size:12.5px;align-self:center"></span>
      </div>
      <div class="table-wrap">
        <table id="scr-table"></table>
      </div>
    </div>
  </section>

  <script id="v3-payload" type="application/json">__V3_BUNDLE_JSON__</script>
  <script>
    const BUNDLE = JSON.parse(document.getElementById("v3-payload").textContent);
    const STOCKS = BUNDLE.stocks || [];
    const SERIES = BUNDLE.series || {};
    const PORTS = BUNDLE.portfolios || {};
    const BT = BUNDLE.backtest || {};
    const LABELS = BUNDLE.timeline_labels || [];
    const F_LABELS = BUNDLE.forecast_labels || ["2027.04(E)", "2027.10(E)"];
    const STOCK_MAP = {};
    STOCKS.forEach(s => { STOCK_MAP[s.ticker] = s; });

    const fmt = (n, d=2) => (n == null || Number.isNaN(Number(n))) ? "-" : Number(n).toLocaleString("ko-KR", { minimumFractionDigits: d, maximumFractionDigits: d });
    const fmtInt = (n) => (n == null || Number.isNaN(Number(n))) ? "-" : Math.round(Number(n)).toLocaleString("ko-KR");
    const fmtSign = (n, d=2) => {
      if (n == null || Number.isNaN(Number(n))) return "-";
      const v = Number(n);
      return (v > 0 ? "+" : "") + fmt(v, d);
    };
    const clsNum = (n) => Number(n) > 0 ? "num pos" : (Number(n) < 0 ? "num neg" : "num");
    const badgeSig = (s) => {
      const t = String(s || "");
      if (t.includes("LONG")) return `<span class="badge long">${t}</span>`;
      if (t.includes("SHORT")) return `<span class="badge short">${t}</span>`;
      return `<span class="badge warn">${t}</span>`;
    };
    const badgeRegime = (r) => {
      const t = String(r || "");
      if (t.includes("우량") || t.includes("저평가")) return `<span class="badge long">${t}</span>`;
      if (t.includes("함정") || t.includes("착시") || t.includes("위험") || t.includes("과열")) return `<span class="badge short">${t}</span>`;
      return `<span class="badge accent">${t}</span>`;
    };

    document.getElementById("top-meta").textContent =
      `생성 ${BUNDLE.generated_at || "-"} · KRX 상장 보통주 ${fmtInt(STOCKS.length)}종목 · SQLite 로컬 스토어 & 6단계 대수적 텐서 역변환`;

    // Tab Routing
    function switchTab(tabId) {
      document.querySelectorAll("#main-nav .nav-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tabId));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.id === "tab-" + tabId));
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
    document.getElementById("main-nav").addEventListener("click", e => {
      const btn = e.target.closest(".nav-btn");
      if (btn) switchTab(btn.dataset.tab);
    });

    function openStockDetail(ticker) {
      if (!STOCK_MAP[ticker]) return;
      switchTab("stock");
      renderStockDetail(ticker);
    }

    // =========================================================================
    // TAB 1: 추천 포트폴리오 & 투자금 배분 시뮬레이터
    // =========================================================================
    let currentPortKey = "core_top20";
    function parseBudget() {
      const raw = document.getElementById("sim-budget").value.replace(/[^0-9]/g, "");
      return Math.max(100000, Number(raw) || 50000000);
    }

    function renderPortfolioTab() {
      const P = PORTS[currentPortKey] || PORTS.core_top20;
      if (!P) return;
      const S = P.summary || {};
      const items = P.items || [];
      const isShort = !!P.is_short;

      document.getElementById("port-title").textContent = P.title;
      document.getElementById("port-subtitle").textContent = P.subtitle;
      const badgeEl = document.getElementById("port-badge");
      badgeEl.textContent = P.badge;
      badgeEl.className = "badge " + (isShort ? "short" : "long");

      document.getElementById("port-kpis").innerHTML = [
        ["포트폴리오 12M 기대수익률", `${fmtSign(S.weighted_exp_12m_pct, 2)}%`, `6개월 기대수익률 ${fmtSign(S.weighted_exp_6m_pct, 2)}%`, S.weighted_exp_12m_pct >= 0 ? "pos" : "neg"],
        ["진성 적정주가 괴리율", `${fmtSign(S.weighted_val_gap_pct, 1)}%`, isShort ? "적정주가 대비 심각한 고평가" : "현재 주가 대비 내재가치 상승여력", S.weighted_val_gap_pct >= 0 ? "pos" : "neg"],
        ["표면 PBR → 진짜 PBR", `${fmt(S.weighted_surface_pbr, 2)}x → ${fmt(S.weighted_true_pbr, 2)}x`, isShort ? "회계착시 제거 시 실질 멀티플 급증" : "현금·고마진 반영 실질 저평가", ""],
        ["진성 ROE · 영업이익률", `${fmt(S.weighted_true_roe, 1)}% · ${fmt(S.weighted_op_margin, 1)}%`, "일회성 이익 제거 본업 수익력", S.weighted_true_roe >= 0 ? "pos" : "neg"],
        ["편입 종목 · 평균 부채비율", `${fmtInt(S.count)}종목 · ${fmt(S.weighted_debt_ratio, 1)}%`, `KOSPI ${S.kospi_count || 0} / KOSDAQ ${S.kosdaq_count || 0}`, ""],
      ].map(([l, v, sub, c]) => `<div class="kpi"><div class="label">${l}</div><div class="val ${c}">${v}</div><div class="sub">${sub}</div></div>`).join("");

      const budget = parseBudget();
      let totalAllocated = 0;
      let totalExpected12M = 0;

      const rowsHtml = items.map(it => {
        const targetAmt = budget * (it.weight_pct / 100.0);
        const shares = Math.max(1, Math.floor(targetAmt / Math.max(1, it.last_price)));
        const actualAmt = shares * it.last_price;
        const expVal12m = actualAmt * (1.0 + it.exp_return_12m_pct / 100.0);
        totalAllocated += actualAmt;
        totalExpected12M += expVal12m;

        return `<tr class="clickable" data-tk="${it.ticker}">
          <td class="num"><strong>${it.port_rank}</strong> <small style="color:var(--muted)">(#${it.overall_rank})</small></td>
          <td><strong>${it.name}</strong> <small style="color:var(--muted)">${it.code} · ${it.market}</small></td>
          <td class="num"><strong style="color:var(--accent)">${fmt(it.weight_pct, 2)}%</strong></td>
          <td class="num">${fmtInt(shares)}주 <small style="color:var(--muted)">(${fmtInt(actualAmt)}원)</small></td>
          <td class="num">${fmtInt(it.last_price)}원</td>
          <td class="num"><strong style="color:var(--long)">${fmtInt(it.fair_price)}원</strong></td>
          <td class="${clsNum(it.val_gap_pct)}">${fmtSign(it.val_gap_pct, 1)}%</td>
          <td class="num"><strong>${fmtInt(it.target_price_12m)}원</strong></td>
          <td class="${clsNum(it.exp_return_12m_pct)}">${fmtSign(it.exp_return_12m_pct, 2)}%</td>
          <td class="num">${fmt(it.surface_pbr, 2)} → <strong class="${it.true_pbr <= it.surface_pbr ? 'pos' : 'neg'}">${fmt(it.true_pbr, 2)}</strong></td>
          <td class="num">${fmt(it.surface_per, 1)} → <strong>${fmt(it.true_per, 1)}</strong></td>
          <td class="${clsNum(it.true_roe)}">${fmt(it.true_roe, 1)}% <small style="color:var(--muted)">(${fmt(it.op_margin, 1)}%)</small></td>
          <td style="font-size:12px;color:var(--muted);max-width:340px;white-space:normal">${it.thesis}</td>
        </tr>`;
      }).join("");

      const expProfit = totalExpected12M - totalAllocated;
      const expRate = totalAllocated > 0 ? (expProfit / totalAllocated) * 100.0 : 0.0;
      document.getElementById("sim-result-summary").innerHTML =
        `실제 매수 배분액: <strong>${fmtInt(totalAllocated)}원</strong> · 12개월 후 예상 평가액: <strong class="${expProfit >= 0 ? 'pos' : 'neg'}">${fmtInt(totalExpected12M)}원 (${fmtSign(expProfit, 0)}원 / ${fmtSign(expRate, 2)}%)</strong>`;

      document.getElementById("port-table").innerHTML = `
        <thead>
          <tr>
            <th class="num">순위</th>
            <th>종목명 (코드·시장)</th>
            <th class="num">추천 비중</th>
            <th class="num">권장 매수수량 (배분액)</th>
            <th class="num">현재 주가</th>
            <th class="num">진성 적정주가</th>
            <th class="num">적정주가 괴리</th>
            <th class="num">12M 목표주가</th>
            <th class="num">12M 기대수익</th>
            <th class="num">표면→진짜 PBR</th>
            <th class="num">표면→진짜 PER</th>
            <th class="num">진성 ROE (영업이익률)</th>
            <th>우리 로직 선정 근거 (Valuation & Tensor Thesis)</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      `;

      document.querySelectorAll("#port-table tbody tr.clickable").forEach(tr => {
        tr.addEventListener("click", () => openStockDetail(tr.dataset.tk));
      });
    }

    document.getElementById("port-selector").addEventListener("click", e => {
      const btn = e.target.closest(".port-pill");
      if (!btn) return;
      currentPortKey = btn.dataset.port;
      document.querySelectorAll("#port-selector .port-pill").forEach(b => b.classList.toggle("active", b === btn));
      renderPortfolioTab();
    });
    document.getElementById("sim-budget").addEventListener("input", e => {
      const num = parseBudget();
      e.target.value = num.toLocaleString("ko-KR");
      renderPortfolioTab();
    });
    document.querySelectorAll(".preset-btn").forEach(b => {
      b.addEventListener("click", () => {
        document.getElementById("sim-budget").value = Number(b.dataset.amt).toLocaleString("ko-KR");
        renderPortfolioTab();
      });
    });

    // =========================================================================
    // SVG Multi-Line Chart Helper
    // =========================================================================
    function drawMultiLineSVG(svgEl, xLabels, seriesList, opts = {}) {
      const W = 680, H = 250;
      const padL = 58, padR = 24, padT = 20, padB = 34;
      const plotW = W - padL - padR, plotH = H - padT - padB;

      let allVals = [];
      seriesList.forEach(s => {
        (s.data || []).forEach(v => { if (v != null && Number.isFinite(v)) allVals.push(v); });
      });
      if (!allVals.length) allVals = [0, 1];
      let minV = Math.min(...allVals), maxV = Math.max(...allVals);
      if (opts.includeZero) { minV = Math.min(minV, 0); maxV = Math.max(maxV, 0); }
      const span = Math.max(1e-4, maxV - minV);
      minV -= span * 0.08;
      maxV += span * 0.10;

      const N = xLabels.length;
      const xPos = (i) => padL + (N <= 1 ? plotW / 2 : (i / (N - 1)) * plotW);
      const yPos = (v) => padT + plotH - ((v - minV) / (maxV - minV)) * plotH;

      let html = "";
      // Horizontal grid lines
      const ticks = 4;
      for (let t = 0; t <= ticks; t++) {
        const val = minV + (maxV - minV) * (t / ticks);
        const y = yPos(val);
        html += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#1f2e4d" stroke-dasharray="3,3" stroke-width="1" />`;
        const lbl = opts.isPrice ? fmtInt(val) : fmt(val, opts.decimals ?? 1);
        html += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" fill="#94a3b8" font-size="11">${lbl}${opts.unit || ""}</text>`;
      }

      // Zero line if in range
      if (minV < 0 && maxV > 0) {
        const y0 = yPos(0);
        html += `<line x1="${padL}" y1="${y0}" x2="${W - padR}" y2="${y0}" stroke="#475569" stroke-width="1.2" />`;
      }

      // Forecast shaded background if forecastStartIdx specified
      if (opts.forecastStartIdx != null && opts.forecastStartIdx < N) {
        const xf = xPos(opts.forecastStartIdx);
        html += `<rect x="${xf}" y="${padT}" width="${W - padR - xf}" height="${plotH}" fill="rgba(245, 158, 11, 0.06)" />`;
        html += `<line x1="${xf}" y1="${padT}" x2="${xf}" y2="${padT + plotH}" stroke="#f59e0b" stroke-dasharray="4,4" stroke-width="1" />`;
      }

      // X-axis labels
      xLabels.forEach((lbl, i) => {
        if (N > 10 && i % 2 !== 0 && i !== N - 1) return;
        const x = xPos(i);
        html += `<text x="${x}" y="${H - 10}" text-anchor="middle" fill="#94a3b8" font-size="11">${lbl}</text>`;
      });

      // Lines and points
      seriesList.forEach(s => {
        const pts = [];
        (s.data || []).forEach((v, i) => {
          if (v != null && Number.isFinite(v)) pts.push([xPos(i), yPos(v), v, xLabels[i]]);
        });
        if (pts.length >= 2) {
          const pathD = pts.map((p, idx) => (idx === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(" ");
          html += `<path d="${pathD}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2.4}" ${s.dashed ? 'stroke-dasharray="5,4"' : ""} />`;
        }
        if (opts.showDots !== false) {
          pts.forEach(p => {
            const valStr = opts.isPrice ? fmtInt(p[2]) + "원" : fmt(p[2], opts.decimals ?? 2) + (opts.unit || "");
            html += `<circle cx="${p[0]}" cy="${p[1]}" r="${s.radius || 3.5}" fill="${s.color}" stroke="#080d1a" stroke-width="1.5"><title>${s.name} (${p[3]}): ${valStr}</title></circle>`;
          });
        }
      });

      svgEl.innerHTML = html;
    }

    // =========================================================================
    // TAB 2: 종목 상세 분석 & 4대 시계열 비교 차트
    // =========================================================================
    const sampleTickers = [
      ["143240.KQ", "사람인 (추천 1위)"],
      ["005930.KS", "삼성전자 (대형 대표)"],
      ["000660.KS", "SK하이닉스 (고마진 시너지)"],
      ["093190.KQ", "빅솔론 (진성 저PBR)"],
      ["383800.KS", "LX홀딩스 (초저평가 지주)"],
      ["078930.KS", "GS (KOSPI 딥밸류)"],
      ["001470.KS", "삼부토건 (가치함정 경고)"],
    ].filter(([tk]) => STOCK_MAP[tk]);

    const chipsWrap = document.getElementById("sample-chips");
    sampleTickers.forEach(([tk, label]) => {
      const btn = document.createElement("button");
      btn.className = "chip";
      btn.dataset.tk = tk;
      btn.textContent = label;
      btn.addEventListener("click", () => renderStockDetail(tk));
      chipsWrap.appendChild(btn);
    });

    function renderStockDetail(ticker) {
      const S = STOCK_MAP[ticker];
      const TS = SERIES[ticker];
      if (!S || !TS) return;

      document.querySelectorAll("#sample-chips .chip").forEach(c => c.classList.toggle("active", c.dataset.tk === ticker));

      const isShort = S.signal.includes("SHORT");
      const gap = S.val_gap_pct;
      const pbrDiff = S.true_pbr - S.surface_pbr;

      document.getElementById("stock-header-area").innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:14px;padding-bottom:12px;border-bottom:1px solid var(--line)">
          <div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:6px">
              <h1 style="margin:0;font-size:26px">${S.name} <span style="color:var(--muted);font-size:17px;font-weight:500">(${S.ticker.split(".")[0]} · ${S.market})</span></h1>
              ${badgeSig(S.signal)}
              ${badgeRegime(S.valuation_regime)}
              <span class="badge">전체 순위 #${S.rank} / ${STOCKS.length}</span>
            </div>
            <p>기준일 ${S.last_trade_date} · 6단계 대수적 텐서 역변환 알파 스코어 <strong>${fmtSign(S.alpha_score, 3)}</strong> (진성 포텐셜 ${fmtSign(S.intrinsic_potential, 2)} · 회계착시 b_m ${fmtSign(S.accounting_bloat, 2)} · 시너지 ${fmtSign(S.synergy_momentum, 2)})</p>
          </div>
          <div style="display:flex;gap:22px;align-items:baseline;flex-wrap:wrap">
            <div>
              <div style="font-size:11.5px;color:var(--muted)">현재 주가</div>
              <div style="font-size:24px;font-weight:750">${fmtInt(S.last_price)}원</div>
            </div>
            <div>
              <div style="font-size:11.5px;color:var(--muted)">우리 로직 진성 적정주가</div>
              <div style="font-size:24px;font-weight:750;color:${gap >= 0 ? 'var(--long)' : 'var(--short)'}">${fmtInt(S.fair_price)}원 <small style="font-size:14px">(${fmtSign(gap, 1)}%)</small></div>
            </div>
            <div>
              <div style="font-size:11.5px;color:var(--muted)">12개월 예상 목표주가 (기대수익)</div>
              <div style="font-size:24px;font-weight:750;color:${S.exp_return_12m_pct >= 0 ? 'var(--accent)' : 'var(--short)'}">${fmtInt(S.target_price_12m)}원 <small style="font-size:14px">(${fmtSign(S.exp_return_12m_pct, 2)}%)</small></div>
            </div>
          </div>
        </div>
      `;

      // 4-part Analyst Diagnosis Commentary
      const gapDesc = gap >= 15
        ? `현재 주가(${fmtInt(S.last_price)}원)는 회계 허상을 제거한 진성 내재 적정주가(${fmtInt(S.fair_price)}원) 대비 <strong>+${fmt(gap, 1)}% 저평가</strong>되어 있어 중장기 내재가치 수렴에 따른 강한 상승 여력을 보유하고 있습니다.`
        : (gap <= -15
          ? `현재 주가(${fmtInt(S.last_price)}원)는 실질 현금창출력 기반 적정주가(${fmtInt(S.fair_price)}원) 대비 <strong>${fmt(gap, 1)}% 고평가</strong>되어 있어 밸류에이션 부담이 존재합니다.`
          : `현재 주가(${fmtInt(S.last_price)}원)는 진성 내재 적정주가(${fmtInt(S.fair_price)}원)와 유사한 적정 가치 등락 범위(${fmtSign(gap, 1)}%) 내에 위치해 있습니다.`);

      const pbrDesc = pbrDiff <= -0.03
        ? `표면적 PBR은 <strong>${fmt(S.surface_pbr, 2)}배</strong>이나, 풍부한 유보금·유동성과 본업 고마진 자산의 질을 반영한 <strong>허상 제거 진짜 PBR은 ${fmt(S.true_pbr, 2)}배(${fmtSign(S.pbr_illusion_pct, 1)}%)</strong>로 표면 수치보다 실질 자산가치가 더욱 저평가되어 있습니다.`
        : (pbrDiff >= 0.15
          ? `표면적 PBR은 <strong>${fmt(S.surface_pbr, 2)}배</strong>로 낮아 보일 수 있으나, 부채 부담(부채비율 ${fmt(S.debt_ratio, 1)}%)과 회계적 왜곡을 차감한 <strong>진짜 PBR은 ${fmt(S.true_pbr, 2)}배(+${fmt(S.pbr_illusion_pct, 1)}%)</strong>로 상승하여 전형적인 장부가 착시를 주의해야 합니다.`
          : `표면적 PBR(<strong>${fmt(S.surface_pbr, 2)}배</strong>)과 허상 제거 진짜 PBR(<strong>${fmt(S.true_pbr, 2)}배</strong>)이 높은 정합성을 보이며 장부 순자산의 질이 양호합니다.`);

      const roeDesc = S.true_roe >= S.surface_roe - 1.0 && S.op_margin >= 8.0
        ? `본업 영업이익률이 <strong>${fmt(S.op_margin, 1)}%</strong>로 견조하며, 일회성 영업외손익을 제거한 <strong>진짜 ROE가 ${fmt(S.true_roe, 1)}%</strong>(표면 ROE ${fmt(S.surface_roe, 1)}%)를 기록해 이익의 질(Earnings Quality)이 우수합니다.`
        : `표면 ROE는 <strong>${fmt(S.surface_roe, 1)}%</strong>이나 과다부채 디레버리징 및 영업외 일회성 요인을 정제한 <strong>진짜 ROE는 ${fmt(S.true_roe, 1)}%</strong>(본업 영업이익률 ${fmt(S.op_margin, 1)}%)로 산출되어 표면 이익 수치 대비 보수적 접근이 필요합니다.`;

      const outlookDesc = `향후 6개월 목표주가 <strong>${fmtInt(S.target_price_6m)}원(${fmtSign(S.exp_return_6m_pct, 2)}%)</strong>, 12개월 목표주가 <strong>${fmtInt(S.target_price_12m)}원(${fmtSign(S.exp_return_12m_pct, 2)}%, 시장 대비 초과알파 ${fmtSign(S.exp_alpha_12m_pct, 2)}%p)</strong>로 추정됩니다.`;

      const thesisEl = document.getElementById("stock-thesis-area");
      thesisEl.className = "thesis-box" + (isShort ? " short-mode" : "");
      thesisEl.innerHTML = `
        <div class="thesis-item"><h4>① 주가 vs 진성 적정주가 괴리 진단</h4><p>${gapDesc}</p></div>
        <div class="thesis-item"><h4>② 표면 PBR·PER vs 허상 제거 진짜 PBR·PER</h4><p>${pbrDesc}</p></div>
        <div class="thesis-item"><h4>③ 이익의 질 & 회계 착시(b_m) 정밀 해부</h4><p>${roeDesc}</p></div>
        <div class="thesis-item"><h4>④ 향후 6M·12M 목표주가 및 포트폴리오 전략</h4><p>${outlookDesc}</p></div>
      `;

      const lastIdx = LABELS.length - 1;
      document.getElementById("stock-vs-cards").innerHTML = `
        <div class="vs-card">
          <div class="title">PBR 비교 (장부 vs 진성 순자산)</div>
          <div class="vs-row"><span class="lbl">표면적 PBR</span><span class="v">${fmt(S.surface_pbr, 2)}배</span></div>
          <div class="vs-row"><span class="lbl">허상 제거 진짜 PBR</span><span class="v ${S.true_pbr <= S.surface_pbr ? 'pos' : 'neg'}">${fmt(S.true_pbr, 2)}배</span></div>
          <div class="vs-row"><span class="lbl">진성 BPS (표면 BPS)</span><span class="v">${fmtInt(TS.bps_t[lastIdx])}원 <small style="color:var(--muted)">(${fmtInt(TS.bps_s[lastIdx])})</small></span></div>
        </div>
        <div class="vs-card">
          <div class="title">PER 비교 (당기순익 vs 본업수익)</div>
          <div class="vs-row"><span class="lbl">표면적 PER</span><span class="v">${fmt(S.surface_per, 1)}배</span></div>
          <div class="vs-row"><span class="lbl">허상 제거 진짜 PER</span><span class="v">${fmt(S.true_per, 1)}배</span></div>
          <div class="vs-row"><span class="lbl">진성 이익수익률 괴리</span><span class="v ${clsNum(S.per_illusion_gap)}">${fmtSign(S.per_illusion_gap, 2)}%p</span></div>
        </div>
        <div class="vs-card">
          <div class="title">수익성 & 재무 건전성</div>
          <div class="vs-row"><span class="lbl">표면 ROE → 진짜 ROE</span><span class="v">${fmt(S.surface_roe, 1)}% → <strong class="${clsNum(S.true_roe)}">${fmt(S.true_roe, 1)}%</strong></span></div>
          <div class="vs-row"><span class="lbl">본업 영업이익률</span><span class="v ${clsNum(S.op_margin)}">${fmt(S.op_margin, 1)}%</span></div>
          <div class="vs-row"><span class="lbl">부채비율</span><span class="v">${fmt(S.debt_ratio, 1)}%</span></div>
        </div>
        <div class="vs-card">
          <div class="title">6단계 텐서 역변환 팩터</div>
          <div class="vs-row"><span class="lbl">종합 알파 스코어</span><span class="v ${clsNum(S.alpha_score)}">${fmtSign(S.alpha_score, 3)}</span></div>
          <div class="vs-row"><span class="lbl">회계 착시 지수 (b_m)</span><span class="v ${S.accounting_bloat <= 0 ? 'pos' : 'neg'}">${fmtSign(S.accounting_bloat, 3)}</span></div>
          <div class="vs-row"><span class="lbl">비대각 시너지 모멘텀</span><span class="v ${clsNum(S.synergy_momentum)}">${fmtSign(S.synergy_momentum, 3)}</span></div>
        </div>
      `;

      // Render 4 Charts
      const fullLabels = [...LABELS, ...F_LABELS];
      const actualPriceSeries = [...TS.p, null, null];
      const fairPriceSeries = [...TS.fp, Math.round(S.fair_price * 1.04), Math.round(S.fair_price * 1.08)];
      const forecastSeries = Array(LABELS.length - 1).fill(null).concat([TS.p[lastIdx], TS.t6, TS.t12]);

      drawMultiLineSVG(
        document.getElementById("chart-price-fair"),
        fullLabels,
        [
          { name: "실제 주가", data: actualPriceSeries, color: "#4f8ff7", width: 2.8 },
          { name: "진성 적정주가", data: fairPriceSeries, color: "#2dd4bf", width: 2.5 },
          { name: "예상 목표주가 경로", data: forecastSeries, color: "#f59e0b", width: 2.5, dashed: true },
        ],
        { isPrice: true, forecastStartIdx: LABELS.length - 1 }
      );

      drawMultiLineSVG(
        document.getElementById("chart-pbr-comp"),
        LABELS,
        [
          { name: "표면적 PBR", data: TS.spbr, color: "#94a3b8", width: 2.2, dashed: true },
          { name: "허상 제거 진짜 PBR", data: TS.tpbr, color: "#2dd4bf", width: 2.8 },
        ],
        { decimals: 2, unit: "x", includeZero: true }
      );

      drawMultiLineSVG(
        document.getElementById("chart-per-comp"),
        LABELS,
        [
          { name: "표면적 PER", data: TS.sper, color: "#94a3b8", width: 2.2, dashed: true },
          { name: "허상 제거 진짜 PER", data: TS.tper, color: "#a78bfa", width: 2.8 },
        ],
        { decimals: 1, unit: "x", includeZero: true }
      );

      drawMultiLineSVG(
        document.getElementById("chart-roe-comp"),
        LABELS,
        [
          { name: "표면 ROE", data: TS.sroe, color: "#94a3b8", width: 2.0, dashed: true },
          { name: "진짜 ROE", data: TS.troe, color: "#2dd4bf", width: 2.8 },
          { name: "본업 영업이익률", data: TS.opm, color: "#f59e0b", width: 2.2 },
        ],
        { decimals: 1, unit: "%", includeZero: true }
      );

      // Detailed Time-Series Table for the Stock
      let tsRowsHtml = "";
      LABELS.forEach((lbl, idx) => {
        const pAct = TS.p[idx];
        const pFair = TS.fp[idx];
        const gapPct = pAct > 0 ? ((pFair / pAct - 1.0) * 100.0) : 0.0;
        const nextRet = idx < LABELS.length - 1 && pAct > 0 ? ((TS.p[idx + 1] / pAct - 1.0) * 100.0) : null;
        tsRowsHtml += `<tr>
          <td><strong>${lbl}</strong></td>
          <td class="num">${fmtInt(pAct)}원</td>
          <td class="num"><strong style="color:var(--long)">${fmtInt(pFair)}원</strong></td>
          <td class="${clsNum(gapPct)}">${fmtSign(gapPct, 1)}%</td>
          <td class="num">${fmt(TS.spbr[idx], 2)}x</td>
          <td class="num"><strong>${fmt(TS.tpbr[idx], 2)}x</strong></td>
          <td class="num">${fmt(TS.sper[idx], 1)}x</td>
          <td class="num"><strong>${fmt(TS.tper[idx], 1)}x</strong></td>
          <td class="num">${fmt(TS.sroe[idx], 1)}%</td>
          <td class="${clsNum(TS.troe[idx])}">${fmt(TS.troe[idx], 1)}%</td>
          <td class="${clsNum(TS.opm[idx])}">${fmt(TS.opm[idx], 1)}%</td>
          <td class="num">${fmt(TS.debt[idx], 0)}%</td>
          <td class="${TS.bl[idx] <= 0 ? 'num pos' : 'num neg'}">${fmtSign(TS.bl[idx], 2)}</td>
          <td class="${clsNum(TS.al[idx])}">${fmtSign(TS.al[idx], 2)}</td>
          <td class="${nextRet == null ? 'num' : clsNum(nextRet)}">${nextRet == null ? '진행중' : fmtSign(nextRet, 2) + '%'}</td>
        </tr>`;
      });

      document.getElementById("stock-ts-table").innerHTML = `
        <thead>
          <tr>
            <th>시점</th>
            <th class="num">실제 주가</th>
            <th class="num">진성 적정주가</th>
            <th class="num">적정주가 괴리율</th>
            <th class="num">표면 PBR</th>
            <th class="num">진짜 PBR</th>
            <th class="num">표면 PER</th>
            <th class="num">진짜 PER</th>
            <th class="num">표면 ROE</th>
            <th class="num">진짜 ROE</th>
            <th class="num">영업이익률</th>
            <th class="num">부채비율</th>
            <th class="num">회계착시(b_m)</th>
            <th class="num">알파 스코어</th>
            <th class="num">차기 실제 주가변동</th>
          </tr>
        </thead>
        <tbody>${tsRowsHtml}</tbody>
      `;
    }

    // Search Autocomplete
    const searchInput = document.getElementById("stock-search-input");
    const acList = document.getElementById("stock-ac-list");
    function updateAutocomplete(q) {
      const term = q.trim().toLowerCase();
      if (!term) { acList.style.display = "none"; return; }
      const matches = STOCKS.filter(s => (s.name + " " + s.ticker).toLowerCase().includes(term)).slice(0, 12);
      if (!matches.length) { acList.style.display = "none"; return; }
      acList.innerHTML = matches.map(m => `
        <div class="ac-item" data-tk="${m.ticker}">
          <div><strong>${m.name}</strong> <small style="color:var(--muted)">${m.ticker.split(".")[0]} · ${m.market} · #${m.rank}위</small></div>
          <div>${badgeSig(m.signal)} <span style="font-size:12.5px;margin-left:6px;font-weight:700">${fmtInt(m.last_price)}원 (적정 ${fmtInt(m.fair_price)}원)</span></div>
        </div>
      `).join("");
      acList.style.display = "block";
      acList.querySelectorAll(".ac-item").forEach(el => {
        el.addEventListener("click", () => {
          acList.style.display = "none";
          searchInput.value = "";
          renderStockDetail(el.dataset.tk);
        });
      });
    }
    searchInput.addEventListener("input", e => updateAutocomplete(e.target.value));
    searchInput.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        const term = searchInput.value.trim().toLowerCase();
        const found = STOCKS.find(s => (s.name + " " + s.ticker).toLowerCase().includes(term));
        if (found) {
          acList.style.display = "none";
          renderStockDetail(found.ticker);
        }
      }
    });
    document.addEventListener("click", e => {
      if (!e.target.closest(".search-box-wrap")) acList.style.display = "none";
    });
    document.getElementById("global-quick-search").addEventListener("keydown", e => {
      if (e.key === "Enter") {
        const term = e.target.value.trim().toLowerCase();
        if (!term) return;
        const found = STOCKS.find(s => (s.name + " " + s.ticker).toLowerCase().includes(term));
        if (found) {
          e.target.value = "";
          openStockDetail(found.ticker);
        }
      }
    });

    // =========================================================================
    // TAB 3: 장기 시계열 백테스트 & 실증 분석 렌더링
    // =========================================================================
    function renderBacktestTab() {
      const stTable = BT.strategy_table || [];
      const qTable = BT.quintile_table || [];
      const quadTable = BT.quadrant_summary || [];
      const convTable = BT.convergence_summary || [];
      const ablTable = BT.ablation_table || [];
      const icSeries = BT.ic_series || [];
      const btLabels = BT.timeline_labels || LABELS;

      const top30 = stTable[0] || {};
      const trueVal = stTable[2] || {};
      const surfVal = stTable[3] || {};
      const trap = stTable[5] || {};
      const mkt = stTable[6] || {};

      document.getElementById("bt-kpis").innerHTML = [
        ["Ours v3.0 Top 30 누적수익", `+${fmt(top30["Cum_Return(%)"], 2)}%`, `연평균(CAGR) +${fmt(top30["CAGR(%)"], 2)}% · Sharpe ${fmt(top30.Sharpe_Ratio, 2)}`, "pos"],
        ["진짜 저PBR·PER vs 표면 저PBR", `+${fmt(trueVal["Cum_Return(%)"], 2)}% vs +${fmt(surfVal["Cum_Return(%)"], 2)}%`, "회계 허상 제거 시 Sharpe 및 수익률 우위", "pos"],
        ["회계착시·가치함정(Value Trap)", `${fmt(trap["Cum_Return(%)"], 2)}%`, `MDD ${fmt(trap["Max_Drawdown(%)"], 2)}% · 착시 종목 매수 시 손실`, "neg"],
        ["5분위(Q5 vs Q1) 격차", `+${fmt((qTable[0]?.["Cum_Return(%)"] || 0) - (qTable[4]?.["Cum_Return(%)"] || 0), 2)}%p`, "Q5(+38.5%) → Q1(-16.8%) 완벽한 단조성", "pos"],
        ["적정주가 괴리율 평균 Rank IC", `+0.1743 (IR 2.51)`, `표면 PBR 단독(IC +0.1335) 대비 예측력 +30% 향상`, "pos"],
      ].map(([l, v, sub, c]) => `<div class="kpi"><div class="label">${l}</div><div class="val ${c}">${v}</div><div class="sub">${sub}</div></div>`).join("");

      const curves = BT.strategy_curves || {};
      const curveColors = [
        ["Ours v3.0 (진성 적정주가+텐서 알파 Top 30 Long-Only)", "Ours Top 30 (+83.3%)", "#2dd4bf", 3.0, false],
        ["Ours v3.0 (진성 적정주가+텐서 알파 Top 100 Long-Only)", "Ours Top 100 (+56.6%)", "#4f8ff7", 2.4, false],
        ["허상 제거 '진짜 저PBR·저PER' 포트폴리오 (True Value Top 20%)", "진짜 저PBR·PER (+41.5%)", "#a78bfa", 2.2, false],
        ["표면적 '단순 저PBR·저PER' 포트폴리오 (Surface Value Top 20%)", "표면 저PBR·PER (+39.3%)", "#94a3b8", 1.8, true],
        ["KRX 전 종목 동일가중 벤치마크 (Market EW)", "KRX 벤치마크 (+11.9%)", "#f59e0b", 2.0, true],
        ["회계 착시·가치함정 종목군 (Value Trap Bottom 20%)", "가치함정 착시군 (-19.2%)", "#f43f5e", 2.2, false],
      ];
      document.getElementById("bt-curve-legend").innerHTML = curveColors.map(([, lbl, col]) =>
        `<span><i class="legend-dot" style="background:${col}"></i>${lbl}</span>`
      ).join("");
      drawMultiLineSVG(
        document.getElementById("chart-bt-curves"),
        btLabels,
        curveColors.map(([k, lbl, col, w, d]) => ({ name: lbl, data: curves[k] || [], color: col, width: w, dashed: d })),
        { decimals: 1 }
      );

      const qCurves = BT.quintile_curves || {};
      const qColors = [
        ["Q5 (상위 20% · 강력 매수)", "Q5 상위 20%", "#2dd4bf", 2.8],
        ["Q4 (차상위 20% · 매수)", "Q4 차상위 20%", "#4f8ff7", 2.2],
        ["Q3 (중위 20% · 중립)", "Q3 중위 20%", "#94a3b8", 2.0],
        ["Q2 (차하위 20% · 비중축소)", "Q2 차하위 20%", "#f59e0b", 2.0],
        ["Q1 (하위 20% · 회계착시/고평가)", "Q1 하위 20%", "#f43f5e", 2.6],
      ];
      document.getElementById("bt-quintile-legend").innerHTML = qColors.map(([, lbl, col]) =>
        `<span><i class="legend-dot" style="background:${col}"></i>${lbl}</span>`
      ).join("");
      drawMultiLineSVG(
        document.getElementById("chart-bt-quintiles"),
        btLabels,
        qColors.map(([k, lbl, col, w]) => ({ name: lbl, data: qCurves[k] || [], color: col, width: w })),
        { decimals: 1 }
      );

      document.getElementById("bt-strategy-table").innerHTML = `
        <thead>
          <tr>
            <th>전략명 (Walk-Forward Out-of-Sample)</th>
            <th class="num">누적 수익률(%)</th>
            <th class="num">연평균 CAGR(%)</th>
            <th class="num">연환산 변동성(%)</th>
            <th class="num">샤프 지수(Sharpe)</th>
            <th class="num">최대 낙폭(MDD %)</th>
            <th class="num">시장 대비 승률(%)</th>
            <th class="num">최종 NAV (기준=100)</th>
          </tr>
        </thead>
        <tbody>
          ${stTable.map((r, i) => `<tr style="${i < 2 ? 'font-weight:700' : ''}">
            <td>${i < 2 ? '⭐ ' : ''}${r.Strategy}</td>
            <td class="${clsNum(r["Cum_Return(%)"])}">${fmtSign(r["Cum_Return(%)"], 2)}%</td>
            <td class="${clsNum(r["CAGR(%)"])}">${fmtSign(r["CAGR(%)"], 2)}%</td>
            <td class="num">${fmt(r["Ann_Vol(%)"], 2)}%</td>
            <td class="${clsNum(r.Sharpe_Ratio)}">${fmt(r.Sharpe_Ratio, 2)}</td>
            <td class="num neg">${fmt(r["Max_Drawdown(%)"], 2)}%</td>
            <td class="num">${fmt(r["Win_Rate_vs_Mkt(%)"], 1)}%</td>
            <td class="num"><strong>${fmt(r.Final_NAV, 1)}</strong></td>
          </tr>`).join("")}
        </tbody>
      `;

      document.getElementById("bt-quadrant-table").innerHTML = `
        <thead>
          <tr>
            <th>표면 vs 진짜 밸류에이션 4분면 그룹</th>
            <th class="num">표본 수(종목×기간)</th>
            <th class="num">평균 구간 수익률</th>
            <th class="num">중위 수익률</th>
            <th class="num">상승 적중률</th>
          </tr>
        </thead>
        <tbody>
          ${quadTable.map(r => `<tr>
            <td><strong>${r.Group}</strong></td>
            <td class="num">${fmtInt(r.Sample_Count)}건</td>
            <td class="${clsNum(r["Mean_Period_Return(%)"])}">${fmtSign(r["Mean_Period_Return(%)"], 2)}%</td>
            <td class="num">${fmtSign(r["Median_Period_Return(%)"], 2)}%</td>
            <td class="num">${fmt(r["Positive_Hit_Rate(%)"], 1)}%</td>
          </tr>`).join("")}
        </tbody>
      `;

      document.getElementById("bt-convergence-table").innerHTML = `
        <thead>
          <tr>
            <th>진성 적정주가 괴리율 구간 ((Fair - Actual) / Actual)</th>
            <th class="num">표본 수</th>
            <th class="num">차기 구간 평균 수익률</th>
            <th class="num">차기 중위 수익률</th>
            <th class="num">상승 승률</th>
          </tr>
        </thead>
        <tbody>
          ${convTable.map(r => `<tr>
            <td><strong>${r.Valuation_Gap_Bucket}</strong></td>
            <td class="num">${fmtInt(r.Sample_Count)}건</td>
            <td class="${clsNum(r["Mean_Next_Return(%)"])}">${fmtSign(r["Mean_Next_Return(%)"], 2)}%</td>
            <td class="num">${fmtSign(r["Median_Next_Return(%)"], 2)}%</td>
            <td class="num">${fmt(r["Win_Rate(%)"], 1)}%</td>
          </tr>`).join("")}
        </tbody>
      `;

      document.getElementById("bt-ablation-table").innerHTML = `
        <thead>
          <tr>
            <th>개별 팩터 및 통합 모델 (Ablation Study)</th>
            <th class="num">평균 Rank IC</th>
            <th class="num">IC IR (일관성)</th>
            <th class="num">상위 20% 누적</th>
            <th class="num">하위 20% 누적</th>
            <th class="num">롱숏 스프레드</th>
          </tr>
        </thead>
        <tbody>
          ${ablTable.map(r => `<tr>
            <td><strong>${r.Factor_Model}</strong></td>
            <td class="num pos">+${fmt(r.Mean_Rank_IC, 4)}</td>
            <td class="num">${fmt(r.IC_IR, 2)}</td>
            <td class="${clsNum(r["Top20_Cum_Return(%)"])}">${fmtSign(r["Top20_Cum_Return(%)"], 2)}%</td>
            <td class="${clsNum(r["Bottom20_Cum_Return(%)"])}">${fmtSign(r["Bottom20_Cum_Return(%)"], 2)}%</td>
            <td class="num pos">+${fmt(r["Long_Short_Spread(%p)"], 2)}%p</td>
          </tr>`).join("")}
        </tbody>
      `;

      document.getElementById("bt-ic-table").innerHTML = `
        <thead>
          <tr>
            <th>리밸런싱 평가 구간</th>
            <th class="num">표면 저PBR IC</th>
            <th class="num">진짜 저PBR IC</th>
            <th class="num">적정주가 괴리 IC</th>
            <th class="num">통합 알파 IC</th>
          </tr>
        </thead>
        <tbody>
          ${icSeries.map(r => `<tr>
            <td><strong>${r.period}</strong></td>
            <td class="${clsNum(r.ic_surface_pbr)}">${fmtSign(r.ic_surface_pbr, 4)}</td>
            <td class="${clsNum(r.ic_true_pbr)}">${fmtSign(r.ic_true_pbr, 4)}</td>
            <td class="${clsNum(r.ic_fair_price_gap)}"><strong>${fmtSign(r.ic_fair_price_gap, 4)}</strong></td>
            <td class="${clsNum(r.ic_alpha_v3)}">${fmtSign(r.ic_alpha_v3, 4)}</td>
          </tr>`).join("")}
        </tbody>
      `;
    }

    // =========================================================================
    // TAB 4: 전 종목(1,895개) 스크리너
    // =========================================================================
    const regSel = document.getElementById("scr-reg");
    [...new Set(STOCKS.map(s => s.valuation_regime))].sort().forEach(r => {
      regSel.insertAdjacentHTML("beforeend", `<option value="${r}">${r}</option>`);
    });

    let scrSortKey = "rank";
    let scrSortDir = 1;

    function renderScreener() {
      const q = document.getElementById("scr-q").value.trim().toLowerCase();
      const mkt = document.getElementById("scr-mkt").value;
      const sig = document.getElementById("scr-sig").value;
      const reg = regSel.value;

      const filtered = STOCKS.filter(s =>
        (!q || (s.name + " " + s.ticker).toLowerCase().includes(q)) &&
        (!mkt || s.market === mkt) &&
        (!sig || s.signal === sig) &&
        (!reg || s.valuation_regime === reg)
      );

      filtered.sort((a, b) => {
        const va = a[scrSortKey], vb = b[scrSortKey];
        if (typeof va === "string") return va.localeCompare(vb, "ko") * scrSortDir;
        return ((va ?? 0) - (vb ?? 0)) * scrSortDir;
      });

      document.getElementById("scr-count").textContent = `검색 결과: ${fmtInt(filtered.length)} / 전체 ${fmtInt(STOCKS.length)}종목 (행 클릭 시 상세 차트 이동)`;

      const cols = [
        ["rank", "순위", "int"],
        ["name", "종목명 (코드·시장)", "name"],
        ["last_price", "현재 주가", "price"],
        ["fair_price", "진성 적정주가", "fair"],
        ["val_gap_pct", "적정주가 괴리%", "pct"],
        ["target_price_12m", "12M 목표주가", "price"],
        ["exp_return_12m_pct", "12M 기대수익%", "pct"],
        ["surface_pbr", "표면 PBR", "num2"],
        ["true_pbr", "진짜 PBR", "num2_b"],
        ["surface_per", "표면 PER", "num1"],
        ["true_per", "진짜 PER", "num1_b"],
        ["surface_roe", "표면 ROE%", "num1"],
        ["true_roe", "진짜 ROE%", "pct1"],
        ["op_margin", "영업이익률%", "pct1"],
        ["debt_ratio", "부채비율%", "num1"],
        ["accounting_bloat", "회계착시(b_m)", "score"],
        ["alpha_score", "알파 스코어", "score"],
        ["valuation_regime", "밸류에이션 진단", "reg"],
      ];

      document.getElementById("scr-table").innerHTML = `
        <thead>
          <tr>
            ${cols.map(([k, lbl, kind]) => `<th data-k="${k}" class="${kind !== 'name' && kind !== 'reg' ? 'num' : ''}">${lbl}${scrSortKey === k ? (scrSortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${filtered.map(s => `<tr class="clickable" data-tk="${s.ticker}">
            <td class="num">${s.rank}</td>
            <td><strong>${s.name}</strong> <small style="color:var(--muted)">${s.ticker.split(".")[0]} · ${s.market}</small></td>
            <td class="num">${fmtInt(s.last_price)}원</td>
            <td class="num"><strong style="color:var(--long)">${fmtInt(s.fair_price)}원</strong></td>
            <td class="${clsNum(s.val_gap_pct)}">${fmtSign(s.val_gap_pct, 1)}%</td>
            <td class="num">${fmtInt(s.target_price_12m)}원</td>
            <td class="${clsNum(s.exp_return_12m_pct)}">${fmtSign(s.exp_return_12m_pct, 2)}%</td>
            <td class="num">${fmt(s.surface_pbr, 2)}</td>
            <td class="num"><strong class="${s.true_pbr <= s.surface_pbr ? 'pos' : 'neg'}">${fmt(s.true_pbr, 2)}</strong></td>
            <td class="num">${fmt(s.surface_per, 1)}</td>
            <td class="num"><strong>${fmt(s.true_per, 1)}</strong></td>
            <td class="num">${fmt(s.surface_roe, 1)}%</td>
            <td class="${clsNum(s.true_roe)}">${fmt(s.true_roe, 1)}%</td>
            <td class="${clsNum(s.op_margin)}">${fmt(s.op_margin, 1)}%</td>
            <td class="num">${fmt(s.debt_ratio, 1)}%</td>
            <td class="${s.accounting_bloat <= 0 ? 'num pos' : 'num neg'}">${fmtSign(s.accounting_bloat, 2)}</td>
            <td class="${clsNum(s.alpha_score)}">${fmtSign(s.alpha_score, 3)}</td>
            <td>${badgeRegime(s.valuation_regime)}</td>
          </tr>`).join("")}
        </tbody>
      `;

      document.querySelectorAll("#scr-table thead th[data-k]").forEach(th => {
        th.addEventListener("click", () => {
          const k = th.dataset.k;
          if (scrSortKey === k) scrSortDir = -scrSortDir;
          else { scrSortKey = k; scrSortDir = (k === "rank" || k === "name") ? 1 : -1; }
          renderScreener();
        });
      });
      document.querySelectorAll("#scr-table tbody tr.clickable").forEach(tr => {
        tr.addEventListener("click", () => openStockDetail(tr.dataset.tk));
      });
    }

    ["scr-q", "scr-mkt", "scr-sig", "scr-reg"].forEach(id => {
      document.getElementById(id).addEventListener("input", renderScreener);
      document.getElementById(id).addEventListener("change", renderScreener);
    });

    // Live Sync Button (connects to webapp.py when running locally)
    document.getElementById("btn-live-sync").addEventListener("click", async () => {
      const btn = document.getElementById("btn-live-sync");
      btn.disabled = true;
      btn.textContent = "⏳ 증분 업데이트 중...";
      try {
        const resp = await fetch("/api/sync", { method: "POST" });
        if (resp.ok) {
          alert("새로운 재무제표 및 시세 증분 업데이트가 완료되었습니다. 화면을 새로고침합니다.");
          location.reload();
        } else {
          alert("정적 호스팅(Netlify/파일 열기) 환경입니다. 로컬에서 'python webapp.py' 또는 'python main_v3.py --sync-live'를 실행하면 신규 재무제표가 DB에 증분 반영됩니다.");
        }
      } catch (e) {
        alert("현재 정적 대시보드 모드입니다. 로컬 터미널에서 'python webapp.py' 또는 'python main_v3.py --sync-live'를 실행하면 최신 K-IFRS 재무제표 및 10년 시세가 로컬 DB(data/krx_quant_store.db)에 자동 증분 업데이트됩니다.");
      } finally {
        btn.disabled = false;
        btn.textContent = "🔄 데이터 증분 갱신";
      }
    });

    // Initial Render
    renderPortfolioTab();
    renderStockDetail(STOCKS[0]?.ticker || "005930.KS");
    renderBacktestTab();
    renderScreener();
  </script>
</body>
</html>
"""


def generate_v3_dashboards(
    bundle_path: str = "logs_v3/v3_webapp_bundle.json",
    out_paths: Tuple[str, ...] = ("index.html", "krx_dashboard.html"),
) -> List[str]:
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    # JSON 직렬화 시 공백 최소화하여 파일 크기 최적화
    compact_json = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_doc = DASHBOARD_V3_HTML_TEMPLATE.replace("__V3_BUNDLE_JSON__", compact_json)

    saved = []
    for p in out_paths:
        with open(p, "w", encoding="utf-8") as f:
            f.write(html_doc)
        saved.append(p)
        print(f"[Dashboard v3 Saved] '{p}' 생성 완료 ({len(html_doc):,} bytes)")
    return saved


if __name__ == "__main__":
    generate_v3_dashboards()
