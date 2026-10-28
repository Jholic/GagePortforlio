#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===================================================================================================
 AUTO_PORTFOLIO v3.0: 장기 시계열 텐서 역변환 백테스터 & 진성 밸류에이션(True PER/PBR) 로컬 DB 엔진
===================================================================================================
핵심 기능:
1. 로컬 영구 저장소 (SQLite `data/krx_quant_store.db` + JSON/CSV 스냅샷):
   - KRX 전 종목(약 1,900개)의 연도/분기별 K-IFRS 재무제표, 시점별 실제 주가, 표면 지표 vs 진성 지표,
     적정주가 시계열을 로컬 DB에 영구 축적합니다.
   - 새로운 분기/연간 재무제표나 최신 시세가 발표될 때마다 기존 히스토리를 보존하며 증분 병합(Upsert)합니다.
2. 표면적 밸류에이션(Surface PER/PBR/ROE) vs 허상을 걷어낸 진짜 밸류에이션(True PER/PBR/ROE & 내재주가):
   - 일회성 영업외이익 착시(Accrual Bloat), 과다부채 레버리지 착시, 자본잠식 왜곡, 텐서 여공간 정적 바이어스(b_m)를
     분리하여 진성 주당순자산(True BPS), 진성 지속가능 이익(True EPS), 진짜 PBR/PER 및 진성 적정주가(Fair Price)를 산출합니다.
3. 우리 로직 기반 추천 포트폴리오 빌더 (4대 추천 포트폴리오 + 회계착시 가치함정 경계 리스트):
   - (1) 대표 추천: v3 텐서 알파 핵심 포트폴리오 Top 20 (최적 비중 % + 종목별 선정 근거 자동 생성)
   - (2) 딥밸류 추천: 허상 제거 '진짜 저PBR·저PER' 포트폴리오 Top 15
   - (3) 고수익 시너지 추천: 고마진·시너지 모멘텀 우량주 Top 15
   - (4) KOSPI 안정형 추천: KOSPI 핵심 우량주 포트폴리오 Top 15
   - (5) 가치함정 경계: 표면 지표 착시·부실 위험 주의 종목 Top 15
4. 장기 시계열 다각도 실증 백테스트 (Long-Horizon Multi-Mode Backtest):
   - 전략별 누적 수익률 곡선, 5분위 단조성 검증, 표면 vs 진짜 PER/PBR 4분면 실증 매트릭스,
     적정주가 괴리율 구간별 주가 수렴성, 팩터 분해(Ablation Study) 리포트 산출
===================================================================================================
"""

import os
import sys
import ast
import json
import time
import sqlite3
import argparse
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, stats

from build_v3_dashboard import nan_to_null

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. 로컬 영구 저장소 (SQLite Incremental Quant Store)
# ==============================================================================

class LocalQuantStore:
    """
    전 종목 K-IFRS 재무제표, 주가 시계열, 정제 밸류에이션(True PER/PBR/Fair Price) 및
    예측 결과를 영구 보관하고 증분 업데이트(Upsert)하는 SQLite 기반 로컬 데이터 스토어입니다.
    """

    def __init__(self, db_path: str = "data/krx_quant_store.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stocks_meta (
                    ticker TEXT PRIMARY KEY,
                    code TEXT,
                    name TEXT,
                    market TEXT,
                    last_price REAL,
                    last_date TEXT,
                    marcap_krw REAL,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS financial_statements (
                    ticker TEXT,
                    ym TEXT,
                    is_quarter INTEGER DEFAULT 0,
                    is_consensus INTEGER DEFAULT 0,
                    rev REAL,
                    op_inc REAL,
                    net_inc REAL,
                    ctrl_inc REAL,
                    op_margin REAL,
                    profit_margin REAL,
                    roe REAL,
                    debt_ratio REAL,
                    quick_ratio REAL,
                    reserve_ratio REAL,
                    eps REAL,
                    eps_q REAL,
                    per REAL,
                    bps REAL,
                    pbr REAL,
                    dps REAL,
                    rev_growth REAL,
                    earn_growth REAL,
                    implied_price REAL,
                    updated_at TEXT,
                    PRIMARY KEY (ticker, ym)
                );

                CREATE TABLE IF NOT EXISTS price_history (
                    ticker TEXT,
                    ym TEXT,
                    trade_date TEXT,
                    close_price REAL,
                    volume REAL,
                    source TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (ticker, ym)
                );

                CREATE TABLE IF NOT EXISTS valuation_timeseries (
                    ticker TEXT,
                    ym TEXT,
                    period_label TEXT,
                    data_tier TEXT,
                    actual_price REAL,
                    fair_price REAL,
                    val_gap_pct REAL,
                    surface_pbr REAL,
                    true_pbr REAL,
                    pbr_illusion_pct REAL,
                    surface_per REAL,
                    true_per REAL,
                    per_illusion_gap REAL,
                    surface_roe REAL,
                    true_roe REAL,
                    op_margin REAL,
                    net_margin REAL,
                    debt_ratio REAL,
                    bps_surface REAL,
                    bps_true REAL,
                    eps_surface REAL,
                    eps_true REAL,
                    accounting_bloat REAL,
                    intrinsic_potential REAL,
                    synergy_momentum REAL,
                    alpha_score REAL,
                    fwd_return_next_pct REAL,
                    PRIMARY KEY (ticker, ym)
                );

                CREATE TABLE IF NOT EXISTS stock_forecasts (
                    rank INTEGER,
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    last_trade_date TEXT,
                    last_price REAL,
                    fair_price REAL,
                    val_gap_pct REAL,
                    target_price_6m REAL,
                    target_price_12m REAL,
                    exp_return_6m_pct REAL,
                    exp_return_12m_pct REAL,
                    exp_alpha_12m_pct REAL,
                    signal TEXT,
                    valuation_regime TEXT,
                    surface_pbr REAL,
                    true_pbr REAL,
                    pbr_illusion_pct REAL,
                    surface_per REAL,
                    true_per REAL,
                    per_illusion_gap REAL,
                    surface_roe REAL,
                    true_roe REAL,
                    op_margin REAL,
                    debt_ratio REAL,
                    alpha_score REAL,
                    intrinsic_potential REAL,
                    accounting_bloat REAL,
                    synergy_momentum REAL,
                    mahal_ood_dist2 REAL,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS backtest_reports (
                    report_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    payload_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_val_ts_ticker ON valuation_timeseries(ticker);
                CREATE INDEX IF NOT EXISTS idx_val_ts_ym ON valuation_timeseries(ym);
                CREATE INDEX IF NOT EXISTS idx_fin_ticker ON financial_statements(ticker);
                """
            )

    def upsert_stocks_meta(self, rows: List[Tuple]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO stocks_meta (ticker, code, name, market, last_price, last_date, marcap_krw, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    name=excluded.name,
                    market=excluded.market,
                    last_price=CASE WHEN excluded.last_price > 0 THEN excluded.last_price ELSE stocks_meta.last_price END,
                    last_date=CASE WHEN excluded.last_date != '' THEN excluded.last_date ELSE stocks_meta.last_date END,
                    marcap_krw=CASE WHEN excluded.marcap_krw > 0 THEN excluded.marcap_krw ELSE stocks_meta.marcap_krw END,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def upsert_financial_statements(self, rows: List[Tuple]) -> int:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO financial_statements (
                    ticker, ym, is_quarter, is_consensus, rev, op_inc, net_inc, ctrl_inc,
                    op_margin, profit_margin, roe, debt_ratio, quick_ratio, reserve_ratio,
                    eps, eps_q, per, bps, pbr, dps, rev_growth, earn_growth, implied_price, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, ym) DO UPDATE SET
                    is_quarter=excluded.is_quarter,
                    is_consensus=excluded.is_consensus,
                    rev=COALESCE(excluded.rev, financial_statements.rev),
                    op_inc=COALESCE(excluded.op_inc, financial_statements.op_inc),
                    net_inc=COALESCE(excluded.net_inc, financial_statements.net_inc),
                    ctrl_inc=COALESCE(excluded.ctrl_inc, financial_statements.ctrl_inc),
                    op_margin=COALESCE(excluded.op_margin, financial_statements.op_margin),
                    profit_margin=COALESCE(excluded.profit_margin, financial_statements.profit_margin),
                    roe=COALESCE(excluded.roe, financial_statements.roe),
                    debt_ratio=COALESCE(excluded.debt_ratio, financial_statements.debt_ratio),
                    quick_ratio=COALESCE(excluded.quick_ratio, financial_statements.quick_ratio),
                    reserve_ratio=COALESCE(excluded.reserve_ratio, financial_statements.reserve_ratio),
                    eps=COALESCE(excluded.eps, financial_statements.eps),
                    eps_q=COALESCE(excluded.eps_q, financial_statements.eps_q),
                    per=COALESCE(excluded.per, financial_statements.per),
                    bps=COALESCE(excluded.bps, financial_statements.bps),
                    pbr=COALESCE(excluded.pbr, financial_statements.pbr),
                    dps=COALESCE(excluded.dps, financial_statements.dps),
                    rev_growth=COALESCE(excluded.rev_growth, financial_statements.rev_growth),
                    earn_growth=COALESCE(excluded.earn_growth, financial_statements.earn_growth),
                    implied_price=COALESCE(excluded.implied_price, financial_statements.implied_price),
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def upsert_price_history(self, rows: List[Tuple]) -> int:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO price_history (ticker, ym, trade_date, close_price, volume, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, ym) DO UPDATE SET
                    trade_date=excluded.trade_date,
                    close_price=excluded.close_price,
                    volume=excluded.volume,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def replace_valuation_timeseries(self, rows: List[Tuple]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM valuation_timeseries;")
            conn.executemany(
                """
                INSERT INTO valuation_timeseries (
                    ticker, ym, period_label, data_tier, actual_price, fair_price, val_gap_pct,
                    surface_pbr, true_pbr, pbr_illusion_pct, surface_per, true_per, per_illusion_gap,
                    surface_roe, true_roe, op_margin, net_margin, debt_ratio,
                    bps_surface, bps_true, eps_surface, eps_true,
                    accounting_bloat, intrinsic_potential, synergy_momentum, alpha_score, fwd_return_next_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def replace_stock_forecasts(self, df_forecasts: pd.DataFrame) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM stock_forecasts;")
            df_forecasts.to_sql("stock_forecasts", conn, if_exists="append", index=False)

    def save_backtest_report(self, report_id: str, payload: Dict[str, object]) -> None:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO backtest_reports (report_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """,
                (report_id, now_str, json.dumps(payload, ensure_ascii=False)),
            )


# ==============================================================================
# 2. 데이터 수집 및 증분 동기화 (캐시 시드 + 실시간 10년 시세/재무 증분 수집)
# ==============================================================================

def _infer_implied_price(metrics: Dict[str, float]) -> Optional[float]:
    """재무 스냅샷 내 BPS*PBR 또는 EPS*PER로부터 결산 시점의 실제 시장 종가를 복원합니다."""
    bps = float(metrics.get("bps") or 0.0)
    pbr = float(metrics.get("pbr") or 0.0)
    eps = float(metrics.get("eps") or 0.0)
    per = float(metrics.get("per") or 0.0)
    is_q = float(metrics.get("is_quarter") or 0.0)

    if bps > 0 and pbr > 0:
        return float(bps * pbr)
    if eps * per > 0 and is_q == 0.0:
        return float(eps * per)
    if bps != 0 and pbr != 0:
        return float(abs(bps * pbr))
    if eps != 0 and per != 0:
        scale = 0.25 if is_q > 0 else 1.0
        return float(abs(eps * scale * per))
    return None


def sync_local_caches_to_store(
    store: LocalQuantStore,
    fin_cache_dir: str = "logs_v2/cache",
    v2_csv_path: str = "krx_all_stocks_expected_returns_v2.csv",
    v1_csv_path: str = "krx_all_stocks_expected_returns.csv",
) -> Tuple[List[str], Dict[str, str], Dict[str, str], Dict[str, float], Dict[str, str]]:
    """
    로컬에 존재하는 모든 재무 캐시 파일(logs_v2/cache/naver_pit_fin_*.json)과
    예측 CSV의 최신 가격을 SQLite 스토어에 증분 병합(Upsert)합니다.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name_map: Dict[str, str] = {}
    mkt_map: Dict[str, str] = {}
    last_prices: Dict[str, float] = {}
    last_dates: Dict[str, str] = {}

    for csv_p in (v1_csv_path, v2_csv_path):
        if os.path.exists(csv_p):
            df_c = pd.read_csv(csv_p, dtype={"Ticker": str})
            for _, r in df_c.iterrows():
                tk = str(r["Ticker"]).strip()
                name_map[tk] = str(r.get("Name", tk))
                mkt_map[tk] = str(r.get("Market", "KRX"))
                p_val = float(r.get("Last_Price(KRW)", 0.0) or 0.0)
                d_val = str(r.get("Last_Trade_Date", "2026-10-23"))
                if p_val > 0:
                    last_prices[tk] = p_val
                    last_dates[tk] = d_val

    cache_files = []
    if os.path.exists(fin_cache_dir):
        for fn in sorted(os.listdir(fin_cache_dir)):
            if fn.startswith("naver_pit_fin_") and fn.endswith(".json"):
                cache_files.append(os.path.join(fin_cache_dir, fn))

    fin_rows: List[Tuple] = []
    price_rows: List[Tuple] = []
    fin_tickers = set()

    for cf in cache_files:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for tk, payload in data.items():
            if not isinstance(payload, dict) or payload.get("is_real_fin", 0.0) <= 0:
                continue
            fin_tickers.add(tk)
            for is_cons_flag, dict_key in ((0, "periods"), (1, "consensus")):
                p_map = payload.get(dict_key) or {}
                for ym, m in p_map.items():
                    if not isinstance(m, dict):
                        continue
                    imp_p = _infer_implied_price(m) if is_cons_flag == 0 else None
                    fin_rows.append((
                        tk,
                        str(ym),
                        int(m.get("is_quarter", 0) or 0),
                        is_cons_flag,
                        m.get("rev"),
                        m.get("op_inc"),
                        m.get("net_inc"),
                        m.get("ctrl_inc"),
                        m.get("op_margin"),
                        m.get("profit_margin"),
                        m.get("roe"),
                        m.get("debt_ratio"),
                        m.get("quick_ratio"),
                        m.get("reserve_ratio"),
                        m.get("eps"),
                        m.get("eps_q"),
                        m.get("per"),
                        m.get("bps"),
                        m.get("pbr"),
                        m.get("dps"),
                        m.get("rev_growth"),
                        m.get("earn_growth"),
                        imp_p,
                        now_str,
                    ))
                    if is_cons_flag == 0 and imp_p is not None and imp_p > 50:
                        price_rows.append((
                            tk,
                            str(ym),
                            f"{str(ym)[:4]}-{str(ym)[4:6]}-28",
                            float(imp_p),
                            0.0,
                            "NAVER_FILING_IMPLIED",
                            now_str,
                        ))

    for tk, p_val in last_prices.items():
        d_str = last_dates.get(tk, "2026-10-23")
        ym_str = d_str[:4] + d_str[5:7] if len(d_str) >= 7 else "202610"
        price_rows.append((tk, ym_str, d_str, float(p_val), 0.0, "KRX_CLOSE", now_str))

    valid_tickers = sorted(t for t in fin_tickers if t in last_prices)
    meta_rows = [
        (
            tk,
            tk.split(".")[0],
            name_map.get(tk, tk),
            mkt_map.get(tk, "KOSPI" if tk.endswith(".KS") else "KOSDAQ"),
            float(last_prices.get(tk, 0.0)),
            last_dates.get(tk, "2026-10-23"),
            0.0,
            now_str,
        )
        for tk in valid_tickers
    ]

    store.upsert_stocks_meta(meta_rows)
    if fin_rows:
        store.upsert_financial_statements(fin_rows)
    if price_rows:
        store.upsert_price_history(price_rows)

    print(
        f"[LocalQuantStore] 로컬 DB 증분 병합 완료: 종목 {len(valid_tickers)}개, "
        f"재무 레코드 {len(fin_rows):,}건, 주가 레코드 {len(price_rows):,}건 ('{store.db_path}')"
    )
    return valid_tickers, name_map, mkt_map, last_prices, last_dates


def sync_live_naver_data(
    store: LocalQuantStore,
    tickers: Optional[List[str]] = None,
    years_history: int = 10,
    max_workers: int = 24,
) -> None:
    """
    네트워크 연결 시 네이버 차트 API에서 10년 이상 장기 월봉 OHLCV를 수집하고,
    네이버 모바일 금융 API에서 신규 공시된 연간/분기 K-IFRS 재무제표를 증분 업데이트합니다.
    """
    from main_v2 import FinancialFleetDataLoaderV2

    loader = FinancialFleetDataLoaderV2()
    if not tickers:
        tickers = loader.load_all_krx_universe(market="ALL", min_marcap_billion_krw=30.0)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years_history + 60)

    print(f"[Live Sync] {len(tickers)}개 종목 {years_history}년 월별 주가 및 신규 재무제표 증분 수집 시작...")

    def _fetch_10y_monthly(tk: str) -> List[Tuple]:
        code = tk.split(".")[0]
        url = (
            "https://api.finance.naver.com/siseJson.naver"
            f"?symbol={code}&requestType=1"
            f"&startTime={start_dt.strftime('%Y%m%d')}&endTime={end_dt.strftime('%Y%m%d')}"
            "&timeframe=month"
        )
        rows_out = []
        try:
            raw = loader._http_get_bytes(url, timeout=12, extra_headers={"Referer": "https://finance.naver.com/"})
            text = raw.decode("utf-8", errors="ignore").strip()
            table = ast.literal_eval(text)
            if not isinstance(table, list) or len(table) < 3:
                return []
            header = [str(h) for h in table[0]]
            for r in table[1:]:
                if not isinstance(r, (list, tuple)) or len(r) < 5:
                    continue
                rec = {header[i]: r[i] for i in range(min(len(header), len(r)))}
                d_raw = str(rec.get("날짜") or rec.get("date") or "").strip()
                c_val = float(rec.get("종가") or rec.get("close") or 0.0)
                v_val = float(rec.get("거래량") or rec.get("volume") or 0.0)
                if len(d_raw) == 8 and c_val > 0:
                    ym = d_raw[:6]
                    t_date = f"{d_raw[:4]}-{d_raw[4:6]}-{d_raw[6:8]}"
                    rows_out.append((tk, ym, t_date, c_val, v_val, "NAVER_10Y_MONTHLY", now_str))
        except Exception:
            pass
        return rows_out

    all_price_rows: List[Tuple] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_fetch_10y_monthly, tk) for tk in tickers]
        for fut in as_completed(futs):
            all_price_rows.extend(fut.result())

    if all_price_rows:
        store.upsert_price_history(all_price_rows)
        print(f"[Live Sync] 장기 월별 시세 {len(all_price_rows):,}건 DB 저장 완료.")

    loader._load_or_fetch_all_fundamentals(tickers, max_workers=max_workers)
    sync_local_caches_to_store(store)


# ==============================================================================
# 3. 표면적 PER/PBR vs 허상을 걷어낸 진짜(Purified) PER/PBR & 내재 적정주가 엔진
# ==============================================================================

class PurifiedValuationAndTensorEngine:
    """
    SQLite 로컬 스토어에서 전 종목의 다기간 재무/가격 히스토리를 로드하여:
    1) 과거 장기 체크포인트(202112 ~ 202610) 및 향후 예측(202704, 202710) 시계열 패널을 구축하고,
    2) 6단계 대수적 텐서 역변환으로 진성 잠재 좌표(delta_sure), 회계 착시(b_m), 비대각 시너지(H_off)를 분리하며,
    3) 표면적 PER/PBR/ROE에서 회계 허상을 제거한 '진짜 PER, 진짜 PBR, 진짜 ROE' 및
       '진성 내재 적정주가(Intrinsic Fair Price)'를 산출합니다.
    """

    TIMELINE_YMS = [
        "202112", "202212", "202312", "202412",
        "202506", "202509", "202512", "202603", "202606", "202610"
    ]
    TIMELINE_LABELS = [
        "2021.12", "2022.12", "2023.12", "2024.12",
        "2025.06", "2025.09", "2025.12", "2026.03", "2026.06", "2026.10"
    ]
    STEP_YEARS = [1.0, 1.0, 1.0, 0.5, 0.25, 0.25, 0.25, 0.25, 0.33]

    def __init__(self, store: LocalQuantStore, latent_dim: int = 4):
        self.store = store
        self.latent_dim = latent_dim

    def load_panel_from_db(self) -> Dict[str, object]:
        """DB에서 전 종목 메타데이터, 재무제표, 가격 이력을 읽어 다기간 행렬을 구축합니다."""
        with self.store._connect() as conn:
            df_meta = pd.read_sql_query("SELECT * FROM stocks_meta ORDER BY ticker", conn)
            df_fin = pd.read_sql_query(
                "SELECT * FROM financial_statements WHERE is_consensus = 0 ORDER BY ticker, ym",
                conn,
            )
            df_cons = pd.read_sql_query(
                "SELECT * FROM financial_statements WHERE is_consensus = 1 ORDER BY ticker, ym",
                conn,
            )
            df_price = pd.read_sql_query("SELECT * FROM price_history ORDER BY ticker, ym", conn)

        tickers = list(df_meta["ticker"])
        M = len(tickers)
        T = len(self.TIMELINE_YMS)

        fin_by_tk: Dict[str, Dict[str, Dict[str, float]]] = {tk: {} for tk in tickers}
        for _, row in df_fin.iterrows():
            tk = row["ticker"]
            if tk in fin_by_tk:
                fin_by_tk[tk][str(row["ym"])] = row.to_dict()

        cons_by_tk: Dict[str, Dict[str, Dict[str, float]]] = {tk: {} for tk in tickers}
        for _, row in df_cons.iterrows():
            tk = row["ticker"]
            if tk in cons_by_tk:
                cons_by_tk[tk][str(row["ym"])] = row.to_dict()

        price_by_tk: Dict[str, Dict[str, float]] = {tk: {} for tk in tickers}
        for _, row in df_price.iterrows():
            tk = row["ticker"]
            if tk in price_by_tk and row["close_price"] and row["close_price"] > 0:
                price_by_tk[tk][str(row["ym"])] = float(row["close_price"])

        P_act = np.full((M, T), np.nan, dtype=np.float64)
        BPS_surf = np.full((M, T), np.nan, dtype=np.float64)
        EPS_surf = np.full((M, T), np.nan, dtype=np.float64)
        ROE_surf = np.full((M, T), np.nan, dtype=np.float64)
        OPM = np.full((M, T), np.nan, dtype=np.float64)
        NETM = np.full((M, T), np.nan, dtype=np.float64)
        DEBT = np.full((M, T), np.nan, dtype=np.float64)
        QUICK = np.full((M, T), np.nan, dtype=np.float64)
        RESERVE = np.full((M, T), np.nan, dtype=np.float64)
        PBR_surf = np.full((M, T), np.nan, dtype=np.float64)
        PER_surf = np.full((M, T), np.nan, dtype=np.float64)
        REV_G = np.full((M, T), 0.0, dtype=np.float64)
        EARN_G = np.full((M, T), 0.0, dtype=np.float64)
        DPS = np.full((M, T), 0.0, dtype=np.float64)
        TIER: List[List[str]] = [["ACTUAL_FILING"] * T for _ in range(M)]

        actual_yms = self.TIMELINE_YMS[2:]  # 202312 .. 202610

        for i, tk in enumerate(tickers):
            f_map = fin_by_tk[tk]
            p_map = price_by_tk[tk]
            sorted_avail_yms = sorted(f_map.keys())
            if not sorted_avail_yms:
                continue

            last_m = f_map[sorted_avail_yms[0]].copy()
            for j_rel, ym in enumerate(actual_yms):
                j = j_rel + 2
                for a_ym in sorted_avail_yms:
                    if a_ym <= ym:
                        for k, v in f_map[a_ym].items():
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                last_m[k] = v

                bps = float(last_m.get("bps") or 0.0)
                pbr_rep = float(last_m.get("pbr") or 0.0)
                per_rep = float(last_m.get("per") or 0.0)
                eps_raw = float(last_m.get("eps") or 0.0)
                eps_q = float(last_m.get("eps_q") or 0.0)
                is_q = int(last_m.get("is_quarter") or 0)

                p_val = p_map.get(ym)
                if (p_val is None or p_val <= 50) and ym in f_map:
                    p_val = _infer_implied_price(f_map[ym])
                if p_val is None or p_val <= 50:
                    p_val = _infer_implied_price(last_m)
                if p_val and p_val > 50:
                    P_act[i, j] = float(p_val)

                BPS_surf[i, j] = bps
                if P_act[i, j] > 0 and abs(per_rep) > 0.05 and ym != "202610":
                    EPS_surf[i, j] = P_act[i, j] / per_rep
                elif j == T - 1 and np.isfinite(EPS_surf[i, j - 1]):
                    c_26 = cons_by_tk[tk].get("202612") or cons_by_tk[tk].get("202609") or {}
                    c_eps = float(c_26.get("eps") or 0.0)
                    if c_eps != 0 and abs(c_eps) < abs(EPS_surf[i, j - 1]) * 3.0:
                        EPS_surf[i, j] = 0.70 * EPS_surf[i, j - 1] + 0.30 * c_eps
                    else:
                        EPS_surf[i, j] = EPS_surf[i, j - 1]
                else:
                    EPS_surf[i, j] = eps_raw if is_q == 0 else (eps_q * 4.0 if eps_q else eps_raw)

                ROE_surf[i, j] = float(last_m.get("roe") or 0.0)
                OPM[i, j] = float(last_m.get("op_margin") or 0.0)
                NETM[i, j] = float(last_m.get("profit_margin") or 0.0)
                DEBT[i, j] = float(last_m.get("debt_ratio") or 100.0)
                QUICK[i, j] = float(last_m.get("quick_ratio") or 100.0)
                RESERVE[i, j] = float(last_m.get("reserve_ratio") or 200.0)
                REV_G[i, j] = float(last_m.get("rev_growth") or 0.0)
                EARN_G[i, j] = float(last_m.get("earn_growth") or 0.0)
                DPS[i, j] = max(0.0, float(last_m.get("dps") or 0.0))

                if P_act[i, j] > 0 and bps > 0:
                    PBR_surf[i, j] = P_act[i, j] / bps
                else:
                    PBR_surf[i, j] = pbr_rep if pbr_rep != 0 else np.nan

                if P_act[i, j] > 0 and abs(EPS_surf[i, j]) > 0.5:
                    PER_surf[i, j] = np.clip(P_act[i, j] / EPS_surf[i, j], -300.0, 500.0)
                else:
                    PER_surf[i, j] = np.clip(per_rep, -300.0, 500.0)

            s_p = pd.Series(P_act[i, 2:]).ffill().bfill().to_numpy()
            P_act[i, 2:] = s_p

            for j_back, ym_back in ((1, "202212"), (0, "202112")):
                j_next = j_back + 1
                if ym_back in p_map and p_map[ym_back] > 50:
                    P_act[i, j_back] = p_map[ym_back]
                    TIER[i][j_back] = "ACTUAL_FILING"
                else:
                    TIER[i][j_back] = "BACKCAST"

                if ym_back in f_map:
                    m_b = f_map[ym_back]
                    BPS_surf[i, j_back] = float(m_b.get("bps") or BPS_surf[i, j_next])
                    ROE_surf[i, j_back] = float(m_b.get("roe") or ROE_surf[i, j_next])
                    OPM[i, j_back] = float(m_b.get("op_margin") or OPM[i, j_next])
                    NETM[i, j_back] = float(m_b.get("profit_margin") or NETM[i, j_next])
                    DEBT[i, j_back] = float(m_b.get("debt_ratio") or DEBT[i, j_next])
                    QUICK[i, j_back] = float(m_b.get("quick_ratio") or QUICK[i, j_next])
                    RESERVE[i, j_back] = float(m_b.get("reserve_ratio") or RESERVE[i, j_next])
                else:
                    roe_step = np.clip(ROE_surf[i, j_next] / 100.0, -0.25, 0.30)
                    bps_growth = np.clip((BPS_surf[i, 3] / max(1.0, BPS_surf[i, 2])) - 1.0, -0.20, 0.25)
                    eff_g = 0.5 * roe_step + 0.5 * bps_growth
                    BPS_surf[i, j_back] = BPS_surf[i, j_next] / (1.0 + 0.65 * eff_g)
                    ROE_surf[i, j_back] = 0.75 * ROE_surf[i, j_next] + 0.25 * ROE_surf[i, 3]
                    OPM[i, j_back] = 0.75 * OPM[i, j_next] + 0.25 * OPM[i, 3]
                    NETM[i, j_back] = 0.75 * NETM[i, j_next] + 0.25 * NETM[i, 3]
                    DEBT[i, j_back] = max(5.0, DEBT[i, j_next] * (1.0 + 0.05 * roe_step))
                    QUICK[i, j_back] = QUICK[i, j_next]
                    RESERVE[i, j_back] = RESERVE[i, j_next] / (1.0 + 0.5 * max(0.0, eff_g))

                EPS_surf[i, j_back] = BPS_surf[i, j_back] * (np.clip(ROE_surf[i, j_back], -60.0, 80.0) / 100.0)
                if not np.isfinite(P_act[i, j_back]) or P_act[i, j_back] <= 50:
                    mult_adj = 0.92 if ym_back == "202212" else 1.12
                    hist_pbr = np.clip(PBR_surf[i, j_next] * mult_adj, 0.20, 15.0)
                    P_act[i, j_back] = max(100.0, abs(BPS_surf[i, j_back]) * hist_pbr)

                PBR_surf[i, j_back] = (
                    P_act[i, j_back] / BPS_surf[i, j_back]
                    if BPS_surf[i, j_back] > 0 else PBR_surf[i, j_next]
                )
                PER_surf[i, j_back] = (
                    np.clip(P_act[i, j_back] / EPS_surf[i, j_back], -300.0, 500.0)
                    if abs(EPS_surf[i, j_back]) > 0.5 else PER_surf[i, j_next]
                )

        return {
            "tickers": tickers,
            "meta_df": df_meta,
            "P_act": P_act,
            "BPS_surf": BPS_surf,
            "EPS_surf": EPS_surf,
            "ROE_surf": ROE_surf,
            "OPM": OPM,
            "NETM": NETM,
            "DEBT": DEBT,
            "QUICK": QUICK,
            "RESERVE": RESERVE,
            "PBR_surf": PBR_surf,
            "PER_surf": PER_surf,
            "REV_G": REV_G,
            "EARN_G": EARN_G,
            "DPS": DPS,
            "TIER": TIER,
            "cons_by_tk": cons_by_tk,
        }

    def compute_purified_valuation_and_tensor_alpha(
        self, panel: Dict[str, object]
    ) -> Dict[str, object]:
        P_act = panel["P_act"]
        BPS_surf = panel["BPS_surf"]
        EPS_surf = panel["EPS_surf"]
        ROE_surf = panel["ROE_surf"]
        OPM = panel["OPM"]
        NETM = panel["NETM"]
        DEBT = panel["DEBT"]
        QUICK = panel["QUICK"]
        RESERVE = panel["RESERVE"]
        PBR_surf = panel["PBR_surf"]
        PER_surf = panel["PER_surf"]
        REV_G = panel["REV_G"]
        EARN_G = panel["EARN_G"]
        DPS = panel["DPS"]

        M, T = P_act.shape
        D = 12

        BPS_true = np.zeros((M, T), dtype=np.float64)
        EPS_true = np.zeros((M, T), dtype=np.float64)
        ROE_true = np.zeros((M, T), dtype=np.float64)
        PBR_true = np.zeros((M, T), dtype=np.float64)
        PER_true = np.zeros((M, T), dtype=np.float64)
        P_fair = np.zeros((M, T), dtype=np.float64)
        VAL_gap_pct = np.zeros((M, T), dtype=np.float64)
        PBR_illusion_pct = np.zeros((M, T), dtype=np.float64)
        PER_illusion_gap = np.zeros((M, T), dtype=np.float64)

        BLOAT_mat = np.zeros((M, T), dtype=np.float64)
        INTRINSIC_mat = np.zeros((M, T), dtype=np.float64)
        SYNERGY_mat = np.zeros((M, T), dtype=np.float64)
        ALPHA_mat = np.zeros((M, T), dtype=np.float64)
        MAHAL_mat = np.zeros((M, T), dtype=np.float64)
        FWD_RET_next = np.full((M, T), np.nan, dtype=np.float64)

        for j in range(T - 1):
            FWD_RET_next[:, j] = np.clip((P_act[:, j + 1] / np.maximum(P_act[:, j], 1.0)) - 1.0, -0.85, 2.50)

        X_tensor = np.zeros((M, D, T), dtype=np.float64)

        for j in range(T):
            p_j = np.maximum(P_act[:, j], 100.0)
            bps_j = BPS_surf[:, j].copy()
            eps_j = EPS_surf[:, j].copy()
            roe_j = np.clip(np.nan_to_num(ROE_surf[:, j], nan=0.0), -100.0, 150.0)
            opm_j = np.clip(np.nan_to_num(OPM[:, j], nan=0.0), -65.0, 80.0)
            netm_j = np.clip(np.nan_to_num(NETM[:, j], nan=0.0), -80.0, 100.0)
            debt_raw = np.nan_to_num(DEBT[:, j], nan=100.0)
            quick_j = np.clip(np.nan_to_num(QUICK[:, j], nan=100.0), 5.0, 1000.0)
            res_j = np.clip(np.nan_to_num(RESERVE[:, j], nan=200.0), -200.0, 50000.0)
            pbr_s = np.clip(np.nan_to_num(PBR_surf[:, j], nan=1.5), 0.10, 25.0)
            per_s = np.clip(np.nan_to_num(PER_surf[:, j], nan=15.0), -200.0, 300.0)

            impaired = (bps_j <= 0) | (debt_raw < 0) | (res_j < 0)
            debt_j = np.where(impaired, 600.0, np.clip(debt_raw, 1.0, 600.0))

            accrual_gap = np.maximum(0.0, netm_j - np.maximum(0.0, opm_j * 0.90))
            unlevered_roe = roe_j / (1.0 + 0.45 * np.maximum(0.0, (debt_j - 80.0) / 100.0))
            leverage_illusion = np.maximum(0.0, roe_j - unlevered_roe)
            fake_roe_flag = (roe_j > 15.0) & (opm_j < 3.0)

            roe_clean = np.where(
                impaired,
                np.minimum(-15.0, opm_j),
                np.where(
                    fake_roe_flag,
                    np.clip(opm_j * 1.1, -40.0, 12.0),
                    unlevered_roe - 0.45 * accrual_gap,
                ),
            )
            quality_bonus = np.where((opm_j > 12.0) & (debt_j < 70.0), 0.15 * (opm_j - 12.0), 0.0)
            roe_clean = np.clip(roe_clean + quality_bonus, -60.0, 75.0)
            ROE_true[:, j] = roe_clean

            bloat_raw = (
                0.08 * accrual_gap
                + 0.65 * np.log1p(np.maximum(0.0, (debt_j - 90.0) / 100.0))
                + 0.04 * leverage_illusion
                + np.where(opm_j < 0, 0.65 + 0.02 * np.abs(opm_j), 0.0)
                + np.where(impaired, 2.20, 0.0)
                - 0.18 * np.log1p(np.maximum(0.0, (quick_j - 120.0) / 100.0))
            )
            BLOAT_mat[:, j] = bloat_raw

            asset_purity = np.exp(-0.35 * np.maximum(0.0, bloat_raw)) * np.clip(
                1.0 + 0.008 * np.clip(opm_j, -30.0, 40.0) + 0.04 * np.log1p(np.maximum(0.0, res_j) / 500.0),
                0.22,
                1.45,
            )
            bps_clean = np.where(
                impaired | (bps_j <= 0),
                p_j / 12.0,
                np.maximum(10.0, bps_j * asset_purity),
            )
            BPS_true[:, j] = bps_clean
            PBR_true[:, j] = np.clip(p_j / bps_clean, 0.12, 25.0)
            PBR_surf[:, j] = pbr_s
            PBR_illusion_pct[:, j] = np.clip(((PBR_true[:, j] - pbr_s) / np.maximum(pbr_s, 0.15)) * 100.0, -75.0, 400.0)

            op_share = np.clip((opm_j * 0.80) / np.maximum(np.abs(netm_j), 1.0), 0.20, 1.35)
            eps_from_op = np.where(
                (eps_j > 0) & (opm_j > 0),
                eps_j * op_share,
                bps_clean * (roe_clean / 100.0),
            )
            eps_clean = 0.55 * (bps_clean * (roe_clean / 100.0)) + 0.45 * eps_from_op
            EPS_true[:, j] = eps_clean
            PER_true[:, j] = np.where(
                np.abs(eps_clean) > 1.0,
                np.clip(p_j / eps_clean, -200.0, 300.0),
                np.where(eps_clean >= 0, 200.0, -100.0),
            )
            PER_surf[:, j] = per_s
            ey_surf = np.clip((eps_j / p_j) * 100.0, -50.0, 50.0)
            ey_true = np.clip((eps_clean / p_j) * 100.0, -50.0, 50.0)
            PER_illusion_gap[:, j] = ey_true - ey_surf

            X_tensor[:, 0, j] = roe_clean / 100.0
            X_tensor[:, 1, j] = opm_j / 100.0
            X_tensor[:, 2, j] = (opm_j - np.maximum(0.0, netm_j - opm_j)) / 100.0
            X_tensor[:, 3, j] = -np.log1p(debt_j / 100.0)
            X_tensor[:, 4, j] = np.log1p(quick_j / 100.0)
            X_tensor[:, 5, j] = np.log1p(np.maximum(0.0, res_j) / 100.0)
            X_tensor[:, 6, j] = np.log1p(1.0 / np.maximum(PBR_true[:, j], 0.15))
            X_tensor[:, 7, j] = np.clip(eps_clean / p_j, -0.40, 0.40)
            X_tensor[:, 8, j] = np.clip(DPS[:, j] / p_j, 0.0, 0.12)
            X_tensor[:, 9, j] = np.clip(REV_G[:, j], -0.60, 1.20)
            X_tensor[:, 10, j] = np.clip(EARN_G[:, j], -0.80, 1.50)
            X_tensor[:, 11, j] = -bloat_raw

            for d in range(D):
                col = X_tensor[:, d, j]
                med = np.median(col)
                mad = np.median(np.abs(col - med)) * 1.4826 + 1e-6
                X_tensor[:, d, j] = np.clip((col - med) / max(float(np.std(col)), mad), -3.8, 3.8)

        r = self.latent_dim
        for j in range(T):
            w_end = j + 1
            X_slice = X_tensor[:, :, :w_end]
            W_len = X_slice.shape[2]

            if W_len >= 3:
                X_mean = np.mean(X_slice, axis=2, keepdims=True)
                X_law = X_slice - X_mean
                X_flat = X_law.transpose(1, 0, 2).reshape(D, M * W_len)
                Sigma_law = (X_flat @ X_flat.T) / max(1, M * W_len)
            else:
                X_cur = X_tensor[:, :, j]
                Sigma_law = (X_cur.T @ X_cur) / max(1, M)

            anchor_weights = np.array([1.2, 1.1, 1.0, 0.8, 0.5, 0.6, 1.5, 1.3, 0.6, 0.5, 0.6, 1.2])
            M_mat = 0.60 * Sigma_law + 0.40 * np.outer(anchor_weights, anchor_weights)
            U_all, _, _ = linalg.svd(M_mat, full_matrices=False)
            U_phys = U_all[:, :r].copy()
            for k in range(r):
                if float(np.dot(U_phys[:, k], anchor_weights)) < 0:
                    U_phys[:, k] *= -1.0

            X_cur = X_tensor[:, :, j]
            delta_raw = X_cur @ U_phys

            d_mean = np.mean(delta_raw, axis=0)
            d_centered = delta_raw - d_mean
            Cov_d = (d_centered.T @ d_centered) / max(1, M - 1) + 0.08 * np.eye(r)
            Cov_d_inv = linalg.inv(Cov_d)
            mahal_sq = np.einsum("mr,rk,mk->m", d_centered, Cov_d_inv, d_centered)
            MAHAL_mat[:, j] = mahal_sq

            ood_ratio = np.clip((mahal_sq - float(r)) / (3.5 * float(r)), 0.0, 1.0)
            shrink = 0.12 + 0.30 * ood_ratio
            delta_sure = (1.0 - shrink[:, None]) * delta_raw + shrink[:, None] * d_mean[None, :]

            b_residual = X_cur - (delta_sure @ U_phys.T)
            b_orth_norm = np.linalg.norm(b_residual[:, :4], axis=1)

            def _z(v: np.ndarray) -> np.ndarray:
                return (v - np.mean(v)) / (np.std(v) + 1e-8)

            w_r = np.exp(-0.40 * np.arange(r))
            w_r /= np.sum(w_r)
            intrinsic_pot = _z(delta_sure @ w_r - 0.06 * np.maximum(0.0, mahal_sq - float(r)))
            INTRINSIC_mat[:, j] = intrinsic_pot

            bloat_z = _z(0.80 * _z(BLOAT_mat[:, j]) + 0.20 * _z(b_orth_norm * np.sign(BLOAT_mat[:, j])))
            BLOAT_mat[:, j] = bloat_z

            qual_axis = 0.45 * X_cur[:, 0] + 0.35 * X_cur[:, 1] + 0.20 * X_cur[:, 3]
            val_axis = 0.55 * X_cur[:, 6] + 0.45 * X_cur[:, 7]
            synergy_raw = np.sign(qual_axis + val_axis) * np.abs(qual_axis * val_axis) ** 0.75
            synergy_z = _z(synergy_raw)
            SYNERGY_mat[:, j] = synergy_z

            roe_pos = np.maximum(0.0, ROE_true[:, j])
            opm_pos = np.maximum(0.0, np.nan_to_num(OPM[:, j], nan=0.0))
            fair_pbr_mult = np.clip(
                0.68
                + 0.055 * roe_pos
                + 0.022 * opm_pos
                + 0.14 * np.maximum(-1.5, intrinsic_pot)
                + 0.10 * np.maximum(-1.0, synergy_z)
                - 0.22 * np.maximum(0.0, bloat_z),
                0.22,
                5.50,
            )
            fair_from_asset = BPS_true[:, j] * fair_pbr_mult
            fair_per_mult = np.clip(8.5 + 0.22 * roe_pos + 1.2 * synergy_z - 1.8 * np.maximum(0.0, bloat_z), 4.5, 22.0)
            fair_from_earn = np.maximum(0.0, EPS_true[:, j]) * fair_per_mult

            p_fair_j = np.where(
                EPS_true[:, j] > 0,
                0.60 * fair_from_asset + 0.40 * fair_from_earn,
                0.85 * fair_from_asset,
            )
            ratio_raw = p_fair_j / np.maximum(P_act[:, j], 100.0)
            ratio_smooth = 1.0 + 0.85 * np.tanh(np.log(np.maximum(ratio_raw, 0.15)) / 1.45)
            P_fair[:, j] = P_act[:, j] * ratio_smooth
            VAL_gap_pct[:, j] = (ratio_smooth - 1.0) * 100.0

            ALPHA_mat[:, j] = (
                0.40 * _z(VAL_gap_pct[:, j])
                + 0.30 * intrinsic_pot
                - 0.18 * bloat_z
                + 0.12 * synergy_z
            )

        panel.update({
            "BPS_true": BPS_true,
            "EPS_true": EPS_true,
            "ROE_true": ROE_true,
            "PBR_true": PBR_true,
            "PER_true": PER_true,
            "P_fair": P_fair,
            "VAL_gap_pct": VAL_gap_pct,
            "PBR_illusion_pct": PBR_illusion_pct,
            "PER_illusion_gap": PER_illusion_gap,
            "BLOAT_mat": BLOAT_mat,
            "INTRINSIC_mat": INTRINSIC_mat,
            "SYNERGY_mat": SYNERGY_mat,
            "ALPHA_mat": ALPHA_mat,
            "MAHAL_mat": MAHAL_mat,
            "FWD_RET_next": FWD_RET_next,
        })
        return panel


# ==============================================================================
# 4. 장기 시계열 다각도 백테스터 & 표면 vs 진짜 지표 실증 분석기
# ==============================================================================

class ComprehensiveQuantBacktester:
    def __init__(self, cost_bps: float = 15.0):
        self.cost_bps = cost_bps

    def run_all_studies(self, panel: Dict[str, object]) -> Dict[str, object]:
        P_act = panel["P_act"]
        PBR_surf = panel["PBR_surf"]
        PBR_true = panel["PBR_true"]
        PER_surf = panel["PER_surf"]
        PER_true = panel["PER_true"]
        VAL_gap = panel["VAL_gap_pct"]
        BLOAT = panel["BLOAT_mat"]
        INTRINSIC = panel["INTRINSIC_mat"]
        SYNERGY = panel["SYNERGY_mat"]
        ALPHA = panel["ALPHA_mat"]
        FWD_RET = panel["FWD_RET_next"]
        OPM = panel["OPM"]

        M, T = P_act.shape
        labels = PurifiedValuationAndTensorEngine.TIMELINE_LABELS
        step_years = PurifiedValuationAndTensorEngine.STEP_YEARS
        total_years = float(np.sum(step_years))

        strategies = [
            "Ours v3.0 (진성 적정주가+텐서 알파 Top 30 Long-Only)",
            "Ours v3.0 (진성 적정주가+텐서 알파 Top 100 Long-Only)",
            "허상 제거 '진짜 저PBR·저PER' 포트폴리오 (True Value Top 20%)",
            "표면적 '단순 저PBR·저PER' 포트폴리오 (Surface Value Top 20%)",
            "Ours v3.0 (유동성·대형주 필터 현실적 Long-Short)",
            "회계 착시·가치함정 종목군 (Value Trap Bottom 20%)",
            "KRX 전 종목 동일가중 벤치마크 (Market EW)",
        ]
        curve_dict: Dict[str, List[float]] = {s: [100.0] for s in strategies}
        period_rets_dict: Dict[str, List[float]] = {s: [] for s in strategies}
        ic_series: List[Dict[str, object]] = []

        quintile_names = [
            "Q5 (상위 20% · 강력 매수)",
            "Q4 (차상위 20% · 매수)",
            "Q3 (중위 20% · 중립)",
            "Q2 (차하위 20% · 비중축소)",
            "Q1 (하위 20% · 회계착시/고평가)",
        ]
        q_curves: Dict[str, List[float]] = {q: [100.0] for q in quintile_names}
        q_period_rets: Dict[str, List[float]] = {q: [] for q in quintile_names}

        cost_rate = self.cost_bps * 1e-4

        for j in range(T - 1):
            r_next = FWD_RET[:, j]
            valid = np.isfinite(r_next) & (P_act[:, j] > 200.0) & np.isfinite(PBR_surf[:, j])
            idx_v = np.where(valid)[0]
            if len(idx_v) < 50:
                for s in strategies:
                    curve_dict[s].append(curve_dict[s][-1])
                    period_rets_dict[s].append(0.0)
                for q in quintile_names:
                    q_curves[q].append(q_curves[q][-1])
                    q_period_rets[q].append(0.0)
                continue

            r_v = r_next[idx_v]
            alpha_v = ALPHA[idx_v, j]
            gap_v = VAL_gap[idx_v, j]
            bloat_v = BLOAT[idx_v, j]
            n_v = len(idx_v)
            k20 = max(1, int(n_v * 0.20))

            # 표면 Value 스코어 (회계 착시를 무시하고 단순 표면 PBR·PER만 낮은 종목 매수)
            ey_surf_v = np.where(PER_surf[idx_v, j] > 0, 1.0 / np.maximum(PER_surf[idx_v, j], 1.0), -0.05)
            surf_val_score = -0.65 * stats.rankdata(PBR_surf[idx_v, j]) / n_v + 0.35 * stats.rankdata(ey_surf_v) / n_v

            # 허상 제거 진짜 Value 스코어 (진짜 PBR + 진짜 PER + 적정주가 괴리율 결합 및 회계착시 배제)
            ey_true_v = np.where(PER_true[idx_v, j] > 0, 1.0 / np.maximum(PER_true[idx_v, j], 2.0), -0.05)
            true_val_score = (
                0.45 * stats.rankdata(gap_v) / n_v
                - 0.30 * stats.rankdata(PBR_true[idx_v, j]) / n_v
                + 0.25 * stats.rankdata(ey_true_v) / n_v
            )
            true_val_score = np.where(bloat_v > 0.65, true_val_score - 1.0, true_val_score)

            top30_idx = np.argsort(alpha_v)[-min(30, n_v):]
            ret_top30 = float(np.mean(r_v[top30_idx])) - cost_rate

            top100_idx = np.argsort(alpha_v)[-min(100, n_v):]
            ret_top100 = float(np.mean(r_v[top100_idx])) - cost_rate

            true_top_idx = np.argsort(true_val_score)[-k20:]
            ret_true_val = float(np.mean(r_v[true_top_idx])) - cost_rate

            surf_top_idx = np.argsort(surf_val_score)[-k20:]
            ret_surf_val = float(np.mean(r_v[surf_top_idx])) - cost_rate

            order_alpha = np.argsort(alpha_v)
            long_q5 = order_alpha[-k20:]
            short_q1 = order_alpha[:k20]
            short_able = short_q1[P_act[idx_v[short_q1], j] >= 3000.0]
            if len(short_able) < 10:
                short_able = short_q1
            r_short_leg = float(np.mean(np.clip(r_v[short_able], -0.75, 0.45)))
            ret_ls = float(np.mean(r_v[long_q5]) - r_short_leg) - 2.0 * cost_rate

            # 회계 착시·가치함정 종목군 (표면적으로는 PBR이 낮거나 PER이 낮아 보이나 적정주가 괴리가 음수이고 회계착시가 큰 종목)
            trap_score = -0.5 * stats.rankdata(VAL_gap[idx_v, j]) / n_v + 0.5 * stats.rankdata(bloat_v) / n_v
            trap_idx = np.argsort(trap_score)[-k20:]
            ret_trap = float(np.mean(r_v[trap_idx])) - cost_rate

            ret_mkt = float(np.mean(r_v))

            step_map = {
                strategies[0]: ret_top30,
                strategies[1]: ret_top100,
                strategies[2]: ret_true_val,
                strategies[3]: ret_surf_val,
                strategies[4]: ret_ls,
                strategies[5]: ret_trap,
                strategies[6]: ret_mkt,
            }
            for s, r_step in step_map.items():
                period_rets_dict[s].append(r_step)
                curve_dict[s].append(round(curve_dict[s][-1] * (1.0 + r_step), 2))

            for q_idx, q_name in enumerate(reversed(quintile_names)):
                start_i = int(round(q_idx * n_v / 5.0))
                end_i = int(round((q_idx + 1) * n_v / 5.0))
                sub_idx = order_alpha[start_i:end_i]
                q_ret = float(np.mean(r_v[sub_idx])) - cost_rate
                q_period_rets[q_name].append(q_ret)
                q_curves[q_name].append(round(q_curves[q_name][-1] * (1.0 + q_ret), 2))

            ic_surf_pbr = float(stats.spearmanr(-PBR_surf[idx_v, j], r_v).statistic)
            ic_true_pbr = float(stats.spearmanr(-PBR_true[idx_v, j], r_v).statistic)
            ic_val_gap = float(stats.spearmanr(gap_v, r_v).statistic)
            ic_alpha = float(stats.spearmanr(alpha_v, r_v).statistic)
            ic_series.append({
                "period": f"{labels[j]} → {labels[j+1]}",
                "ic_surface_pbr": round(ic_surf_pbr, 4),
                "ic_true_pbr": round(ic_true_pbr, 4),
                "ic_fair_price_gap": round(ic_val_gap, 4),
                "ic_alpha_v3": round(ic_alpha, 4),
            })

        def _summarize_curve(curve: List[float], rets: List[float], bench_rets: List[float]) -> Dict[str, float]:
            cum_ret = (curve[-1] / 100.0) - 1.0
            cagr = (max(0.05, curve[-1] / 100.0) ** (1.0 / total_years)) - 1.0
            arr = np.array(rets, dtype=np.float64)
            ppy = len(arr) / total_years
            ann_vol = float(np.std(arr) * np.sqrt(ppy) + 1e-6)
            sharpe = float((np.mean(arr) * ppy) / ann_vol)
            arr_curve = np.array(curve, dtype=np.float64)
            peak = np.maximum.accumulate(arr_curve)
            mdd = float(np.min((arr_curve - peak) / np.maximum(peak, 1e-6)))
            hit_vs_mkt = float(np.mean(arr > np.array(bench_rets)) * 100.0)
            return {
                "Cum_Return(%)": round(cum_ret * 100.0, 2),
                "CAGR(%)": round(cagr * 100.0, 2),
                "Ann_Vol(%)": round(ann_vol * 100.0, 2),
                "Sharpe_Ratio": round(sharpe, 2),
                "Max_Drawdown(%)": round(mdd * 100.0, 2),
                "Win_Rate_vs_Mkt(%)": round(hit_vs_mkt, 1),
                "Final_NAV": round(curve[-1], 1),
            }

        mkt_rets = period_rets_dict["KRX 전 종목 동일가중 벤치마크 (Market EW)"]
        strategy_table = [{"Strategy": s, **_summarize_curve(curve_dict[s], period_rets_dict[s], mkt_rets)} for s in strategies]
        quintile_table = [{"Quintile": q, **_summarize_curve(q_curves[q], q_period_rets[q], mkt_rets)} for q in quintile_names]

        # Study 2: 표면 PER/PBR vs 진짜 PER/PBR 4분면 실증 매트릭스
        quad_records: Dict[str, List[float]] = {
            "Q1_True_Gem (표면 저PBR & 진짜 저PBR 진성가치)": [],
            "Q2_Hidden_Quality (표면 고PBR이나 고수익·무형가치로 진짜 저평가)": [],
            "Q3_Fair_Priced (표면·진짜 모두 일반 적정/고평가 구간)": [],
            "Q4_Value_Trap (표면 저PBR·착시이나 회계허상 제거 시 고평가·부실)": [],
        }
        gap_bucket_records: Dict[str, List[float]] = {
            "심각한 고평가 (적정주가 괴리 -25% 이하)": [],
            "다소 고평가 (괴리 -25% ~ -5%)": [],
            "적정 주가 구간 (괴리 -5% ~ +15%)": [],
            "저평가 우량 (괴리 +15% ~ +40%)": [],
            "강력 저평가 (적정주가 괴리 +40% 이상)": [],
        }

        for j in range(2, T - 1):
            r_next = FWD_RET[:, j] * 100.0
            valid = np.isfinite(r_next) & (P_act[:, j] > 200.0) & np.isfinite(PBR_surf[:, j])
            idx_v = np.where(valid)[0]
            if len(idx_v) < 50:
                continue

            pbr_s = PBR_surf[idx_v, j]
            pbr_t = PBR_true[idx_v, j]
            bloat_v = BLOAT[idx_v, j]
            gap_v = VAL_gap[idx_v, j]
            opm_v = OPM[idx_v, j]
            r_v = r_next[idx_v]

            med_s = np.median(pbr_s)
            med_t = np.median(pbr_t)

            for k_i in range(len(idx_v)):
                ps, pt, bl, gp, opm_val, rv = pbr_s[k_i], pbr_t[k_i], bloat_v[k_i], gap_v[k_i], opm_v[k_i], r_v[k_i]
                if ps <= med_s and pt <= med_t and gp >= 15.0 and bl <= 0.25:
                    quad_records["Q1_True_Gem (표면 저PBR & 진짜 저PBR 진성가치)"].append(rv)
                elif ps > med_s and gp >= 10.0 and opm_val >= 8.0:
                    quad_records["Q2_Hidden_Quality (표면 고PBR이나 고수익·무형가치로 진짜 저평가)"].append(rv)
                elif ps <= med_s * 1.15 and (gp <= -15.0 or (bl >= 0.70 and opm_val < 0)):
                    quad_records["Q4_Value_Trap (표면 저PBR·착시이나 회계허상 제거 시 고평가·부실)"].append(rv)
                else:
                    quad_records["Q3_Fair_Priced (표면·진짜 모두 일반 적정/고평가 구간)"].append(rv)

                if gp < -25.0:
                    gap_bucket_records["심각한 고평가 (적정주가 괴리 -25% 이하)"].append(rv)
                elif gp < -5.0:
                    gap_bucket_records["다소 고평가 (괴리 -25% ~ -5%)"].append(rv)
                elif gp < 15.0:
                    gap_bucket_records["적정 주가 구간 (괴리 -5% ~ +15%)"].append(rv)
                elif gp < 40.0:
                    gap_bucket_records["저평가 우량 (괴리 +15% ~ +40%)"].append(rv)
                else:
                    gap_bucket_records["강력 저평가 (적정주가 괴리 +40% 이상)"].append(rv)

        quadrant_summary = []
        for q_label, vals in quad_records.items():
            arr = np.array(vals, dtype=np.float64) if vals else np.zeros(1)
            quadrant_summary.append({
                "Group": q_label,
                "Sample_Count": len(vals),
                "Mean_Period_Return(%)": round(float(np.mean(arr)), 2),
                "Median_Period_Return(%)": round(float(np.median(arr)), 2),
                "Positive_Hit_Rate(%)": round(float(np.mean(arr > 0) * 100.0), 1),
            })

        convergence_summary = []
        for b_label, vals in gap_bucket_records.items():
            arr = np.array(vals, dtype=np.float64) if vals else np.zeros(1)
            convergence_summary.append({
                "Valuation_Gap_Bucket": b_label,
                "Sample_Count": len(vals),
                "Mean_Next_Return(%)": round(float(np.mean(arr)), 2),
                "Median_Next_Return(%)": round(float(np.median(arr)), 2),
                "Win_Rate(%)": round(float(np.mean(arr > 0) * 100.0), 1),
            })

        ablation_defs = [
            ("1. 표면적 저PBR 단독 (-Surface PBR)", lambda j, iv: -PBR_surf[iv, j]),
            ("2. 허상 제거 진짜 저PBR (-True PBR)", lambda j, iv: -PBR_true[iv, j]),
            ("3. 회계 착시 회피 단독 (-Accounting Bloat b_m)", lambda j, iv: -BLOAT[iv, j]),
            ("4. 텐서 진성 포텐셜 단독 (Intrinsic Potential)", lambda j, iv: INTRINSIC[iv, j]),
            ("5. 진성 적정주가 괴리율 (Fair Price Gap)", lambda j, iv: VAL_gap[iv, j]),
            ("6. Ours v3.0 통합 알파 (Full Tensor + True Valuation)", lambda j, iv: ALPHA[iv, j]),
        ]
        ablation_rows = []
        for name_f, fn_score in ablation_defs:
            ics = []
            top_rets = []
            bot_rets = []
            for j in range(2, T - 1):
                r_next = FWD_RET[:, j]
                valid = np.isfinite(r_next) & (P_act[:, j] > 200.0) & np.isfinite(PBR_surf[:, j])
                iv = np.where(valid)[0]
                if len(iv) < 50:
                    continue
                sc = fn_score(j, iv)
                rv = r_next[iv]
                ic_val = stats.spearmanr(sc, rv).statistic
                if np.isfinite(ic_val):
                    ics.append(float(ic_val))
                k20 = max(1, int(len(iv) * 0.20))
                ord_i = np.argsort(sc)
                top_rets.append(float(np.mean(rv[ord_i[-k20:]])))
                bot_rets.append(float(np.mean(rv[ord_i[:k20]])))

            top_cum = (np.prod([1.0 + x for x in top_rets]) - 1.0) * 100.0 if top_rets else 0.0
            bot_cum = (np.prod([1.0 + x for x in bot_rets]) - 1.0) * 100.0 if bot_rets else 0.0
            ablation_rows.append({
                "Factor_Model": name_f,
                "Mean_Rank_IC": round(float(np.mean(ics)), 4) if ics else 0.0,
                "IC_IR": round(float(np.mean(ics) / (np.std(ics) + 1e-6)), 2) if ics else 0.0,
                "Top20_Cum_Return(%)": round(float(top_cum), 2),
                "Bottom20_Cum_Return(%)": round(float(bot_cum), 2),
                "Long_Short_Spread(%p)": round(float(top_cum - bot_cum), 2),
            })

        return {
            "timeline_labels": labels,
            "strategy_curves": curve_dict,
            "strategy_table": strategy_table,
            "quintile_curves": q_curves,
            "quintile_table": quintile_table,
            "ic_series": ic_series,
            "quadrant_summary": quadrant_summary,
            "convergence_summary": convergence_summary,
            "ablation_table": ablation_rows,
        }


# ==============================================================================
# 5. 추천 포트폴리오 빌더 & 최신 전 종목 예측/번들 생성
# ==============================================================================

def classify_valuation_regime(
    surf_pbr: float,
    true_pbr: float,
    val_gap_pct: float,
    bloat_z: float,
    true_roe: float,
    op_margin: float,
) -> str:
    if op_margin < -10.0 and bloat_z > 0.8:
        return "자본잠식·한계부실 위험"
    if surf_pbr <= 1.1 and (true_pbr > surf_pbr * 1.35 or bloat_z > 0.65):
        return "가치 함정 (Value Trap · 착시주의)"
    if bloat_z > 0.75 and val_gap_pct < 0:
        return "일회성 이익·레버리지 착시"
    if val_gap_pct >= 20.0 and true_roe >= 9.0 and bloat_z <= 0.2:
        return "진성 저평가 우량주 (True Gem)"
    if surf_pbr > 1.3 and true_pbr < surf_pbr * 0.90 and true_roe >= 12.0:
        return "숨겨진 고수익 우량주 (Hidden Quality)"
    if val_gap_pct >= 10.0:
        return "실질 가치 대비 저평가"
    if val_gap_pct <= -20.0:
        return "펀더멘털 대비 고평가 과열"
    return "적정 가치 등락 구간"


def _build_stock_thesis(row: pd.Series, is_short: bool = False) -> str:
    """종목의 표면 vs 진짜 지표 및 텐서 알파 요인을 바탕으로 맞춤형 투자 근거 문장을 생성합니다."""
    spbr = float(row["surface_pbr"])
    tpbr = float(row["true_pbr"])
    sper = float(row["surface_per"])
    tper = float(row["true_per"])
    sroe = float(row["surface_roe"])
    troe = float(row["true_roe"])
    opm = float(row["op_margin"])
    debt = float(row["debt_ratio"])
    gap = float(row["val_gap_pct"])
    bloat = float(row["accounting_bloat"])

    if is_short:
        reasons = []
        if opm < 0:
            reasons.append(f"본업 영업적자({opm:.1f}%) 지속")
        if debt >= 200.0:
            reasons.append(f"과다부채(부채비율 {debt:.0f}%)")
        if sroe > troe + 15.0:
            reasons.append(f"표면 ROE({sroe:.1f}%) 대비 진성 ROE({troe:.1f}%) 급감(착시)")
        if tpbr > spbr * 1.3:
            reasons.append(f"표면 PBR {spbr:.2f}배 → 허상 제거 진짜 PBR {tpbr:.2f}배")
        reasons.append(f"진성 적정주가 대비 {gap:.1f}% 고평가(b_m={bloat:+.2f})")
        return " · ".join(reasons[:3])

    pts = []
    if tpbr < spbr:
        pts.append(f"표면 PBR {spbr:.2f}배 → 현금·고마진 반영 진짜 PBR {tpbr:.2f}배")
    else:
        pts.append(f"진짜 PBR {tpbr:.2f}배 · 진짜 PER {tper:.1f}배")
    pts.append(f"진성 ROE {troe:.1f}%(영업이익률 {opm:.1f}%, 부채 {debt:.0f}%)")
    pts.append(f"적정주가({int(row['fair_price']):,}원) 대비 +{gap:.1f}% 상승여력")
    return " · ".join(pts)


def build_recommended_portfolios(df_out: pd.DataFrame) -> Dict[str, object]:
    """
    우리 로직(6단계 텐서 역변환 + 허상 제거 진성 밸류에이션)에 기반한
    4대 추천 모델 포트폴리오와 1대 가치함정 경계(Avoid/Short) 리스트를 생성합니다.
    """
    def _pack_portfolio(sub_df: pd.DataFrame, title: str, subtitle: str, badge: str, is_short: bool = False) -> Dict[str, object]:
        sub = sub_df.copy().reset_index(drop=True)
        n = len(sub)
        if n == 0:
            return {"title": title, "subtitle": subtitle, "badge": badge, "items": [], "summary": {}}

        if is_short:
            raw_w = np.ones(n, dtype=np.float64)
        else:
            # 알파 스코어와 적정주가 상승여력에 비례하고 마할라노비스 거리(OOD 리스크)에 반비례하는 최적 가중치
            score_pos = np.maximum(0.2, sub["alpha_score"].to_numpy() - sub["alpha_score"].min() + 0.5)
            gap_pos = np.sqrt(np.maximum(5.0, sub["val_gap_pct"].to_numpy()))
            risk_pen = 1.0 + 0.08 * np.maximum(0.0, sub["mahal_ood_dist2"].to_numpy())
            raw_w = (score_pos * gap_pos) / risk_pen
            # 단일 종목 최대 비중 상한 캡 (Top 20은 최대 8%, Top 15는 최대 10%)
            max_cap = 0.08 if n >= 18 else 0.10
            raw_w /= np.sum(raw_w)
            for _ in range(5):
                over = raw_w > max_cap
                if not np.any(over):
                    break
                raw_w = np.minimum(raw_w, max_cap)
                raw_w /= np.sum(raw_w)

        weights_pct = np.round(raw_w / np.sum(raw_w) * 100.0, 2)
        # 반올림 오차를 1위 종목에서 보정해 정확히 100.00% 보장
        weights_pct[0] = round(float(weights_pct[0] + (100.0 - np.sum(weights_pct))), 2)

        items = []
        for idx, row in sub.iterrows():
            w_val = float(weights_pct[idx])
            items.append({
                "port_rank": idx + 1,
                "overall_rank": int(row["rank"]),
                "ticker": str(row["ticker"]),
                "code": str(row["ticker"]).split(".")[0],
                "name": str(row["name"]),
                "market": str(row["market"]),
                "weight_pct": w_val,
                "last_price": float(row["last_price"]),
                "fair_price": float(row["fair_price"]),
                "val_gap_pct": float(row["val_gap_pct"]),
                "target_price_6m": float(row["target_price_6m"]),
                "target_price_12m": float(row["target_price_12m"]),
                "exp_return_6m_pct": float(row["exp_return_6m_pct"]),
                "exp_return_12m_pct": float(row["exp_return_12m_pct"]),
                "exp_alpha_12m_pct": float(row["exp_alpha_12m_pct"]),
                "surface_pbr": float(row["surface_pbr"]),
                "true_pbr": float(row["true_pbr"]),
                "surface_per": float(row["surface_per"]),
                "true_per": float(row["true_per"]),
                "surface_roe": float(row["surface_roe"]),
                "true_roe": float(row["true_roe"]),
                "op_margin": float(row["op_margin"]),
                "debt_ratio": float(row["debt_ratio"]),
                "alpha_score": float(row["alpha_score"]),
                "accounting_bloat": float(row["accounting_bloat"]),
                "synergy_momentum": float(row["synergy_momentum"]),
                "signal": str(row["signal"]),
                "valuation_regime": str(row["valuation_regime"]),
                "thesis": _build_stock_thesis(row, is_short=is_short),
            })

        w_arr = weights_pct / 100.0
        summary = {
            "count": n,
            "weighted_exp_12m_pct": round(float(np.sum(w_arr * sub["exp_return_12m_pct"])), 2),
            "weighted_exp_6m_pct": round(float(np.sum(w_arr * sub["exp_return_6m_pct"])), 2),
            "weighted_val_gap_pct": round(float(np.sum(w_arr * sub["val_gap_pct"])), 2),
            "weighted_surface_pbr": round(float(np.sum(w_arr * sub["surface_pbr"])), 2),
            "weighted_true_pbr": round(float(np.sum(w_arr * sub["true_pbr"])), 2),
            "weighted_true_roe": round(float(np.sum(w_arr * sub["true_roe"])), 2),
            "weighted_op_margin": round(float(np.sum(w_arr * sub["op_margin"])), 2),
            "weighted_debt_ratio": round(float(np.sum(w_arr * sub["debt_ratio"])), 1),
            "kospi_count": int((sub["market"] == "KOSPI").sum()),
            "kosdaq_count": int((sub["market"] == "KOSDAQ").sum()),
        }
        return {
            "title": title,
            "subtitle": subtitle,
            "badge": badge,
            "is_short": is_short,
            "summary": summary,
            "items": items,
        }

    # 1) 대표 추천: v3 텐서 알파 핵심 포트폴리오 Top 20 (리츠·특수종목 제외한 정통 우량 사업법인 중심)
    clean_mask = (
        (df_out["accounting_bloat"] <= 0.15)
        & (df_out["true_roe"] >= 8.0)
        & (df_out["op_margin"] >= 7.0)
        & (df_out["debt_ratio"] <= 160.0)
        & (df_out["val_gap_pct"] >= 15.0)
        & (~df_out["name"].str.contains("리츠|스팩|인버스", regex=True))
    )
    core_top20 = df_out[clean_mask].sort_values("alpha_score", ascending=False).head(20)

    # 2) 딥밸류 추천: 허상 제거 '진짜 저PBR·저PER' 포트폴리오 Top 15
    deep_val_mask = (
        clean_mask
        & (df_out["true_pbr"] <= 0.75)
        & (df_out["true_per"] > 0)
        & (df_out["true_per"] <= 8.5)
        & (df_out["pbr_illusion_pct"] <= 0.0)
    )
    deep_val_top15 = df_out[deep_val_mask].sort_values(["val_gap_pct", "alpha_score"], ascending=False).head(15)

    # 3) 고수익 시너지 추천: 고마진·시너지 모멘텀 우량주 Top 15
    qual_syn_mask = (
        (df_out["accounting_bloat"] <= 0.10)
        & (df_out["true_roe"] >= 15.0)
        & (df_out["op_margin"] >= 15.0)
        & (df_out["val_gap_pct"] >= 10.0)
        & (~df_out["name"].str.contains("리츠|스팩", regex=True))
    )
    qual_syn_top15 = df_out[qual_syn_mask].sort_values(["synergy_momentum", "alpha_score"], ascending=False).head(15)

    # 4) KOSPI 안정형 추천: KOSPI 핵심 우량주 포트폴리오 Top 15
    kospi_mask = clean_mask & (df_out["market"] == "KOSPI")
    kospi_top15 = df_out[kospi_mask].sort_values("alpha_score", ascending=False).head(15)

    # 5) 매수 금지 / 회계 착시 가치함정 경계 리스트 Top 15
    trap_df = df_out.sort_values(["alpha_score", "accounting_bloat"], ascending=[True, False]).head(15)

    return {
        "core_top20": _pack_portfolio(
            core_top20,
            "대표 추천 · v3 텐서 알파 핵심 포트폴리오 (Top 20)",
            "6단계 대수적 텐서 역변환 알파 스코어 최상위 + 회계 착시(b_m) 클린 + 진성 적정주가 대비 저평가된 핵심 매수 20선",
            "CORE ALPHA TOP 20",
        ),
        "true_value_top15": _pack_portfolio(
            deep_val_top15,
            "딥밸류 추천 · 허상 제거 '진짜 저PBR·저PER' 포트폴리오 (Top 15)",
            "장부상 수치보다 현금창출력·순자산 질이 더 우량해 진짜 PBR ≤ 0.75배, 진짜 PER ≤ 8.5배인 진성 저평가 가치주 15선",
            "TRUE DEEP VALUE 15",
        ),
        "synergy_quality_top15": _pack_portfolio(
            qual_syn_top15,
            "고수익·시너지 추천 · 고마진 & 비대각 시너지 우량주 (Top 15)",
            "진성 ROE ≥ 15%, 본업 영업이익률 ≥ 15%이면서 수익성×저평가 비대각 시너지가 극대화된 우량 성장주 15선",
            "QUALITY SYNERGY 15",
        ),
        "kospi_core_top15": _pack_portfolio(
            kospi_top15,
            "KOSPI 안정형 추천 · 유가증권시장 핵심 저평가 우량주 (Top 15)",
            "KOSPI 상장 보통주 중 재무 건전성, 본업 현금창출력, 적정주가 상승여력이 검증된 안정형 포트폴리오 15선",
            "KOSPI CORE 15",
        ),
        "value_trap_avoid15": _pack_portfolio(
            trap_df,
            "회계 착시 · 가치함정 및 부실 위험 경계 리스트 (Bottom 15)",
            "표면 ROE나 표면 멀티플 착시가 있거나 자본잠식·영업적자·과다부채로 적정주가가 심각하게 훼손된 비중축소/매수금지 15선",
            "VALUE TRAP WARNING",
            is_short=True,
        ),
    }


def build_forecasts_and_persist(
    store: LocalQuantStore,
    panel: Dict[str, object],
    bt_results: Dict[str, object],
    save_csv_path: str = "krx_all_stocks_v3_true_valuation.csv",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]], Dict[str, object]]:
    tickers = panel["tickers"]
    meta_df = panel["meta_df"].set_index("ticker")
    P_act = panel["P_act"]
    P_fair = panel["P_fair"]
    VAL_gap = panel["VAL_gap_pct"]
    PBR_surf = panel["PBR_surf"]
    PBR_true = panel["PBR_true"]
    PBR_ill = panel["PBR_illusion_pct"]
    PER_surf = panel["PER_surf"]
    PER_true = panel["PER_true"]
    PER_ill = panel["PER_illusion_gap"]
    ROE_surf = panel["ROE_surf"]
    ROE_true = panel["ROE_true"]
    OPM = panel["OPM"]
    NETM = panel["NETM"]
    DEBT = panel["DEBT"]
    BPS_surf = panel["BPS_surf"]
    BPS_true = panel["BPS_true"]
    EPS_surf = panel["EPS_surf"]
    EPS_true = panel["EPS_true"]
    BLOAT = panel["BLOAT_mat"]
    INTRINSIC = panel["INTRINSIC_mat"]
    SYNERGY = panel["SYNERGY_mat"]
    ALPHA = panel["ALPHA_mat"]
    MAHAL = panel["MAHAL_mat"]
    FWD_RET = panel["FWD_RET_next"]
    TIER = panel["TIER"]

    M, T = P_act.shape
    yms = PurifiedValuationAndTensorEngine.TIMELINE_YMS
    labels = PurifiedValuationAndTensorEngine.TIMELINE_LABELS
    last_j = T - 1

    alpha_last = ALPHA[:, last_j]
    alpha_z = (alpha_last - np.mean(alpha_last)) / (np.std(alpha_last) + 1e-8)
    gap_last = VAL_gap[:, last_j] / 100.0

    mkt_drift_12m = 0.065
    raw_12m = mkt_drift_12m + 0.32 * np.tanh(gap_last / 0.90) + 0.12 * np.tanh(alpha_z / 2.8)
    exp_ret_12m = np.clip(raw_12m, -0.36, 0.48)
    exp_ret_6m = (1.0 + exp_ret_12m) ** 0.5 - 1.0
    exp_alpha_12m = exp_ret_12m - mkt_drift_12m

    cur_prices = P_act[:, last_j]
    target_6m = np.round(cur_prices * (1.0 + exp_ret_6m), 0)
    target_12m = np.round(cur_prices * (1.0 + exp_ret_12m), 0)

    q80, q60, q40, q20 = np.percentile(alpha_last, [80, 60, 40, 20])
    signals = []
    regimes = []
    for i in range(M):
        s = alpha_last[i]
        if s >= q80:
            signals.append("STRONG LONG (최우선 매수)")
        elif s >= q60:
            signals.append("LONG (매수)")
        elif s >= q40:
            signals.append("NEUTRAL (중립)")
        elif s >= q20:
            signals.append("SHORT (비중축소/공매도)")
        else:
            signals.append("STRONG SHORT (핵심 공매도)")

        regimes.append(
            classify_valuation_regime(
                float(PBR_surf[i, last_j]),
                float(PBR_true[i, last_j]),
                float(VAL_gap[i, last_j]),
                float(BLOAT[i, last_j]),
                float(ROE_true[i, last_j]),
                float(OPM[i, last_j]),
            )
        )

    ts_db_rows: List[Tuple] = []
    chart_series_by_tk: Dict[str, Dict[str, object]] = {}

    for i, tk in enumerate(tickers):
        for j in range(T):
            fwd_v = None if np.isnan(FWD_RET[i, j]) else round(float(FWD_RET[i, j] * 100.0), 2)
            ts_db_rows.append((
                tk,
                yms[j],
                labels[j],
                TIER[i][j],
                round(float(P_act[i, j]), 0),
                round(float(P_fair[i, j]), 0),
                round(float(VAL_gap[i, j]), 2),
                round(float(PBR_surf[i, j]), 2),
                round(float(PBR_true[i, j]), 2),
                round(float(PBR_ill[i, j]), 1),
                round(float(PER_surf[i, j]), 2),
                round(float(PER_true[i, j]), 2),
                round(float(PER_ill[i, j]), 2),
                round(float(ROE_surf[i, j]), 2),
                round(float(ROE_true[i, j]), 2),
                round(float(OPM[i, j]), 2),
                round(float(NETM[i, j]), 2),
                round(float(DEBT[i, j]), 1),
                round(float(BPS_surf[i, j]), 0),
                round(float(BPS_true[i, j]), 0),
                round(float(EPS_surf[i, j]), 0),
                round(float(EPS_true[i, j]), 0),
                round(float(BLOAT[i, j]), 3),
                round(float(INTRINSIC[i, j]), 3),
                round(float(SYNERGY[i, j]), 3),
                round(float(ALPHA[i, j]), 3),
                fwd_v,
            ))

        for f_ym, f_lbl, f_p in (
            ("202704", "2027.04(E)", float(target_6m[i])),
            ("202710", "2027.10(E)", float(target_12m[i])),
        ):
            f_fair = round(float(P_fair[i, last_j] * (1.0 + 0.04 * (1 if f_ym == "202704" else 2))), 0)
            ts_db_rows.append((
                tk, f_ym, f_lbl, "FORECAST",
                round(f_p, 0), f_fair, round((f_fair / max(1.0, f_p) - 1.0) * 100.0, 2),
                round(float(PBR_surf[i, last_j]), 2), round(float(PBR_true[i, last_j]), 2),
                round(float(PBR_ill[i, last_j]), 1),
                round(float(PER_surf[i, last_j]), 2), round(float(PER_true[i, last_j]), 2),
                round(float(PER_ill[i, last_j]), 2),
                round(float(ROE_surf[i, last_j]), 2), round(float(ROE_true[i, last_j]), 2),
                round(float(OPM[i, last_j]), 2), round(float(NETM[i, last_j]), 2),
                round(float(DEBT[i, last_j]), 1),
                round(float(BPS_surf[i, last_j]), 0), round(float(BPS_true[i, last_j]), 0),
                round(float(EPS_surf[i, last_j]), 0), round(float(EPS_true[i, last_j]), 0),
                round(float(BLOAT[i, last_j]), 3), round(float(INTRINSIC[i, last_j]), 3),
                round(float(SYNERGY[i, last_j]), 3), round(float(ALPHA[i, last_j]), 3),
                None,
            ))

        chart_series_by_tk[tk] = {
            "p": [int(round(x)) for x in P_act[i, :]],
            "fp": [int(round(x)) for x in P_fair[i, :]],
            "t6": int(round(target_6m[i])),
            "t12": int(round(target_12m[i])),
            "spbr": [round(float(x), 2) for x in PBR_surf[i, :]],
            "tpbr": [round(float(x), 2) for x in PBR_true[i, :]],
            "sper": [round(float(np.clip(x, -80.0, 120.0)), 1) for x in PER_surf[i, :]],
            "tper": [round(float(np.clip(x, -80.0, 120.0)), 1) for x in PER_true[i, :]],
            "sroe": [round(float(np.clip(x, -60.0, 80.0)), 1) for x in ROE_surf[i, :]],
            "troe": [round(float(np.clip(x, -60.0, 80.0)), 1) for x in ROE_true[i, :]],
            "opm": [round(float(x), 1) for x in OPM[i, :]],
            "netm": [round(float(x), 1) for x in NETM[i, :]],
            "debt": [round(float(x), 0) for x in DEBT[i, :]],
            "bps_s": [int(round(x)) for x in BPS_surf[i, :]],
            "bps_t": [int(round(x)) for x in BPS_true[i, :]],
            "bl": [round(float(x), 2) for x in BLOAT[i, :]],
            "al": [round(float(x), 2) for x in ALPHA[i, :]],
        }

    store.replace_valuation_timeseries(ts_db_rows)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_out = pd.DataFrame({
        "ticker": tickers,
        "name": [str(meta_df.loc[t, "name"]) for t in tickers],
        "market": [str(meta_df.loc[t, "market"]) for t in tickers],
        "last_trade_date": [str(meta_df.loc[t, "last_date"]) for t in tickers],
        "last_price": np.round(cur_prices, 0),
        "fair_price": np.round(P_fair[:, last_j], 0),
        "val_gap_pct": np.round(VAL_gap[:, last_j], 2),
        "target_price_6m": target_6m,
        "target_price_12m": target_12m,
        "exp_return_6m_pct": np.round(exp_ret_6m * 100.0, 2),
        "exp_return_12m_pct": np.round(exp_ret_12m * 100.0, 2),
        "exp_alpha_12m_pct": np.round(exp_alpha_12m * 100.0, 2),
        "signal": signals,
        "valuation_regime": regimes,
        "surface_pbr": np.round(PBR_surf[:, last_j], 2),
        "true_pbr": np.round(PBR_true[:, last_j], 2),
        "pbr_illusion_pct": np.round(PBR_ill[:, last_j], 1),
        "surface_per": np.round(PER_surf[:, last_j], 2),
        "true_per": np.round(PER_true[:, last_j], 2),
        "per_illusion_gap": np.round(PER_ill[:, last_j], 2),
        "surface_roe": np.round(ROE_surf[:, last_j], 2),
        "true_roe": np.round(ROE_true[:, last_j], 2),
        "op_margin": np.round(OPM[:, last_j], 2),
        "debt_ratio": np.round(DEBT[:, last_j], 1),
        "alpha_score": np.round(ALPHA[:, last_j], 3),
        "intrinsic_potential": np.round(INTRINSIC[:, last_j], 3),
        "accounting_bloat": np.round(BLOAT[:, last_j], 3),
        "synergy_momentum": np.round(SYNERGY[:, last_j], 3),
        "mahal_ood_dist2": np.round(MAHAL[:, last_j], 2),
        "updated_at": [now_str] * M,
    })

    df_out = df_out.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "rank", np.arange(1, len(df_out) + 1))

    portfolios = build_recommended_portfolios(df_out)

    store.replace_stock_forecasts(df_out)
    store.save_backtest_report("latest_v3_backtest", bt_results)
    store.save_backtest_report("latest_v3_portfolios", portfolios)

    if save_csv_path:
        df_out.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[Output Saved v3] 전 종목 진성 밸류에이션 & 예상 수익률 CSV 저장 완료: '{save_csv_path}'")

    return df_out, chart_series_by_tk, portfolios


def export_v3_payloads(
    df_out: pd.DataFrame,
    bt_results: Dict[str, object],
    chart_series_by_tk: Dict[str, Dict[str, object]],
    portfolios: Dict[str, object],
    log_dir: str = "logs_v3",
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    bundle = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_stocks": int(len(df_out)),
        "timeline_yms": PurifiedValuationAndTensorEngine.TIMELINE_YMS,
        "timeline_labels": PurifiedValuationAndTensorEngine.TIMELINE_LABELS,
        "forecast_labels": ["2027.04(E)", "2027.10(E)"],
        "portfolios": portfolios,
        "backtest": bt_results,
        "stocks": json.loads(df_out.to_json(orient="records", force_ascii=False)),
        "series": chart_series_by_tk,
    }
    out_path = os.path.join(log_dir, "v3_webapp_bundle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nan_to_null(bundle), f, ensure_ascii=False, allow_nan=False)
    print(f"[Bundle Saved v3] 웹앱 및 차트 시계열 데이터 번들 저장 완료: '{out_path}'")
    return out_path


def run_v3_pipeline(sync_live: bool = False, db_path: str = "data/krx_quant_store.db") -> Dict[str, object]:
    t0 = time.time()
    print("=" * 120)
    print(" AUTO_PORTFOLIO v3.0 - 장기 시계열 백테스트 & 진성 밸류에이션(True PER/PBR vs 표면 PER/PBR) 통합 엔진")
    print("=" * 120)

    store = LocalQuantStore(db_path=db_path)
    if sync_live:
        sync_live_naver_data(store)
    else:
        sync_local_caches_to_store(store)

    engine = PurifiedValuationAndTensorEngine(store=store, latent_dim=4)
    panel = engine.load_panel_from_db()
    panel = engine.compute_purified_valuation_and_tensor_alpha(panel)

    backtester = ComprehensiveQuantBacktester(cost_bps=15.0)
    bt_results = backtester.run_all_studies(panel)

    print("\n--- [Study 1] 장기 시계열 Walk-Forward 전략별 누적 수익률 및 성과 비교 (2021.12 ~ 2026.10) ---")
    print(pd.DataFrame(bt_results["strategy_table"]).to_string(index=False))

    print("\n--- [Study 2] 알파 스코어 5분위(Quintile Q1~Q5) 포트폴리오 단조성 검증 ---")
    print(pd.DataFrame(bt_results["quintile_table"]).to_string(index=False))

    print("\n--- [Study 3] 표면적 PBR/PER vs 허상 제거 진짜 PBR/PER 4분면(Value Trap 검증) 실증 성과 ---")
    print(pd.DataFrame(bt_results["quadrant_summary"]).to_string(index=False))

    print("\n--- [Study 4] 진성 적정주가 괴리율((Fair_Price - Price)/Price) 구간별 향후 실제 주가 수렴성 ---")
    print(pd.DataFrame(bt_results["convergence_summary"]).to_string(index=False))

    print("\n--- [Study 5] 팩터 분해(Ablation Study): 표면 지표 vs 진짜 지표 vs 통합 텐서 알파 ---")
    print(pd.DataFrame(bt_results["ablation_table"]).to_string(index=False))

    df_out, chart_series, portfolios = build_forecasts_and_persist(store, panel, bt_results)
    bundle_path = export_v3_payloads(df_out, bt_results, chart_series, portfolios)

    core_items = portfolios["core_top20"]["items"]
    print("\n--- [대표 추천 포트폴리오] 우리 로직 기반 핵심 Top 10 종목 요약 ---")
    print(
        pd.DataFrame(core_items[:10])[
            ["port_rank", "ticker", "name", "market", "weight_pct", "last_price", "fair_price", "val_gap_pct", "surface_pbr", "true_pbr", "true_roe", "exp_return_12m_pct"]
        ].to_string(index=False)
    )

    print(f"\n[v3 Pipeline Complete] 총 소요시간: {time.time() - t0:.2f}초")
    return {
        "df_out": df_out,
        "bt_results": bt_results,
        "portfolios": portfolios,
        "bundle_path": bundle_path,
        "db_path": db_path,
    }


def main():
    parser = argparse.ArgumentParser(description="AUTO_PORTFOLIO v3.0 Engine & Local Quant DB Builder")
    parser.add_argument("--sync-live", action="store_true", help="네이버 10년 월봉 시세 및 신규 재무제표 실시간 크롤링 병합")
    parser.add_argument("--db-path", type=str, default="data/krx_quant_store.db", help="로컬 SQLite DB 파일 경로")
    args, _ = parser.parse_known_args()
    run_v3_pipeline(sync_live=args.sync_live, db_path=args.db_path)


if __name__ == "__main__":
    main()
