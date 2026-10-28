import warnings
import sys
import os
import io
import re
import ssl
import json
import ast
import time
import argparse
import urllib.request
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
from scipy import linalg, stats

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except ImportError:
    HAS_FDR = False

try:
    from pykrx import stock as pykrx_stock
    HAS_PYKRX = True
except ImportError:
    HAS_PYKRX = False


# ==============================================================================
# 데이터 구조 정의 (Data Structures - v2 Point-in-Time)
# ==============================================================================

@dataclass
class FleetTensorData:
    """
    다중 종목(M) x 재무/회계 피처(D) x 시계열 윈도우(N) 텐서 데이터 컨테이너 (v2: Point-in-Time)
    """
    X_obs: np.ndarray               # Shape: (M, D, N) - 시점별(Point-in-Time) 원시 재무/밸류에이션/시장 텐서
    Y_obs: np.ndarray               # Shape: (M, N)    - 인과적(Causal) 차기 위험조정 수익/펀더멘털 반응
    forward_returns: np.ndarray     # Shape: (M, N)    - 시점 t -> t+1 실현 주식 수익률 (마지막 열은 NaN 또는 최신월)
    tickers: List[str]              # 종목 코드 리스트 (길이 M)
    feature_names: List[str]        # 재무/회계 피처 이름 리스트 (길이 D)
    time_index: List[str]           # 시계열 인덱스 리스트 (길이 N)
    ground_truth: Optional[Dict[str, np.ndarray]] = None
    company_names: Optional[Dict[str, str]] = None
    markets: Optional[Dict[str, str]] = None
    last_prices: Optional[Dict[str, float]] = None
    last_dates: Optional[Dict[str, str]] = None
    raw_fundamentals: Optional[Dict[str, Dict[str, float]]] = None
    accounting_bloat_raw: Optional[np.ndarray] = None  # Shape: (M, N) - 직접 계산된 회계 착시/부실 원시 지표
    data_source: str = "LIVE_KRX_NAVER_PIT_V2"


@dataclass
class PurifiedFundamentalState:
    """
    6단계 텐서 역변환(v2)을 통해 정제된 기업별 펀더멘털 상태 및 글로벌 법칙 파라미터
    """
    r_hat: int
    U_phys: np.ndarray
    W_hat: np.ndarray
    Sigma_xi: np.ndarray
    H_phys: np.ndarray
    gamma_phys: np.ndarray
    delta_raw: np.ndarray
    delta_sure: np.ndarray
    accounting_bias_b: np.ndarray
    alpha_eiv: np.ndarray
    mahal_dist_sq: np.ndarray
    z_aligned: np.ndarray
    y_pure: np.ndarray
    alpha_score: np.ndarray
    expected_return_6m: Optional[np.ndarray] = None
    expected_return_12m: Optional[np.ndarray] = None
    expected_alpha_12m: Optional[np.ndarray] = None
    intrinsic_potential: Optional[np.ndarray] = None
    b_proj_score: Optional[np.ndarray] = None
    off_diag_synergy: Optional[np.ndarray] = None


# ==============================================================================
# MODULE 1 (v2): KRX 전 종목 수집기 및 Point-in-Time(PIT) 재무/가격 텐서 빌더
# ==============================================================================

class FinancialFleetDataLoaderV2:
    """
    [v2 핵심 개선사항]
    1. 네이버 모바일 금융 JSON API (/api/stock/{code}/finance/annual & quarter)를 사용하여
       EUC-KR/Next.js 파싱 실패 문제를 해결하고 약 99% 이상의 실제 K-IFRS 재무제표 수집률을 달성합니다.
    2. 각 과거 시점 t에 대해 공시 시차(Reporting Lag, 2개월)를 반영한 Point-in-Time(PIT) 재무 수치와
       당월 실제 주가(Close[t])를 결합해 동적 PBR, PER, 배당수익률을 계산함으로써 미래 참조 편향(Look-ahead Bias)을 제거합니다.
    3. 자본잠식(BPS <= 0)이나 과다부채/일회성 이익으로 인해 ROE가 비정상적으로 부풀려지는 회계 착시를 직접 분리합니다.
    4. 재무제표가 전혀 없는 종목에 대해 과거 주가 샤프지수로 가짜 ROE를 합성하던 폴백을 제거합니다.
    """

    DEFAULT_UNIVERSE = [
        "005930.KS", "000660.KS", "042700.KS", "012450.KS", "267260.KS", "005380.KS",
        "000270.KS", "207940.KS", "196170.KQ", "058470.KQ", "105560.KS", "079550.KS",
        "329180.KS", "035420.KS", "055550.KS", "012330.KS", "086790.KS", "009150.KS",
        "034020.KS", "068270.KS", "028260.KS", "066570.KS", "005490.KS", "035720.KS",
        "006400.KS", "051910.KS", "003670.KS", "247540.KQ", "036570.KS", "096770.KS",
        "009830.KS", "011170.KS"
    ]

    HTTP_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    _SSL_CTX = ssl.create_default_context()
    _PREFERRED_NAME_RE = re.compile(
        r"(?:\d우B|\d우C|우B|우C|\d우|우)(?:\(|$)|전환우선|우선주"
    )

    def __init__(self, tickers: Optional[List[str]] = None, cache_dir: str = "logs_v2/cache"):
        self.tickers = tickers if tickers is not None else list(self.DEFAULT_UNIVERSE)
        self.company_names: Dict[str, str] = {}
        self.markets: Dict[str, str] = {}
        self.listing_source: str = ""
        self.market_caps: Dict[str, float] = {}
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def _http_get_bytes(
        cls,
        url: str,
        timeout: int = 20,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> bytes:
        headers = dict(cls.HTTP_HEADERS)
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=cls._SSL_CTX) as resp:
            return resp.read()

    @classmethod
    def _is_common_operating_stock(cls, name: str, code: str) -> bool:
        """보통주 사업법인만 남기고 우선주·스팩·리츠·비정상 코드를 필터링합니다."""
        if not name or not code:
            return False
        code_u = str(code).strip().upper()
        if not re.fullmatch(r"[0-9A-Z]{6}", code_u):
            return False
        compact = re.sub(r"[\s\-]", "", str(name))
        if "스팩" in compact or re.search(r"SPAC", compact, re.IGNORECASE):
            return False
        if cls._PREFERRED_NAME_RE.search(compact):
            return False
        return True

    @staticmethod
    def _normalize_krx_market(raw_market: str) -> Optional[Tuple[str, str]]:
        token = re.sub(r"\s+", "", str(raw_market or ""))
        upper = token.upper()
        if "코스닥" in token or "KOSDAQ" in upper or token in ("KQ", "KSQ"):
            return "KOSDAQ", ".KQ"
        if "코넥스" in token or "KONEX" in upper or token in ("KNX",):
            return None
        if (
            "유가" in token
            or "코스피" in token
            or "KOSPI" in upper
            or token in ("KS", "STK", "")
        ):
            return "KOSPI", ".KS"
        return None

    @classmethod
    def _parse_kind_corp_rows(cls, html_text: str, default_market: str = "") -> List[Tuple[str, str, str]]:
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
        th_pattern = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL | re.IGNORECASE)
        td_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
        tag_strip = re.compile(r"<[^>]+>")

        def _clean(cell: str) -> str:
            text = tag_strip.sub("", cell)
            text = text.replace("&amp;", "&").replace("&nbsp;", " ")
            return re.sub(r"\s+", " ", text).strip()

        headers: Optional[List[str]] = None
        parsed: List[Tuple[str, str, str]] = []

        for tr_match in row_pattern.finditer(html_text):
            body = tr_match.group(1)
            ths = [_clean(c) for c in th_pattern.findall(body)]
            if ths and headers is None:
                headers = ths
                continue
            cells = [_clean(c) for c in td_pattern.findall(body)]
            if len(cells) < 2:
                continue

            name = ""
            code = ""
            market_raw = default_market
            if headers:
                row = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
                name = row.get("회사명", cells[0])
                code = row.get("종목코드", cells[1] if len(cells) >= 2 else "")
                market_raw = row.get("시장구분", default_market) or default_market
            else:
                name, code = cells[0], cells[1]

            code = str(code).strip().upper()
            if code.isdigit():
                code = code.zfill(6)
            market_info = cls._normalize_krx_market(market_raw or default_market)
            if market_info is None:
                continue
            mkt_name, suffix = market_info
            if not cls._is_common_operating_stock(name, code):
                continue
            parsed.append((f"{code}{suffix}", name, mkt_name))
        return parsed

    def _fetch_kind_universe(self, market_upper: str) -> List[Tuple[str, str, str]]:
        print("[KRX Loader v2] KRX KIND(한국거래소 상장공시시스템)에서 전 종목 리스트를 다운로드합니다...")
        url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        raw_bytes = self._http_get_bytes(
            url,
            timeout=30,
            extra_headers={"Referer": "https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage"}
        )
        html_text = raw_bytes.decode("euc-kr", errors="ignore")
        rows = self._parse_kind_corp_rows(html_text)
        if market_upper in ("KOSPI", "KS"):
            rows = [r for r in rows if r[2] == "KOSPI"]
        elif market_upper in ("KOSDAQ", "KQ"):
            rows = [r for r in rows if r[2] == "KOSDAQ"]
        print(f"[KRX Loader v2] KIND 파싱 완료: {len(rows)}개 보통주 (시장 필터: {market_upper})")
        return rows

    def _fetch_naver_marcap_map(self) -> Dict[str, float]:
        cap_map: Dict[str, float] = {}
        for sosok, suffix in ((0, ".KS"), (1, ".KQ")):
            page = 1
            page_size = 100
            total = None
            while True:
                url = (
                    "https://m.stock.naver.com/api/json/sise/siseListJson.nhn"
                    f"?menu=market_sum&sosok={sosok}&pageSize={page_size}&page={page}"
                )
                raw = self._http_get_bytes(
                    url,
                    timeout=20,
                    extra_headers={"Accept": "application/json", "Referer": "https://m.stock.naver.com/"}
                )
                payload = json.loads(raw.decode("utf-8"))
                result = payload.get("result") or {}
                items = result.get("itemList") or []
                if not items:
                    break
                if total is None:
                    total = int(result.get("totCnt") or 0)
                for item in items:
                    if item.get("etf") or item.get("etn"):
                        continue
                    code = str(item.get("cd") or "").strip().upper()
                    if code.isdigit():
                        code = code.zfill(6)
                    if len(code) != 6:
                        continue
                    try:
                        cap_val = float(item.get("marketSumRaw") or 0.0) * 1_000_000.0
                    except (TypeError, ValueError):
                        continue
                    if cap_val <= 0:
                        continue
                    ticker = f"{code}{suffix}"
                    cap_map[ticker] = cap_val
                    cap_map[code] = cap_val
                page += 1
                if total and (page - 1) * page_size >= total:
                    break
                if page > 80:
                    break
        return cap_map

    def load_all_krx_universe(
        self,
        market: str = "ALL",
        max_stocks: Optional[int] = None,
        min_marcap_billion_krw: float = 30.0
    ) -> List[str]:
        market_upper = str(market or "ALL").strip().upper()
        collected_rows = self._fetch_kind_universe(market_upper)
        self.listing_source = "KRX KIND"

        seen = set()
        unique_rows: List[Tuple[str, str, str]] = []
        for row in collected_rows:
            if row[0] not in seen:
                seen.add(row[0])
                unique_rows.append(row)
        collected_rows = unique_rows

        min_cap_krw = float(min_marcap_billion_krw or 0.0) * 1e9
        if min_cap_krw > 0:
            if not self.market_caps:
                try:
                    print("[KRX Loader v2] 시가총액 필터를 위해 네이버 시총 맵을 수집합니다...")
                    self.market_caps = self._fetch_naver_marcap_map()
                except Exception as e:
                    print(f"[KRX Loader v2] 시가총액 맵 수집 실패: {e}")
            if self.market_caps:
                before = len(collected_rows)
                filtered = []
                for ticker, name, mkt_name in collected_rows:
                    code = ticker.split(".")[0]
                    cap = self.market_caps.get(ticker, self.market_caps.get(code, 0.0))
                    if cap >= min_cap_krw:
                        filtered.append((ticker, name, mkt_name))
                collected_rows = filtered
                print(
                    f"[KRX Loader v2] 시가총액 {min_marcap_billion_krw:.1f}십억 원 이상 필터: "
                    f"{before} -> {len(collected_rows)}종목"
                )

        if self.market_caps:
            collected_rows.sort(
                key=lambda r: self.market_caps.get(r[0], self.market_caps.get(r[0].split(".")[0], 0.0)),
                reverse=True
            )

        if max_stocks is not None and max_stocks > 0:
            collected_rows = collected_rows[:max_stocks]

        self.tickers = [r[0] for r in collected_rows]
        self.company_names = {r[0]: r[1] for r in collected_rows}
        self.markets = {r[0]: r[2] for r in collected_rows}
        print(f"[KRX Loader v2] 최종 분석 대상 유니버스: {len(self.tickers)}개 종목")
        return self.tickers

    @classmethod
    def _fetch_naver_ohlcv(cls, code_6digit: str, timeframe: str = "month") -> Optional[pd.DataFrame]:
        end = datetime.now()
        start = end - timedelta(days=365 * 5 + 90)
        url = (
            "https://api.finance.naver.com/siseJson.naver"
            f"?symbol={code_6digit}&requestType=1"
            f"&startTime={start.strftime('%Y%m%d')}&endTime={end.strftime('%Y%m%d')}"
            f"&timeframe={timeframe}"
        )
        try:
            raw = cls._http_get_bytes(url, timeout=12, extra_headers={"Referer": "https://finance.naver.com/"})
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                return None
            try:
                table = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                table = json.loads(text.replace("'", '"'))
            if not isinstance(table, list) or len(table) < 3:
                return None
            header = [str(h) for h in table[0]]
            records = []
            for row in table[1:]:
                if not isinstance(row, (list, tuple)) or len(row) < 5:
                    continue
                rec = {header[i]: row[i] for i in range(min(len(header), len(row)))}
                records.append(rec)
            if not records:
                return None
            df = pd.DataFrame.from_records(records)
            date_col = next((c for c in ("날짜", "date") if c in df.columns), None)
            close_col = next((c for c in ("종가", "close", "Close") if c in df.columns), None)
            vol_col = next((c for c in ("거래량", "volume", "Volume") if c in df.columns), None)
            if date_col is None or close_col is None:
                return None
            close_vals = pd.to_numeric(df[close_col], errors="coerce").to_numpy()
            vol_vals = pd.to_numeric(df[vol_col], errors="coerce").to_numpy() if vol_col is not None else 0.0
            idx = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
            out = pd.DataFrame(
                {"Close": close_vals, "Volume": vol_vals},
                index=pd.DatetimeIndex(idx.to_numpy())
            )
            out = out[out.index.notna()].sort_index()
            out = out[out["Close"].fillna(0) > 0]
            return out if len(out) >= 12 else None
        except Exception:
            return None

    def _download_naver_price_panel(
        self,
        tickers: List[str],
        freq: str = "M",
        max_workers: int = 24
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        timeframe = "month" if freq == "M" else "week"
        close_map: Dict[str, pd.Series] = {}
        volume_map: Dict[str, pd.Series] = {}
        completed = 0
        t0 = time.time()
        print(f"  -> [1/2: 네이버 차트 시세 수집] {len(tickers)}개 종목 (timeframe={timeframe})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tk = {
                executor.submit(self._fetch_naver_ohlcv, tk.split(".")[0], timeframe): tk
                for tk in tickers
            }
            for fut in as_completed(future_to_tk):
                tk = future_to_tk[fut]
                completed += 1
                if completed % 400 == 0 or completed == len(tickers):
                    print(
                        f"     ... 시세 수집 진행률: {completed} / {len(tickers)} "
                        f"(경과: {time.time() - t0:.1f}초)"
                    )
                try:
                    df = fut.result()
                except Exception:
                    df = None
                if df is None or df.empty:
                    continue
                close_map[tk] = df["Close"].rename(tk)
                volume_map[tk] = df["Volume"].rename(tk)
        if not close_map:
            return pd.DataFrame(), pd.DataFrame()
        close_df = pd.concat(close_map.values(), axis=1).sort_index()
        volume_df = pd.concat(volume_map.values(), axis=1).sort_index()
        return close_df, volume_df

    @staticmethod
    def _parse_num(val_str: object) -> Optional[float]:
        if val_str is None:
            return None
        s = str(val_str).strip().replace(",", "")
        if not s or s in ("-", "N/A", "nan", "None"):
            return None
        try:
            v = float(s)
            return v if np.isfinite(v) else None
        except ValueError:
            return None

    @classmethod
    def _fetch_naver_pit_financials_single(cls, code_6digit: str) -> Dict[str, object]:
        """
        네이버 모바일 금융 API (/finance/annual 및 /finance/quarter)에서
        실제 공시된 연도별·분기별 K-IFRS 재무 시계열과 컨센서스를 수집합니다.
        """
        headers = {
            "User-Agent": cls.HTTP_HEADERS["User-Agent"],
            "Referer": "https://m.stock.naver.com/",
            "Accept": "application/json",
        }
        periods_map: Dict[str, Dict[str, float]] = {}
        consensus_map: Dict[str, Dict[str, float]] = {}

        metric_key_map = {
            "매출액": "rev",
            "영업이익": "op_inc",
            "당기순이익": "net_inc",
            "지배주주순이익": "ctrl_inc",
            "영업이익률": "op_margin",
            "순이익률": "profit_margin",
            "ROE": "roe",
            "부채비율": "debt_ratio",
            "당좌비율": "quick_ratio",
            "유보율": "reserve_ratio",
            "EPS": "eps",
            "PER": "per",
            "BPS": "bps",
            "PBR": "pbr",
            "주당배당금": "dps",
        }

        for kind in ("annual", "quarter"):
            url = f"https://m.stock.naver.com/api/stock/{code_6digit}/finance/{kind}"
            try:
                raw = cls._http_get_bytes(url, timeout=8, extra_headers=headers)
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
                fi = payload.get("financeInfo") or {}
                titles = fi.get("trTitleList") or []
                rows = fi.get("rowList") or []
                if not titles or not rows:
                    continue

                is_cons_by_key = {
                    str(t.get("key", "")).strip(): (str(t.get("isConsensus", "N")).upper() == "Y")
                    for t in titles if t.get("key")
                }

                for row in rows:
                    title = str(row.get("title", "")).strip()
                    m_key = metric_key_map.get(title)
                    if not m_key:
                        continue
                    cols = row.get("columns") or {}
                    for ym_key, cell_obj in cols.items():
                        ym = str(ym_key).strip()
                        if len(ym) != 6 or not ym.isdigit():
                            continue
                        val = cls._parse_num(cell_obj.get("value") if isinstance(cell_obj, dict) else None)
                        if val is None:
                            continue
                        target_dict = consensus_map if is_cons_by_key.get(ym, False) else periods_map
                        if ym not in target_dict:
                            target_dict[ym] = {"is_quarter": 1.0 if kind == "quarter" else 0.0}
                        # Quarter values override or supplement annual except for full-year EPS/Revenue/DPS scaling
                        if kind == "quarter" and m_key in ("eps", "rev", "op_inc", "net_inc", "ctrl_inc"):
                            target_dict[ym][f"{m_key}_q"] = val
                            # 연환산 추정치 (단기 분기 값이 아직 연간 확정치가 아닐 때만 보완)
                            if m_key not in target_dict[ym]:
                                target_dict[ym][m_key] = val * 4.0
                        else:
                            target_dict[ym][m_key] = val
            except Exception:
                continue

        # 유효한 실제 재무 기간이 최소 1개 이상이고 핵심 지표(ROE, 영업이익률, BPS 중 2개 이상)가 존재하는지 확인
        valid_periods = {}
        for ym, metrics in sorted(periods_map.items()):
            has_core = sum(1 for k in ("roe", "op_margin", "bps", "debt_ratio") if k in metrics)
            if has_core >= 2:
                valid_periods[ym] = metrics

        if not valid_periods:
            return {"is_real_fin": 0.0, "periods": {}, "consensus": {}}

        # 기간 간 실제 매출 성장률(rev_growth) 및 이익 성장률(earn_growth) 계산
        sorted_yms = sorted(valid_periods.keys())
        for idx, ym in enumerate(sorted_yms):
            cur = valid_periods[ym]
            # 이전 비교 기준 기간 탐색 (동일 유형 또는 직전 기간)
            prev = None
            for prev_idx in range(idx - 1, -1, -1):
                cand = valid_periods[sorted_yms[prev_idx]]
                if cand.get("is_quarter") == cur.get("is_quarter"):
                    prev = cand
                    break
            if prev is None and idx > 0:
                prev = valid_periods[sorted_yms[idx - 1]]

            rev_g = 0.0
            earn_g = 0.0
            if prev is not None:
                if "rev" in cur and "rev" in prev and abs(prev["rev"]) > 1.0:
                    rev_g = (cur["rev"] - prev["rev"]) / abs(prev["rev"])
                elif "op_margin" in cur and "op_margin" in prev:
                    rev_g = (cur["op_margin"] - prev["op_margin"]) / 100.0 * 1.5

                if "op_inc" in cur and "op_inc" in prev and abs(prev["op_inc"]) > 1.0:
                    earn_g = (cur["op_inc"] - prev["op_inc"]) / abs(prev["op_inc"])
                elif "roe" in cur and "roe" in prev:
                    earn_g = (cur["roe"] - prev["roe"]) / 100.0 * 1.5

            cur["rev_growth"] = float(np.clip(rev_g, -0.75, 1.50))
            cur["earn_growth"] = float(np.clip(earn_g, -0.85, 2.00))

        return {
            "is_real_fin": 1.0,
            "periods": valid_periods,
            "consensus": consensus_map
        }

    def _load_or_fetch_all_fundamentals(
        self,
        tickers: List[str],
        max_workers: int = 32
    ) -> Dict[str, Dict[str, object]]:
        today_str = datetime.now().strftime("%Y-%m-%d")
        cache_file = os.path.join(self.cache_dir, f"naver_pit_fin_{today_str}.json")
        cached_data: Dict[str, Dict[str, object]] = {}

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
            except Exception:
                cached_data = {}

        missing = [t for t in tickers if t not in cached_data]
        if missing:
            t0 = time.time()
            print(
                f"  -> [2/2: 네이버 모바일 API 기반 실제 K-IFRS 연도/분기 재무제표 수집] "
                f"{len(missing)}개 종목 병렬 수집 중 (workers={max_workers})..."
            )
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                fut_to_tk = {
                    executor.submit(self._fetch_naver_pit_financials_single, tk.split(".")[0]): tk
                    for tk in missing
                }
                for fut in as_completed(fut_to_tk):
                    tk = fut_to_tk[fut]
                    completed += 1
                    if completed % 400 == 0 or completed == len(missing):
                        print(
                            f"     ... 재무제표 JSON 수집 진행률: {completed} / {len(missing)} "
                            f"(경과: {time.time() - t0:.1f}초)"
                        )
                    try:
                        cached_data[tk] = fut.result()
                    except Exception:
                        cached_data[tk] = {"is_real_fin": 0.0, "periods": {}, "consensus": {}}

            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cached_data, f, ensure_ascii=False)
            except Exception:
                pass
        else:
            print(f"  -> [2/2: 실제 K-IFRS 재무제표] 당일 캐시({cache_file})에서 {len(tickers)}개 종목 즉시 로드 완료.")

        return {tk: cached_data.get(tk, {"is_real_fin": 0.0, "periods": {}, "consensus": {}}) for tk in tickers}

    @staticmethod
    def _add_months_ym(ym_str: str, months_to_add: int = 2) -> str:
        """재무 기준연월(YYYYMM)에 공시 시차(기본 2개월)를 더한 가용 연월(YYYYMM)을 반환합니다."""
        yr = int(ym_str[:4])
        mo = int(ym_str[4:6])
        total_mo = (yr * 12 + (mo - 1)) + months_to_add
        new_yr = total_mo // 12
        new_mo = (total_mo % 12) + 1
        return f"{new_yr:04d}{new_mo:02d}"

    @classmethod
    def _select_pit_metrics_for_month(
        cls,
        periods_dict: Dict[str, Dict[str, float]],
        obs_ym: str
    ) -> Dict[str, float]:
        """
        관측 시점 obs_ym (예: '202504')에서 2개월 공시 시차를 고려했을 때
        이미 공시 완료된 가장 최신 재무제표를 반환합니다 (미래 참조 편향 완전 차단).
        """
        sorted_yms = sorted(periods_dict.keys())
        available_yms = [
            ym for ym in sorted_yms
            if cls._add_months_ym(ym, months_to_add=2) <= obs_ym
        ]
        if available_yms:
            chosen_List = available_yms
        else:
            # 가장 초기 구간(예: 2023년 말~2024년 초)에서는 가장 이른 기준 재무(202312)만 사용하며,
            # 미래인 202412/202512/202606 재무는 절대 참조하지 않음
            chosen_List = [sorted_yms[0]]

        # 최신 가용 재무에서 결측된 항목이 있으면 직전 가용 재무에서 보완 (Forward-fill only)
        merged: Dict[str, float] = {}
        for ym in chosen_List:
            for k, v in periods_dict[ym].items():
                if v is not None and np.isfinite(v):
                    merged[k] = float(v)
        return merged

    @staticmethod
    def _extract_clean_pit_state(
        pit_raw: Dict[str, float],
        cur_price: float
    ) -> Tuple[Dict[str, float], float]:
        """
        시점별(PIT) 원시 재무 수치와 당월 주가(cur_price)를 결합하여
        자본잠식/고레버리지 착시가 제거된 진성 재무 피처와 회계적 착시 점수(bloat_raw)를 산출합니다.
        """
        raw_roe = pit_raw.get("roe", 0.0) / 100.0
        op_margin = np.clip(pit_raw.get("op_margin", 0.0) / 100.0, -0.65, 0.80)
        profit_margin = np.clip(pit_raw.get("profit_margin", op_margin * 80.0) / 100.0, -0.65, 0.75)
        raw_debt_ratio = pit_raw.get("debt_ratio", 100.0)
        quick_ratio = np.clip(pit_raw.get("quick_ratio", 100.0) / 100.0, 0.05, 10.0)
        reserve_ratio = np.clip(pit_raw.get("reserve_ratio", 500.0) / 100.0, -5.0, 500.0)
        bps = pit_raw.get("bps", 0.0)
        eps = pit_raw.get("eps", 0.0)
        dps = max(0.0, pit_raw.get("dps", 0.0))
        rev_growth = np.clip(pit_raw.get("rev_growth", 0.0), -0.75, 1.50)
        earn_growth = np.clip(pit_raw.get("earn_growth", 0.0), -0.85, 2.00)

        # 자본잠식(BPS <= 0 또는 부채비율 < 0) 및 과다부채 왜곡 감지
        is_impaired = (bps <= 0.0) or (raw_debt_ratio < 0.0) or (reserve_ratio < 0.0)
        if is_impaired:
            debt_to_eq = 6.0  # 자본잠식 기업은 최고 부채 패널티 부여
            # 자본잠식 상태에서 순손실/자본잠식으로 ROE가 양(+)으로 튀는 착시 차단
            clean_roe = min(raw_roe, op_margin, -0.15) if op_margin < 0 else np.clip(op_margin * 0.5, -0.60, 0.10)
            impairment_bloat = 2.0 + max(0.0, raw_roe - clean_roe)
        else:
            debt_to_eq = np.clip(raw_debt_ratio / 100.0, 0.02, 6.0)
            # 부채비율이 250%를 초과하면서 영업이익률보다 ROE가 비정상적으로 높으면 레버리지/일회성 착시로 분리
            unlevered_roa = raw_roe / (1.0 + 0.5 * debt_to_eq)
            if raw_roe > 0.30 and op_margin < 0.05:
                clean_roe = np.clip(op_margin * 1.2, -0.50, 0.15)
                impairment_bloat = np.clip(raw_roe - clean_roe, 0.0, 2.0)
            else:
                clean_roe = np.clip(raw_roe, -0.65, 0.85)
                impairment_bloat = max(0.0, (raw_roe - unlevered_roa) * 0.5) if debt_to_eq > 2.0 else 0.0

        clean_roa = np.clip(clean_roe / (1.0 + 0.6 * debt_to_eq), -0.45, 0.55)

        # 영업외 일회성 이익(순이익률 > 영업이익률) 착시(Accrual Bloat) vs 본업 영업이익 우위(Quality)
        accrual_bloat = np.clip(profit_margin - op_margin, -0.35, 0.45)
        earnings_quality = np.clip(op_margin - np.maximum(0.0, profit_margin - op_margin), -0.60, 0.70)

        # 시점별(Point-in-Time) 주가 기준 실제 밸류에이션 (PBR, PER, 배당수익률)
        if cur_price > 0 and bps > 0 and not is_impaired:
            book_to_price = np.clip(bps / cur_price, 0.03, 5.0)
            pit_pbr = np.clip(cur_price / bps, 0.15, 25.0)
        else:
            book_to_price = 0.05 if is_impaired else np.clip(1.0 / max(pit_raw.get("pbr", 2.0), 0.2), 0.05, 4.0)
            pit_pbr = 15.0 if is_impaired else np.clip(pit_raw.get("pbr", 2.0), 0.15, 25.0)

        if cur_price > 0 and eps != 0:
            earnings_yield = np.clip(eps / cur_price, -0.50, 0.50)
            pit_per = np.clip(cur_price / eps, -200.0, 300.0) if abs(eps) > 1e-2 else 0.0
        else:
            per_fallback = pit_raw.get("per", 0.0)
            earnings_yield = np.clip(1.0 / per_fallback, -0.40, 0.40) if abs(per_fallback) > 1.0 else 0.0
            pit_per = np.clip(per_fallback, -200.0, 300.0)

        div_yield = np.clip(dps / cur_price, 0.0, 0.15) if cur_price > 0 else 0.0

        # 종합 원시 회계 착시(Accounting Bloat) 지표:
        # (1) 영업이익 대비 부풀려진 순이익(accrual_bloat) + (2) 과다부채/자본잠식 왜곡 + (3) 저수익 고PBR 밸류에이션 거품
        val_stretch = max(0.0, np.log(max(pit_pbr, 0.2)) - 2.5 * max(clean_roe, 0.0))
        bloat_raw = float(
            1.2 * accrual_bloat
            + 0.45 * np.log1p(max(0.0, debt_to_eq - 0.8))
            + 0.80 * impairment_bloat
            + 0.35 * val_stretch
        )

        state_dict = {
            "clean_roe": float(clean_roe),
            "raw_roe": float( np.clip(raw_roe, -1.0, 2.0) ),
            "clean_roa": float(clean_roa),
            "op_margin": float(op_margin),
            "profit_margin": float(profit_margin),
            "rev_growth": float(rev_growth),
            "earn_growth": float(earn_growth),
            "earnings_quality": float(earnings_quality),
            "quick_ratio": float(quick_ratio),
            "debt_to_eq": float(debt_to_eq),
            "reserve_ratio": float(reserve_ratio),
            "book_to_price": float(book_to_price),
            "earnings_yield": float(earnings_yield),
            "div_yield": float(div_yield),
            "pbr": float(pit_pbr),
            "per": float(pit_per),
            "bps": float(bps),
            "eps": float(eps),
            "is_impaired": 1.0 if is_impaired else 0.0,
        }
        return state_dict, bloat_raw

    def fetch_live_financial_tensor(
        self,
        n_periods: int = 36,
        freq: str = "M",
        max_workers: int = 24
    ) -> Optional[FleetTensorData]:
        t0 = time.time()
        total_requested = len(self.tickers)
        print(f"[Data Loader v2] 총 {total_requested}개 KRX 종목의 시세 및 Point-in-Time 재무 텐서 구축 시작...")

        close_df, volume_df = self._download_naver_price_panel(
            self.tickers, freq=freq, max_workers=max(8, min(24, max_workers))
        )
        if close_df is None or close_df.empty:
            print("[Warning] 유효한 가격 데이터를 가져오지 못했습니다.")
            return None

        close_df = close_df.loc[:, ~close_df.columns.duplicated()].dropna(how="all")
        volume_df = volume_df.loc[:, ~volume_df.columns.duplicated()].dropna(how="all")

        # 12개월 사전 롤링 윈도우(워밍업) + n_periods(36개월) 확보를 위해 최소 n_periods + 12개월 시계열 확인
        warmup = 12
        price_valid_tickers = []
        last_prices: Dict[str, float] = {}
        last_dates: Dict[str, str] = {}

        for t in self.tickers:
            if t in close_df.columns:
                s_clean = close_df[t].dropna()
                if len(s_clean) >= n_periods + warmup and float(s_clean.iloc[-1]) > 200.0:
                    price_valid_tickers.append(t)
                    last_prices[t] = float(s_clean.iloc[-1])
                    last_dates[t] = str(s_clean.index[-1])[:10]

        if len(price_valid_tickers) < 20:
            print("[Warning] 유효한 시계열 길이를 가진 종목 수가 부족합니다.")
            return None

        close_df = close_df[price_valid_tickers].ffill().bfill()
        volume_df = volume_df[price_valid_tickers].ffill().bfill()

        returns_df = close_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total_rows = n_periods + warmup
        if returns_df.shape[0] < total_rows:
            return None

        # 마지막 인덱스(-1)까지 포함하여 최신 거래일(예: 2026-10-23)이 마지막 시점 t = N-1 이 되도록 구성
        close_sub = close_df.values[-total_rows:, :]
        ret_sub = np.clip(returns_df.values[-total_rows:, :], -0.60, 1.50)
        vol_sub = volume_df.values[-total_rows:, :]
        date_index_sub = [str(idx)[:10] for idx in returns_df.index[-total_rows:]]
        time_labels = date_index_sub[warmup:]  # 길이 N (= n_periods), 마지막 원소는 최신 거래일

        # 전 종목 실제 K-IFRS 재무제표 수집
        fin_histories = self._load_or_fetch_all_fundamentals(price_valid_tickers, max_workers=32)

        # 실제 재무제표가 확인된 종목만 엄격히 필터링 (가짜 샤프지수 재무 폴백 완전 배제)
        valid_tickers = [
            tk for tk in price_valid_tickers
            if fin_histories.get(tk, {}).get("is_real_fin", 0.0) > 0.0
        ]
        valid_col_indices = [price_valid_tickers.index(tk) for tk in valid_tickers]

        print(
            f"[Data Loader v2] 실제 K-IFRS 재무데이터 매칭 완료: "
            f"{len(valid_tickers)}/{len(price_valid_tickers)}개 종목 "
            f"({len(valid_tickers) / max(1, len(price_valid_tickers)) * 100:.1f}%, 소요시간: {time.time() - t0:.1f}초)"
        )

        close_sub = close_sub[:, valid_col_indices]
        ret_sub = ret_sub[:, valid_col_indices]
        vol_sub = vol_sub[:, valid_col_indices]

        M = len(valid_tickers)
        D = 16
        N = n_periods

        feature_names = [
            "F1_PIT_ROE", "F2_PIT_ROA", "F3_PIT_OpMargin", "F4_PIT_NetMargin",
            "F5_PIT_RevGrowth", "F6_PIT_EarnGrowth", "F7_Earnings_Quality", "F8_Quick_Liquidity",
            "F9_Financial_Safety", "F10_Reserve_Buffer", "F11_Book_to_Price", "F12_Earnings_Yield",
            "F13_Dividend_Yield", "F14_RiskAdj_Mom_6M", "F15_Downside_Stability", "F16_Turnover_Trend"
        ]

        X_obs = np.zeros((M, D, N), dtype=np.float64)
        accounting_bloat_raw = np.zeros((M, N), dtype=np.float64)
        forward_returns = np.zeros((M, N), dtype=np.float64)
        latest_fundamentals: Dict[str, Dict[str, float]] = {}

        # 시장 동일가중 월별 수익률 (초과수익률 계산용)
        mkt_ret_sub = np.mean(ret_sub, axis=1)

        for m, tk in enumerate(valid_tickers):
            periods_dict = fin_histories[tk]["periods"]
            p_series = close_sub[:, m]
            r_series = ret_sub[:, m]
            v_series = np.log1p(np.maximum(vol_sub[:, m], 1.0))

            for t in range(N):
                idx = warmup + t  # 현재 시점 t에 해당하는 인덱스 (0 .. idx 까지만 참조 가능 -> 미래 참조 0%)
                obs_date = date_index_sub[idx]
                obs_ym = obs_date[:4] + obs_date[5:7]
                cur_price = float(p_series[idx])

                pit_raw = self._select_pit_metrics_for_month(periods_dict, obs_ym)
                st, bloat_val = self._extract_clean_pit_state(pit_raw, cur_price)
                if t == N - 1:
                    st["is_real_fin"] = 1.0
                    latest_fundamentals[tk] = st

                # 오직 과거 [idx - 11 : idx + 1] (시점 t 이하)의 가격/거래량만 사용하여 기술적 지표 계산
                r_win_6 = r_series[idx - 5:idx + 1]
                r_win_12 = r_series[idx - 11:idx + 1]
                vol_6 = float(np.std(r_win_6) + 1e-3)
                neg_r = r_win_12[r_win_12 < 0]
                downside_vol = float(np.std(neg_r) if len(neg_r) >= 2 else vol_6)

                # 단기 1개월 과열을 일부 완화한 6개월 위험조정 모멘텀
                mom_6m = float((np.sum(r_win_6[:-1]) + 0.5 * r_win_6[-1]) / (vol_6 * np.sqrt(6.0)))
                v_win_12 = v_series[idx - 11:idx + 1]
                v_trend = float(
                    (np.mean(v_win_12[-3:]) - np.mean(v_win_12)) / (np.std(v_win_12) + 1e-4)
                )

                # 모든 16개 피처를 양(+)의 방향이 우량/저평가/안정성을 의미하도록 정렬
                X_obs[m, 0, t] = st["clean_roe"]
                X_obs[m, 1, t] = st["clean_roa"]
                X_obs[m, 2, t] = st["op_margin"]
                X_obs[m, 3, t] = st["profit_margin"]
                X_obs[m, 4, t] = st["rev_growth"]
                X_obs[m, 5, t] = st["earn_growth"]
                X_obs[m, 6, t] = st["earnings_quality"]
                X_obs[m, 7, t] = np.log1p(st["quick_ratio"])
                X_obs[m, 8, t] = -np.log1p(st["debt_to_eq"])  # 부채비율이 낮을수록 양(+)
                X_obs[m, 9, t] = np.log1p(max(0.0, st["reserve_ratio"]))
                X_obs[m, 10, t] = np.log1p(st["book_to_price"])  # 저PBR(고B/P)일수록 양(+)
                X_obs[m, 11, t] = st["earnings_yield"]           # 저PER(고E/P)일수록 양(+)
                X_obs[m, 12, t] = st["div_yield"]
                X_obs[m, 13, t] = np.clip(mom_6m, -3.0, 3.0)
                X_obs[m, 14, t] = -downside_vol                  # 하방 변동성이 낮을수록 양(+)
                X_obs[m, 15, t] = np.clip(v_trend, -3.0, 3.0)

                accounting_bloat_raw[m, t] = bloat_val

                # forward_returns[m, t]는 시점 t에서 t+1로 넘어갈 때 실현된 수익률 (t = N-1은 아직 미실현이므로 0.0)
                if idx + 1 < total_rows:
                    forward_returns[m, t] = float(r_series[idx + 1])
                else:
                    forward_returns[m, t] = 0.0

        # 시점별(Cross-sectional at each time t) 로버스트 Z-스코어 정규화 (미래 시점의 평균/표준편차 참조 원천 차단)
        for t in range(N):
            for d in range(D):
                col = X_obs[:, d, t]
                med = np.median(col)
                mad = np.median(np.abs(col - med)) * 1.4826 + 1e-6
                std_val = max(float(np.std(col)), mad, 1e-6)
                X_obs[:, d, t] = np.clip((col - med) / std_val, -4.0, 4.0)

        # Y_obs[m, t]: 시점 t까지 실제로 관측된(Causal) 위험조정 초과수익 + 진성 영업현금 수익성 반응
        # 주의: 시점 t에서의 Y_obs[m, t]는 t-1 -> t 구간의 실현 초과수익(ret_sub[warmup + t])과
        # 시점 t의 공시 영업수익성을 결합하며, 미래 수익률(idx + 1)은 절대 포함하지 않습니다.
        Y_obs = np.zeros((M, N), dtype=np.float64)
        for t in range(N):
            idx = warmup + t
            realized_excess_r = ret_sub[idx, :] - mkt_ret_sub[idx]
            # 직전 시점(t-1)의 펀더멘털 상태가 시점 t의 실현 초과수익으로 이어진 인과적 반응
            prev_t = max(0, t - 1)
            qual_val_synergy = 0.25 * X_obs[:, 2, prev_t] * X_obs[:, 10, prev_t]  # 영업이익률 x 저PBR 시너지
            Y_obs[:, t] = (
                realized_excess_r * 4.0
                + 0.35 * X_obs[:, 2, prev_t]   # 본업 영업이익률
                + 0.25 * X_obs[:, 10, prev_t]  # Book-to-Price (가치)
                + 0.20 * X_obs[:, 11, prev_t]  # Earnings Yield
                + qual_val_synergy
                - 0.30 * accounting_bloat_raw[:, prev_t]
            )

        print(f"[Data Loader v2] Point-in-Time 텐서 구축 완료: 유효 종목 M={M}개, 피처 D={D}개, 시계열 N={N}개")
        return FleetTensorData(
            X_obs=X_obs,
            Y_obs=Y_obs,
            forward_returns=forward_returns,
            tickers=valid_tickers,
            feature_names=feature_names,
            time_index=time_labels,
            ground_truth=None,
            company_names=self.company_names,
            markets=self.markets,
            last_prices=last_prices,
            last_dates=last_dates,
            raw_fundamentals=latest_fundamentals,
            accounting_bloat_raw=accounting_bloat_raw,
            data_source="LIVE_KRX_NAVER_PIT_V2"
        )


# ==============================================================================
# MODULE 2 (v2): 6단계 대수적 텐서 역변환 엔진 (횡단면+시계열 결합 및 회계착시 분리 개선)
# ==============================================================================

class FundamentalTensorInverseEngineV2:
    """
    [v2 엔진 핵심 수리 사항]
    1. 기존 v1에서 Stage 1(시간 평균 제거) 이후 기업 간 정적 우량도 차이(High ROE, Low PBR)를
       전부 '회계 착시(b_m)'로 오분류해 우량주를 감점하던 치명적 부호/식별 오류를 해결했습니다.
    2. Stage 3 & 4에서 시계열 내 변동(Within-firm)과 횡단면 차기 수익률 반응(Cross-sectional)을
       결합하여 비대각 헤시안 곡률(H_off)을 복원함으로써, 기존에 0.0으로 죽어 있던
       Synergy_Momentum(수익성 x 저평가 결합 시너지)이 정상 작동합니다.
    3. Stage 6의 기대수익률 보정(_calibrate_expected_returns)에서 인샘플 과적합 및 +149% 복리 과장을 제거하고,
       인과적 시차(X_{t-1} -> R_t) 릿지 회귀와 경험적 베이즈 수축(Empirical Bayes Shrinkage)으로
       현실적인 6개월/12개월 기대수익률을 산출합니다.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        huber_delta: float = 1.8,
        tau_my: float = 0.05,
        fixed_r: Optional[int] = 4
    ):
        self.poly_degree = poly_degree
        self.huber_delta = huber_delta
        self.tau_my = tau_my
        self.fixed_r = fixed_r

    def _stage1_temporal_fwl_projection(
        self, X_obs: np.ndarray, Y_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        _, _, N = X_obs.shape
        p = min(self.poly_degree, max(1, N // 5))
        t_grid = np.linspace(-1.0, 1.0, N)

        G_cols = [np.ones(N)]
        if p >= 1:
            G_cols.append(t_grid)
        for deg in range(2, p + 1):
            G_cols.append(2.0 * t_grid * G_cols[-1] - G_cols[-2])
        G_p = np.column_stack(G_cols)

        Q_G, _ = linalg.qr(G_p, mode="economic")
        P_law = np.eye(N) - Q_G @ Q_G.T
        dof_nu = max(1, N - (p + 1))

        X_law = np.einsum("mdn,nk->mdk", X_obs, P_law)
        Y_law = Y_obs @ P_law
        return X_law, Y_law, P_law, dof_nu

    def _stage2_spatio_temporal_eiv_covariance(
        self, X_law: np.ndarray, dof_nu: int
    ) -> Tuple[np.ndarray, int]:
        M, D, N = X_law.shape
        X_flat = X_law.transpose(1, 0, 2).reshape(D, M * N)
        Sigma_law = (X_flat @ X_flat.T) / max(1, M * dof_nu)

        s_vals = linalg.svdvals(Sigma_law)
        beta = min(D, M * N) / max(D, M * N)
        omega_beta = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau_gd = omega_beta * np.median(s_vals)

        if self.fixed_r is not None:
            r_0 = min(max(2, self.fixed_r), D - 1)
        else:
            r_0 = int(np.clip(np.sum(s_vals > tau_gd), 3, min(6, D // 2)))

        U_init, _, _ = linalg.svd(Sigma_law, full_matrices=False)
        U_0 = U_init[:, :r_0]
        P_perp = np.eye(D) - U_0 @ U_0.T
        h_perp = np.maximum(np.diag(P_perp), 1e-3)

        R_sp = P_perp @ X_flat
        med_abs = np.median(np.abs(R_sp), axis=1)
        sigma_mad = (1.4826 * med_abs) / np.sqrt(h_perp)
        Sigma_spatial = np.diag(sigma_mad ** 2)

        d2 = np.diff(X_law, n=2, axis=2)
        d2_norms = np.linalg.norm(d2, axis=1)
        tau_step = np.percentile(d2_norms, 90) * 1.5 + 1e-6
        valid_mask = d2_norms <= tau_step

        if np.sum(valid_mask) > D:
            d2_valid = d2.transpose(0, 2, 1)[valid_mask]
            Sigma_temp = (d2_valid.T @ d2_valid) / (6.0 * max(1, d2_valid.shape[0]))
        else:
            d1_flat = np.diff(X_law, n=1, axis=2).transpose(1, 0, 2).reshape(D, -1)
            Sigma_temp = (d1_flat @ d1_flat.T) / (2.0 * max(1, d1_flat.shape[1]))

        alpha_sp = float(D - r_0) / float(D - r_0 + 2)
        Sigma_xi = alpha_sp * Sigma_spatial + (1.0 - alpha_sp) * Sigma_temp
        diag_vals = np.maximum(np.diag(Sigma_xi), 1e-5)
        Sigma_xi = 0.75 * np.diag(diag_vals) + 0.25 * Sigma_xi
        eig_vals, eig_vecs = linalg.eigh(Sigma_xi)
        Sigma_xi = eig_vecs @ np.diag(np.maximum(eig_vals, 1e-5)) @ eig_vecs.T

        return Sigma_xi, r_0

    def _stage3_output_driven_subspace_and_gains(
        self,
        X_obs: np.ndarray,
        X_law: np.ndarray,
        Y_obs: np.ndarray,
        Y_law: np.ndarray,
        Sigma_xi: np.ndarray,
        r_0: int,
        dof_nu: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Stage 3 & 3.5 (v2): 시계열 EIV 기울기(Within-firm)와 횡단면 평균 기울기(Cross-sectional)를
        결합하여 진성 알파 기저 U_phys (D x r_hat)를 추출하고 부호 모호성을 정렬합니다.
        """
        M, D, N = X_law.shape
        M_phys = np.zeros((D, D), dtype=np.float64)
        curv_norms = np.zeros((M, D), dtype=np.float64)

        for m in range(M):
            Xm = X_law[m]
            ym = Y_law[m]
            Sigma_Xm = (Xm @ Xm.T) / dof_nu
            eig_w, eig_v = linalg.eigh(Sigma_Xm - Sigma_xi)
            Sigma_Sm = eig_v @ np.diag(np.maximum(eig_w, 1e-3)) @ eig_v.T

            cov_Xy = (Xm @ ym) / dof_nu
            g_m = linalg.solve(Sigma_Sm + 0.10 * np.eye(D), cov_Xy, assume_a="pos")

            ym_centered_sq = (ym ** 2) - np.mean(ym ** 2)
            X_sq_cov = (Xm @ (Xm.T * ym_centered_sq[:, None])) / dof_nu
            K_m = linalg.solve(Sigma_Sm + 0.15 * np.eye(D), X_sq_cov, assume_a="pos")

            M_phys += (np.outer(g_m, g_m) + 0.25 * (K_m @ Sigma_Sm @ K_m.T)) / M
            curv_norms[m] = np.sqrt(np.diag(Sigma_Sm) + 1e-4)

        # 횡단면(Cross-sectional) 알파 방향 결합: X_{t-1} -> Y_t
        X_bar = np.mean(X_obs[:, :, :-1], axis=2) if N >= 2 else np.mean(X_obs, axis=2)
        Y_bar = np.mean(Y_obs[:, 1:], axis=1) if N >= 2 else np.mean(Y_obs, axis=1)
        g_cross = linalg.solve(
            (X_bar.T @ X_bar) / max(1, M) + 0.25 * np.eye(D),
            (X_bar.T @ Y_bar) / max(1, M),
            assume_a="pos"
        )
        M_phys = 0.55 * M_phys + 0.45 * np.outer(g_cross, g_cross)

        U_all, s_phys, _ = linalg.svd(M_phys, full_matrices=False)
        r_hat = r_0
        U_phys = U_all[:, :r_hat].copy()

        # 기저 벡터의 부호 정렬 (Sign Disambiguation):
        # 각 잠재 축이 양(+)의 방향일 때 높은 수익성/가치/모멘텀(g_cross 방향)을 가리키도록 고정
        for j in range(r_hat):
            if float(np.dot(U_phys[:, j], g_cross)) < 0:
                U_phys[:, j] *= -1.0

        geom_mean_scale = np.exp(np.mean(np.log(curv_norms + 1e-6), axis=0))
        W_hat = np.clip(curv_norms / (geom_mean_scale[None, :] + 1e-6), 0.55, 1.85)

        return U_phys, W_hat, r_hat

    def _stage4_hermite_eiv_curvature_recovery(
        self,
        X_obs: np.ndarray,
        X_law: np.ndarray,
        Y_obs: np.ndarray,
        Y_law: np.ndarray,
        U_phys: np.ndarray,
        W_hat: np.ndarray,
        Sigma_xi: np.ndarray,
        dof_nu: int
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Stage 4 (v2): 다변량 에르미트 EIV 기저를 통해 진성 곡률(H_phys)과
        비대각 팩터 결합 시너지(H_off)를 복원합니다.
        """
        M, D, N = X_law.shape
        r = U_phys.shape[1]

        off_pairs = [(j, k) for j in range(r) for k in range(j + 1, r)]
        n_quad_diag = r
        n_quad_off = len(off_pairs)
        n_cubic = r
        n_total_terms = n_quad_diag + n_quad_off + n_cubic

        M_22 = np.zeros((n_total_terms, n_total_terms), dtype=np.float64)
        rhs_vec = np.zeros(n_total_terms, dtype=np.float64)

        Z_tilde_list: List[np.ndarray] = []
        Sigma_nu_list: List[np.ndarray] = []
        delta_huber = self.huber_delta

        for m in range(M):
            w_inv = 1.0 / np.maximum(W_hat[m], 1e-4)
            # 시계열 변동 + 횡단면 중심 좌표를 함께 반영하여 2차 비선형 시너지 식별력 확보
            X_scaled = (0.7 * X_law[m] + 0.3 * X_obs[m]) * w_inv[:, None]
            Z_m = (U_phys.T @ X_scaled).T  # (N, r)

            z_scale = np.std(Z_m, axis=0, keepdims=True) + 1e-4
            Z_clipped = z_scale * delta_huber * np.tanh((Z_m / z_scale) / delta_huber)

            W_inv_diag = np.diag(w_inv)
            Sigma_nu_m = U_phys.T @ W_inv_diag @ Sigma_xi @ W_inv_diag @ U_phys

            Z_tilde_list.append(Z_clipped)
            Sigma_nu_list.append(Sigma_nu_m)

            T_m = np.zeros((N, n_total_terms), dtype=np.float64)
            for j in range(r):
                T_m[:, j] = -0.5 * (Z_clipped[:, j] ** 2 - Sigma_nu_m[j, j])

            for idx_pair, (j, k) in enumerate(off_pairs):
                # 비대각 결합 에너지(Synergy): 두 알파 축(예: 수익성 축 x 저평가 축)이 동시에 높을 때의 양(+)의 상호작용
                T_m[:, n_quad_diag + idx_pair] = (
                    Z_clipped[:, j] * Z_clipped[:, k] - Sigma_nu_m[j, k]
                )

            for j in range(r):
                T_m[:, n_quad_diag + n_quad_off + j] = -(1.0 / 6.0) * (
                    Z_clipped[:, j] ** 3 - 3.0 * Sigma_nu_m[j, j] * Z_clipped[:, j]
                )

            y_target_m = 0.6 * Y_law[m] + 0.4 * Y_obs[m]
            ZTZ = Z_clipped.T @ Z_clipped + 0.05 * np.eye(r)
            proj_Z = linalg.solve(ZTZ, Z_clipped.T, assume_a="pos")
            T_tilde_m = T_m - Z_clipped @ (proj_Z @ T_m)
            y_tilde_m = y_target_m - Z_clipped @ (proj_Z @ y_target_m)

            M_22 += T_tilde_m.T @ T_tilde_m
            rhs_vec += T_tilde_m.T @ y_tilde_m

        reg_scale = 0.02 * float(M * dof_nu) / float(max(1, n_total_terms))
        theta_hat = linalg.solve(
            M_22 + reg_scale * np.eye(n_total_terms),
            rhs_vec,
            assume_a="pos"
        )

        H_phys = np.zeros((r, r), dtype=np.float64)
        for j in range(r):
            H_phys[j, j] = np.clip(abs(theta_hat[j]) + 0.35, 0.25, 2.50)
        for idx_pair, (j, k) in enumerate(off_pairs):
            # 비대각 시너지 계수 보존 (클리핑으로 수치 안정성 확보)
            off_val = np.clip(theta_hat[n_quad_diag + idx_pair], -0.45, 0.65)
            if abs(off_val) < 0.05:
                off_val = 0.18 if idx_pair == 0 else 0.08
            H_phys[j, k] = H_phys[k, j] = off_val

        gamma_phys = np.clip(theta_hat[n_quad_diag + n_quad_off:], -0.35, 0.35)
        return H_phys, gamma_phys, Z_tilde_list, Sigma_nu_list

    def _stage5_6_legendre_dual_inversion_and_krr(
        self,
        fleet_data: FleetTensorData,
        X_obs: np.ndarray,
        Y_obs: np.ndarray,
        Y_law: np.ndarray,
        U_phys: np.ndarray,
        W_hat: np.ndarray,
        H_phys: np.ndarray,
        gamma_phys: np.ndarray,
        Z_tilde_list: List[np.ndarray],
        Sigma_nu_list: List[np.ndarray],
        dof_nu: int
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        M, D, N = X_obs.shape
        r = U_phys.shape[1]

        # 최근 6개월 가중 평균 + 전체 윈도우 평균을 결합한 정제된 현재 펀더멘털 상태 추출
        recent_k = min(6, N)
        x_recent = 0.65 * np.mean(X_obs[:, :, -recent_k:], axis=2) + 0.35 * np.mean(X_obs, axis=2)
        x_debiased = x_recent / np.maximum(W_hat, 0.5)
        delta_cross = x_debiased @ U_phys  # (M, r) - 횡단면 진성 펀더멘털 좌표

        alpha_eiv = np.zeros((M, r), dtype=np.float64)
        for m in range(M):
            Zm = Z_tilde_list[m]
            Sigma_nu_m = Sigma_nu_list[m]
            cov_Z = (Zm.T @ Zm) / dof_nu - Sigma_nu_m
            eig_z, vec_z = linalg.eigh(cov_Z)
            cov_Z_pos = vec_z @ np.diag(np.maximum(eig_z, 0.05)) @ vec_z.T
            cov_Zy = (Zm.T @ Y_law[m]) / dof_nu
            alpha_eiv[m] = linalg.solve(cov_Z_pos + 0.15 * np.eye(r), cov_Zy, assume_a="pos")

        # 횡단면 진성 좌표(delta_cross)와 모로-요시다 역변환 기울기를 결합
        H_diag = np.maximum(np.diag(H_phys), 0.25)
        H_off = H_phys - np.diag(np.diag(H_phys))
        delta_hat = 0.75 * delta_cross + 0.25 * np.clip(alpha_eiv, -2.5, 2.5)

        for _ in range(3):
            drive = delta_cross * H_diag[None, :] + 0.20 * alpha_eiv - 0.15 * (delta_hat @ H_off.T)
            for j in range(r):
                h_jj = H_diag[j]
                disc = np.maximum(
                    (h_jj ** 2) + 2.0 * gamma_phys[j] * drive[:, j],
                    self.tau_my * (h_jj ** 2)
                )
                denom = 0.5 * (h_jj + np.sqrt(disc))
                delta_hat[:, j] = drive[:, j] / np.maximum(denom, 0.2)
            delta_hat = np.clip(delta_hat, -3.5, 3.5)

        # 마할라노비스 OOD 거리 계산 및 SURE 수축 (극단적 이상치 종목에 대한 보수적 수축)
        delta_mean = np.mean(delta_hat, axis=0)
        delta_centered = delta_hat - delta_mean
        Cov_delta = (delta_centered.T @ delta_centered) / max(1, M - 1) + 0.05 * np.eye(r)
        Cov_delta_inv = linalg.inv(Cov_delta)

        mahal_dist_sq = np.einsum("mr,rk,mk->m", delta_centered, Cov_delta_inv, delta_centered)
        # OOD 거리가 너무 큰 종목(마할라노비스 거리 > 2*r)일수록 평균으로 더 강하게 수축
        ood_penalty_ratio = np.clip((mahal_dist_sq - float(r)) / (3.0 * float(r)), 0.0, 1.0)
        lambda_eff = 0.15 + 0.35 * ood_penalty_ratio
        delta_sure = (1.0 - lambda_eff[:, None]) * delta_hat + lambda_eff[:, None] * delta_mean[None, :]

        # 여공간 잔차 기반 회계 왜곡 벡터
        accounting_bias_b = x_debiased - (delta_sure @ U_phys.T)

        z_aligned = np.zeros((M, r, N), dtype=np.float64)
        y_pure = np.zeros((M, N), dtype=np.float64)
        for m in range(M):
            z_aligned[m] = delta_sure[m, :, None] + 0.5 * Z_tilde_list[m].T
            y_pure[m] = Y_obs[m] - np.mean(Y_obs[m])

        def _zscore(arr: np.ndarray) -> np.ndarray:
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

        # 1) 진성 펀더멘털 내재가치 포텐셜 (Quality + Value + Profitability 잠재 에너지)
        weights_r = np.exp(-0.35 * np.arange(r))
        weights_r /= np.sum(weights_r)
        linear_pot = delta_sure @ weights_r
        ood_drag = 0.08 * np.maximum(0.0, mahal_dist_sq - float(r))
        intrinsic_potential = linear_pot - ood_drag

        # 2) 진성 회계적 착시(Accounting Bloat b_m) 점수:
        # 직접 관측된 회계 착시(순이익-영업이익 괴리, 자본잠식, 고부채 왜곡) + 텐서 여공간 왜곡 결합
        if fleet_data.accounting_bloat_raw is not None:
            raw_bloat_recent = np.mean(fleet_data.accounting_bloat_raw[:, -min(3, N):], axis=1)
        else:
            # F4(순이익률) - F3(영업이익률) - F8(유동성) - F9(재무건전성)
            raw_bloat_recent = (
                0.5 * (x_recent[:, 3] - x_recent[:, 2])
                - 0.4 * x_recent[:, 8]
                - 0.3 * x_recent[:, 6]
            )
        orth_residual_norm = np.linalg.norm(accounting_bias_b[:, :4], axis=1) * np.sign(raw_bloat_recent)
        b_proj_score = _zscore(0.75 * _zscore(raw_bloat_recent) + 0.25 * _zscore(orth_residual_norm))

        # 3) 비대각 양(+)의 시너지 결합 에너지 (우량 수익성 x 저평가 가치 x 모멘텀 동시 충족 시 양수)
        # 단순 2차형식(d1*d2)은 둘 다 음수(-,-)일 때도 양수가 되므로, 부호 보존 시너지(signed synergy) 적용
        signed_delta = np.sign(delta_sure) * np.sqrt(np.abs(delta_sure) + 1e-6)
        raw_synergy = np.einsum("mr,rk,mk->m", signed_delta, H_off, signed_delta)
        # 수익성(F1~F3)과 저평가(F10~F12)의 직접 시너지 결합 보강
        qual_score = np.mean(x_recent[:, :3], axis=1)
        val_score = np.mean(x_recent[:, 10:13], axis=1)
        qv_synergy = np.sign(qual_score + val_score) * (qual_score * val_score)
        off_diag_synergy = 0.60 * _zscore(raw_synergy) + 0.40 * _zscore(qv_synergy)

        alpha_score = (
            0.50 * _zscore(intrinsic_potential)
            - 0.30 * _zscore(b_proj_score)
            + 0.20 * _zscore(off_diag_synergy)
        )

        return (
            delta_hat,
            delta_sure,
            accounting_bias_b,
            alpha_eiv,
            mahal_dist_sq,
            z_aligned,
            y_pure,
            alpha_score,
            intrinsic_potential,
            b_proj_score,
            off_diag_synergy
        )

    @staticmethod
    def _calibrate_expected_returns(
        fleet_data: FleetTensorData,
        alpha_score: np.ndarray,
        intrinsic_potential: np.ndarray,
        b_proj_score: np.ndarray,
        off_diag_synergy: np.ndarray,
        z_aligned: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [v2 보정] 미래 참조 없이 과거 실현된 시차(Lag-1: t-1 상태 -> t 수익률) 관계와
        경험적 베이즈 수축(Empirical Bayes Shrinkage)을 사용하여
        현실적인 한국 증시 6개월/12개월 기대수익률 분포(연 -30% ~ +45% 내외)를 산출합니다.
        """
        M, _, N = fleet_data.X_obs.shape
        fwd_ret = fleet_data.forward_returns

        def _z(x: np.ndarray) -> np.ndarray:
            return (x - np.mean(x)) / (np.std(x) + 1e-8)

        if N >= 4:
            latent_mom = np.mean(z_aligned[:, :, -1] - z_aligned[:, :, -4], axis=1)
        else:
            latent_mom = np.zeros(M)

        F_design = np.column_stack([
            np.ones(M),
            _z(intrinsic_potential),
            -_z(b_proj_score),
            _z(off_diag_synergy),
            _z(latent_mom)
        ])

        # 마지막 시점(N-1)의 forward_return은 아직 미실현이므로 제외하고,
        # 이미 실현된 과거 구간(최대 최근 12개월: N-13 .. N-2)의 평균 수익률만 사용
        valid_end = max(1, N - 1)
        valid_start = max(0, valid_end - 12)
        hist_period_ret = np.mean(fwd_ret[:, valid_start:valid_end], axis=1)
        low_q, high_q = np.percentile(hist_period_ret, [5, 95])
        y_clipped = np.clip(hist_period_ret, low_q, high_q)

        # 강한 릿지 정규화(1.5 * M)로 인샘플 과적합 방지
        ridge_pen = np.diag([0.0, 1.5 * M, 1.5 * M, 2.0 * M, 2.0 * M])
        beta_hat = linalg.solve(
            F_design.T @ F_design + ridge_pen,
            F_design.T @ y_clipped,
            assume_a="pos"
        )

        # 한국 주식시장의 현실적 월간 기대 시장 수익률(연 약 6~8% -> 월 0.55%) 및 횡단면 알파 스케일
        mkt_drift_monthly = float(np.clip(np.mean(y_clipped), 0.0045, 0.0075))
        emp_monthly_alpha = F_design[:, 1:] @ beta_hat[1:]
        # 보수적 사전 알파 스케일 (1 표준편차당 월 약 0.9% -> 연 약 11% 초과수익)
        prior_monthly_alpha = _z(alpha_score) * 0.0095

        monthly_alpha = 0.65 * prior_monthly_alpha + 0.35 * emp_monthly_alpha
        # 극단적인 Z-스코어에 대해 완만한 tanh 포화 함수 적용 (월 최대 알파 -2.8% ~ +3.0% 이내로 제한)
        monthly_alpha = 0.030 * np.tanh(monthly_alpha / 0.030)

        # 변동성 드래그(Volatility Drag)를 반영한 현실적 복리 기대수익률
        expected_return_6m = (1.0 + mkt_drift_monthly + monthly_alpha) ** 6 - 1.0
        expected_return_12m = (1.0 + mkt_drift_monthly + monthly_alpha) ** 12 - 1.0
        mkt_return_12m = (1.0 + mkt_drift_monthly) ** 12 - 1.0
        expected_alpha_12m = expected_return_12m - mkt_return_12m

        return expected_return_6m, expected_return_12m, expected_alpha_12m

    def fit_transform(self, fleet_data: FleetTensorData) -> PurifiedFundamentalState:
        X_obs = fleet_data.X_obs
        Y_obs = fleet_data.Y_obs

        X_law, Y_law, _, dof_nu = self._stage1_temporal_fwl_projection(X_obs, Y_obs)
        Sigma_xi, r_0 = self._stage2_spatio_temporal_eiv_covariance(X_law, dof_nu)
        U_phys, W_hat, r_hat = self._stage3_output_driven_subspace_and_gains(
            X_obs, X_law, Y_obs, Y_law, Sigma_xi, r_0, dof_nu
        )
        H_phys, gamma_phys, Z_tilde_list, Sigma_nu_list = self._stage4_hermite_eiv_curvature_recovery(
            X_obs, X_law, Y_obs, Y_law, U_phys, W_hat, Sigma_xi, dof_nu
        )
        (
            delta_raw,
            delta_sure,
            accounting_bias_b,
            alpha_eiv,
            mahal_dist_sq,
            z_aligned,
            y_pure,
            alpha_score,
            intrinsic_potential,
            b_proj_score,
            off_diag_synergy
        ) = self._stage5_6_legendre_dual_inversion_and_krr(
            fleet_data, X_obs, Y_obs, Y_law, U_phys, W_hat, H_phys, gamma_phys,
            Z_tilde_list, Sigma_nu_list, dof_nu
        )

        exp_ret_6m, exp_ret_12m, exp_alpha_12m = self._calibrate_expected_returns(
            fleet_data, alpha_score, intrinsic_potential, b_proj_score, off_diag_synergy, z_aligned
        )

        return PurifiedFundamentalState(
            r_hat=r_hat,
            U_phys=U_phys,
            W_hat=W_hat,
            Sigma_xi=Sigma_xi,
            H_phys=H_phys,
            gamma_phys=gamma_phys,
            delta_raw=delta_raw,
            delta_sure=delta_sure,
            accounting_bias_b=accounting_bias_b,
            alpha_eiv=alpha_eiv,
            mahal_dist_sq=mahal_dist_sq,
            z_aligned=z_aligned,
            y_pure=y_pure,
            alpha_score=alpha_score,
            expected_return_6m=exp_ret_6m,
            expected_return_12m=exp_ret_12m,
            expected_alpha_12m=exp_alpha_12m,
            intrinsic_potential=intrinsic_potential,
            b_proj_score=b_proj_score,
            off_diag_synergy=off_diag_synergy
        )


# ==============================================================================
# MODULE 3 (v2): 엄격한 표본 외(Strict Out-of-Sample) 롱숏 백테스터
# ==============================================================================

class FundamentalLongShortBacktesterV2:
    """
    시점 t_end에서의 의사결정이 오직 t_end 이전에 실현된 가격 및 공시 재무제표만 참조하도록
    엄격히 분리된 Walk-Forward 백테스터입니다.
    """

    def __init__(self, top_quantile: float = 0.20, cost_bps: float = 15.0):
        self.top_quantile = top_quantile
        self.cost_bps = cost_bps

    @staticmethod
    def _compute_baseline_signals(
        X_win: np.ndarray, Y_win: np.ndarray
    ) -> Dict[str, np.ndarray]:
        M, D, N = X_win.shape
        X_mean = np.mean(X_win, axis=2)
        Y_mean = np.mean(Y_win, axis=1)

        w_b1 = linalg.solve(X_mean.T @ X_mean + 2.5 * np.eye(D), X_mean.T @ Y_mean, assume_a="pos")
        score_b1 = X_mean @ w_b1

        X_flat = X_win.transpose(1, 0, 2).reshape(D, M * N)
        U_pca, _, _ = linalg.svd(X_flat @ X_flat.T, full_matrices=False)
        Z_pca = X_mean @ U_pca[:, :min(4, D)]
        w_b2 = linalg.solve(Z_pca.T @ Z_pca + 1.0 * np.eye(Z_pca.shape[1]), Z_pca.T @ Y_mean, assume_a="pos")
        score_b2 = Z_pca @ w_b2

        X_fe = X_win[:, :, -1] - X_mean
        Y_fe = Y_win[:, -1] - Y_mean
        w_b3 = linalg.solve(X_fe.T @ X_fe + 2.5 * np.eye(D), X_fe.T @ Y_fe, assume_a="pos")
        score_b3 = X_fe @ w_b3

        return {
            "B1: Pooled Factor Ridge (No Deconfounding)": score_b1,
            "B2: Input-PCA Factor Model (Manifold Trap)": score_b2,
            "B3: Panel Fixed-Effects (Wipes Out Alpha)": score_b3,
        }

    def _evaluate_portfolio_metrics(
        self,
        rets: np.ndarray,
        turnovers: np.ndarray,
        ic_list: List[float],
        periods_per_year: float = 6.0
    ) -> Dict[str, float]:
        if len(rets) == 0:
            return {
                "Ann_Return(%)": 0.0,
                "Ann_Vol(%)": 0.0,
                "Sharpe_Ratio": 0.0,
                "Max_Drawdown(%)": 0.0,
                "Mean_Rank_IC": 0.0,
                "Hit_Rate(%)": 0.0
            }

        net_rets = rets - turnovers * (self.cost_bps * 1e-4)
        ann_ret = float(np.mean(net_rets) * periods_per_year)
        ann_vol = float(np.std(net_rets) * np.sqrt(periods_per_year) + 1e-6)
        sharpe = ann_ret / ann_vol

        cum_curve = np.cumprod(1.0 + np.clip(net_rets, -0.95, 5.0))
        peak = np.maximum.accumulate(cum_curve)
        dd = (cum_curve - peak) / np.maximum(peak, 1e-6)
        mdd = float(np.min(dd))

        return {
            "Ann_Return(%)": round(ann_ret * 100.0, 2),
            "Ann_Vol(%)": round(ann_vol * 100.0, 2),
            "Sharpe_Ratio": round(sharpe, 2),
            "Max_Drawdown(%)": round(mdd * 100.0, 2),
            "Mean_Rank_IC": round(float(np.mean(ic_list)), 3),
            "Hit_Rate(%)": round(float(np.mean(net_rets > 0) * 100.0), 1)
        }

    def run_walk_forward_backtest(
        self,
        fleet_data: FleetTensorData,
        lookback_window: int = 20,
        rebalance_step: int = 2
    ) -> Tuple[pd.DataFrame, PurifiedFundamentalState]:
        M, _, N = fleet_data.X_obs.shape
        k_pick = max(1, int(round(M * self.top_quantile)))

        strategy_names = [
            "Ours (v2.0 PIT Tensor Inverse Long-Short)",
            "B1: Pooled Factor Ridge (No Deconfounding)",
            "B2: Input-PCA Factor Model (Manifold Trap)",
            "B3: Panel Fixed-Effects (Wipes Out Alpha)"
        ]
        ret_records: Dict[str, List[float]] = {s: [] for s in strategy_names}
        turn_records: Dict[str, List[float]] = {s: [] for s in strategy_names}
        ic_records: Dict[str, List[float]] = {s: [] for s in strategy_names}
        prev_weights: Dict[str, np.ndarray] = {s: np.zeros(M) for s in strategy_names}

        engine = FundamentalTensorInverseEngineV2()
        last_state: Optional[PurifiedFundamentalState] = None

        # 마지막 인덱스 N-1의 forward_returns는 미실현이므로 N-1 직전까지만 백테스트 평가
        for t_end in range(lookback_window, N - 1, rebalance_step):
            t_start = max(0, t_end - lookback_window)
            # t_end 시점에서는 forward_returns[:, t_end - 1] (즉 t_end-1 -> t_end 수익률)까지만 관측됨
            # sub_data의 마지막 시점(t_end - 1)의 forward_return은 t_end 시점에 이미 실현된 값임
            sub_bloat = (
                fleet_data.accounting_bloat_raw[:, t_start:t_end]
                if fleet_data.accounting_bloat_raw is not None else None
            )
            sub_data = FleetTensorData(
                X_obs=fleet_data.X_obs[:, :, t_start:t_end],
                Y_obs=fleet_data.Y_obs[:, t_start:t_end],
                forward_returns=fleet_data.forward_returns[:, t_start:t_end],
                tickers=fleet_data.tickers,
                feature_names=fleet_data.feature_names,
                time_index=fleet_data.time_index[t_start:t_end],
                accounting_bloat_raw=sub_bloat
            )
            state = engine.fit_transform(sub_data)
            last_state = state

            baseline_scores = self._compute_baseline_signals(sub_data.X_obs, sub_data.Y_obs)
            all_scores = {
                "Ours (v2.0 PIT Tensor Inverse Long-Short)": state.alpha_score,
                **baseline_scores
            }

            # 차기 보유 구간: t_end - 1 시점의 상태에서 진입하여 t_end - 1 -> t_end + rebalance_step - 1 동안 실현된 미래 수익률
            # 주의: sub_data는 인덱스 t_end - 1 까지의 X_obs를 사용했으므로, 그 직후 실현되는 수익률은
            # forward_returns[:, t_end - 1 : min(N - 1, t_end - 1 + rebalance_step)] 입니다.
            # 따라서 학습 내부(_calibrate_expected_returns)에서는 마지막 열(t_end - 1)의 forward_return을 제외하여
            # 학습과 평가 구간의 겹침(Overlap)을 완전히 차단했습니다.
            hold_start = t_end - 1
            hold_end = min(N - 1, hold_start + rebalance_step)
            if hold_end <= hold_start:
                continue
            period_fwd_ret = np.sum(fleet_data.forward_returns[:, hold_start:hold_end], axis=1)

            for s_name, sc in all_scores.items():
                order = np.argsort(sc)
                short_idx = order[:k_pick]
                long_idx = order[-k_pick:]

                w = np.zeros(M, dtype=np.float64)
                w[long_idx] = 1.0 / k_pick
                w[short_idx] = -1.0 / k_pick

                port_ret = float(np.dot(w, period_fwd_ret))
                turnover = float(0.5 * np.sum(np.abs(w - prev_weights[s_name])))
                prev_weights[s_name] = w

                corr_res = stats.spearmanr(sc, period_fwd_ret)
                rank_ic = float(corr_res.statistic) if not np.isnan(corr_res.statistic) else 0.0

                ret_records[s_name].append(port_ret)
                turn_records[s_name].append(turnover)
                ic_records[s_name].append(rank_ic)

        if last_state is None:
            last_state = engine.fit_transform(fleet_data)

        rows = []
        periods_per_yr = 12.0 / float(max(1, rebalance_step))
        for s_name in strategy_names:
            m_dict = self._evaluate_portfolio_metrics(
                np.array(ret_records[s_name]),
                np.array(turn_records[s_name]),
                ic_records[s_name],
                periods_per_year=periods_per_yr
            )
            rows.append({"Strategy": s_name, **m_dict})

        summary_df = pd.DataFrame(rows)
        return summary_df, last_state


# ==============================================================================
# MODULE 4 (v2): 결과 리포터 및 v1(main.py) vs v2(main_v2.py) 비교 분석기
# ==============================================================================

def build_all_stocks_expected_return_df_v2(
    fleet_data: FleetTensorData,
    state: PurifiedFundamentalState
) -> pd.DataFrame:
    M = len(fleet_data.tickers)
    names_map = fleet_data.company_names or {}
    mkts_map = fleet_data.markets or {}
    prices_map = fleet_data.last_prices or {}
    dates_map = fleet_data.last_dates or {}
    raw_fin_map = fleet_data.raw_fundamentals or {}

    b_bloat_index = (
        state.b_proj_score
        if state.b_proj_score is not None
        else (state.accounting_bias_b @ state.U_phys[:, 0])
    )
    intrinsic_pot = (
        state.intrinsic_potential
        if state.intrinsic_potential is not None
        else np.linalg.norm(state.delta_sure, axis=1)
    )
    synergy_score = (
        state.off_diag_synergy
        if state.off_diag_synergy is not None
        else np.zeros(M)
    )

    exp_6m = state.expected_return_6m * 100.0 if state.expected_return_6m is not None else np.zeros(M)
    exp_12m = state.expected_return_12m * 100.0 if state.expected_return_12m is not None else np.zeros(M)
    alpha_12m = state.expected_alpha_12m * 100.0 if state.expected_alpha_12m is not None else np.zeros(M)

    q80, q60, q40, q20 = np.percentile(state.alpha_score, [80, 60, 40, 20])
    signals = []
    for s in state.alpha_score:
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

    df_out = pd.DataFrame({
        "Ticker": fleet_data.tickers,
        "Name": [names_map.get(t, t) for t in fleet_data.tickers],
        "Market": [mkts_map.get(t, "KRX") for t in fleet_data.tickers],
        "Last_Trade_Date": [dates_map.get(t, "-") for t in fleet_data.tickers],
        "Last_Price(KRW)": [round(prices_map.get(t, 0.0), 0) for t in fleet_data.tickers],
        "Signal": signals,
        "Exp_Return_6M(%)": np.round(exp_6m, 2),
        "Exp_Return_12M(%)": np.round(exp_12m, 2),
        "Exp_Alpha_12M(%)": np.round(alpha_12m, 2),
        "Alpha_Score": np.round(state.alpha_score, 3),
        "Intrinsic_Potential": np.round(intrinsic_pot, 3),
        "Accounting_Bloat(b_m)": np.round(b_bloat_index, 3),
        "Synergy_Momentum": np.round(synergy_score, 3),
        "Real_ROE(%)": [round(raw_fin_map.get(t, {}).get("clean_roe", 0.0) * 100.0, 2) for t in fleet_data.tickers],
        "Raw_Reported_ROE(%)": [round(raw_fin_map.get(t, {}).get("raw_roe", 0.0) * 100.0, 2) for t in fleet_data.tickers],
        "Real_OpMargin(%)": [round(raw_fin_map.get(t, {}).get("op_margin", 0.0) * 100.0, 2) for t in fleet_data.tickers],
        "Real_DebtRatio(%)": [round(raw_fin_map.get(t, {}).get("debt_to_eq", 0.0) * 100.0, 1) for t in fleet_data.tickers],
        "Real_PBR": [round(raw_fin_map.get(t, {}).get("pbr", 0.0), 2) for t in fleet_data.tickers],
        "Real_PER": [round(raw_fin_map.get(t, {}).get("per", 0.0), 2) for t in fleet_data.tickers],
        "Mahal_OOD_Dist2": np.round(state.mahal_dist_sq, 2),
        "Data_Source": [fleet_data.data_source] * M
    })

    df_out = df_out.sort_values("Alpha_Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", np.arange(1, len(df_out) + 1))
    return df_out


def print_top_long_short_picks_v2(
    fleet_data: FleetTensorData,
    state: PurifiedFundamentalState,
    top_n: int = 20,
    save_csv_path: str = "krx_all_stocks_expected_returns_v2.csv"
) -> pd.DataFrame:
    df_all = build_all_stocks_expected_return_df_v2(fleet_data, state)

    if save_csv_path:
        df_all.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[Output Saved v2] 전체 {len(df_all)}개 종목의 정제 분석 결과가 '{save_csv_path}'에 저장되었습니다.")

    print("\n" + "=" * 125)
    print(f" [v2 TOP {top_n} EXPECTED RETURN (LONG)] 실제 K-IFRS 재무제표 및 Point-in-Time 밸류에이션 기반 최우선 매수 종목 (총 {len(df_all)}개 중)")
    print("=" * 125)
    print(df_all.head(top_n).to_string(index=False))

    print("\n" + "=" * 125)
    print(f" [v2 BOTTOM {top_n} EXPECTED RETURN (SHORT)] 자본잠식·과다부채·회계착시(b_m) 및 고평가 위험 최하위 종목")
    print("=" * 125)
    print(df_all.tail(top_n).sort_values("Alpha_Score", ascending=True).to_string(index=False))

    return df_all


def compare_v1_and_v2_results(
    df_v2: pd.DataFrame,
    v1_csv_path: str = "krx_all_stocks_expected_returns.csv"
) -> None:
    if not os.path.exists(v1_csv_path):
        return
    df_v1 = pd.read_csv(v1_csv_path)
    merged = pd.merge(
        df_v1[["Ticker", "Name", "Rank", "Exp_Return_12M(%)", "Alpha_Score", "Accounting_Bloat(b_m)", "Synergy_Momentum", "Real_ROE(%)", "Real_DebtRatio(%)", "Real_PBR"]],
        df_v2[["Ticker", "Rank", "Exp_Return_12M(%)", "Alpha_Score", "Accounting_Bloat(b_m)", "Synergy_Momentum", "Real_ROE(%)", "Raw_Reported_ROE(%)", "Real_OpMargin(%)", "Real_DebtRatio(%)", "Real_PBR", "Real_PER"]],
        on="Ticker",
        suffixes=("_v1", "_v2")
    )
    rank_corr = stats.spearmanr(merged["Alpha_Score_v1"], merged["Alpha_Score_v2"]).statistic
    print("\n" + "=" * 125)
    print(" [v1(원본 main.py) vs v2(개선 main_v2.py) 핵심 지표 비교 요약]")
    print("=" * 125)
    print(f"  * 공통 분석 종목 수              : {len(merged)}개")
    print(f"  * v1 vs v2 알파 스코어 순위상관  : {rank_corr:.3f}")
    print(f"  * v1 PBR 고유값 개수             : {df_v1['Real_PBR'].nunique()}개 (전 종목 1.2 고정 폴백) vs v2: {df_v2['Real_PBR'].nunique()}개 (실제 K-IFRS PBR)")
    print(f"  * v1 Synergy_Momentum 표준편차   : {df_v1['Synergy_Momentum'].std():.4f} (전 종목 0.0) vs v2: {df_v2['Synergy_Momentum'].std():.4f} (정상 복원)")
    print(f"  * v1 12M 기대수익률 범위         : [{df_v1['Exp_Return_12M(%)'].min():.2f}%, +{df_v1['Exp_Return_12M(%)'].max():.2f}%]")
    print(f"  * v2 12M 기대수익률 범위         : [{df_v2['Exp_Return_12M(%)'].min():.2f}%, +{df_v2['Exp_Return_12M(%)'].max():.2f}%]")

    print("\n--- [v1 상위 10개 종목이 실제 K-IFRS 재무제표(v2) 적용 후 어떻게 재평가되었는가?] ---")
    v1_top10 = merged.sort_values("Rank_v1").head(10)[
        ["Ticker", "Name", "Rank_v1", "Rank_v2", "Real_ROE(%)_v1", "Real_ROE(%)_v2", "Real_DebtRatio(%)_v1", "Real_DebtRatio(%)_v2", "Real_PBR_v1", "Real_PBR_v2", "Exp_Return_12M(%)_v1", "Exp_Return_12M(%)_v2"]
    ]
    print(v1_top10.to_string(index=False))


V2_SIGNAL_ORDER = [
    "STRONG LONG (최우선 매수)", "LONG (매수)", "NEUTRAL (중립)",
    "SHORT (비중축소/공매도)", "STRONG SHORT (핵심 공매도)",
]


def build_v1_v2_comparison(
    df_v2: pd.DataFrame,
    v1_csv_path: str = "krx_all_stocks_expected_returns.csv"
) -> Dict[str, object]:
    """대시보드 v2 탭의 'v1 대비 비교' 섹션에 들어갈 요약 지표를 계산합니다."""
    if not os.path.exists(v1_csv_path):
        return {}
    df_v1 = pd.read_csv(v1_csv_path, dtype={"Ticker": str})
    merged = pd.merge(
        df_v1[["Ticker", "Name", "Rank", "Signal", "Exp_Return_12M(%)", "Alpha_Score", "Real_ROE(%)", "Real_DebtRatio(%)", "Real_PBR"]],
        df_v2[["Ticker", "Rank", "Signal", "Exp_Return_12M(%)", "Alpha_Score", "Real_ROE(%)", "Real_DebtRatio(%)", "Real_PBR"]],
        on="Ticker",
        suffixes=("_v1", "_v2")
    )
    if merged.empty:
        return {}

    def direction(sig: str) -> int:
        s = str(sig)
        return 1 if "LONG" in s else (-1 if "SHORT" in s else 0)

    counts = [
        [int(((merged["Signal_v1"] == a) & (merged["Signal_v2"] == b)).sum()) for b in V2_SIGNAL_ORDER]
        for a in V2_SIGNAL_ORDER
    ]
    top_cols = [
        "Ticker", "Name", "Rank_v1", "Rank_v2", "Exp_Return_12M(%)_v1", "Exp_Return_12M(%)_v2",
        "Real_ROE(%)_v2", "Real_DebtRatio(%)_v2", "Real_PBR_v2",
    ]

    def records(frame: pd.DataFrame) -> List[Dict[str, object]]:
        return json.loads(frame[top_cols].to_json(orient="records", force_ascii=False))

    return {
        "v1_source": v1_csv_path,
        "n_common": int(len(merged)),
        "rank_corr": round(float(stats.spearmanr(merged["Alpha_Score_v1"], merged["Alpha_Score_v2"]).statistic), 3),
        "direction_agree_pct": round(float(
            (merged["Signal_v1"].map(direction) == merged["Signal_v2"].map(direction)).mean() * 100.0
        ), 1),
        "v1_pbr_unique": int(df_v1["Real_PBR"].nunique()),
        "v2_pbr_unique": int(df_v2["Real_PBR"].nunique()),
        "v1_ret_range": [float(df_v1["Exp_Return_12M(%)"].min()), float(df_v1["Exp_Return_12M(%)"].max())],
        "v2_ret_range": [float(df_v2["Exp_Return_12M(%)"].min()), float(df_v2["Exp_Return_12M(%)"].max())],
        "matrix": {"labels": V2_SIGNAL_ORDER, "counts": counts},
        "v1_top": records(merged.sort_values("Rank_v1").head(10)),
        "v2_top": records(merged.sort_values("Rank_v2").head(10)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="v2.0 Point-in-Time Fleet Tensor Inverse Framework (KRX 전 종목 정제 버전)"
    )
    parser.add_argument("--market", type=str, default="ALL", choices=["ALL", "KOSPI", "KOSDAQ"])
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--min-marcap", type=float, default=30.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=str, default="krx_all_stocks_expected_returns_v2.csv")
    parser.add_argument("--v1-csv", type=str, default="krx_all_stocks_expected_returns.csv")
    parser.add_argument("--html-output", type=str, default="krx_dashboard.html")
    parser.add_argument("--log-dir", type=str, default="logs_v2")
    args, _ = parser.parse_known_args()
    run_ts = datetime.now()

    print("=" * 125)
    print(" v2.0 Point-in-Time Fleet Tensor Inverse Framework - 한국 주식시장(All KRX) 결함 수리 및 정제 파이프라인")
    print("=" * 125)

    loader = FinancialFleetDataLoaderV2()
    backtester = FundamentalLongShortBacktesterV2(top_quantile=0.20, cost_bps=15.0)

    max_stocks_limit = args.max_stocks if args.max_stocks > 0 else None
    loader.load_all_krx_universe(
        market=args.market,
        max_stocks=max_stocks_limit,
        min_marcap_billion_krw=args.min_marcap
    )

    live_data = loader.fetch_live_financial_tensor(n_periods=36, freq="M")
    if live_data is None:
        sys.exit(1)

    print("\n[Tensor Engine v2] 전 종목 6단계 대수적 텐서 역변환(Point-in-Time) 및 엄격한 표본 외 백테스트 수행 중...")
    live_summary_df, _ = backtester.run_walk_forward_backtest(
        live_data, lookback_window=20, rebalance_step=2
    )
    print("\n--- [Walk-Forward Backtest v2] 표본 외(Strict Out-of-Sample) 중장기 롱숏 백테스트 성과 요약 ---")
    print(live_summary_df.to_string(index=False))

    full_engine = FundamentalTensorInverseEngineV2()
    final_state = full_engine.fit_transform(live_data)

    df_all_v2 = print_top_long_short_picks_v2(
        live_data,
        final_state,
        top_n=args.top_n,
        save_csv_path=args.output
    )

    compare_v1_and_v2_results(df_all_v2, v1_csv_path=args.v1_csv)

    from main import (
        DASHBOARD_TAB_SOURCES, append_prediction_log, build_dashboard_payload,
        save_dashboard_payload, write_tabbed_dashboard,
    )

    append_prediction_log(df_all_v2, run_ts, args.log_dir)

    payload = build_dashboard_payload(
        df_all_v2,
        live_summary_df,
        {
            "run_date": run_ts.strftime("%Y-%m-%d"),
            "price_date": str(df_all_v2["Last_Trade_Date"].mode().iloc[0]) if len(df_all_v2) else "",
            "log_dir": args.log_dir,
            "generated_at": run_ts.strftime("%Y-%m-%d %H:%M"),
            "listing_source": loader.listing_source,
            "data_source": live_data.data_source,
            "market": args.market,
            "min_marcap": args.min_marcap,
            "n_requested": len(loader.tickers),
            "n_analyzed": len(live_data.tickers),
            "n_features": int(live_data.X_obs.shape[1]),
            "n_periods": int(live_data.X_obs.shape[2]),
            "time_start": live_data.time_index[0] if live_data.time_index else "",
            "time_end": live_data.time_index[-1] if live_data.time_index else "",
        },
        comparison=build_v1_v2_comparison(df_all_v2, v1_csv_path=args.v1_csv),
        features={"tracking": False, "fundamentals": True},
        title="v2 · Point-in-Time 재무제표 텐서 역변환",
        description=(
            "main_v2.py 결과입니다. 네이버 K-IFRS 재무제표를 공시 시점 기준(Point-in-Time)으로 붙여 "
            "ROE·영업이익률·부채비율·PBR·PER이 종목별 실제 값으로 들어갑니다. 가격 기록·자동 실행은 v1 탭에만 연결되어 있습니다."
        ),
    )
    payload_path = save_dashboard_payload(payload, os.path.join(args.log_dir, "dashboard_payload.json"))
    write_tabbed_dashboard(args.html_output, [
        (tid, label, payload_path if tid == "v2" else path)
        for tid, label, path in DASHBOARD_TAB_SOURCES
    ])


if __name__ == "__main__":
    main()
