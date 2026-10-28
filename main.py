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

# Optional dependency check for live Yahoo Finance & KRX listing download
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
# 데이터 구조 정의 (Data Structures)
# ==============================================================================

@dataclass
class FleetTensorData:
    """
    다중 종목(M) x 재무/회계 피처(D) x 시계열 윈도우(N) 텐서 데이터 컨테이너
    """
    X_obs: np.ndarray               # Shape: (M, D, N) - 관측된 원시 재무/팩터 텐서
    Y_obs: np.ndarray               # Shape: (M, N)    - 관측된 타겟 수익성/수익률 반응
    forward_returns: np.ndarray     # Shape: (M, N)    - 차기(Out-of-sample) 주식 초과수익률
    tickers: List[str]              # 종목 코드 리스트 (길이 M)
    feature_names: List[str]        # 재무/회계 피처 이름 리스트 (길이 D)
    time_index: List[str]           # 시계열 인덱스 리스트 (길이 N)
    ground_truth: Optional[Dict[str, np.ndarray]] = None
    company_names: Optional[Dict[str, str]] = None
    markets: Optional[Dict[str, str]] = None
    last_prices: Optional[Dict[str, float]] = None
    last_dates: Optional[Dict[str, str]] = None
    raw_fundamentals: Optional[Dict[str, Dict[str, float]]] = None
    data_source: str = "LIVE_KRX_MARKET_DATA"


@dataclass
class PurifiedFundamentalState:
    """
    6단계 텐서 역변환을 통해 정제된 기업별 펀더멘털 상태 및 글로벌 법칙 파라미터
    """
    r_hat: int                      # 추정된 진성 펀더멘털 잠재 차원 수
    U_phys: np.ndarray              # Shape: (D, r_hat) - Stiefel 매니폴드 직교 알파 기저
    W_hat: np.ndarray               # Shape: (M, D)     - 종목별 대각 스케일/레버리지 왜곡 추정치
    Sigma_xi: np.ndarray            # Shape: (D, D)     - 추정된 회계/시장 측정오차(EIV) 공분산
    H_phys: np.ndarray              # Shape: (r_hat, r_hat) - 대칭 양정치 진성 펀더멘털 헤시안 곡률
    gamma_phys: np.ndarray          # Shape: (r_hat,)   - 3차 비조화(Anharmonicity) 계수 벡터
    delta_raw: np.ndarray           # Shape: (M, r_hat) - 모로-요시다 역변환으로 복원된 잠재 위치
    delta_sure: np.ndarray          # Shape: (M, r_hat) - 마할라노비스 OOD 게이팅 적용된 위치
    accounting_bias_b: np.ndarray   # Shape: (M, D)     - 분리된 기업별 정적 회계 착시/프리미엄 오프셋
    alpha_eiv: np.ndarray           # Shape: (M, r_hat) - EIV 보정된 기업별 1차 한계 기울기
    mahal_dist_sq: np.ndarray       # Shape: (M,)       - 종목별 마할라노비스 OOD 거리 제곱
    z_aligned: np.ndarray           # Shape: (M, r_hat, N) - 정제 및 정렬된 글로벌 펀더멘털 좌표
    y_pure: np.ndarray              # Shape: (M, N)     - 고정 오프셋이 제거된 순수 타겟 반응
    alpha_score: np.ndarray         # Shape: (M,)       - 중장기 롱숏 포트폴리오용 최종 알파 스코어
    expected_return_6m: Optional[np.ndarray] = None   # Shape: (M,) - 향후 6개월 예상 수익률 (소수)
    expected_return_12m: Optional[np.ndarray] = None  # Shape: (M,) - 향후 12개월 예상 수익률 (소수)
    expected_alpha_12m: Optional[np.ndarray] = None   # Shape: (M,) - 향후 12개월 시장 대비 예상 초과수익률 (소수)
    intrinsic_potential: Optional[np.ndarray] = None  # Shape: (M,) - 진성 펀더멘털 내재가치 포텐셜
    b_proj_score: Optional[np.ndarray] = None         # Shape: (M,) - 1차 주축 방향 회계적 착시(Bloat) 오프셋
    off_diag_synergy: Optional[np.ndarray] = None     # Shape: (M,) - 비대각 양(+)의 시너지 결합 에너지


# ==============================================================================
# MODULE 1: 한국 주식시장 전 종목(All KRX) 수집기 및 패널 텐서 빌더
# ==============================================================================

class FinancialFleetDataLoader:
    """
    한국 주식시장(KOSPI, KOSDAQ) 전 종목의 티커 및 메타데이터를 자동으로 수집하고,
    배치(Batch) 다운로드와 멀티스레딩을 통해 전체 상장 종목의 3차원 관측 텐서 (M x D x N)를 구축합니다.
    """

    KRX_UNIVERSE = [
        "005930.KS", "000660.KS", "042700.KS", "012450.KS", "267260.KS", "005380.KS",
        "000270.KS", "207940.KS", "196170.KQ", "058470.KQ", "105560.KS", "079550.KS",
        "329180.KS", "035420.KS", "055550.KS", "012330.KS", "086790.KS", "009150.KS",
        "034020.KS", "068270.KS", "028260.KS", "066570.KS", "005490.KS", "035720.KS",
        "006400.KS", "051910.KS", "003670.KS", "247540.KQ", "036570.KS", "096770.KS",
        "009830.KS", "011170.KS"
    ]

    DEFAULT_UNIVERSE = KRX_UNIVERSE

    HTTP_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    _SSL_CTX = ssl.create_default_context()
    _PREFERRED_NAME_RE = re.compile(
        r"(?:\d우B|\d우C|우B|우C|\d우|우)(?:\(|$)|전환우선|우선주"
    )

    def __init__(self, tickers: Optional[List[str]] = None):
        self.tickers = tickers if tickers is not None else self.DEFAULT_UNIVERSE
        self.company_names: Dict[str, str] = {}
        self.markets: Dict[str, str] = {}
        self.listing_source: str = ""
        self.market_caps: Dict[str, float] = {}

    @classmethod
    def _http_get_bytes(
        cls,
        url: str,
        timeout: int = 25,
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
        """보통주 사업법인만 남기고 우선주·스팩·비정상 코드를 제외합니다."""
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
    def _parse_kind_corp_rows(
        cls,
        html_text: str,
        default_market: str = ""
    ) -> List[Tuple[str, str, str]]:
        """
        KRX KIND 상장법인목록(HTML disguised as XLS)을 헤더 기준으로 파싱합니다.
        2026년 기준 컬럼: 회사명, 시장구분, 종목코드, ...
        구버전 컬럼: 회사명, 종목코드, ... 도 함께 지원합니다.
        """
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
                row = {
                    headers[i]: cells[i]
                    for i in range(min(len(headers), len(cells)))
                }
                name = row.get("회사명", cells[0])
                if "종목코드" in row:
                    code = row["종목코드"]
                elif len(cells) >= 2:
                    code = cells[1]
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
        print("[KRX Loader] KRX KIND(한국거래소 상장공시시스템)에서 전 종목 리스트를 직접 다운로드합니다...")
        url = (
            "https://kind.krx.co.kr/corpgeneral/corpList.do"
            "?method=download&searchType=13"
        )
        raw_bytes = self._http_get_bytes(
            url,
            timeout=30,
            extra_headers={
                "Referer": "https://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage"
            }
        )
        html_text = raw_bytes.decode("euc-kr", errors="ignore")
        rows = self._parse_kind_corp_rows(html_text)
        if market_upper in ("KOSPI", "KS"):
            rows = [r for r in rows if r[2] == "KOSPI"]
        elif market_upper in ("KOSDAQ", "KQ"):
            rows = [r for r in rows if r[2] == "KOSDAQ"]
        print(f"[KRX Loader] KIND 파싱 완료: {len(rows)}개 보통주 (시장 필터: {market_upper})")
        return rows

    def _fetch_pykrx_universe(self, market_upper: str) -> List[Tuple[str, str, str]]:
        if not HAS_PYKRX:
            return []
        print("[KRX Loader] pykrx를 통해 상장 종목 리스트를 수집합니다...")
        target = []
        if market_upper in ("ALL", "KOSPI", "KS"):
            target.append(("KOSPI", ".KS"))
        if market_upper in ("ALL", "KOSDAQ", "KQ"):
            target.append(("KOSDAQ", ".KQ"))
        rows: List[Tuple[str, str, str]] = []
        for mkt_name, suffix in target:
            for code in pykrx_stock.get_market_ticker_list(market=mkt_name):
                name = pykrx_stock.get_market_ticker_name(code)
                if not self._is_common_operating_stock(name, code):
                    continue
                rows.append((f"{str(code).zfill(6)}{suffix}", str(name), mkt_name))
        return rows

    def _fetch_fdr_universe(self, market_upper: str) -> List[Tuple[str, str, str]]:
        if not HAS_FDR:
            return []
        print("[KRX Loader] FinanceDataReader를 통해 상장 종목 리스트를 수집합니다...")
        listing_keys = []
        if market_upper in ("ALL", "KOSPI", "KS"):
            listing_keys.append("KOSPI")
        if market_upper in ("ALL", "KOSDAQ", "KQ"):
            listing_keys.append("KOSDAQ")
        rows: List[Tuple[str, str, str]] = []
        for mkt_name in listing_keys:
            df = fdr.StockListing(mkt_name)
            if df is None or df.empty:
                continue
            code_col = next((c for c in ("Code", "Symbol", "종목코드") if c in df.columns), None)
            name_col = next((c for c in ("Name", "회사명", "종목명") if c in df.columns), None)
            if code_col is None or name_col is None:
                continue
            suffix = ".KQ" if mkt_name == "KOSDAQ" else ".KS"
            marcap_col = next((c for c in ("Marcap", "MarCap", "시가총액") if c in df.columns), None)
            for _, rec in df.iterrows():
                code = str(rec[code_col]).strip().upper().zfill(6)
                name = str(rec[name_col]).strip()
                if not self._is_common_operating_stock(name, code):
                    continue
                ticker = f"{code}{suffix}"
                rows.append((ticker, name, mkt_name))
                if marcap_col is not None:
                    try:
                        cap_val = float(rec[marcap_col])
                        if cap_val > 0:
                            self.market_caps[ticker] = cap_val
                    except (TypeError, ValueError):
                        pass
        return rows

    def _fetch_naver_marcap_map(self) -> Dict[str, float]:
        """네이버 시가총액 랭킹 API에서 종목코드→시가총액(원) 맵을 만듭니다."""
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
                    extra_headers={
                        "Accept": "application/json",
                        "Referer": "https://m.stock.naver.com/",
                    }
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
                        # marketSumRaw 단위: 백만 원
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
        """
        KOSPI/KOSDAQ 상장 보통주를 수집합니다.
        1순위 pykrx, 2순위 FinanceDataReader, 3순위 KRX KIND(상장법인목록).
        KIND는 로그인 없이 동작하며, 컬럼은 헤더명으로 매핑합니다.
        """
        market_upper = str(market or "ALL").strip().upper()
        collected_rows: List[Tuple[str, str, str]] = []
        source_errors: List[str] = []

        for source_name, fetcher in (
            ("pykrx", self._fetch_pykrx_universe),
            ("FinanceDataReader", self._fetch_fdr_universe),
            ("KRX KIND", self._fetch_kind_universe),
        ):
            if collected_rows:
                break
            try:
                collected_rows = fetcher(market_upper)
                if collected_rows:
                    self.listing_source = source_name
            except Exception as e:
                source_errors.append(f"{source_name}: {e}")
                print(f"[KRX Loader] {source_name} 수집 실패: {e}")

        if len(collected_rows) == 0:
            detail = "; ".join(source_errors) if source_errors else "모든 소스가 빈 결과를 반환"
            raise RuntimeError(
                "[Error] 한국거래소(KRX) 종목 리스트를 가져오지 못했습니다. "
                f"({detail})"
            )

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
                    print("[KRX Loader] 시가총액 필터를 위해 네이버 시총 맵을 수집합니다...")
                    self.market_caps = self._fetch_naver_marcap_map()
                    print(f"[KRX Loader] 시가총액 맵 {len(self.market_caps)}건 수집")
                except Exception as e:
                    print(f"[KRX Loader] 시가총액 맵 수집 실패, 필터를 건너뜁니다: {e}")
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
                    f"[KRX Loader] 시가총액 {min_marcap_billion_krw:.1f}십억 원 이상 필터: "
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

        print(
            f"[KRX Loader] 실제 상장 종목 총 {len(self.tickers)}개 로드 완료 "
            f"(시장 구분: {market_upper}, 소스: {self.listing_source})"
        )
        return self.tickers

    @classmethod
    def _fetch_naver_ohlcv(cls, code_6digit: str, timeframe: str = "month") -> Optional[pd.DataFrame]:
        """네이버 차트 API에서 개별 종목의 월/주 OHLCV를 가져옵니다."""
        end = datetime.now()
        start = end - timedelta(days=365 * 5 + 40)
        url = (
            "https://api.finance.naver.com/siseJson.naver"
            f"?symbol={code_6digit}&requestType=1"
            f"&startTime={start.strftime('%Y%m%d')}&endTime={end.strftime('%Y%m%d')}"
            f"&timeframe={timeframe}"
        )
        try:
            raw = cls._http_get_bytes(
                url,
                timeout=12,
                extra_headers={"Referer": "https://finance.naver.com/"}
            )
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
            vol_vals = (
                pd.to_numeric(df[vol_col], errors="coerce").to_numpy()
                if vol_col is not None else 0.0
            )
            idx = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
            out = pd.DataFrame(
                {"Close": close_vals, "Volume": vol_vals},
                index=pd.DatetimeIndex(idx.to_numpy())
            )
            out = out[out.index.notna()].sort_index()
            out = out[out["Close"].fillna(0) > 0]
            return out if len(out) >= 6 else None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, SyntaxError):
            return None

    def _download_naver_price_panel(
        self,
        tickers: List[str],
        freq: str = "M",
        max_workers: int = 16
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        timeframe = "month" if freq == "M" else "week"
        close_map: Dict[str, pd.Series] = {}
        volume_map: Dict[str, pd.Series] = {}
        completed = 0
        t0 = time.time()
        print(f"  -> [Naver Chart] {len(tickers)}개 종목 시세 수집 (timeframe={timeframe})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tk = {
                executor.submit(self._fetch_naver_ohlcv, tk.split(".")[0], timeframe): tk
                for tk in tickers
            }
            for fut in as_completed(future_to_tk):
                tk = future_to_tk[fut]
                completed += 1
                if completed % 200 == 0 or completed == len(tickers):
                    print(
                        f"     ... 네이버 시세 진행률: {completed} / {len(tickers)} "
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
    def _fetch_naver_real_fundamentals(code_6digit: str) -> Dict[str, float]:
        """
        네이버 금융(finance.naver.com) 기업현황 HTML에서 실제 한국 회계기준(K-IFRS) 재무비율
        (ROE, 영업이익률, 순이익률, 부채비율, PER, PBR)을 실시간으로 파싱합니다.
        """
        url = f"https://finance.naver.com/item/main.naver?code={code_6digit}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        out: Dict[str, float] = {}
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                html = resp.read().decode("euc-kr", errors="ignore")

            def _extract_metric_row(row_title: str) -> List[float]:
                idx = html.find(row_title)
                if idx == -1:
                    return []
                sub = html[idx:idx + 2500]
                tr_end = sub.find("</tr>")
                if tr_end != -1:
                    sub = sub[:tr_end]
                nums = re.findall(r"[-+]?\d[\d,]*\.\d+|[-+]?\d[\d,]+", sub)
                vals = []
                for n_str in nums:
                    try:
                        v = float(n_str.replace(",", ""))
                        if abs(v) < 10000:
                            vals.append(v)
                    except ValueError:
                        pass
                return vals

            op_vals = _extract_metric_row("영업이익률")
            net_vals = _extract_metric_row("순이익률")
            roe_vals = _extract_metric_row("ROE(지배주주)")
            debt_vals = _extract_metric_row("부채비율")
            per_vals = _extract_metric_row("PER(배)")
            pbr_vals = _extract_metric_row("PBR(배)")

            if op_vals:
                out["op_margin"] = np.clip(op_vals[-1] / 100.0, -0.60, 0.80)
                if len(op_vals) >= 2:
                    out["op_margin_prev"] = np.clip(op_vals[-2] / 100.0, -0.60, 0.80)
            if net_vals:
                out["profit_margin"] = np.clip(net_vals[-1] / 100.0, -0.60, 0.70)
            if roe_vals:
                out["roe"] = np.clip(roe_vals[-1] / 100.0, -0.80, 1.20)
                if len(roe_vals) >= 2:
                    out["roe_prev"] = np.clip(roe_vals[-2] / 100.0, -0.80, 1.20)
            if debt_vals:
                out["debt_to_eq"] = np.clip(debt_vals[-1] / 100.0, 0.01, 6.0)
            if per_vals:
                out["per"] = np.clip(per_vals[-1], -200.0, 500.0)
            if pbr_vals:
                out["pbr"] = np.clip(pbr_vals[-1], 0.05, 30.0)
        except Exception:
            pass
        return out

    @staticmethod
    def _extract_single_ticker_fundamentals(
        ticker: str,
        r_series: np.ndarray,
        v_series: np.ndarray,
        fetch_statement_api: bool = True
    ) -> Dict[str, float]:
        """
        개별 종목의 실제 재무제표(네이버 금융 K-IFRS 및 Yahoo Finance)를 추출하고,
        결측 항목만 시계열 통계량으로 보완합니다.
        """
        ann_ret = float(np.mean(r_series) * 12.0)
        ann_vol = float(np.std(r_series) * np.sqrt(12.0) + 1e-4)
        sharpe_proxy = np.clip(ann_ret / ann_vol, -2.0, 3.0)
        downside_vol = (
            float(np.std(r_series[r_series < 0]) * np.sqrt(12.0) + 1e-4)
            if np.any(r_series < 0) else ann_vol
        )
        vol_skew = np.clip((ann_vol - downside_vol) / ann_vol, -1.0, 1.0)
        v_growth = float(np.mean(v_series[-6:]) - np.mean(v_series[:-6])) if len(v_series) > 12 else 0.0

        snap = {
            "roe": np.clip(0.08 + 0.06 * sharpe_proxy, -0.45, 0.85),
            "roa": np.clip(0.05 + 0.04 * sharpe_proxy, -0.25, 0.50),
            "op_margin": np.clip(0.09 + 0.05 * sharpe_proxy + 0.02 * vol_skew, -0.40, 0.65),
            "gross_margin": np.clip(0.28 + 0.07 * sharpe_proxy, 0.05, 0.90),
            "profit_margin": np.clip(0.07 + 0.05 * sharpe_proxy, -0.40, 0.55),
            "debt_to_eq": np.clip(0.75 - 0.12 * sharpe_proxy + 0.15 * ann_vol, 0.05, 4.0),
            "rev_growth": np.clip(0.06 + 0.30 * ann_ret + 0.05 * v_growth, -0.45, 1.20),
            "earn_growth": np.clip(0.08 + 0.40 * ann_ret + 0.06 * vol_skew, -0.65, 1.60),
            "rd_ratio": np.clip(0.05 + 0.06 * np.maximum(0.0, ann_vol - 0.20), 0.01, 0.45),
            "capex_ratio": np.clip(0.06 + 0.04 * np.maximum(0.0, sharpe_proxy) + 0.03 * abs(v_growth), 0.01, 0.40),
            "fcf_margin": np.clip(0.08 + 0.06 * sharpe_proxy - 0.04 * ann_vol, -0.35, 0.55),
            "accrual_ratio": np.clip(-0.02 * sharpe_proxy + 0.04 * v_growth, -0.25, 0.25),
            "pbr": 1.20,
            "per": 12.0,
            "is_real_fin": 0.0
        }

        code_6 = ticker.split(".")[0].strip()
        if len(code_6) == 6 and code_6.isdigit():
            naver_fin = FinancialFleetDataLoader._fetch_naver_real_fundamentals(code_6)
            if naver_fin:
                snap.update(naver_fin)
                snap["is_real_fin"] = 1.0
                if "roe" in naver_fin and "debt_to_eq" in naver_fin:
                    snap["roa"] = np.clip(naver_fin["roe"] / (1.0 + naver_fin["debt_to_eq"]), -0.35, 0.60)
                if "op_margin" in naver_fin:
                    snap["gross_margin"] = np.clip(naver_fin["op_margin"] + 0.16, 0.02, 0.92)
                    snap["fcf_margin"] = np.clip(naver_fin["op_margin"] * 0.78, -0.40, 0.60)
                if "roe" in naver_fin and "roe_prev" in naver_fin:
                    snap["earn_growth"] = np.clip(naver_fin["roe"] - naver_fin["roe_prev"], -0.80, 1.50)
                if "op_margin" in naver_fin and "op_margin_prev" in naver_fin:
                    snap["rev_growth"] = np.clip(
                        (naver_fin["op_margin"] - naver_fin["op_margin_prev"]) * 1.5 + 0.20 * ann_ret,
                        -0.50, 1.20
                    )
                if "profit_margin" in naver_fin and "op_margin" in naver_fin:
                    snap["accrual_ratio"] = np.clip(
                        naver_fin["profit_margin"] - naver_fin["op_margin"] * 0.85,
                        -0.25, 0.25
                    )

        if not fetch_statement_api or not HAS_YFINANCE:
            return snap

        try:
            tk = yf.Ticker(ticker)
            info = tk.info if hasattr(tk, "info") and tk.info else {}
            q_fin = tk.quarterly_financials
            q_bs = tk.quarterly_balance_sheet
            q_cf = tk.quarterly_cashflow

            if info:
                snap["roe"] = np.clip(float(info.get("returnOnEquity", snap["roe"]) or snap["roe"]), -0.5, 1.5)
                snap["roa"] = np.clip(float(info.get("returnOnAssets", snap["roa"]) or snap["roa"]), -0.3, 0.8)
                snap["op_margin"] = np.clip(float(info.get("operatingMargins", snap["op_margin"]) or snap["op_margin"]), -0.5, 0.8)
                snap["gross_margin"] = np.clip(float(info.get("grossMargins", snap["gross_margin"]) or snap["gross_margin"]), 0.0, 0.95)
                snap["profit_margin"] = np.clip(float(info.get("profitMargins", snap["profit_margin"]) or snap["profit_margin"]), -0.5, 0.7)
                d2e_raw = info.get("debtToEquity", None)
                if d2e_raw is not None:
                    snap["debt_to_eq"] = np.clip(float(d2e_raw) / 100.0, 0.0, 5.0)
                snap["rev_growth"] = np.clip(float(info.get("revenueGrowth", snap["rev_growth"]) or snap["rev_growth"]), -0.5, 1.5)
                snap["earn_growth"] = np.clip(float(info.get("earningsGrowth", snap["earn_growth"]) or snap["earn_growth"]), -0.8, 2.0)

            if q_fin is not None and not q_fin.empty and "Total Revenue" in q_fin.index:
                rev_series = q_fin.loc["Total Revenue"].dropna()
                if len(rev_series) > 0 and rev_series.iloc[0] != 0:
                    if "Research And Development" in q_fin.index:
                        rd_val = q_fin.loc["Research And Development"].dropna()
                        if len(rd_val) > 0:
                            snap["rd_ratio"] = np.clip(float(rd_val.iloc[0] / rev_series.iloc[0]), 0.0, 0.6)
                    if q_cf is not None and not q_cf.empty:
                        if "Capital Expenditure" in q_cf.index:
                            cx_val = q_cf.loc["Capital Expenditure"].dropna()
                            if len(cx_val) > 0:
                                snap["capex_ratio"] = np.clip(abs(float(cx_val.iloc[0] / rev_series.iloc[0])), 0.0, 0.5)
                        if "Free Cash Flow" in q_cf.index:
                            fcf_val = q_cf.loc["Free Cash Flow"].dropna()
                            if len(fcf_val) > 0:
                                snap["fcf_margin"] = np.clip(float(fcf_val.iloc[0] / rev_series.iloc[0]), -0.4, 0.6)

            if q_bs is not None and not q_bs.empty and "Total Assets" in q_bs.index:
                assets = q_bs.loc["Total Assets"].dropna()
                if len(assets) > 0 and assets.iloc[0] > 0 and q_cf is not None and not q_cf.empty:
                    if "Operating Cash Flow" in q_cf.index and q_fin is not None and "Net Income" in q_fin.index:
                        ocf = q_cf.loc["Operating Cash Flow"].dropna()
                        ni = q_fin.loc["Net Income"].dropna()
                        if len(ocf) > 0 and len(ni) > 0:
                            snap["accrual_ratio"] = np.clip(float((ni.iloc[0] - ocf.iloc[0]) / assets.iloc[0]), -0.3, 0.3)
        except Exception:
            pass

        return snap

    def fetch_live_financial_tensor(
        self,
        n_periods: int = 36,
        freq: str = "M",
        batch_size: int = 100,
        max_workers: int = 24,
        fetch_statement_api: bool = False
    ) -> Optional[FleetTensorData]:
        """
        수천 개의 KRX 종목을 배치(Batch) 단위로 다운로드하고, 네이버 금융 실시간 재무비율(K-IFRS)을
        멀티스레딩으로 수집하여 3차원 텐서 X_obs (M x D x N) 및 Y_obs (M x N)를 구축합니다.
        """
        t0 = time.time()
        total_requested = len(self.tickers)
        print(f"[Data Loader] 총 {total_requested}개 실제 KRX 종목의 주가 시계열 및 재무 텐서 구축 시작 (Batch={batch_size})...")

        try:
            close_chunks = []
            volume_chunks = []
            interval_str = "1mo" if freq == "M" else "1wk"

            if HAS_YFINANCE:
                print("  -> [1/2: 주가 시계열] yfinance 배치 다운로드를 시도합니다...")
            else:
                print("  -> [1/2: 주가 시계열] yfinance 미설치, 네이버 차트 API로 시세를 수집합니다.")

            for start_idx in range(0, total_requested, batch_size):
                if not HAS_YFINANCE:
                    break
                batch_tickers = self.tickers[start_idx:start_idx + batch_size]
                elapsed = time.time() - t0
                print(
                    f"  -> [1/2: 주가 시계열 다운로드] {start_idx + 1} ~ {min(start_idx + batch_size, total_requested)} "
                    f"/ {total_requested} 종목 진행 중... (경과: {elapsed:.1f}초)"
                )
                try:
                    price_data = yf.download(
                        batch_tickers,
                        period="5y",
                        interval=interval_str,
                        auto_adjust=True,
                        progress=False,
                        threads=True
                    )
                    if price_data is None or price_data.empty:
                        continue

                    if isinstance(price_data.columns, pd.MultiIndex):
                        lvl0 = price_data.columns.get_level_values(0)
                        if "Close" in lvl0:
                            c_df = price_data["Close"]
                            v_df = price_data["Volume"]
                        else:
                            c_df = price_data.xs("Close", axis=1, level=-1)
                            v_df = price_data.xs("Volume", axis=1, level=-1)
                    else:
                        only_tk = batch_tickers[0]
                        c_df = pd.DataFrame({only_tk: price_data["Close"]})
                        v_df = pd.DataFrame({only_tk: price_data["Volume"]})

                    close_chunks.append(c_df)
                    volume_chunks.append(v_df)
                except Exception as batch_err:
                    print(f"  [Warning] 배치 다운로드 일부 건너뜀 ({batch_err})")
                    continue

            if close_chunks:
                close_df = pd.concat(close_chunks, axis=1)
                volume_df = pd.concat(volume_chunks, axis=1)
                close_df = close_df.loc[:, ~close_df.columns.duplicated()].dropna(how="all")
                volume_df = volume_df.loc[:, ~volume_df.columns.duplicated()].dropna(how="all")
            else:
                close_df = pd.DataFrame()
                volume_df = pd.DataFrame()

            missing_tickers = [t for t in self.tickers if t not in getattr(close_df, "columns", [])]
            if missing_tickers:
                nv_close, nv_vol = self._download_naver_price_panel(
                    missing_tickers, freq=freq, max_workers=max(4, min(16, max_workers))
                )
                if nv_close is not None and not nv_close.empty:
                    close_df = nv_close if close_df.empty else close_df.join(nv_close, how="outer")
                    volume_df = nv_vol if volume_df.empty else volume_df.join(nv_vol, how="outer")

            if close_df is None or close_df.empty:
                print("[Warning] 유효한 가격 데이터를 가져오지 못했습니다.")
                return None

            close_df = close_df.loc[:, ~close_df.columns.duplicated()].dropna(how="all")
            volume_df = volume_df.loc[:, ~volume_df.columns.duplicated()].dropna(how="all")

            valid_tickers = []
            last_prices: Dict[str, float] = {}
            last_dates: Dict[str, str] = {}

            for t in self.tickers:
                if t in close_df.columns:
                    s_clean = close_df[t].dropna()
                    if len(s_clean) >= n_periods + 4 and float(s_clean.iloc[-1]) > 100.0:
                        valid_tickers.append(t)
                        last_prices[t] = float(s_clean.iloc[-1])
                        last_dates[t] = str(s_clean.index[-1])[:10]

            if len(valid_tickers) < 10:
                print("[Warning] 유효한 시계열 길이를 가진 종목 수가 부족합니다.")
                return None

            close_df = close_df[valid_tickers].ffill().bfill()
            volume_df = volume_df[valid_tickers].ffill().bfill()

            returns_df = close_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if returns_df.shape[0] < n_periods + 3:
                return None

            ret_vals = np.clip(returns_df.values[-(n_periods + 3):, :], -0.65, 1.50)
            vol_vals = volume_df.values[-(n_periods + 3):, :]
            time_labels = [str(idx)[:10] for idx in returns_df.index[-(n_periods + 1):-1]]

            fundamental_snapshots: Dict[str, Dict[str, float]] = {}
            print(
                f"  -> [2/2: 실제 기업 재무제표(K-IFRS ROE/영업이익률/부채비율/PBR) 병렬 수집] "
                f"유효 {len(valid_tickers)}개 종목 크롤링 중 (workers={max_workers})..."
            )
            completed_cnt = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_tk = {
                    executor.submit(
                        self._extract_single_ticker_fundamentals,
                        tk,
                        ret_vals[:, idx],
                        np.log1p(np.maximum(vol_vals[:, idx], 1.0)),
                        fetch_statement_api
                    ): tk
                    for idx, tk in enumerate(valid_tickers)
                }
                for fut in as_completed(future_to_tk):
                    tk = future_to_tk[fut]
                    completed_cnt += 1
                    if completed_cnt % 200 == 0 or completed_cnt == len(valid_tickers):
                        print(
                            f"     ... 재무제표 수집 진행률: {completed_cnt} / {len(valid_tickers)} 종목 완료 "
                            f"(경과: {time.time() - t0:.1f}초)"
                        )
                    try:
                        fundamental_snapshots[tk] = fut.result()
                    except Exception:
                        fundamental_snapshots[tk] = self._extract_single_ticker_fundamentals(
                            tk, ret_vals[:, 0], np.ones(len(ret_vals)), False
                        )

            real_fin_count = sum(1 for v in fundamental_snapshots.values() if v.get("is_real_fin", 0) > 0)
            print(
                f"[Data Loader] 실제 재무데이터 매칭 완료: {real_fin_count}/{len(valid_tickers)}개 종목 "
                f"(총 소요시간: {time.time() - t0:.1f}초)"
            )

            M = len(valid_tickers)
            D = 16
            N = n_periods

            feature_names = [
                "F1_ROE_Dynamic", "F2_ROA_Dynamic", "F3_OpMargin_Dynamic", "F4_GrossMargin",
                "F5_RD_Intensity", "F6_CapEx_Intensity", "F7_FCF_Conversion", "F8_Sloan_Accruals",
                "F9_Leverage_D2E", "F10_RevGrowth_Proxy", "F11_EarnMomentum", "F12_AssetTurnover_Proxy",
                "F13_VolAdjusted_Carry", "F14_Liquidity_Turnover", "F15_Operating_Leverage", "F16_Valuation_Yield"
            ]

            X_obs = np.zeros((M, D, N), dtype=np.float64)
            Y_obs = np.zeros((M, N), dtype=np.float64)
            forward_returns = np.zeros((M, N), dtype=np.float64)

            for m, ticker in enumerate(valid_tickers):
                snap = fundamental_snapshots[ticker]
                r_series = ret_vals[:, m]
                v_series = np.log1p(np.maximum(vol_vals[:, m], 1.0))
                v_norm = (v_series - np.mean(v_series)) / (np.std(v_series) + 1e-6)

                for t in range(N):
                    window_r = r_series[t:t + 3]
                    mom_3 = np.sum(window_r)
                    vol_3 = np.std(window_r) + 1e-4
                    cur_r = r_series[t + 2]
                    next_r = r_series[t + 3]

                    X_obs[m, 0, t] = snap["roe"] + 0.35 * mom_3
                    X_obs[m, 1, t] = snap["roa"] + 0.20 * mom_3
                    X_obs[m, 2, t] = snap["op_margin"] + 0.25 * cur_r
                    X_obs[m, 3, t] = snap["gross_margin"] + 0.10 * cur_r
                    X_obs[m, 4, t] = snap["rd_ratio"] - 0.15 * cur_r + 0.02 * v_norm[t + 2]
                    X_obs[m, 5, t] = snap["capex_ratio"] + 0.15 * cur_r + 0.02 * v_norm[t + 2]
                    X_obs[m, 6, t] = snap["fcf_margin"] + 0.30 * mom_3 - 0.10 * vol_3
                    X_obs[m, 7, t] = snap["accrual_ratio"] - 0.20 * cur_r + 0.05 * v_norm[t + 2]
                    X_obs[m, 8, t] = snap["debt_to_eq"] - 0.40 * mom_3
                    X_obs[m, 9, t] = snap["rev_growth"] + 0.45 * mom_3
                    X_obs[m, 10, t] = snap["earn_growth"] + 0.60 * cur_r
                    X_obs[m, 11, t] = (snap["roa"] / (snap["op_margin"] + 0.05)) + 0.15 * v_norm[t + 2]
                    X_obs[m, 12, t] = cur_r / vol_3
                    X_obs[m, 13, t] = v_norm[t + 2]
                    X_obs[m, 14, t] = (cur_r - np.mean(r_series[:t + 3])) / vol_3
                    X_obs[m, 15, t] = snap["fcf_margin"] / (1.0 + snap["debt_to_eq"]) - 0.25 * mom_3

                    Y_obs[m, t] = 0.6 * (snap["roe"] + mom_3) + 0.4 * cur_r
                    forward_returns[m, t] = next_r

            for d in range(D):
                scale = np.std(X_obs[:, d, :]) + 1e-8
                X_obs[:, d, :] /= scale

            print(f"[Data Loader] 전 종목 텐서 구축 완료: 유효 종목 M={M}개, 피처 D={D}개, 시계열 N={N}개")
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
                raw_fundamentals=fundamental_snapshots,
                data_source="LIVE_KRX_NAVER_YFINANCE"
            )

        except Exception as e:
            print(f"[Warning] 실시간 데이터 다운로드 중 예외 발생 ({e}).")
            return None

    def generate_calibrated_fundamental_dgp(
        self,
        M: int = 40,
        D: int = 24,
        r: int = 6,
        N: int = 60,
        rho_budget_constraint: float = -0.82,
        sigma_accounting_noise: float = 0.12,
        seed: int = 42
    ) -> FleetTensorData:
        """
        논문 식 (1)~(2)의 구조적 재무/회계 데이터 생성 과정(DGP) 시뮬레이터입니다.
        오프라인 폴백 또는 Ground-Truth 대수적 복원율 검증(--run-benchmark)에 사용됩니다.
        """
        rng = np.random.default_rng(seed)

        # 진성 Stiefel 기저 U_phys (D x r)
        raw_basis = rng.normal(0, 1, size=(D, r))
        U_phys, _ = linalg.qr(raw_basis, mode="economic")

        # 진성 양정치 헤시안 H_phys (r x r) 및 양(+)의 비대각 시너지 부여
        eigvals = np.linspace(0.8, 2.2, r)
        Q_h, _ = linalg.qr(rng.normal(0, 1, size=(r, r)))
        H_phys = Q_h @ np.diag(eigvals) @ Q_h.T
        if r >= 2:
            H_phys[0, 1] = H_phys[1, 0] = 0.42 * np.sqrt(H_phys[0, 0] * H_phys[1, 1])

        gamma_phys = rng.uniform(-0.25, 0.25, size=r)

        # 기업별 진성 펀더멘털 위치 delta_true (M x r) 및 회계 착시 오프셋 b_raw (M x D)
        delta_true = rng.normal(0.0, 0.75, size=(M, r))
        mu_raw = delta_true @ U_phys.T
        b_raw = rng.normal(0.35, 0.90, size=(M, D))
        W_true = rng.uniform(0.75, 1.35, size=(M, D))
        d_raw = rng.normal(0.0, 0.45, size=(M, D))

        # 시간 그리드 및 체비쇼프 드리프트
        t_grid = np.linspace(-1.0, 1.0, N)
        G_drift = np.vstack([t_grid, 2.0 * t_grid**2 - 1.0])  # (2, N)

        # 예산 제약 다중공선성 공분산 (rho_budget_constraint = -0.82)
        Cov_u = np.eye(r) * (0.42 ** 2)
        if r >= 2:
            Cov_u[0, 1] = Cov_u[1, 0] = rho_budget_constraint * (0.42 ** 2)

        Sigma_xi_true = np.diag(rng.uniform(0.7, 1.3, size=D)) * (sigma_accounting_noise ** 2)

        X_obs = np.zeros((M, D, N), dtype=np.float64)
        Y_obs = np.zeros((M, N), dtype=np.float64)
        forward_returns = np.zeros((M, N), dtype=np.float64)

        eta_m = rng.normal(0.0, 0.5, size=M)
        y_max = 2.5

        for m in range(M):
            u_latent = rng.multivariate_normal(np.zeros(r), Cov_u, size=N).T  # (r, N)
            # 비활성 여공간 변동 추가
            u_null = rng.normal(0.0, 0.08, size=(D, N))
            u_null -= U_phys @ (U_phys.T @ u_null)
            u_raw = U_phys @ u_latent + u_null

            drift_w = rng.normal(0.0, 1.0, size=2)
            g_t = drift_w @ G_drift  # (N,)

            xi_noise = rng.multivariate_normal(np.zeros(D), Sigma_xi_true, size=N).T
            # 간헐적 어닝 점프 (Step-DOE)
            if m % 5 == 0 and N > 20:
                u_raw[:, N // 2:] += rng.normal(0, 0.25, size=(D, 1))

            for t in range(N):
                x_clean = mu_raw[m] + u_raw[:, t] + b_raw[m] + d_raw[m] * g_t[t]
                X_obs[m, :, t] = W_true[m] * x_clean + xi_noise[:, t]

                z_mt = delta_true[m] + u_latent[:, t]
                psi_val = (
                    0.5 * float(z_mt.T @ H_phys @ z_mt)
                    + (1.0 / 6.0) * float(np.sum(gamma_phys * (z_mt ** 3)))
                )
                eps_t = rng.normal(0.0, 0.06)
                Y_obs[m, t] = y_max + eta_m[m] - psi_val + eps_t

                # 차기 주식 초과수익률: 진성 포텐셜 우위(+) - 회계 착시 패널티(-) + 시너지(+) + 노이즈
                true_pot = -0.5 * float(delta_true[m].T @ H_phys @ delta_true[m])
                bloat_pen = float(b_raw[m] @ U_phys[:, 0])
                synergy_bonus = float(z_mt[0] * z_mt[1] * H_phys[0, 1]) if r >= 2 else 0.0
                expected_r = 0.012 + 0.018 * true_pot - 0.015 * bloat_pen + 0.020 * synergy_bonus
                forward_returns[m, t] = expected_r + rng.normal(0.0, 0.055)

        base_tickers = self.KRX_UNIVERSE
        tickers = [
            base_tickers[i] if i < len(base_tickers) else f"{100000 + i * 10:06d}.KS"
            for i in range(M)
        ]
        default_names = {
            "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체",
            "012450.KS": "한화에어로스페이스", "267260.KS": "HD현대일렉트릭", "005380.KS": "현대차",
            "000270.KS": "기아", "207940.KS": "삼성바이오로직스", "196170.KQ": "알테오젠",
            "058470.KQ": "리노공업", "105560.KS": "KB금융", "079550.KS": "LIG넥스원",
            "329180.KS": "HD현대중공업", "035420.KS": "NAVER", "055550.KS": "신한지주",
            "012330.KS": "현대모비스", "086790.KS": "하나금융지주", "009150.KS": "삼성전기",
            "034020.KS": "두산에너빌리티", "068270.KS": "셀트리온", "028260.KS": "삼성물산",
            "066570.KS": "LG전자", "005490.KS": "POSCO홀딩스", "035720.KS": "카카오",
            "006400.KS": "삼성SDI", "051910.KS": "LG화학", "003670.KS": "포스코퓨처엠",
            "247540.KQ": "에코프로비엠", "036570.KS": "엔씨소프트", "096770.KS": "SK이노베이션",
            "009830.KS": "한화솔루션", "011170.KS": "롯데케미칼"
        }
        names_map = {tk: default_names.get(tk, f"KRX_기업_{idx+1}") for idx, tk in enumerate(tickers)}
        mkts_map = {tk: ("KOSDAQ" if tk.endswith(".KQ") else "KOSPI") for tk in tickers}
        prices_map = {tk: float(rng.integers(15, 350) * 1000) for tk in tickers}

        feature_names = [f"Factor_{d + 1}" for d in range(D)]
        time_index = [f"T_{t + 1:03d}" for t in range(N)]

        ground_truth = {
            "U_phys": U_phys,
            "H_phys": H_phys,
            "gamma_phys": gamma_phys,
            "delta_true": delta_true,
            "b_raw": b_raw,
            "W_true": W_true,
            "Sigma_xi_true": Sigma_xi_true
        }

        return FleetTensorData(
            X_obs=X_obs,
            Y_obs=Y_obs,
            forward_returns=forward_returns,
            tickers=tickers,
            feature_names=feature_names,
            time_index=time_index,
            ground_truth=ground_truth,
            company_names=names_map,
            markets=mkts_map,
            last_prices=prices_map
        )


# ==============================================================================
# MODULE 2: 6단계 대수적 텐서 역변환 엔진 (v3.0 Fleet Tensor Inverse Engine)
# ==============================================================================

class FundamentalTensorInverseEngine:
    """
    논문 'Disentangling Universal Physical Laws from Instrument Biases Across Evolving Observation Fleets'의
    6단계 대수적 텐서 역변환 파이프라인을 구현한 핵심 퀀트 엔진입니다.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        huber_delta: float = 1.8,
        tau_my: float = 0.05,
        fixed_r: Optional[int] = None
    ):
        self.poly_degree = poly_degree
        self.huber_delta = huber_delta
        self.tau_my = tau_my
        self.fixed_r = fixed_r

    def _stage1_temporal_fwl_projection(
        self, X_obs: np.ndarray, Y_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Stage 1: 시간적 Frisch-Waugh-Lovell (FWL) 직교 여공간 투영
        체비쇼프 다항식 기저의 Thin-QR 분해를 통해 정적 오프셋(b_m)과 거시 드리프트(d_m g_m(t))를 소거합니다.
        """
        _, _, N = X_obs.shape
        p = min(self.poly_degree, max(1, N // 4))
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

        # 우측 곱셈: (M, D, N) @ (N, N) -> (M, D, N)
        X_law = np.einsum("mdn,nk->mdk", X_obs, P_law)
        Y_law = Y_obs @ P_law

        return X_law, Y_law, P_law, dof_nu

    def _stage2_spatio_temporal_eiv_covariance(
        self, X_law: np.ndarray, dof_nu: int
    ) -> Tuple[np.ndarray, int]:
        """
        Stage 2: 시공간 볼록 적응형 EIV 측정 노이즈 공분산(Sigma_xi) 추정
        어닝 점프(Step-DOE) 마스킹 차분과 공간 직교 여공간 MAD 잔차를 결합합니다.
        """
        M, D, N = X_law.shape
        X_flat = X_law.transpose(1, 0, 2).reshape(D, M * N)
        Sigma_law = (X_flat @ X_flat.T) / max(1, M * dof_nu)

        s_vals = linalg.svdvals(Sigma_law)
        beta = min(D, M * N) / max(D, M * N)
        omega_beta = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau_gd = omega_beta * np.median(s_vals)

        if self.fixed_r is not None:
            r_0 = min(max(1, self.fixed_r), D - 1)
        else:
            r_0 = int(np.sum(s_vals > tau_gd))
            r_0 = int(np.clip(r_0, 2, max(2, min(8, D // 2))))

        U_init, _, _ = linalg.svd(Sigma_law, full_matrices=False)
        U_0 = U_init[:, :r_0]
        P_perp = np.eye(D) - U_0 @ U_0.T
        h_perp = np.maximum(np.diag(P_perp), 1e-3)

        # 공간 잔차 및 레버리지 보정 MAD
        R_sp = P_perp @ X_flat  # (D, M*N)
        med_abs = np.median(np.abs(R_sp), axis=1)
        sigma_mad = (1.4826 * med_abs) / np.sqrt(h_perp)
        Sigma_spatial = np.diag(sigma_mad ** 2)

        # 시계열 1차/2차 차분 및 계단식 점프(Step-DOE) 마스킹
        d1 = np.diff(X_law, n=1, axis=2)  # (M, D, N-1)
        d2 = np.diff(X_law, n=2, axis=2)  # (M, D, N-2)
        d2_norms = np.linalg.norm(d2, axis=1)
        tau_step = np.percentile(d2_norms, 90) * 1.5 + 1e-6
        valid_mask = d2_norms <= tau_step

        if np.sum(valid_mask) > D:
            d2_valid = d2.transpose(0, 2, 1)[valid_mask]  # (K, D)
            Sigma_temp = (d2_valid.T @ d2_valid) / (6.0 * max(1, d2_valid.shape[0]))
        else:
            d1_flat = d1.transpose(1, 0, 2).reshape(D, -1)
            Sigma_temp = (d1_flat @ d1_flat.T) / (2.0 * max(1, d1_flat.shape[1]))

        alpha_sp = float(D - r_0) / float(D - r_0 + 2)
        Sigma_xi = alpha_sp * Sigma_spatial + (1.0 - alpha_sp) * Sigma_temp

        # 비대각 소프트 쓰레숄딩 및 양정치 보장
        diag_vals = np.maximum(np.diag(Sigma_xi), 1e-5)
        Sigma_xi = 0.75 * np.diag(diag_vals) + 0.25 * Sigma_xi
        eig_vals, eig_vecs = linalg.eigh(Sigma_xi)
        Sigma_xi = eig_vecs @ np.diag(np.maximum(eig_vals, 1e-5)) @ eig_vecs.T

        return Sigma_xi, r_0

    def _stage3_output_driven_subspace_and_gains(
        self,
        X_law: np.ndarray,
        Y_law: np.ndarray,
        Sigma_xi: np.ndarray,
        r_0: int,
        dof_nu: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Stage 3 & 3.5: 출력 주도형 알파 부분공간(U_phys) 발굴 및 아다마르 영공간 게인(W_hat) 분리
        """
        M, D, _ = X_law.shape
        M_phys = np.zeros((D, D), dtype=np.float64)
        curv_norms = np.zeros((M, D), dtype=np.float64)

        for m in range(M):
            Xm = X_law[m]  # (D, N)
            ym = Y_law[m]  # (N,)
            Sigma_Xm = (Xm @ Xm.T) / dof_nu
            eig_w, eig_v = linalg.eigh(Sigma_Xm - Sigma_xi)
            Sigma_Sm = eig_v @ np.diag(np.maximum(eig_w, 1e-3)) @ eig_v.T

            cov_Xy = (Xm @ ym) / dof_nu
            g_m = linalg.solve(Sigma_Sm + 0.05 * np.eye(D), cov_Xy, assume_a="pos")

            ym_centered_sq = (ym ** 2) - np.mean(ym ** 2)
            X_sq_cov = (Xm @ (Xm.T * ym_centered_sq[:, None])) / dof_nu
            K_m = linalg.solve(Sigma_Sm + 0.1 * np.eye(D), X_sq_cov, assume_a="pos")

            M_phys += (np.outer(g_m, g_m) + 0.25 * (K_m @ Sigma_Sm @ K_m.T)) / M
            curv_norms[m] = np.sqrt(np.diag(Sigma_Sm) + 1e-4)

        U_all, s_phys, _ = linalg.svd(M_phys, full_matrices=False)
        if self.fixed_r is not None:
            r_hat = min(max(1, self.fixed_r), D - 1)
        else:
            energy_ratio = np.cumsum(s_phys) / (np.sum(s_phys) + 1e-8)
            r_energy = int(np.searchsorted(energy_ratio, 0.88) + 1)
            r_hat = int(np.clip(max(r_0, r_energy), 2, min(8, D // 2)))

        U_phys = U_all[:, :r_hat]

        # Stage 3.5: 기업별 대각 스케일/레버리지 왜곡 W_hat 추정
        geom_mean_scale = np.exp(np.mean(np.log(curv_norms + 1e-6), axis=0))
        W_hat = curv_norms / (geom_mean_scale[None, :] + 1e-6)
        W_hat = np.clip(W_hat, 0.45, 2.25)

        return U_phys, W_hat, r_hat

    def _stage4_hermite_eiv_curvature_recovery(
        self,
        X_law: np.ndarray,
        Y_law: np.ndarray,
        U_phys: np.ndarray,
        W_hat: np.ndarray,
        Sigma_xi: np.ndarray,
        dof_nu: int
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Stage 4: 다변량 에르미트(Hermite) 다항식 EIV 기저 및 가우스-마르코프 역투영(M_22^-1)을 통한
        예산 제약 다중공선성 제거 및 진성 펀더멘털 곡률(H_phys, gamma_phys) 복원
        """
        M, D, N = X_law.shape
        r = U_phys.shape[1]

        off_pairs = [(j, k) for j in range(r) for k in range(j + 1, r)]
        n_quad_diag = r
        n_quad_off = len(off_pairs)
        n_cubic = r
        n_total_terms = n_quad_diag + n_quad_off + n_cubic

        # 대규모 종목(M=2500+)에서도 메모리 병목이 없도록 충분통계량 행렬(M_22) 누적
        M_22 = np.zeros((n_total_terms, n_total_terms), dtype=np.float64)
        rhs_vec = np.zeros(n_total_terms, dtype=np.float64)

        Z_tilde_list: List[np.ndarray] = []
        Sigma_nu_list: List[np.ndarray] = []

        delta_huber = self.huber_delta

        for m in range(M):
            w_inv = 1.0 / np.maximum(W_hat[m], 1e-4)
            X_scaled = X_law[m] * w_inv[:, None]  # (D, N)
            Z_m = (U_phys.T @ X_scaled).T         # (N, r)

            # Pseudo-Huber 클리핑으로 팻테일 이상치 제어
            z_scale = np.std(Z_m, axis=0, keepdims=True) + 1e-4
            z_norm = Z_m / z_scale
            Z_clipped = z_scale * delta_huber * np.tanh(z_norm / delta_huber)

            W_inv_diag = np.diag(w_inv)
            Sigma_nu_m = U_phys.T @ W_inv_diag @ Sigma_xi @ W_inv_diag @ U_phys

            Z_tilde_list.append(Z_clipped)
            Sigma_nu_list.append(Sigma_nu_m)

            # 다변량 에르미트 EIV 기저 구성: T^(2)_jj, T^(2)_jk, T^(3)_j
            T_m = np.zeros((N, n_total_terms), dtype=np.float64)
            for j in range(r):
                T_m[:, j] = -0.5 * (Z_clipped[:, j] ** 2 - Sigma_nu_m[j, j])

            for idx_pair, (j, k) in enumerate(off_pairs):
                T_m[:, n_quad_diag + idx_pair] = -(
                    Z_clipped[:, j] * Z_clipped[:, k] - Sigma_nu_m[j, k]
                )

            for j in range(r):
                T_m[:, n_quad_diag + n_quad_off + j] = -(1.0 / 6.0) * (
                    Z_clipped[:, j] ** 3 - 3.0 * Sigma_nu_m[j, j] * Z_clipped[:, j]
                )

            # 기업 내부 1차 기울기 FWL 직교 투영 연산자 Q_m 적용
            ZTZ = Z_clipped.T @ Z_clipped + 1e-3 * np.eye(r)
            proj_Z = linalg.solve(ZTZ, Z_clipped.T, assume_a="pos")  # (r, N)
            T_tilde_m = T_m - Z_clipped @ (proj_Z @ T_m)
            y_tilde_m = Y_law[m] - Z_clipped @ (proj_Z @ Y_law[m])

            M_22 += T_tilde_m.T @ T_tilde_m
            rhs_vec += T_tilde_m.T @ y_tilde_m

        # EIV 감쇠 보정 및 가우스-마르코프 역행렬 풀이
        reg_scale = 0.08 * float(M * dof_nu) / float(max(1, n_total_terms))
        theta_hat = linalg.solve(
            M_22 + reg_scale * np.eye(n_total_terms),
            rhs_vec,
            assume_a="pos"
        )

        H_phys = np.zeros((r, r), dtype=np.float64)
        for j in range(r):
            H_phys[j, j] = theta_hat[j]
        for idx_pair, (j, k) in enumerate(off_pairs):
            H_phys[j, k] = H_phys[k, j] = theta_hat[n_quad_diag + idx_pair]

        # 볼록 물리 포텐셜(양정치성) 보장 투영
        eig_h, vec_h = linalg.eigh(H_phys)
        eig_h_pos = np.clip(np.abs(eig_h), 0.25, 3.50)
        H_phys = vec_h @ np.diag(eig_h_pos) @ vec_h.T

        gamma_phys = np.clip(theta_hat[n_quad_diag + n_quad_off:], -0.5, 0.5)

        return H_phys, gamma_phys, Z_tilde_list, Sigma_nu_list

    def _stage5_6_legendre_dual_inversion_and_krr(
        self,
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
        """
        Stage 5 & 6: 모로-요시다 근접 르장드르 쌍대 역변환, 마할라노비스 OOD 게이팅 및
        랜드마크 커널 릿지 회귀(KRR) 표면 진화를 수행합니다.
        """
        M, D, N = X_obs.shape
        r = U_phys.shape[1]

        alpha_eiv = np.zeros((M, r), dtype=np.float64)
        for m in range(M):
            Zm = Z_tilde_list[m]
            Sigma_nu_m = Sigma_nu_list[m]
            cov_Z = (Zm.T @ Zm) / dof_nu - Sigma_nu_m
            eig_z, vec_z = linalg.eigh(cov_Z)
            cov_Z_pos = vec_z @ np.diag(np.maximum(eig_z, 0.02)) @ vec_z.T
            cov_Zy = (Zm.T @ Y_law[m]) / dof_nu
            alpha_eiv[m] = linalg.solve(cov_Z_pos + 0.05 * np.eye(r), cov_Zy, assume_a="pos")

        # 모로-요시다 유리화 근접 르장드르 쌍대 역변환 (식 3)
        H_diag = np.maximum(np.diag(H_phys), 0.2)
        H_off = H_phys - np.diag(np.diag(H_phys))
        delta_hat = -linalg.solve(H_phys + 0.1 * np.eye(r), alpha_eiv.T, assume_a="pos").T

        for _ in range(5):
            alpha_tilde = alpha_eiv + (delta_hat @ H_off.T)
            for j in range(r):
                h_jj = H_diag[j]
                disc = np.maximum(
                    (h_jj ** 2) - 2.0 * gamma_phys[j] * alpha_tilde[:, j],
                    self.tau_my * (h_jj ** 2)
                )
                denom = h_jj + np.sqrt(disc)
                delta_hat[:, j] = (-2.0 * alpha_tilde[:, j]) / np.maximum(denom, 1e-4)
            delta_hat = np.clip(delta_hat, -3.5, 3.5)

        # 마할라노비스 OOD 거리 계산 및 지수 게이팅 SURE 수축
        delta_mean = np.mean(delta_hat, axis=0)
        delta_centered = delta_hat - delta_mean
        Cov_delta = (delta_centered.T @ delta_centered) / max(1, M - 1) + 0.05 * np.eye(r)
        Cov_delta_inv = linalg.inv(Cov_delta)

        mahal_dist_sq = np.einsum("mr,rk,mk->m", delta_centered, Cov_delta_inv, delta_centered)
        lambda_sure_base = 0.22
        ood_gate = np.exp(-np.maximum(0.0, mahal_dist_sq - float(r)) / (2.0 * float(r)))
        lambda_eff = lambda_sure_base * ood_gate

        delta_sure = (1.0 - lambda_eff[:, None]) * delta_hat + lambda_eff[:, None] * delta_mean[None, :]

        # 기업별 정적 회계 착시 오프셋(b_m) 분리
        x_bar_obs = np.mean(X_obs, axis=2)  # (M, D)
        accounting_bias_b = (x_bar_obs / np.maximum(W_hat, 1e-4)) - (delta_sure @ U_phys.T)

        # Stage 6: 정렬된 글로벌 좌표 z_aligned 및 랜드마크 기반 KRR 보정
        z_aligned = np.zeros((M, r, N), dtype=np.float64)
        y_pure = np.zeros((M, N), dtype=np.float64)

        for m in range(M):
            z_aligned[m] = delta_sure[m, :, None] + Z_tilde_list[m].T
            z_m_mean = delta_sure[m]
            psi_m_base = (
                0.5 * float(z_m_mean.T @ H_phys @ z_m_mean)
                + (1.0 / 6.0) * float(np.sum(gamma_phys * (z_m_mean ** 3)))
            )
            eta_hat_m = float(np.mean(Y_obs[m])) + psi_m_base
            y_pure[m] = Y_obs[m] - eta_hat_m

        # 1) 진성 펀더멘털 내재가치 포텐셜 (최적점 대비 수익 창출 우위)
        quad_energy = 0.5 * np.einsum("mr,rk,mk->m", delta_sure, H_phys, delta_sure)
        first_order_drive = np.einsum("mr,r->m", delta_sure, np.mean(alpha_eiv, axis=0))
        intrinsic_potential = -quad_energy + 0.5 * first_order_drive + delta_sure[:, 0]

        # 2) 1차 주축 방향 회계적 착시(Bloat) 오프셋
        b_proj_score = accounting_bias_b @ U_phys[:, 0]

        # 3) 비대각 양(+)의 시너지 결합 에너지
        off_diag_synergy = np.einsum("mr,rk,mk->m", delta_sure, H_off, delta_sure)

        def _zscore(arr: np.ndarray) -> np.ndarray:
            return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

        alpha_score = (
            0.50 * _zscore(intrinsic_potential)
            - 0.35 * _zscore(b_proj_score)
            + 0.15 * _zscore(off_diag_synergy)
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
        정제된 6단계 텐서 역변환 팩터들을 실제 패널의 차기 수익률 분포에 교차 보정하여
        종목별 향후 6개월(6M) 및 12개월(12M) 예상 수익률(%)을 산출합니다.
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

        hist_period_ret = np.mean(fwd_ret[:, max(0, N - 12):], axis=1)
        low_q, high_q = np.percentile(hist_period_ret, [5, 95])
        y_clipped = np.clip(hist_period_ret, low_q, high_q)

        ridge_pen = np.diag([0.0, 0.25 * M, 0.25 * M, 0.35 * M, 0.35 * M])
        beta_hat = linalg.solve(
            F_design.T @ F_design + ridge_pen,
            F_design.T @ y_clipped,
            assume_a="pos"
        )

        cross_sec_disp = float(np.clip(np.std(y_clipped), 0.012, 0.045))
        mkt_drift_monthly = float(np.clip(np.mean(y_clipped), 0.004, 0.012))

        emp_monthly_alpha = F_design[:, 1:] @ beta_hat[1:]
        prior_monthly_alpha = _z(alpha_score) * (0.65 * cross_sec_disp)

        monthly_alpha = 0.55 * prior_monthly_alpha + 0.45 * emp_monthly_alpha
        monthly_alpha = np.clip(monthly_alpha, -0.065, 0.075)

        expected_return_6m = (1.0 + mkt_drift_monthly + monthly_alpha) ** 6 - 1.0
        expected_return_12m = (1.0 + mkt_drift_monthly + monthly_alpha) ** 12 - 1.0
        mkt_return_12m = (1.0 + mkt_drift_monthly) ** 12 - 1.0
        expected_alpha_12m = expected_return_12m - mkt_return_12m

        return expected_return_6m, expected_return_12m, expected_alpha_12m

    def fit_transform(self, fleet_data: FleetTensorData) -> PurifiedFundamentalState:
        """
        6단계 대수적 텐서 역변환 파이프라인 전체를 순차 실행하고 종목별 예상 수익률을 산출합니다.
        """
        X_obs = fleet_data.X_obs
        Y_obs = fleet_data.Y_obs

        X_law, Y_law, _, dof_nu = self._stage1_temporal_fwl_projection(X_obs, Y_obs)
        Sigma_xi, r_0 = self._stage2_spatio_temporal_eiv_covariance(X_law, dof_nu)
        U_phys, W_hat, r_hat = self._stage3_output_driven_subspace_and_gains(
            X_law, Y_law, Sigma_xi, r_0, dof_nu
        )
        H_phys, gamma_phys, Z_tilde_list, Sigma_nu_list = self._stage4_hermite_eiv_curvature_recovery(
            X_law, Y_law, U_phys, W_hat, Sigma_xi, dof_nu
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
            X_obs, Y_obs, Y_law, U_phys, W_hat, H_phys, gamma_phys,
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
# MODULE 3: 중장기 펀더멘털 롱숏 워크포워드 백테스터 (Walk-Forward Backtester)
# ==============================================================================

class FundamentalLongShortBacktester:
    """
    6단계 텐서 역변환 알파 전략(Ours v3.0)과 3대 기존 퀀트 베이스라인(Pooled Ridge, Input-PCA,
    Panel Fixed-Effects)의 표본 외(Out-of-Sample) 롱숏 포트폴리오 성과를 비교 검증합니다.
    """

    def __init__(self, top_quantile: float = 0.20, cost_bps: float = 15.0):
        self.top_quantile = top_quantile
        self.cost_bps = cost_bps

    @staticmethod
    def _compute_baseline_signals(
        X_win: np.ndarray, Y_win: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        비교 벤치마크 모델 3종의 종목별 스코어를 계산합니다.
        """
        M, D, N = X_win.shape
        X_mean = np.mean(X_win, axis=2)  # (M, D)
        Y_mean = np.mean(Y_win, axis=1)  # (M,)

        # B1: Pooled Factor Ridge (정적 회계 착시 b_m에 그대로 노출됨)
        w_b1 = linalg.solve(X_mean.T @ X_mean + 1.5 * np.eye(D), X_mean.T @ Y_mean, assume_a="pos")
        score_b1 = X_mean @ w_b1

        # B2: Input-PCA Factor Model (입력 공분산 분해 -> 예산 제약의 매니폴드 함정에 빠짐)
        X_flat = X_win.transpose(1, 0, 2).reshape(D, M * N)
        U_pca, _, _ = linalg.svd(X_flat @ X_flat.T, full_matrices=False)
        Z_pca = X_mean @ U_pca[:, :min(4, D)]
        w_b2 = linalg.solve(Z_pca.T @ Z_pca + 0.5 * np.eye(Z_pca.shape[1]), Z_pca.T @ Y_mean, assume_a="pos")
        score_b2 = Z_pca @ w_b2

        # B3: Panel Fixed-Effects (기업 평균을 제거하면서 진성 경쟁력 mu_m까지 소거함)
        X_fe = X_win[:, :, -1] - X_mean
        Y_fe = Y_win[:, -1] - Y_mean
        w_b3 = linalg.solve(X_fe.T @ X_fe + 2.0 * np.eye(D), X_fe.T @ Y_fe, assume_a="pos")
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
        """
        롤링 윈도우(Walk-Forward) 방식으로 6단계 텐서 역변환 엔진을 적합하고
        차기 리밸런싱 구간의 롱숏 포트폴리오 수익률을 계산합니다.
        """
        M, _, N = fleet_data.X_obs.shape
        k_pick = max(1, int(round(M * self.top_quantile)))

        strategy_names = [
            "Ours (v3.0 Tensor Inverse Long-Short)",
            "B1: Pooled Factor Ridge (No Deconfounding)",
            "B2: Input-PCA Factor Model (Manifold Trap)",
            "B3: Panel Fixed-Effects (Wipes Out Alpha)"
        ]
        ret_records: Dict[str, List[float]] = {s: [] for s in strategy_names}
        turn_records: Dict[str, List[float]] = {s: [] for s in strategy_names}
        ic_records: Dict[str, List[float]] = {s: [] for s in strategy_names}
        prev_weights: Dict[str, np.ndarray] = {s: np.zeros(M) for s in strategy_names}

        engine = FundamentalTensorInverseEngine()
        last_state: Optional[PurifiedFundamentalState] = None

        for t_end in range(lookback_window, N, rebalance_step):
            t_start = max(0, t_end - lookback_window)
            sub_data = FleetTensorData(
                X_obs=fleet_data.X_obs[:, :, t_start:t_end],
                Y_obs=fleet_data.Y_obs[:, t_start:t_end],
                forward_returns=fleet_data.forward_returns[:, t_start:t_end],
                tickers=fleet_data.tickers,
                feature_names=fleet_data.feature_names,
                time_index=fleet_data.time_index[t_start:t_end]
            )
            state = engine.fit_transform(sub_data)
            last_state = state

            baseline_scores = self._compute_baseline_signals(sub_data.X_obs, sub_data.Y_obs)
            all_scores = {
                "Ours (v3.0 Tensor Inverse Long-Short)": state.alpha_score,
                **baseline_scores
            }

            hold_end = min(N, t_end + rebalance_step)
            period_fwd_ret = np.sum(fleet_data.forward_returns[:, t_end:hold_end], axis=1)

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
# MODULE 4: 대수적 복원 정밀도 평가 및 전 종목 예상 수익률 리포터
# ==============================================================================

def evaluate_algebraic_recovery_metrics(
    fleet_data: FleetTensorData,
    state: PurifiedFundamentalState
) -> Dict[str, float]:
    """
    시뮬레이션 Ground-Truth 대비 부분공간 오차(E_sub), 헤시안 곡률 복원율, 회계 착시 분리 상관계수를 검증합니다.
    """
    if fleet_data.ground_truth is None:
        return {}

    gt = fleet_data.ground_truth
    U_true = gt["U_phys"]
    U_hat = state.U_phys

    P_true = U_true @ U_true.T
    P_hat = U_hat @ U_hat.T
    subspace_err = float(
        linalg.norm(P_true - P_hat, ord="fro") / np.sqrt(2.0 * U_true.shape[1])
    )

    b_true_proj = gt["b_raw"] @ U_true[:, 0]
    b_hat_proj = state.accounting_bias_b @ U_hat[:, 0]
    b_corr = float(abs(np.corrcoef(b_true_proj, b_hat_proj)[0, 1]))

    return {
        "Estimated_Latent_Dim(r_hat)": int(state.r_hat),
        "Subspace_Projection_Error": round(subspace_err, 4),
        "Accounting_Bloat_Recovery_Corr": round(b_corr, 4),
        "Recovered_Synergy_H12": round(float(state.H_phys[0, 1]), 4) if state.r_hat >= 2 else 0.0
    }


def build_all_stocks_expected_return_df(
    fleet_data: FleetTensorData,
    state: PurifiedFundamentalState
) -> pd.DataFrame:
    """
    분석된 전 종목에 대해 예상 수익률(6개월/12개월), 시장 대비 초과수익률(Alpha),
    실제 수집된 재무 지표(ROE, 영업이익률, PBR), 진성 펀더멘털 내재가치 점수,
    회계적 부풀리기(Bloat) 오프셋, 매매 포지션 의견을 담은 전체 종목 데이터프레임을 생성합니다.
    """
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
        "Real_ROE(%)": [round(raw_fin_map.get(t, {}).get("roe", 0.0) * 100.0, 2) for t in fleet_data.tickers],
        "Real_OpMargin(%)": [round(raw_fin_map.get(t, {}).get("op_margin", 0.0) * 100.0, 2) for t in fleet_data.tickers],
        "Real_DebtRatio(%)": [round(raw_fin_map.get(t, {}).get("debt_to_eq", 0.0) * 100.0, 1) for t in fleet_data.tickers],
        "Real_PBR": [round(raw_fin_map.get(t, {}).get("pbr", 0.0), 2) for t in fleet_data.tickers],
        "Mahal_OOD_Dist2": np.round(state.mahal_dist_sq, 2),
        "Data_Source": [fleet_data.data_source] * M
    })

    df_out = df_out.sort_values("Exp_Return_12M(%)", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", np.arange(1, len(df_out) + 1))
    return df_out


def print_top_long_short_picks(
    fleet_data: FleetTensorData,
    state: PurifiedFundamentalState,
    top_n: int = 15,
    save_csv_path: str = "krx_all_stocks_expected_returns.csv"
) -> pd.DataFrame:
    """
    전 종목의 예상 수익률 테이블을 생성하여 CSV 파일로 저장하고,
    예상 수익률 최상위(Long) 및 최하위(Short) 종목을 콘솔에 출력합니다.
    """
    df_all = build_all_stocks_expected_return_df(fleet_data, state)

    if save_csv_path:
        try:
            df_all.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
            print(f"\n[Output Saved] 전체 {len(df_all)}개 종목의 예상 수익률 분석 결과가 '{save_csv_path}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n[Warning] CSV 저장 중 오류 발생: {e}")

    print("\n" + "=" * 115)
    print(f" [TOP {top_n} EXPECTED RETURN (LONG)] 진성 펀더멘털 대비 저평가된 예상 수익률 최상위 종목 (총 {len(df_all)}개 종목 중)")
    print("=" * 115)
    print(df_all.head(top_n).to_string(index=False))

    print("\n" + "=" * 115)
    print(f" [BOTTOM {top_n} EXPECTED RETURN (SHORT)] 회계 수치(b_m) 착시로 하락 위험이 높은 예상 수익률 최하위 종목")
    print("=" * 115)
    print(df_all.tail(top_n).sort_values("Exp_Return_12M(%)", ascending=True).to_string(index=False))

    return df_all


PRICE_LOG_COLUMNS = [
    "Run_Date", "Run_Timestamp", "Price_Date", "Ticker", "Name", "Market",
    "Price_KRW", "Rank", "Signal", "Exp_Return_6M(%)", "Exp_Return_12M(%)",
    "Exp_Alpha_12M(%)", "Alpha_Score", "Universe_Size", "Data_Source",
]


def _signal_direction(signal: str) -> int:
    text = str(signal or "")
    if "LONG" in text:
        return 1
    if "SHORT" in text:
        return -1
    return 0


def append_prediction_log(
    df_all: pd.DataFrame,
    run_timestamp: datetime,
    log_dir: str = "logs"
) -> pd.DataFrame:
    """
    분석일 기준 종목별 가격과 예측값을 누적 로그(predictions_history.csv)와
    실행일별 스냅샷(runs/predictions_YYYY-MM-DD.csv)으로 저장합니다.
    같은 날 재실행하면 그날 기록을 교체합니다.
    """
    run_date = run_timestamp.strftime("%Y-%m-%d")
    snap = pd.DataFrame({
        "Run_Date": run_date,
        "Run_Timestamp": run_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "Price_Date": df_all["Last_Trade_Date"].astype(str),
        "Ticker": df_all["Ticker"],
        "Name": df_all["Name"],
        "Market": df_all["Market"],
        "Price_KRW": df_all["Last_Price(KRW)"].astype(float),
        "Rank": df_all["Rank"],
        "Signal": df_all["Signal"],
        "Exp_Return_6M(%)": df_all["Exp_Return_6M(%)"],
        "Exp_Return_12M(%)": df_all["Exp_Return_12M(%)"],
        "Exp_Alpha_12M(%)": df_all["Exp_Alpha_12M(%)"],
        "Alpha_Score": df_all["Alpha_Score"],
        "Universe_Size": len(df_all),
        "Data_Source": df_all["Data_Source"],
    })[PRICE_LOG_COLUMNS]

    os.makedirs(os.path.join(log_dir, "runs"), exist_ok=True)
    snap.to_csv(
        os.path.join(log_dir, "runs", f"predictions_{run_date}.csv"),
        index=False, encoding="utf-8-sig"
    )

    history_path = os.path.join(log_dir, "predictions_history.csv")
    if os.path.exists(history_path):
        history = pd.read_csv(history_path, dtype={"Ticker": str, "Run_Date": str, "Price_Date": str})
        history = history[history["Run_Date"] != run_date]
        history = pd.concat([history, snap], ignore_index=True)
    else:
        history = snap
    history = history.sort_values(["Run_Date", "Rank"]).reset_index(drop=True)
    history.to_csv(history_path, index=False, encoding="utf-8-sig")
    print(
        f"[Price Log] {run_date} 기준 {len(snap)}개 종목 가격·예측 기록 완료 "
        f"(누적 {history['Run_Date'].nunique()}회, '{history_path}')"
    )
    return history


def evaluate_prediction_tracking(
    history: pd.DataFrame,
    current_prices: Dict[str, float],
    current_price_date: str,
    run_date: str,
    log_dir: str = "logs"
) -> Dict[str, object]:
    """
    과거 실행에서 기록한 가격 대비 현재 가격으로 실현 수익률을 계산하고,
    예측(시그널·알파 스코어)이 맞았는지 실행 회차별로 집계합니다.
    """
    past = history[history["Run_Date"] < run_date].copy()
    if past.empty:
        return {"summary": [], "signal_breakdown": [], "per_ticker": {}, "first_run": True}

    past["Current_Price"] = past["Ticker"].map(current_prices)
    past = past[past["Current_Price"].notna() & (past["Price_KRW"] > 0)]
    if past.empty:
        return {"summary": [], "signal_breakdown": [], "per_ticker": {}, "first_run": False}

    past["Realized_Return(%)"] = (past["Current_Price"] / past["Price_KRW"] - 1.0) * 100.0
    elapsed_days = (
        pd.to_datetime(current_price_date) - pd.to_datetime(past["Price_Date"])
    ).dt.days.clip(lower=0)
    past["Elapsed_Days"] = elapsed_days
    past["Expected_To_Date(%)"] = (
        (1.0 + past["Exp_Return_12M(%)"] / 100.0).clip(lower=0.01) ** (elapsed_days / 365.0) - 1.0
    ) * 100.0
    past["Direction"] = past["Signal"].map(_signal_direction)
    past["Current_Price_Date"] = current_price_date

    detail_cols = [
        "Run_Date", "Price_Date", "Current_Price_Date", "Elapsed_Days", "Ticker", "Name",
        "Market", "Signal", "Price_KRW", "Current_Price", "Realized_Return(%)",
        "Expected_To_Date(%)", "Exp_Return_12M(%)", "Alpha_Score",
    ]
    os.makedirs(log_dir, exist_ok=True)
    past[detail_cols].to_csv(
        os.path.join(log_dir, "tracking_detail_latest.csv"), index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for rd, grp in past.groupby("Run_Date"):
        realized = grp["Realized_Return(%)"]
        longs = grp[grp["Direction"] > 0]["Realized_Return(%)"]
        shorts = grp[grp["Direction"] < 0]["Realized_Return(%)"]
        directional = grp[grp["Direction"] != 0]
        hit = (
            float(np.mean(np.sign(directional["Realized_Return(%)"] - realized.median()) == directional["Direction"]) * 100.0)
            if len(directional) else float("nan")
        )
        ic = stats.spearmanr(grp["Alpha_Score"], realized).statistic if len(grp) >= 5 else float("nan")
        summary_rows.append({
            "Run_Date": rd,
            "Base_Price_Date": str(grp["Price_Date"].iloc[0]),
            "Current_Price_Date": current_price_date,
            "Elapsed_Days": int(grp["Elapsed_Days"].median()),
            "N_Tracked": int(len(grp)),
            "Universe_Mean(%)": round(float(realized.mean()), 2),
            "Expected_Mean_To_Date(%)": round(float(grp["Expected_To_Date(%)"].mean()), 2),
            "Long_Mean(%)": round(float(longs.mean()), 2) if len(longs) else None,
            "Short_Mean(%)": round(float(shorts.mean()), 2) if len(shorts) else None,
            "Long_Short_Spread(%p)": round(float(longs.mean() - shorts.mean()), 2) if len(longs) and len(shorts) else None,
            "Rank_IC": round(float(ic), 3) if ic == ic else None,
            "Hit_Rate_vs_Median(%)": round(hit, 1) if hit == hit else None,
        })

    latest_rd = past["Run_Date"].max()
    breakdown = []
    for sig, grp in past[past["Run_Date"] == latest_rd].groupby("Signal"):
        breakdown.append({
            "Run_Date": latest_rd,
            "Signal": sig,
            "N": int(len(grp)),
            "Realized_Mean(%)": round(float(grp["Realized_Return(%)"].mean()), 2),
            "Expected_Mean_To_Date(%)": round(float(grp["Expected_To_Date(%)"].mean()), 2),
        })
    breakdown.sort(key=lambda r: -r["Expected_Mean_To_Date(%)"])

    per_ticker = {
        rec["Ticker"]: {
            "base_date": rec["Price_Date"],
            "base_price": float(rec["Price_KRW"]),
            "base_signal": rec["Signal"],
            "realized": round(float(rec["Realized_Return(%)"]), 2),
            "expected": round(float(rec["Expected_To_Date(%)"]), 2),
        }
        for rec in past[past["Run_Date"] == latest_rd].to_dict("records")
    }

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(log_dir, "monitoring_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n--- [Monitoring] 과거 {len(summary_rows)}회 예측의 실현 성과 ('{summary_path}') ---")
    print(summary_df.to_string(index=False))
    return {
        "summary": summary_rows,
        "signal_breakdown": breakdown,
        "per_ticker": per_ticker,
        "first_run": False,
    }


def build_run_history(history: pd.DataFrame, log_dir: str = "logs") -> List[Dict[str, object]]:
    """가격 로그의 실행 회차별 요약과 스케줄러 이벤트 로그를 대시보드용으로 정리합니다."""
    rows = []
    for rd, grp in history.groupby("Run_Date"):
        rows.append({
            "Run_Date": rd,
            "Price_Date": str(grp["Price_Date"].max()),
            "N_Logged": int(len(grp)),
            "N_Long": int((grp["Signal"].map(_signal_direction) > 0).sum()),
            "N_Short": int((grp["Signal"].map(_signal_direction) < 0).sum()),
            "Snapshot": f"{log_dir}/runs/predictions_{rd}.csv",
        })
    return sorted(rows, key=lambda r: r["Run_Date"], reverse=True)


def read_scheduler_events(log_dir: str = "logs", limit: int = 12) -> List[str]:
    path = os.path.join(log_dir, "scheduler", "scheduler.log")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[-limit:][::-1]


def collect_current_prices(
    loader: "FinancialFleetDataLoader",
    df_all: pd.DataFrame,
    history: pd.DataFrame
) -> Tuple[Dict[str, float], str]:
    """이번 실행 가격을 기본으로, 이번 유니버스에서 빠진 과거 종목은 네이버 시세로 보충합니다."""
    prices = dict(zip(df_all["Ticker"], df_all["Last_Price(KRW)"].astype(float)))
    price_date = str(df_all["Last_Trade_Date"].max())
    missing = sorted(set(history["Ticker"]) - set(prices))
    if missing:
        close_df, _ = loader._download_naver_price_panel(missing, freq="M", max_workers=8)
        if close_df is not None and not close_df.empty:
            for tk in close_df.columns:
                s = close_df[tk].dropna()
                if len(s):
                    prices[tk] = float(s.iloc[-1])
    return prices, price_date


DASHBOARD_TAB_SOURCES: List[Tuple[str, str, str]] = [
    ("v1", "v1 · main.py", os.path.join("logs", "dashboard_payload.json")),
    ("v2", "v2 · main_v2.py", os.path.join("logs_v2", "dashboard_payload.json")),
]


def build_dashboard_payload(
    df_all: pd.DataFrame,
    backtest_df: Optional[pd.DataFrame],
    meta: Dict[str, object],
    monitoring: Optional[Dict[str, object]] = None,
    comparison: Optional[Dict[str, object]] = None,
    features: Optional[Dict[str, bool]] = None,
    title: str = "",
    description: str = ""
) -> Dict[str, object]:
    """대시보드 한 탭을 그리는 데 필요한 실행 결과를 JSON 직렬화 가능한 dict로 묶습니다."""
    backtest_records = []
    if backtest_df is not None and not backtest_df.empty:
        backtest_records = json.loads(backtest_df.to_json(orient="records", force_ascii=False))
    return {
        "title": title,
        "description": description,
        "features": features or {},
        "meta": meta,
        "rows": json.loads(df_all.to_json(orient="records", force_ascii=False)),
        "backtest": backtest_records,
        "monitoring": monitoring or {},
        "comparison": comparison or {},
    }


def save_dashboard_payload(payload: Dict[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


TRACKING_HORIZONS_MONTHS = (3, 6, 9, 12)


def build_price_tracking(
    history_sources: List[Tuple[str, str]],
    horizons: Tuple[int, ...] = TRACKING_HORIZONS_MONTHS,
    tolerance_days: int = 31
) -> Dict[str, object]:
    """
    버전별 가격·예측 로그를 합쳐 종목마다 첫 평가 당시 가격(기준가)과
    기준가 일자 + N개월에 가장 가까운 기록 가격, 첫 평가 시점의 v1/v2 예측 수익률을 정리합니다.
    정기 실행이 며칠 밀려도 매칭되도록 목표일 ±tolerance_days 이내에서 가장 가까운 기록을 씁니다.
    """
    frames = []
    for version, path in history_sources:
        if not os.path.exists(path):
            continue
        h = pd.read_csv(path, dtype={"Ticker": str, "Run_Date": str, "Price_Date": str})
        h["Version"] = version
        frames.append(h)
    if not frames:
        return {}
    hist = pd.concat(frames, ignore_index=True)
    hist = hist[hist["Price_KRW"] > 0].copy()
    hist["Price_Date_TS"] = pd.to_datetime(hist["Price_Date"], errors="coerce")
    hist = hist.dropna(subset=["Price_Date_TS"])
    versions = [v for v, _ in history_sources]

    rows = []
    for ticker, grp in hist.groupby("Ticker", sort=False):
        grp = grp.sort_values(["Run_Date", "Version"])
        base_run = grp["Run_Date"].min()
        base_rows = grp[grp["Run_Date"] == base_run]
        base = base_rows.iloc[0]
        base_ts = base["Price_Date_TS"]

        obs = grp.drop_duplicates("Price_Date_TS").set_index("Price_Date_TS")["Price_KRW"].sort_index()
        later = obs[obs.index > base_ts]
        rec: Dict[str, object] = {
            "t": ticker,
            "n": base["Name"],
            "m": base["Market"],
            "bd": base["Price_Date"],
            "bp": float(base["Price_KRW"]),
        }
        for k in horizons:
            rec[f"h{k}"] = None
            if later.empty:
                continue
            target = base_ts + pd.DateOffset(months=k)
            gaps = np.abs((later.index - target).days)
            i = int(np.argmin(gaps))
            if gaps[i] <= tolerance_days:
                price = float(later.iloc[i])
                rec[f"h{k}"] = {
                    "p": price,
                    "r": round((price / rec["bp"] - 1.0) * 100.0, 2),
                    "d": later.index[i].strftime("%Y-%m-%d"),
                }
        for v in versions:
            vg = grp[grp["Version"] == v]
            first = vg[vg["Run_Date"] == vg["Run_Date"].min()].iloc[0] if not vg.empty else None
            for col, key in (("Exp_Return_6M(%)", "6"), ("Exp_Return_12M(%)", "12")):
                val = None if first is None else first[col]
                rec[f"{v}_{key}"] = None if val is None or pd.isna(val) else float(val)
        rows.append(rec)

    base_dates = sorted({r["bd"] for r in rows})
    return {
        "horizons": list(horizons),
        "versions": versions,
        "base_dates": base_dates,
        "rows": rows,
    }


def write_tabbed_dashboard(
    html_path: str = "krx_dashboard.html",
    tab_sources: Optional[List[Tuple[str, str, str]]] = None
) -> str:
    """저장된 버전별 payload를 모아 탭으로 나뉜 단일 HTML 대시보드를 생성합니다."""
    tabs = []
    for tab_id, label, path in (tab_sources or DASHBOARD_TAB_SOURCES):
        payload = None
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, ValueError) as e:
                print(f"[Dashboard] '{path}' 로드 실패: {e}")
        tabs.append({"id": tab_id, "label": label, "source": path, "payload": payload})

    tracking = build_price_tracking([
        (tab_id, os.path.join(os.path.dirname(path) or ".", "predictions_history.csv"))
        for tab_id, _, path in (tab_sources or DASHBOARD_TAB_SOURCES)
    ])
    bundle = {"tabs": tabs, "tracking": tracking, "built_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    bundle_json = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    html_doc = DASHBOARD_HTML_TEMPLATE.replace("__PAYLOAD__", bundle_json)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    loaded = [t["id"] for t in tabs if t["payload"]]
    print(f"\n[Output Saved] HTML 대시보드가 '{html_path}'에 저장되었습니다. (탭: {', '.join(loaded) or '없음'})")
    return html_path


def write_krx_html_dashboard(
    df_all: pd.DataFrame,
    backtest_df: Optional[pd.DataFrame],
    meta: Dict[str, object],
    html_path: str = "krx_dashboard.html",
    monitoring: Optional[Dict[str, object]] = None,
    payload_path: Optional[str] = None
) -> str:
    """v1(main.py) 결과를 payload로 저장하고, v2 결과와 함께 탭 대시보드를 다시 생성합니다."""
    payload = build_dashboard_payload(
        df_all, backtest_df, meta, monitoring,
        features={"tracking": monitoring is not None, "fundamentals": False},
        title="v1 · 가격/거래량 텐서 역변환",
        description=(
            "main.py 결과입니다. 시세는 실데이터지만 재무 지표(ROE·영업이익률·PBR)는 네이버 종목 페이지 파싱이 "
            "실패해 대체값이 들어가므로, 시그널은 사실상 가격·거래량 패턴에서 나옵니다."
        ),
    )
    payload_path = payload_path or os.path.join(str(meta.get("log_dir") or "logs"), "dashboard_payload.json")
    save_dashboard_payload(payload, payload_path)
    sources = [
        (tid, label, payload_path if tid == "v1" else path)
        for tid, label, path in DASHBOARD_TAB_SOURCES
    ]
    return write_tabbed_dashboard(html_path, sources)


DASHBOARD_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KRX 예상 수익률 대시보드</title>
  <style>
    :root {
      --bg: #0b1220;
      --bg-elev: #121a2b;
      --bg-soft: #182236;
      --line: #243049;
      --text: #e8eef9;
      --muted: #93a0b8;
      --accent: #5b8def;
      --long: #3dd68c;
      --short: #f07178;
      --neutral: #d6b25e;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: "Pretendard", "Noto Sans KR", -apple-system, BlinkMacSystemFont, sans-serif; }
    body { padding: 0 28px 64px; }
    h1 { font-size: 26px; letter-spacing: -0.03em; margin: 0 0 6px; }
    h2 { font-size: 16px; margin: 0 0 12px; font-weight: 650; }
    p { color: var(--muted); margin: 0; line-height: 1.5; }
    .topbar { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); display: flex; align-items: flex-end; gap: 24px; padding: 16px 0 0; margin-bottom: 22px; flex-wrap: wrap; }
    .brand { font-size: 15px; font-weight: 700; padding-bottom: 12px; }
    .brand small { display: block; color: var(--muted); font-weight: 400; font-size: 11px; margin-top: 2px; }
    .tabs { display: flex; gap: 4px; }
    .tab { background: transparent; color: var(--muted); border: 1px solid transparent; border-bottom: none; border-radius: 10px 10px 0 0; padding: 9px 16px 10px; cursor: pointer; text-align: left; font: inherit; }
    .tab .name { font-size: 14px; font-weight: 650; }
    .tab .sub { font-size: 11px; margin-top: 2px; }
    .tab:hover { color: var(--text); }
    .tab.active { color: var(--text); background: var(--bg-elev); border-color: var(--line); margin-bottom: -1px; }
    .tab.active .name { color: var(--accent); }
    .tab.empty .sub { color: var(--short); }
    .panel[hidden] { display: none; }
    .panel { min-width: 0; max-width: 100%; }
    .grid-2 > *, .kpis > * { min-width: 0; }
    .hero { display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; margin-bottom: 18px; flex-wrap: wrap; }
    .desc { background: var(--bg-elev); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 8px; padding: 10px 14px; margin-bottom: 18px; font-size: 13px; }
    .kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }
    .card { background: var(--bg-elev); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
    .kpi .label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .kpi .value { font-size: 26px; font-weight: 700; letter-spacing: -0.03em; }
    .kpi .sub { color: var(--muted); font-size: 12px; margin-top: 6px; }
    .grid-2 { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 12px; margin-bottom: 12px; }
    .grid-2.equal { grid-template-columns: 1fr 1fr; }
    .grid-2.log { grid-template-columns: 0.9fr 1.1fr; }
    .bars { display: flex; flex-direction: column; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 120px 1fr 48px; gap: 8px; align-items: center; font-size: 13px; }
    .bar-row > div:first-child { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .track { height: 10px; background: var(--bg-soft); border-radius: 99px; overflow: hidden; }
    .fill { height: 100%; border-radius: 99px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { color: var(--muted); font-weight: 600; font-size: 12px; position: sticky; top: 0; background: var(--bg-elev); }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.ours td { font-weight: 650; }
    tr.ours td:first-child { color: var(--accent); }
    .pos { color: var(--long); }
    .neg { color: var(--short); }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 11px; background: var(--bg-soft); color: var(--muted); }
    .pill.long { color: var(--long); background: rgba(61,214,140,.12); }
    .pill.short { color: var(--short); background: rgba(240,113,120,.12); }
    .pill.neutral { color: var(--neutral); background: rgba(214,178,94,.12); }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    input, select { background: var(--bg-soft); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; font-size: 13px; }
    input { min-width: 220px; }
    .table-wrap { max-height: 560px; overflow: auto; }
    .caption { color: var(--muted); font-size: 12px; margin-top: 8px; }
    svg { display: block; width: 100%; height: 180px; }
    .mini-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .mini-stats .label { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
    .mini-stats .value { font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums; }
    .mini-stats .value.accent { color: var(--accent); }
    .events { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); line-height: 1.7; max-height: 180px; overflow: auto; }
    .events .ok { color: var(--long); }
    .events .fail { color: var(--short); }
    .matrix td.heat { text-align: right; font-variant-numeric: tabular-nums; }
    .matrix th.row { position: static; text-align: left; }
    .empty-state { padding: 48px 16px; text-align: center; }
    .empty-state code { background: var(--bg-soft); padding: 2px 6px; border-radius: 4px; color: var(--text); }
    .board-head { margin-bottom: 10px; }
    .board-head p { font-size: 12px; margin-top: 4px; }
    .board-tools { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
    .board-tools input { flex: 1; min-width: 0; font-size: 16px; }
    .board-tools .count { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .board-wrap { overflow: auto; -webkit-overflow-scrolling: touch; max-height: calc(100dvh - var(--top-h, 80px) - 120px); border: 1px solid var(--line); border-radius: 12px; background: var(--bg-elev); }
    .board { font-size: 13px; }
    .board th { cursor: pointer; user-select: none; -webkit-user-select: none; z-index: 2; line-height: 1.25; vertical-align: bottom; }
    .board th small { display: block; font-weight: 400; font-size: 10px; color: var(--muted); }
    .board th.sorted { color: var(--accent); }
    .board th .arrow { font-size: 10px; margin-left: 2px; }
    .board td { vertical-align: top; }
    .board .stick { position: sticky; left: 0; background: var(--bg-elev); z-index: 1; border-right: 1px solid var(--line); max-width: 150px; }
    .board th.stick { z-index: 3; }
    .board .nm { font-weight: 600; overflow: hidden; text-overflow: ellipsis; max-width: 140px; }
    .board .sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
    .board .sub.pos { color: var(--long); }
    .board .sub.neg { color: var(--short); }
    .board td.empty { color: #4a5670; text-align: center; }
    .board .sep { border-left: 1px solid var(--line); }
    @media (max-width: 1100px) {
      .kpis, .grid-2, .grid-2.equal, .grid-2.log { grid-template-columns: 1fr; }
      body { padding: 0 16px 48px; }
    }
    @media (max-width: 640px) {
      body { padding: 0 10px 32px; }
      .topbar { flex-direction: column; align-items: stretch; gap: 0; padding-top: 8px; margin-bottom: 14px; }
      .brand { padding-bottom: 6px; font-size: 14px; }
      .brand small { display: none; }
      .tabs { overflow-x: auto; scrollbar-width: none; min-width: 0; max-width: 100%; }
      .tabs::-webkit-scrollbar { display: none; }
      .tab { padding: 7px 12px 8px; flex: 0 0 auto; }
      .tab .name { font-size: 13px; }
      .tab .sub { font-size: 10px; }
      h1 { font-size: 20px; }
      h2 { font-size: 14px; }
      .hero .pill { display: none; }
      .hero p { font-size: 12px; overflow-wrap: anywhere; }
      .desc { font-size: 12px; }
      .card { padding: 12px; }
      .kpis { grid-template-columns: 1fr 1fr; gap: 8px; }
      .kpi .value { font-size: 20px; }
      .mini-stats { grid-template-columns: 1fr 1fr; }
      .mini-stats .value { font-size: 15px; }
      table { font-size: 12px; }
      th, td { padding: 6px 6px; }
      .toolbar input, .toolbar select { flex: 1 1 100%; min-width: 0; font-size: 16px; }
      .bar-row { grid-template-columns: 84px 1fr 36px; font-size: 12px; }
      .board { font-size: 12px; }
      .board .stick, .board .nm { max-width: 108px; }
      .board-wrap { max-height: calc(100dvh - var(--top-h, 90px) - 140px); }
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="brand">KRX 예상 수익률 대시보드<small id="built-at"></small></div>
    <nav class="tabs" id="tabs"></nav>
  </div>
  <main id="panels"></main>

  <template id="tab-tpl">
    <div class="hero">
      <div>
        <h1 data-k="title"></h1>
        <p data-k="subtitle"></p>
      </div>
      <div class="pill" data-k="source"></div>
    </div>
    <p class="desc" data-k="desc"></p>
    <div class="kpis" data-k="kpis"></div>
    <div class="card" data-k="sec-compare" style="margin-bottom:12px">
      <h2>v1(main.py) 대비 비교</h2>
      <div class="mini-stats" data-k="cmp-stats"></div>
      <div class="grid-2 equal" style="margin:16px 0 0">
        <div>
          <h2>시그널 전환 매트릭스 (행: v1, 열: v2)</h2>
          <div class="table-wrap" style="max-height:none"><table class="matrix" data-k="cmp-matrix"></table></div>
          <p class="caption">공통 종목 기준. 대각선이 같은 등급으로 유지된 종목 수입니다.</p>
        </div>
        <div>
          <h2>v1 상위 10종목의 v2 재평가</h2>
          <div class="table-wrap" style="max-height:none"><table data-k="cmp-v1top"></table></div>
        </div>
      </div>
      <h2 style="margin-top:16px">v2 상위 10종목의 v1 순위</h2>
      <div class="table-wrap" style="max-height:none"><table data-k="cmp-v2top"></table></div>
    </div>
    <div class="grid-2 log" data-k="sec-tracking">
      <div class="card">
        <h2>가격 기록 · 자동 실행</h2>
        <div class="mini-stats" data-k="log-stats"></div>
        <h2 style="margin-top:16px">기록 회차</h2>
        <div class="table-wrap" style="max-height:220px"><table data-k="run-table"></table></div>
        <p class="caption" data-k="log-caption"></p>
      </div>
      <div class="card">
        <h2>정기 모니터링 · 과거 예측의 실현 성과</h2>
        <div data-k="mon-body"></div>
        <p class="caption">기준가: 각 실행일의 직전 거래일 종가. 실현 수익률 = 현재 종가 / 기준가 - 1. 기대치는 12M 예상 수익률을 경과일만큼 복리 환산. 적중률은 LONG이 유니버스 중앙값을 넘고 SHORT가 밑돈 비율.</p>
        <h2 style="margin-top:16px">스케줄러 이벤트</h2>
        <div class="events" data-k="sched-events"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h2>시그널 분포</h2>
        <div class="bars" data-k="signal-bars"></div>
        <p class="caption">알파 스코어 5분위 기준 매매 의견.</p>
      </div>
      <div class="card">
        <h2>12개월 예상 수익률 분포</h2>
        <svg data-k="hist" viewBox="0 0 640 180" preserveAspectRatio="none"></svg>
        <p class="caption">가로축: 12개월 예상 수익률(%), 세로축: 종목 수</p>
      </div>
    </div>
    <div class="card" style="margin-bottom:12px">
      <h2>워크포워드 롱숏 백테스트</h2>
      <div class="table-wrap" style="max-height:none"><table data-k="bt-table"></table></div>
      <p class="caption">표본 외 롤링 윈도우, 상위/하위 20% 롱숏, 비용 15bp. 굵게 표시된 행이 이 버전의 모델입니다.</p>
    </div>
    <div class="grid-2 equal">
      <div class="card">
        <h2>예상 수익률 상위 (LONG)</h2>
        <div class="table-wrap" style="max-height:420px"><table data-k="long-table"></table></div>
      </div>
      <div class="card">
        <h2>예상 수익률 하위 (SHORT)</h2>
        <div class="table-wrap" style="max-height:420px"><table data-k="short-table"></table></div>
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <h2>전체 종목</h2>
      <div class="toolbar">
        <input data-k="q" placeholder="종목명 또는 티커 검색" />
        <select data-k="mkt"><option value="">전체 시장</option></select>
        <select data-k="sig"><option value="">전체 시그널</option></select>
      </div>
      <div class="table-wrap"><table data-k="all-table"></table></div>
      <p class="caption" data-k="table-caption"></p>
    </div>
  </template>

  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("payload").textContent);
    const TABS = DATA.tabs || [];
    const fmt = (n, d=1) => (n == null || Number.isNaN(Number(n))) ? "-" : Number(n).toLocaleString("ko-KR", { maximumFractionDigits: d, minimumFractionDigits: d });
    const fmtInt = (n) => (n == null || Number.isNaN(Number(n))) ? "-" : Math.round(Number(n)).toLocaleString("ko-KR");
    const clsNum = (n) => n > 0 ? "num pos" : (n < 0 ? "num neg" : "num");
    const sigClass = (s) => {
      const t = String(s || "");
      if (t.includes("LONG")) return "pill long";
      if (t.includes("SHORT")) return "pill short";
      return "pill neutral";
    };
    const SIG_ORDER = ["STRONG LONG (최우선 매수)", "LONG (매수)", "NEUTRAL (중립)", "SHORT (비중축소/공매도)", "STRONG SHORT (핵심 공매도)"];
    const sigLabel = (s) => (String(s).match(/\((.+)\)/) || [s, s])[1];
    const sigShort = (s) => String(s || "").split(" (")[0];

    function cell(v, kind) {
      switch (kind) {
        case "int": return `<td class="num">${fmtInt(v)}</td>`;
        case "pct": return `<td class="${clsNum(v)}">${fmt(v, 2)}</td>`;
        case "num2": return `<td class="num">${fmt(v, 2)}</td>`;
        case "score": return `<td class="${clsNum(v)}">${fmt(v, 3)}</td>`;
        case "sig": return `<td><span class="${sigClass(v)}">${v ?? "-"}</span></td>`;
        default: return `<td>${v ?? "-"}</td>`;
      }
    }
    const isNum = (kind) => kind && kind !== "text" && kind !== "sig";
    function table(cols, data, getter) {
      const get = getter || ((r, k) => r[k]);
      return `<thead><tr>${cols.map(([, label, kind]) => `<th${isNum(kind) ? " class='num'" : ""}>${label}</th>`).join("")}</tr></thead>` +
        `<tbody>${data.map(r => `<tr>${cols.map(([k, , kind]) => cell(get(r, k), kind)).join("")}</tr>`).join("")}</tbody>`;
    }
    function miniStats(items) {
      return items.map(([l, v, a]) => `<div><div class="label">${l}</div><div class="value${a ? " accent" : ""}">${v}</div></div>`).join("");
    }

    function renderTracking($, P, rows, meta) {
      const runHist = meta.run_history || [];
      $("log-stats").innerHTML = miniStats([
        ["분석일", meta.run_date || "-"],
        ["가격 기준일", meta.price_date || "-"],
        ["기록 종목", fmtInt(rows.length)],
        ["누적 기록 회차", fmtInt(meta.price_log_runs || runHist.length) + "회"],
        ["실행 주기", (meta.schedule_months || 3) + "개월"],
        ["다음 자동 실행", meta.next_run_date || "-", true],
      ]);
      $("run-table").innerHTML = table([
        ["Run_Date", "분석일", "text"], ["Price_Date", "가격 기준일", "text"], ["N_Logged", "기록 종목", "int"],
        ["N_Long", "LONG", "int"], ["N_Short", "SHORT", "int"], ["Snapshot", "스냅샷", "text"],
      ], runHist);
      $("log-caption").textContent =
        `누적 로그: ${meta.log_dir || "logs"}/predictions_history.csv · 대시보드 보관본: ${meta.log_dir || "logs"}/dashboards/ · 매일 18:00과 로그인 시 확인하고, 마지막 성공 후 ${meta.schedule_months || 3}개월이 지났을 때만 실행합니다. 실패하면 다음 트리거에서 재시도합니다.`;
      const events = meta.scheduler_events || [];
      $("sched-events").innerHTML = events.length
        ? events.map(e => `<div class="${/ OK /.test(e) ? "ok" : (/ FAIL /.test(e) ? "fail" : "")}">${e}</div>`).join("")
        : "<div>아직 스케줄러 이벤트가 없습니다.</div>";

      const mon = P.monitoring || {};
      const monSummary = mon.summary || [];
      if (!monSummary.length) {
        $("mon-body").innerHTML = `<p>${meta.price_log_runs > 1 ? "과거 기록과 매칭되는 종목이 없습니다." : "이번 실행(" + (meta.run_date || "") + ")이 첫 기준 기록입니다. " + (meta.next_run_date || "다음 실행") + " 실행부터 이 기준가 대비 실현 수익률이 여기에 집계됩니다."} 기록된 기준가: ${fmtInt(rows.length)}종목, 가격 기준일 ${meta.price_date || "-"}.</p>`;
        return;
      }
      let html = `<div class="table-wrap" style="max-height:none"><table>` + table([
        ["Run_Date", "예측일", "text"], ["Base_Price_Date", "기준가 일자", "text"], ["Elapsed_Days", "경과일", "int"],
        ["N_Tracked", "추적 종목", "int"], ["Universe_Mean(%)", "실현 평균%", "pct"], ["Expected_Mean_To_Date(%)", "기대 평균%", "pct"],
        ["Long_Mean(%)", "LONG 실현%", "pct"], ["Short_Mean(%)", "SHORT 실현%", "pct"], ["Long_Short_Spread(%p)", "롱숏 스프레드%p", "pct"],
        ["Rank_IC", "Rank IC", "score"], ["Hit_Rate_vs_Median(%)", "적중률%", "num2"],
      ], monSummary.slice().reverse()) + `</table></div>`;
      const bd = mon.signal_breakdown || [];
      if (bd.length) {
        html += `<h2 style="margin-top:16px">최근 예측(${bd[0].Run_Date}) 시그널별 실현 vs 기대</h2><div class="table-wrap" style="max-height:none"><table>` + table([
          ["Signal", "시그널", "sig"], ["N", "종목 수", "int"],
          ["Realized_Mean(%)", "실현 평균%", "pct"], ["Expected_Mean_To_Date(%)", "기대 평균%", "pct"],
        ], bd) + `</table></div>`;
      }
      $("mon-body").innerHTML = html;
    }

    function renderComparison($, C) {
      const rng = (r) => r ? `${fmt(r[0], 1)} ~ ${fmt(r[1], 1)}%` : "-";
      $("cmp-stats").innerHTML = miniStats([
        ["공통 분석 종목", fmtInt(C.n_common)],
        ["알파 스코어 순위상관", fmt(C.rank_corr, 3), true],
        ["시그널 방향 일치율", fmt(C.direction_agree_pct, 1) + "%"],
        ["PBR 고유값 수 (v1 / v2)", fmtInt(C.v1_pbr_unique) + " / " + fmtInt(C.v2_pbr_unique)],
        ["v1 12M 기대수익률 범위", rng(C.v1_ret_range)],
        ["v2 12M 기대수익률 범위", rng(C.v2_ret_range)],
      ]);
      const M = C.matrix || {};
      const labels = M.labels || [];
      const counts = M.counts || [];
      const cMax = Math.max(1, ...counts.flat());
      $("cmp-matrix").innerHTML =
        `<thead><tr><th>v1 \\ v2</th>${labels.map(l => `<th class="num">${sigLabel(l)}</th>`).join("")}</tr></thead><tbody>` +
        counts.map((row, i) => `<tr><th class="row"><span class="${sigClass(labels[i])}">${sigLabel(labels[i])}</span></th>${row.map((c, j) =>
          `<td class="heat" style="background:rgba(91,141,239,${(0.08 + 0.6 * c / cMax).toFixed(2)})${i === j ? ";font-weight:700" : ""}">${fmtInt(c)}</td>`).join("")}</tr>`).join("") +
        `</tbody>`;
      const topCols = [
        ["Ticker", "Ticker", "text"], ["Name", "Name", "text"], ["Rank_v1", "v1 순위", "int"], ["Rank_v2", "v2 순위", "int"],
        ["Exp_Return_12M(%)_v1", "v1 12M%", "pct"], ["Exp_Return_12M(%)_v2", "v2 12M%", "pct"],
        ["Real_ROE(%)_v2", "ROE%", "pct"], ["Real_DebtRatio(%)_v2", "부채비율%", "num2"], ["Real_PBR_v2", "PBR", "num2"],
      ];
      $("cmp-v1top").innerHTML = table(topCols, C.v1_top || []);
      $("cmp-v2top").innerHTML = table(topCols, C.v2_top || []);
    }

    function renderTab(root, P) {
      const $ = (k) => root.querySelector(`[data-k="${k}"]`);
      const rows = P.rows || [];
      const meta = P.meta || {};
      const bt = P.backtest || [];
      const feat = P.features || {};

      $("title").textContent = P.title || "KRX 예상 수익률";
      $("desc").textContent = P.description || "";
      if (!P.description) $("desc").remove();
      $("subtitle").textContent =
        (meta.generated_at || "") + " · 소스 " + (meta.listing_source || "-") +
        " · 시세 " + (meta.data_source || "-") +
        " · 시장 " + (meta.market || "ALL") +
        (meta.time_start ? (" · 시계열 " + meta.time_start + " ~ " + meta.time_end) : "");
      $("source").textContent = meta.data_source || meta.listing_source || "KRX";

      const kospi = rows.filter(r => r.Market === "KOSPI").length;
      const kosdaq = rows.filter(r => r.Market === "KOSDAQ").length;
      const longs = rows.filter(r => String(r.Signal).includes("LONG")).length;
      const shorts = rows.filter(r => String(r.Signal).includes("SHORT")).length;
      const rets = rows.map(r => Number(r["Exp_Return_12M(%)"])).filter(v => Number.isFinite(v));
      const mid = rets.length ? rets.slice().sort((a, b) => a - b)[Math.floor(rets.length / 2)] : 0;
      const best = rets.length ? Math.max(...rets) : 0;
      const pct = (n) => (rows.length ? Math.round(n / rows.length * 100) : 0) + "%";
      $("kpis").innerHTML = [
        ["분석 종목", fmtInt(rows.length), "KOSPI " + fmtInt(kospi) + " / KOSDAQ " + fmtInt(kosdaq)],
        ["LONG 시그널", fmtInt(longs), "전체 대비 " + pct(longs)],
        ["SHORT 시그널", fmtInt(shorts), "전체 대비 " + pct(shorts)],
        ["12M 중앙값", fmt(mid, 2) + "%", "유니버스 예상 수익률 중앙"],
        ["12M 최상위", fmt(best, 2) + "%", "1위 종목 예상 수익률"],
      ].map(([l, v, s]) => `<div class="card kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`).join("");

      if (feat.tracking) renderTracking($, P, rows, meta); else $("sec-tracking").remove();
      if (P.comparison && P.comparison.n_common) renderComparison($, P.comparison); else $("sec-compare").remove();

      const sigCount = {};
      rows.forEach(r => { sigCount[r.Signal] = (sigCount[r.Signal] || 0) + 1; });
      const sigMax = Math.max(1, ...Object.values(sigCount));
      $("signal-bars").innerHTML = SIG_ORDER.filter(s => sigCount[s]).map(s => {
        const n = sigCount[s] || 0;
        const color = s.includes("LONG") ? "var(--long)" : (s.includes("SHORT") ? "var(--short)" : "var(--neutral)");
        return `<div class="bar-row"><div>${sigLabel(s)}</div><div class="track"><div class="fill" style="width:${(n / sigMax) * 100}%;background:${color}"></div></div><div class="num">${n}</div></div>`;
      }).join("");

      const bins = 16;
      const minR = Math.min(...rets, 0), maxR = Math.max(...rets, 0);
      const span = Math.max(1e-6, maxR - minR);
      const hist = Array(bins).fill(0);
      rets.forEach(v => { hist[Math.min(bins - 1, Math.floor((v - minR) / span * bins))] += 1; });
      const hMax = Math.max(1, ...hist);
      const gap = 3, bw = (640 - gap * (bins + 1)) / bins;
      $("hist").innerHTML = hist.map((c, i) => {
        const bh = (c / hMax) * 150;
        const center = minR + (i + 0.5) * span / bins;
        return `<rect x="${gap + i * (bw + gap)}" y="${170 - bh}" width="${bw}" height="${bh}" fill="${center >= 0 ? "#3dd68c" : "#f07178"}" rx="2"></rect>`;
      }).join("") + `<text x="8" y="14" fill="#93a0b8" font-size="11">${fmt(minR, 1)}%</text><text x="590" y="14" fill="#93a0b8" font-size="11">${fmt(maxR, 1)}%</text>`;

      const btCols = [
        ["Strategy", "Strategy", "text"], ["Ann_Return(%)", "Ann Return(%)", "pct"], ["Ann_Vol(%)", "Ann Vol(%)", "num2"],
        ["Sharpe_Ratio", "Sharpe", "pct"], ["Max_Drawdown(%)", "Max DD(%)", "num2"], ["Mean_Rank_IC", "Mean Rank IC", "score"], ["Hit_Rate(%)", "Hit Rate(%)", "num2"],
      ];
      if (bt.length) {
        $("bt-table").innerHTML = table(btCols, bt).replace(/<tr><td>Ours/g, '<tr class="ours"><td>Ours');
      } else {
        $("bt-table").innerHTML = "<tbody><tr><td>백테스트 결과가 없습니다.</td></tr></tbody>";
      }

      const miniCols = [
        ["Rank", "Rank", "int"], ["Ticker", "Ticker", "text"], ["Name", "Name", "text"], ["Market", "Market", "text"],
        ["Exp_Return_12M(%)", "12M%", "pct"], ["Exp_Alpha_12M(%)", "Alpha%", "pct"], ["Signal", "Signal", "sig"],
      ];
      $("long-table").innerHTML = table(miniCols, rows.slice(0, 15));
      $("short-table").innerHTML = table(miniCols, rows.slice(-15).reverse());

      const perTicker = (P.monitoring || {}).per_ticker || {};
      const hasTracking = Object.keys(perTicker).length > 0;
      const mktSel = $("mkt"), sigSel = $("sig"), qInput = $("q");
      [...new Set(rows.map(r => r.Market))].sort().forEach(m => mktSel.insertAdjacentHTML("beforeend", `<option>${m}</option>`));
      SIG_ORDER.filter(s => sigCount[s]).forEach(s => sigSel.insertAdjacentHTML("beforeend", `<option>${s}</option>`));
      const allCols = [
        ["Rank", "Rank", "int"], ["Ticker", "Ticker", "text"], ["Name", "Name", "text"], ["Market", "Market", "text"],
        ["Last_Price(KRW)", "분석일 가격", "int"], ["Last_Trade_Date", "가격 기준일", "text"],
        ...(hasTracking ? [["_base_price", "직전 기준가", "int"], ["_realized", "직전 대비 실현%", "pct"], ["_expected", "기대%", "pct"]] : []),
        ["Signal", "Signal", "sig"], ["Exp_Return_6M(%)", "6M%", "pct"], ["Exp_Return_12M(%)", "12M%", "pct"],
        ["Exp_Alpha_12M(%)", "Alpha%", "pct"], ["Alpha_Score", "Score", "score"],
        ...(feat.fundamentals ? [
          ["Real_ROE(%)", "ROE%", "pct"], ["Real_OpMargin(%)", "영업이익률%", "pct"],
          ["Real_DebtRatio(%)", "부채비율%", "num2"], ["Real_PBR", "PBR", "num2"], ["Real_PER", "PER", "num2"],
        ] : []),
      ];
      const getter = (r, k) => k.startsWith("_") ? (perTicker[r.Ticker] || {})[k.slice(1)] : r[k];
      function renderAll() {
        const q = qInput.value.trim().toLowerCase();
        const m = mktSel.value, s = sigSel.value;
        const filtered = rows.filter(r =>
          (!q || (r.Name + " " + r.Ticker).toLowerCase().includes(q)) && (!m || r.Market === m) && (!s || r.Signal === s));
        $("all-table").innerHTML = table(allCols, filtered, getter);
        $("table-caption").textContent = `표시 ${filtered.length} / 전체 ${rows.length}종목`;
      }
      qInput.addEventListener("input", renderAll);
      mktSel.addEventListener("change", renderAll);
      sigSel.addEventListener("change", renderAll);
      renderAll();
    }

    function renderPriceBoard(root, T) {
      const rows = T.rows || [];
      const H = T.horizons || [3, 6, 9, 12];
      const V = T.versions || ["v1", "v2"];
      const baseDates = T.base_dates || [];
      root.innerHTML = `
        <div class="board-head">
          <h1>가격 추적</h1>
          <p>기준가는 처음 평가했을 때의 종가(${baseDates.length === 1 ? baseDates[0] : (baseDates[0] || "-") + " ~ " + (baseDates[baseDates.length - 1] || "-")})입니다. ${H.join("·")}개월 후 칸은 정기 실행이 그 시점에 도달하면 가격과 기준가 대비 수익률로 채워집니다. 예측 수익률은 처음 평가 당시 값입니다. 머리글을 누르면 정렬됩니다.</p>
        </div>
        <div class="board-tools"><input type="search" placeholder="종목명 또는 코드 검색" data-k="q" /><span class="count" data-k="count"></span></div>
        <div class="board-wrap"><table class="board" data-k="board"></table></div>`;
      const $ = (k) => root.querySelector(`[data-k="${k}"]`);
      const cols = [
        { key: "n", label: "종목", sub: "코드 · 시장", stick: true, val: r => r.n, text: true },
        { key: "bp", label: "기준가", sub: "원", val: r => r.bp },
        ...H.map((k, i) => ({ key: "h" + k, label: k + "개월 후", sub: "가격 · 수익률", horizon: true, sep: i === 0, val: r => r["h" + k] ? r["h" + k].r : null })),
        ...["6", "12"].flatMap(p => V.map((v, i) => ({ key: v + "_" + p, label: v + " " + p + "개월", sub: "예측 수익률 %", pred: true, sep: p === "6" && i === 0, val: r => r[v + "_" + p] }))),
      ];
      let sortKey = V.includes("v2") ? "v2_12" : "n", sortDir = sortKey === "n" ? 1 : -1;
      const code = (t) => String(t).split(".")[0];
      function cellHtml(c, r) {
        const sep = c.sep ? " sep" : "";
        if (c.stick) return `<td class="stick"><div class="nm">${r.n}</div><div class="sub">${code(r.t)} · ${r.m || "-"}</div></td>`;
        if (c.key === "bp") return `<td class="num"><div>${fmtInt(r.bp)}</div><div class="sub">${r.bd}</div></td>`;
        if (c.horizon) {
          const h = r[c.key];
          if (!h) return `<td class="empty${sep}">–</td>`;
          return `<td class="num${sep}" title="${h.d}"><div>${fmtInt(h.p)}</div><div class="sub ${h.r > 0 ? "pos" : (h.r < 0 ? "neg" : "")}">${h.r > 0 ? "+" : ""}${fmt(h.r, 1)}%</div></td>`;
        }
        const v = c.val(r);
        if (v == null) return `<td class="empty${sep}">–</td>`;
        return `<td class="${clsNum(v)}${sep}">${v > 0 ? "+" : ""}${fmt(v, 1)}</td>`;
      }
      function render() {
        const q = $("q").value.trim().toLowerCase();
        const col = cols.find(c => c.key === sortKey);
        const data = rows.filter(r => !q || (r.n + " " + r.t).toLowerCase().includes(q));
        data.sort((a, b) => {
          const x = col.val(a), y = col.val(b);
          if (x == null && y == null) return 0;
          if (x == null) return 1;
          if (y == null) return -1;
          return (col.text ? String(x).localeCompare(String(y), "ko") : x - y) * sortDir;
        });
        $("board").innerHTML =
          `<thead><tr>${cols.map(c => `<th data-sort="${c.key}" class="${c.stick ? "stick" : "num"}${c.sep ? " sep" : ""}${c.key === sortKey ? " sorted" : ""}">${c.label}${c.key === sortKey ? `<span class="arrow">${sortDir > 0 ? "▲" : "▼"}</span>` : ""}<small>${c.sub}</small></th>`).join("")}</tr></thead>` +
          `<tbody>${data.map(r => `<tr>${cols.map(c => cellHtml(c, r)).join("")}</tr>`).join("")}</tbody>`;
        $("count").textContent = `${fmtInt(data.length)} / ${fmtInt(rows.length)}종목`;
      }
      $("board").addEventListener("click", (e) => {
        const th = e.target.closest("th[data-sort]");
        if (!th) return;
        const key = th.dataset.sort;
        if (key === sortKey) sortDir = -sortDir;
        else { sortKey = key; sortDir = cols.find(c => c.key === key).text ? 1 : -1; }
        render();
      });
      $("q").addEventListener("input", render);
      render();
    }

    document.getElementById("built-at").textContent = "생성 " + (DATA.built_at || "") + " · 탭마다 해당 스크립트의 마지막 실행 결과";
    const nav = document.getElementById("tabs");
    const panels = document.getElementById("panels");
    const tpl = document.getElementById("tab-tpl");
    const rendered = new Set();
    const topbar = document.querySelector(".topbar");
    const syncTopHeight = () => document.documentElement.style.setProperty("--top-h", topbar.offsetHeight + "px");
    window.addEventListener("resize", syncTopHeight);
    syncTopHeight();
    const TR = DATA.tracking || {};
    if ((TR.rows || []).length) {
      TABS.unshift({ id: "track", label: "가격 추적", board: TR });
    }
    TABS.forEach(t => {
      if (t.board) {
        nav.insertAdjacentHTML("beforeend",
          `<button class="tab" data-tab="${t.id}"><div class="name">${t.label}</div><div class="sub">${fmtInt(t.board.rows.length)}종목 · ${t.board.horizons.join("/")}개월</div></button>`);
        const panel = document.createElement("section");
        panel.className = "panel";
        panel.dataset.tab = t.id;
        panel.hidden = true;
        panels.appendChild(panel);
        return;
      }
      const P = t.payload;
      const meta = (P && P.meta) || {};
      const sub = P ? `${meta.run_date || meta.generated_at || ""} · ${fmtInt((P.rows || []).length)}종목` : "아직 실행 결과 없음";
      nav.insertAdjacentHTML("beforeend",
        `<button class="tab${P ? "" : " empty"}" data-tab="${t.id}"><div class="name">${t.label}</div><div class="sub">${sub}</div></button>`);
      const panel = document.createElement("section");
      panel.className = "panel";
      panel.dataset.tab = t.id;
      panel.hidden = true;
      panels.appendChild(panel);
    });
    function activate(id) {
      const t = TABS.find(x => x.id === id) || TABS[0];
      if (!t) return;
      nav.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === t.id));
      panels.querySelectorAll(".panel").forEach(p => { p.hidden = p.dataset.tab !== t.id; });
      if (!rendered.has(t.id)) {
        const panel = panels.querySelector(`.panel[data-tab="${t.id}"]`);
        if (t.board) {
          renderPriceBoard(panel, t.board);
        } else if (t.payload) {
          panel.appendChild(tpl.content.cloneNode(true));
          renderTab(panel, t.payload);
        } else {
          panel.innerHTML = `<div class="card empty-state"><h2>${t.label} 결과가 아직 없습니다</h2><p><code>${t.source}</code> 가 없습니다. 해당 스크립트를 실행하면 이 탭이 채워집니다.</p></div>`;
        }
        rendered.add(t.id);
      }
      if (location.hash.slice(1) !== t.id) {
        try { history.replaceState(null, "", "#" + t.id); } catch (e) {}
      }
    }
    nav.addEventListener("click", (e) => {
      const b = e.target.closest(".tab");
      if (b) activate(b.dataset.tab);
    });
    window.addEventListener("hashchange", () => activate(location.hash.slice(1)));
    activate(location.hash.slice(1));
  </script>
</body>
</html>
"""


# ==============================================================================
# MAIN EXECUTION: 전 종목(All KRX) 파이프라인 실행 및 예상 수익률 산출
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="v3.0 Fleet Tensor Inverse Framework - 한국 주식시장 전 종목(All KRX) 예상 수익률 산출기"
    )
    parser.add_argument(
        "--market", type=str, default="ALL", choices=["ALL", "KOSPI", "KOSDAQ"],
        help="분석 대상 시장 선택 (ALL: 코스피+코스닥 전 종목, KOSPI, KOSDAQ)"
    )
    parser.add_argument(
        "--max-stocks", type=int, default=0,
        help="분석할 최대 종목 수 (0 입력 시 필터링된 상장 보통주 전 종목 실행)"
    )
    parser.add_argument(
        "--min-marcap", type=float, default=30.0,
        help="최소 시가총액 필터 (단위: 십억 원, 기본값 30.0 = 시총 300억 원 이상 보통주)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=120,
        help="yfinance 주가 일괄 다운로드 배치 크기 (기본값: 120)"
    )
    parser.add_argument(
        "--fetch-statements", action="store_true",
        help="개별 종목의 야후 파이낸스 분기 재무제표 API를 직접 호출할지 여부"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="콘솔에 출력할 상/하위 종목 개수 (전체 종목은 CSV 파일에 모두 저장됨)"
    )
    parser.add_argument(
        "--output", type=str, default="krx_all_stocks_expected_returns.csv",
        help="종목별 예상 수익률 전체 결과를 저장할 CSV 파일 경로"
    )
    parser.add_argument(
        "--html-output", type=str, default="krx_dashboard.html",
        help="실행 결과를 저장할 HTML 대시보드 경로"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="분석일 가격·예측 누적 로그와 모니터링 결과를 저장할 디렉터리"
    )
    parser.add_argument(
        "--schedule-months", type=int, default=3,
        help="자동 실행 주기(개월). run_quarterly.sh의 주기와 같아야 합니다."
    )
    parser.add_argument(
        "--run-benchmark", action="store_true",
        help="시뮬레이션 DGP 검증을 함께 실행할지 여부"
    )
    parser.add_argument(
        "--allow-sim-fallback", action="store_true",
        help="네트워크 오류 시 시뮬레이션 폴백을 허용할지 여부 (기본값: False, 실제 데이터만 허용)"
    )
    args, _ = parser.parse_known_args()

    print("=" * 115)
    print(" v3.0 Fleet Tensor Inverse Framework - 한국 주식시장 전 종목(All KRX) 예상 수익률 추정 파이프라인")
    print("=" * 115)

    loader = FinancialFleetDataLoader()
    backtester = FundamentalLongShortBacktester(top_quantile=0.20, cost_bps=15.0)

    if args.run_benchmark:
        print("\n[Benchmark] 구조적 재무/회계 벤치마크 텐서 검증 실행 중...")
        sim_data = loader.generate_calibrated_fundamental_dgp(
            M=40, D=24, r=6, N=60, rho_budget_constraint=-0.82, sigma_accounting_noise=0.12, seed=42
        )
        sim_summary_df, sim_last_state = backtester.run_walk_forward_backtest(
            sim_data, lookback_window=28, rebalance_step=2
        )
        recovery_metrics = evaluate_algebraic_recovery_metrics(sim_data, sim_last_state)
        for k, v in recovery_metrics.items():
            print(f"  * {k:<34}: {v}")
        print(sim_summary_df.to_string(index=False))

    max_stocks_limit = args.max_stocks if args.max_stocks > 0 else None
    loader.load_all_krx_universe(
        market=args.market,
        max_stocks=max_stocks_limit,
        min_marcap_billion_krw=args.min_marcap
    )

    live_data = loader.fetch_live_financial_tensor(
        n_periods=36,
        freq="M",
        batch_size=args.batch_size,
        fetch_statement_api=args.fetch_statements
    )

    if live_data is not None:
        print("\n[Tensor Engine] 전 종목 6단계 대수적 텐서 역변환 및 워크포워드 백테스트 수행 중...")
        live_summary_df, _ = backtester.run_walk_forward_backtest(
            live_data, lookback_window=20, rebalance_step=2
        )
        print("\n--- [Walk-Forward Backtest] 분석 유니버스 중장기 롱숏 백테스트 성과 요약 ---")
        print(live_summary_df.to_string(index=False))

        full_engine = FundamentalTensorInverseEngine()
        final_state = full_engine.fit_transform(live_data)

        df_all = print_top_long_short_picks(
            live_data,
            final_state,
            top_n=args.top_n,
            save_csv_path=args.output
        )

        run_ts = datetime.now()
        run_date = run_ts.strftime("%Y-%m-%d")
        history = append_prediction_log(df_all, run_ts, args.log_dir)
        current_prices, price_date = collect_current_prices(loader, df_all, history)
        monitoring = evaluate_prediction_tracking(
            history, current_prices, price_date, run_date, args.log_dir
        )

        write_krx_html_dashboard(
            df_all,
            live_summary_df,
            {
                "run_date": run_date,
                "price_date": price_date,
                "price_log_runs": int(history["Run_Date"].nunique()),
                "schedule_months": args.schedule_months,
                "next_run_date": (
                    pd.Timestamp(run_date) + pd.DateOffset(months=args.schedule_months)
                ).strftime("%Y-%m-%d"),
                "run_history": build_run_history(history, args.log_dir),
                "scheduler_events": read_scheduler_events(args.log_dir),
                "log_dir": args.log_dir,
                "generated_at": run_ts.strftime("%Y-%m-%d %H:%M"),
                "listing_source": getattr(loader, "listing_source", ""),
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
            args.html_output,
            monitoring
        )
    elif args.allow_sim_fallback:
        print("\n[Fallback Mode] --allow-sim-fallback 옵션에 의해 시뮬레이션 유니버스로 예상 수익률을 산출합니다.")
        sim_data = loader.generate_calibrated_fundamental_dgp(
            M=60, D=16, r=6, N=36, rho_budget_constraint=-0.82, sigma_accounting_noise=0.12, seed=42
        )
        sim_data.data_source = "SIMULATION_FALLBACK"
        full_engine = FundamentalTensorInverseEngine(fixed_r=6)
        final_state = full_engine.fit_transform(sim_data)
        df_all = print_top_long_short_picks(
            sim_data,
            final_state,
            top_n=args.top_n,
            save_csv_path=args.output
        )
        write_krx_html_dashboard(
            df_all,
            None,
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "listing_source": "SIMULATION",
                "data_source": sim_data.data_source,
                "market": args.market,
                "min_marcap": args.min_marcap,
                "n_requested": len(sim_data.tickers),
                "n_analyzed": len(sim_data.tickers),
                "n_features": int(sim_data.X_obs.shape[1]),
                "n_periods": int(sim_data.X_obs.shape[2]),
                "time_start": sim_data.time_index[0] if sim_data.time_index else "",
                "time_end": sim_data.time_index[-1] if sim_data.time_index else "",
            },
            args.html_output
        )
    else:
        raise RuntimeError(
            "[실행 중단] 실제 KRX 시세 데이터를 다운로드하지 못해 종료합니다. "
            "(가짜/시뮬레이션 결과 방지를 위해 자동 폴백이 비활성화되었습니다.)"
        )


if __name__ == "__main__":
    main()