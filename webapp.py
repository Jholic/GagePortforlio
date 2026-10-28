#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTO_PORTFOLIO v3.0 로컬 웹 애플리케이션 서버 (`webapp.py`)
- 실행: ./.venv/bin/python webapp.py --port 8765
- 기능:
  1) `http://127.0.0.1:8765` 에서 인터랙티브 대시보드(`index.html`) 서빙
  2) `/api/stock?ticker=005930.KS` : SQLite DB(`data/krx_quant_store.db`)에서 개별 종목의
     전체 K-IFRS 재무제표, 실제 주가 vs 진성 적정주가, 표면 vs 진짜 PER/PBR 시계열 즉시 조회
  3) `/api/search?q=삼성` : 종목명/코드 실시간 검색
  4) `/api/sync` (POST) : 새로운 분기/연간 재무제표와 최신 10년 월봉 시세를 네이버 금융에서
     증분 수집(Upsert)한 뒤, 텐서 역변환 예측 및 대시보드를 자동 갱신
"""

import os
import json
import sqlite3
import argparse
import webbrowser
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler

from main_v3 import LocalQuantStore, run_v3_pipeline
from build_v3_dashboard import generate_v3_dashboards


class QuantWebAppHandler(SimpleHTTPRequestHandler):
    db_path = "data/krx_quant_store.db"

    def _send_json(self, data: object, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json({"status": "ok", "db_path": self.db_path})
            return

        if path == "/api/search":
            q = (qs.get("q", [""])[0] or "").strip()
            store = LocalQuantStore(self.db_path)
            with store._connect() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                if q:
                    like_q = f"%{q}%"
                    cur.execute(
                        """
                        SELECT * FROM stock_forecasts
                        WHERE name LIKE ? OR ticker LIKE ?
                        ORDER BY rank ASC LIMIT 30
                        """,
                        (like_q, like_q),
                    )
                else:
                    cur.execute("SELECT * FROM stock_forecasts ORDER BY rank ASC LIMIT 30")
                rows = [dict(r) for r in cur.fetchall()]
            self._send_json({"results": rows})
            return

        if path == "/api/stock":
            tk = (qs.get("ticker", ["005930.KS"])[0] or "005930.KS").strip()
            store = LocalQuantStore(self.db_path)
            with store._connect() as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM stock_forecasts WHERE ticker = ?", (tk,))
                summary = cur.fetchone()
                cur.execute(
                    "SELECT * FROM valuation_timeseries WHERE ticker = ? ORDER BY ym ASC",
                    (tk,),
                )
                ts_rows = [dict(r) for r in cur.fetchall()]
                cur.execute(
                    "SELECT * FROM financial_statements WHERE ticker = ? ORDER BY ym ASC",
                    (tk,),
                )
                fin_rows = [dict(r) for r in cur.fetchall()]
            self._send_json({
                "ticker": tk,
                "summary": dict(summary) if summary else None,
                "valuation_timeseries": ts_rows,
                "financial_statements": fin_rows,
            })
            return

        if path in ("/", "/index.html", "/krx_dashboard.html"):
            self.path = "/index.html" if os.path.exists("index.html") else "/krx_dashboard.html"

        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/sync":
            try:
                run_v3_pipeline(sync_live=True, db_path=self.db_path)
                generate_v3_dashboards()
                self._send_json({"status": "ok", "message": "Incremental sync & prediction completed."})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return
        self._send_json({"status": "not_found"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="AUTO_PORTFOLIO v3.0 Local Web Application Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db-path", type=str, default="data/krx_quant_store.db")
    parser.add_argument("--open-browser", action="store_true", help="서버 시작 시 브라우저 자동 실행")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.db_path) or not os.path.exists("index.html"):
        run_v3_pipeline(sync_live=False, db_path=args.db_path)
        generate_v3_dashboards()

    QuantWebAppHandler.db_path = args.db_path
    url = f"http://{args.host}:{args.port}"
    print("=" * 90)
    print(f" [AUTO_PORTFOLIO v3.0 WebApp] 로컬 서버 구동 중: {url}")
    print(f" - 로컬 SQLite DB : {args.db_path}")
    print(f" - 종목 상세 API   : {url}/api/stock?ticker=005930.KS")
    print("=" * 90)

    if args.open_browser:
        webbrowser.open(url)

    server = HTTPServer((args.host, args.port), QuantWebAppHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[WebApp] 서버를 종료합니다.")
        server.server_close()


if __name__ == "__main__":
    main()
