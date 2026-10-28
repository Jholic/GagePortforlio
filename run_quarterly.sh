#!/bin/bash
# launchd가 매일 18:00과 로그인 시 호출합니다.
# main.py(v1) → main_v2.py(v2) → main_v3.py(v3 장기 시계열 DB 증분 업데이트) → build_v3_dashboard.py 순서로 실행합니다.
# 버전마다 마지막 성공일을 따로 기록하고, INTERVAL_MONTHS개월이 지난 버전만 실행합니다.
# 실패한 버전은 성공일이 갱신되지 않으므로 다음 트리거에서 그 버전만 재시도됩니다.
# 강제 실행: ./run_quarterly.sh --force

set -u
INTERVAL_MONTHS=3
PROJECT_DIR="/Users/jabiseu/project/PROJECT/AUTO_PORTFOLIO"
PYTHON="$PROJECT_DIR/.venv/bin/python"
SCHED_DIR="$PROJECT_DIR/logs/scheduler"
V1_SUCCESS_FILE="$SCHED_DIR/last_success"
V2_SUCCESS_FILE="$SCHED_DIR/last_success_v2"
V3_SUCCESS_FILE="$SCHED_DIR/last_success_v3"
LOCK_DIR="$SCHED_DIR/.lock"
EVENT_LOG="$SCHED_DIR/scheduler.log"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONUNBUFFERED=1
mkdir -p "$SCHED_DIR" "$PROJECT_DIR/logs/dashboards" "$PROJECT_DIR/data"
cd "$PROJECT_DIR" || exit 1

log_event() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$EVENT_LOG"; }
notify() { osascript -e "display notification \"$1\" with title \"AUTO_PORTFOLIO\"" >/dev/null 2>&1 || true; }

TODAY="$(date +%Y-%m-%d)"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

is_due() {
    [ "$FORCE" -eq 1 ] && return 0
    [ -f "$1" ] || return 0
    local last due
    last="$(tr -d '[:space:]' < "$1")"
    due="$(date -j -v+${INTERVAL_MONTHS}m -f %Y-%m-%d "$last" +%Y-%m-%d 2>/dev/null || echo "$TODAY")"
    [[ ! "$TODAY" < "$due" ]]
}

RUN_V1=0; RUN_V2=0; RUN_V3=0
is_due "$V1_SUCCESS_FILE" && RUN_V1=1
is_due "$V2_SUCCESS_FILE" && RUN_V2=1
is_due "$V3_SUCCESS_FILE" && RUN_V3=1
[ $RUN_V1 -eq 0 ] && [ $RUN_V2 -eq 0 ] && [ $RUN_V3 -eq 0 ] && exit 0

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log_event "SKIP 이미 실행 중"
    exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

STAMP="$(date +%Y-%m-%d_%H%M%S)"
OVERALL=0
DONE=()
FAILED=()

run_job() {
    local name="$1" success_file="$2"; shift 2
    local run_log="$SCHED_DIR/run_${STAMP}_${name}.log"
    log_event "START $name $run_log"
    "$PYTHON" "$@" >> "$run_log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "$TODAY" > "$success_file"
        log_event "OK $name status=0"
        DONE+=("$name")
    else
        log_event "FAIL $name status=$status 다음 트리거에서 재시도 (log: $run_log)"
        FAILED+=("$name")
        OVERALL=$status
    fi
}

[ $RUN_V1 -eq 1 ] && run_job v1 "$V1_SUCCESS_FILE" main.py --top-n 20 --schedule-months "$INTERVAL_MONTHS"
[ $RUN_V2 -eq 1 ] && run_job v2 "$V2_SUCCESS_FILE" main_v2.py --top-n 20
if [ $RUN_V3 -eq 1 ]; then
    run_job v3 "$V3_SUCCESS_FILE" main_v3.py --sync-live
    "$PYTHON" build_v3_dashboard.py >> "$SCHED_DIR/run_${STAMP}_v3.log" 2>&1 || true
fi

if [ ${#DONE[@]} -gt 0 ]; then
    cp krx_dashboard.html "logs/dashboards/krx_dashboard_$TODAY.html"
    log_event "DASHBOARD logs/dashboards/krx_dashboard_$TODAY.html (갱신: ${DONE[*]})"
fi

if [ ${#FAILED[@]} -eq 0 ]; then
    notify "$TODAY 정기 분석 완료 (${DONE[*]}). index.html / krx_dashboard.html 갱신"
else
    notify "정기 분석 일부 실패 (${FAILED[*]}). 다음 트리거에서 재시도"
fi
exit $OVERALL
