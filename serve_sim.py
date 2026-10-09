"""Serve the connectome-informed virtual fly experiment in Google Chrome."""

import csv
import json
import os
import shutil
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8080
PAGE = "point_simulation.html"
CONNECTIONS_PATH = ROOT / "file_csv" / "connections_princeton.csv"
NEURONS_PATH = ROOT / "file_csv" / "neurons.csv"
ROLE_ASSIGNMENTS_PATH = ROOT / "file_csv" / "neuron_role_assignments.csv"


def _require_columns(
    reader: csv.DictReader[str],
    required: set[str],
    file_path: Path,
) -> None:
    columns = set(reader.fieldnames or ())
    missing = required - columns
    if missing:
        raise ValueError(
            f"{file_path.name}에 필요한 열이 없습니다: {', '.join(sorted(missing))}"
        )


def load_circuit_data() -> dict[str, object]:
    roles: list[dict[str, str]] = []
    with ROLE_ASSIGNMENTS_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_columns(
            reader,
            {"root_id", "cell_type", "soma_side", "model_role"},
            ROLE_ASSIGNMENTS_PATH,
        )
        seen_roles: set[str] = set()
        for row in reader:
            root_id = (row["root_id"] or "").strip()
            model_role = (row["model_role"] or "").strip()
            side = (row["soma_side"] or "").strip().lower()
            if not root_id or root_id in seen_roles:
                raise ValueError(
                    f"{ROLE_ASSIGNMENTS_PATH.name}에 비어 있거나 중복된 root_id가 있습니다."
                )
            if model_role not in {"steering", "forward"}:
                raise ValueError(
                    f"지원하지 않는 model_role입니다: {model_role or '(empty)'}"
                )
            if side not in {"left", "right"}:
                raise ValueError(f"뉴런 {root_id}의 soma_side가 올바르지 않습니다.")
            seen_roles.add(root_id)
            roles.append(
                {
                    "root_id": root_id,
                    "cell_type": (row["cell_type"] or "").strip(),
                    "side": side,
                    "model_role": model_role,
                }
            )

    if not roles:
        raise ValueError(f"{ROLE_ASSIGNMENTS_PATH.name}에 역할 뉴런이 없습니다.")

    role_ids = {role["root_id"] for role in roles}
    edge_counts: dict[tuple[str, str], int] = {}
    connection_rows = 0
    with CONNECTIONS_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_columns(
            reader,
            {"pre_root_id", "post_root_id", "syn_count"},
            CONNECTIONS_PATH,
        )
        for line_number, row in enumerate(reader, start=2):
            post_id = (row["post_root_id"] or "").strip()
            if post_id not in role_ids:
                continue
            pre_id = (row["pre_root_id"] or "").strip()
            if not pre_id:
                raise ValueError(
                    f"{CONNECTIONS_PATH.name}:{line_number}에 pre_root_id가 없습니다."
                )
            try:
                synapse_count = int(row["syn_count"] or "")
            except ValueError as error:
                raise ValueError(
                    f"{CONNECTIONS_PATH.name}:{line_number}의 syn_count가 정수가 아닙니다."
                ) from error
            if synapse_count <= 0:
                raise ValueError(
                    f"{CONNECTIONS_PATH.name}:{line_number}의 syn_count는 양수여야 합니다."
                )
            edge_key = (pre_id, post_id)
            edge_counts[edge_key] = edge_counts.get(edge_key, 0) + synapse_count
            connection_rows += 1

    if not edge_counts:
        raise ValueError(
            "연결 파일에서 역할 뉴런으로 들어오는 시냅스 연결을 찾지 못했습니다."
        )

    presynaptic_ids = {pre_id for pre_id, _ in edge_counts}
    presynaptic_sides: dict[str, str] = {}
    with NEURONS_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        _require_columns(reader, {"Root ID", "Soma side"}, NEURONS_PATH)
        for row in reader:
            root_id = (row["Root ID"] or "").strip()
            if root_id not in presynaptic_ids:
                continue
            side = (row["Soma side"] or "").strip().lower()
            if side in {"left", "right"}:
                presynaptic_sides[root_id] = side

    edges = [
        {
            "pre_id": pre_id,
            "pre_side": presynaptic_sides.get(pre_id, "unknown"),
            "post_id": post_id,
            "synapse_count": synapse_count,
        }
        for (pre_id, post_id), synapse_count in sorted(edge_counts.items())
    ]
    metadata_coverage = len(presynaptic_sides)
    return {
        "roles": roles,
        "visual_edges": edges,
        "stats": {
            "connection_rows": connection_rows,
            "aggregated_edges": len(edges),
            "total_synapses": sum(edge_counts.values()),
            "presynaptic_neurons": len(presynaptic_ids),
            "metadata_coverage": metadata_coverage,
            "source": CONNECTIONS_PATH.name,
        },
    }


def make_request_handler(circuit_data: dict[str, object]) -> type[SimpleHTTPRequestHandler]:
    class CircuitRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/api/circuit":
                super().do_GET()
                return

            payload = json.dumps(circuit_data, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return CircuitRequestHandler


def find_chrome() -> str | None:
    candidates = [
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
    ]
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(
                str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
            )
    return next((path for path in candidates if path and Path(path).is_file()), None)


def main() -> int:
    chrome = find_chrome()
    if chrome is None:
        print(
            "Google Chrome을 찾지 못했습니다. Chrome을 설치했는지 확인하세요.",
            file=sys.stderr,
        )
        return 1

    try:
        circuit_data = load_circuit_data()
    except (OSError, csv.Error, ValueError) as error:
        print(f"연결 데이터를 준비하지 못했습니다: {error}", file=sys.stderr)
        return 1

    url = f"http://{HOST}:{PORT}/{quote(PAGE)}"
    try:
        server = ThreadingHTTPServer(
            (HOST, PORT), make_request_handler(circuit_data)
        )
    except OSError as error:
        print(
            f"{HOST}:{PORT} 포트를 열 수 없습니다: {error}\n"
            "8080 포트를 사용 중인 프로그램을 닫고 다시 실행하세요.",
            file=sys.stderr,
        )
        return 1

    stats = circuit_data["stats"]
    print(
        f"회로 연결 로드 완료: {stats['aggregated_edges']}개 연결, "
        f"시냅스 {stats['total_synapses']:,}개"
    )
    print(f"Chrome에서 가상 초파리 실험을 엽니다: {url}")
    print("서버를 종료하려면 이 터미널에서 Ctrl+C를 누르세요.")
    try:
        subprocess.Popen([chrome, url])
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n시뮬레이션 서버를 종료합니다.")
    except OSError as error:
        print(f"Chrome을 실행하지 못했습니다: {error}", file=sys.stderr)
        return 1
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
