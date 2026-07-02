import base64
import json
import os
import re
import shlex
import subprocess
from systems.base import BaseSystemAdapter


class X265DB(BaseSystemAdapter):
    def __init__(self, args):
        self.args = args
        self.sys_name = "x265"
        self.host = args.get("host", "localhost")
        self.port = args.get("port", "")
        self.user = args.get("user", "")
        self.password = args.get("password", "")
        self.sudopassword = args.get("sudopassword", "")
        self.container_name = args.get("container_name", args.get("dockername", "x265-worker"))
        self.docker_image = args.get("docker_image", args.get("dockerimage", "x265_sampler_image"))
        self.docker_memory = args.get("docker_memory", "")
        self.docker_cpus = args.get("docker_cpus", "")
        self.docker_cpuset = args.get("docker_cpuset", "")
        self.docker_log_max_size = args.get("docker_log_max_size", "100m")
        self.docker_log_max_file = args.get("docker_log_max_file", "3")
        self.ready_timeout = int(args.get("ready_timeout", 60))
        self.current_config_string = ""
        self.requested_config = {}
        self.applied_config = {}
        self.normalization_notes = []

        self.knobs_info = self.load_knobs(
            args["knob_config_file"],
            int(args["knob_num"]),
            normalize=self._normalize_knob_schema,
        )
        self.default_knobs = self.build_default_knobs(self.knobs_info)

    @staticmethod
    def _normalize_knob_schema(raw):
        normalized = {}
        for name, info in raw.items():
            item = dict(info)
            if item.get("type") == "enum" and "enum_values" not in item and "options" in item:
                item["enum_values"] = item["options"]
            normalized[name] = item
        return normalized

    def _sudo_prefix(self):
        if self.sudopassword:
            return f"echo {shlex.quote(str(self.sudopassword))} | sudo -S "
        return ""

    def _run(self, cmd, check=True):
        return subprocess.run(
            self._sudo_prefix() + cmd,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _docker_inspect_running(self):
        res = self._run(
            f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(self.container_name)}",
            check=False,
        )
        return res.returncode == 0 and res.stdout.strip().lower() == "true"

    def _docker_image_exists(self):
        res = self._run(f"docker image inspect {shlex.quote(self.docker_image)}", check=False)
        return res.returncode == 0

    def check_connection_alive(self):
        return self._docker_image_exists()

    def start_container(self):
        if self._docker_inspect_running():
            return True

        self.stop_container()
        parts = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--pull=never",
        ]
        if self.docker_memory:
            parts.extend(["--memory", str(self.docker_memory)])
        if self.docker_cpus:
            parts.extend(["--cpus", str(self.docker_cpus)])
        if self.docker_cpuset:
            parts.extend(["--cpuset-cpus", str(self.docker_cpuset)])
        if self.docker_log_max_size:
            parts.extend(["--log-driver", "json-file", "--log-opt", f"max-size={self.docker_log_max_size}"])
        if self.docker_log_max_file:
            parts.extend(["--log-opt", f"max-file={self.docker_log_max_file}"])
        parts.extend([self.docker_image, "bash", "-c", "while true; do sleep 1000; done"])

        cmd = " ".join(shlex.quote(str(part)) for part in parts)
        res = self._run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"启动 x265 容器失败: {res.stderr.strip()}")
        return True

    def stop_container(self):
        self._run(f"docker stop {shlex.quote(self.container_name)}", check=False)
        self._run(f"docker rm -v {shlex.quote(self.container_name)}", check=False)

    def manage_database(self, action, dbname=None):
        if action == "start":
            self.start_container()
        elif action == "stop":
            self.stop_container()
        elif action == "restart":
            self.stop_container()
            self.start_container()

    def modify_config(self, config):
        self.requested_config = dict(config)
        self.normalization_notes = []
        self._normalize_dependent_knobs(config, self.normalization_notes)
        self.applied_config = dict(config)
        params = []
        for key, value in config.items():
            if key in ("id", "frame-threads"):
                continue

            if key in {"wpp", "open-gop", "deblock", "rskip"}:
                params.append(f"--{key}" if int(value) == 1 else f"--no-{key}")
                continue

            if key in {"amp", "rect", "sao", "weightb", "weightp"}:
                if int(value) == 1:
                    params.append(f"--{key}")
                continue

            params.append(f"--{key} {shlex.quote(str(value))}")

        self.current_config_string = " ".join(params).strip()

    @staticmethod
    def _normalize_dependent_knobs(config, notes=None):
        notes = notes if notes is not None else []
        try:
            ctu = int(config.get("ctu", 64))
        except (TypeError, ValueError):
            ctu = 64

        # x265 rejects TU depth 4 with ctu=16:
        # QuadtreeTUMaxDepth* must fit within the current max CU size.
        if ctu <= 16:
            for key in ("tu-inter-depth", "tu-intra-depth"):
                if key not in config:
                    continue
                try:
                    if int(config[key]) > 3:
                        notes.append(
                            f"{key} requested={config[key]} normalized=3 because ctu=16 cannot use TU depth >3"
                        )
                        config[key] = 3
                except (TypeError, ValueError):
                    notes.append(f"{key} requested={config.get(key)} normalized=3 because ctu=16")
                    config[key] = 3

    def set_db_knob(self, config):
        try:
            self.modify_config(config)
            self.start_container()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to set x265 knobs: {e}")
            return False

    def run_benchmark(self, factors):
        video_name = self._construct_video_filename(factors)
        video_basename = os.path.splitext(video_name)[0]
        script = self._build_benchmark_script(
            input_video_path=f"/app/inputs/{video_name}",
            output_path_raw=f"/tmp/{video_basename}.hevc",
            output_path_muxed=f"/tmp/{video_basename}.mp4",
        )
        encoded_script = base64.b64encode(script.encode("utf-8")).decode("utf-8")
        cmd = (
            f"docker exec {shlex.quote(self.container_name)} bash -c "
            f"{shlex.quote(f'echo {encoded_script} | base64 --decode | bash')}"
        )
        return self._run(cmd, check=True)

    def collect_runtime_evidence(self, since=None, tail=200):
        text = self._format_dependency_evidence()
        return [{"source": "config_state", "text": text}] if text else []

    def _format_dependency_evidence(self):
        cfg = self.applied_config or {}
        requested = self.requested_config or {}

        def num(name, default=0):
            try:
                return float(cfg.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        def text(name, default=""):
            return str(cfg.get(name, default)).lower()

        lines = [
            "X265_CONFIG_SNAPSHOT "
            f"requested={json.dumps(requested, sort_keys=True)} "
            f"applied={json.dumps(cfg, sort_keys=True)} "
            f"cli={self.current_config_string}"
        ]
        for note in self.normalization_notes:
            lines.append(
                "X265_NORMALIZATION_EVIDENCE x265_ctu_16_should_limit_tu_depth active: " + note
            )

        ctu = int(num("ctu", 64))
        rskip = int(num("rskip", 1))
        rdoq_level = int(num("rdoq-level", 1))
        rd = int(num("rd", 3))
        b_adapt = int(num("b-adapt", 2))
        bframes = int(num("bframes", 4))
        amp = int(num("amp", 1))
        ref = int(num("ref", 3))
        limit_refs = int(num("limit-refs", 3))
        aq_mode = int(num("aq-mode", 2))
        me = text("me", "hex")
        subme = int(num("subme", 2))
        tu_inter_depth = int(num("tu-inter-depth", 1))
        tu_intra_depth = int(num("tu-intra-depth", 1))
        limit_tu = int(num("limit-tu", 0))

        if ctu == 16:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_ctu_16_should_keep_rskip active: "
                f"ctu=16 increases CU recursion; rskip={rskip} controls early exit from recursion."
            )
            lines.append(
                "X265_CONFIG_EVIDENCE x265_ctu_16_should_limit_tu_depth active: "
                f"ctu=16 constrains transform-unit depth; tu-inter-depth={tu_inter_depth}, tu-intra-depth={tu_intra_depth}."
            )
        else:
            lines.append("X265_CONFIG_EVIDENCE x265_ctu_16_should_keep_rskip inactive: ctu is not 16.")
            lines.append("X265_CONFIG_EVIDENCE x265_ctu_16_should_limit_tu_depth inactive: ctu is not 16.")

        if rdoq_level == 0:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_rdoq_disabled_should_disable_psy_rdoq active: "
                "rdoq-level=0 disables RDOQ; psy-rdoq has no RDOQ path."
            )
        else:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_rdoq_disabled_should_disable_psy_rdoq inactive: rdoq-level is enabled."
            )

        if rd < 3:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_low_rd_should_disable_psy_rd active: "
                "rd<3 does not use RDO-based mode decisions; psy-rd has no RDO mode-decision path."
            )
        else:
            lines.append("X265_CONFIG_EVIDENCE x265_low_rd_should_disable_psy_rd inactive: rd is at least 3.")

        if b_adapt == 2:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_trellis_b_adapt_should_limit_bframes active: "
                f"b-adapt=2 enables trellis lookahead; bframes={bframes} controls B-frame lookahead work."
            )
        else:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_trellis_b_adapt_should_limit_bframes inactive: b-adapt is not 2."
            )

        if rd >= 5:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_high_rd_should_avoid_amp active: "
                f"rd>=5 uses expensive analysis; amp={amp} expands inter partition analysis."
            )
        else:
            lines.append("X265_CONFIG_EVIDENCE x265_high_rd_should_avoid_amp inactive: rd is below 5.")

        if ref >= 6:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_high_ref_should_limit_refs active: "
                f"ref>=6 expands reference search; limit-refs={limit_refs} controls reference analysis limiting."
            )
        else:
            lines.append("X265_CONFIG_EVIDENCE x265_high_ref_should_limit_refs inactive: ref is below 6.")

        if aq_mode == 0:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_aq_disabled_should_disable_aq_strength active: "
                "aq-mode=0 disables adaptive quantization; aq-strength has no AQ path."
            )
        else:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_aq_disabled_should_disable_aq_strength inactive: aq-mode is enabled."
            )

        if me == "full":
            lines.append(
                "X265_CONFIG_EVIDENCE x265_full_me_should_limit_subme active: "
                f"me=full uses exhaustive motion search; subme={subme} adds subpel refinement work."
            )
        else:
            lines.append("X265_CONFIG_EVIDENCE x265_full_me_should_limit_subme inactive: me is not full.")

        if tu_inter_depth >= 4:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_deep_tu_should_enable_limit_tu active: "
                f"tu-inter-depth>=4 enables deeper TU recursion; limit-tu={limit_tu} controls TU recursion early exit."
            )
        else:
            lines.append(
                "X265_CONFIG_EVIDENCE x265_deep_tu_should_enable_limit_tu inactive: tu-inter-depth is below 4."
            )
        return "\n".join(lines)

    @staticmethod
    def extract_data_from_output(output):
        fps_match = re.search(r"FPS_RESULT:([0-9]+\.?[0-9]*)", output)
        time_match = re.search(r"real\s*(\S+)\s*user\s*(\S+)\s*sys\s*(\S+)", output, re.DOTALL)
        size_match = re.search(r"SIZE_BYTES_RESULT:(\d+)", output)
        duration_match = re.search(r"DURATION_RESULT:([0-9]+\.?[0-9]*)", output)
        if not all([fps_match, time_match, size_match, duration_match]):
            raise ValueError(f"无法解析 x265 输出指标:\n{output}")

        encode_time = float(time_match.group(1))
        user_cpu = float(time_match.group(2))
        sys_cpu = float(time_match.group(3))
        avg_cpu_util = ((user_cpu + sys_cpu) / encode_time) * 100 if encode_time > 0 else 0
        file_size_bytes = int(size_match.group(1))
        output_file_mb = file_size_bytes / (1024 * 1024)
        duration = float(duration_match.group(1))
        bitrate_kbps = (file_size_bytes * 8) / duration / 1000 if duration > 0 else 0
        encode_fps = float(fps_match.group(1))

        return {
            "EncodeTime": round(encode_time, 2),
            "AvgCPUUtil": round(avg_cpu_util, 2),
            "OutputFileMB": round(output_file_mb, 3),
            "BitrateKbps": round(bitrate_kbps, 2),
            "EncodeFPS": round(encode_fps, 2),
        }

    @staticmethod
    def _construct_video_filename(factors):
        video_type = str(factors["Video Type"]).replace(" ", "_")
        resolution = factors["Resolution"]
        frame_rate = factors["Frame Rate"]
        duration = factors["Time"]
        return f"{video_type}_{resolution}_{frame_rate}fps_{duration}.mp4"

    def _build_benchmark_script(self, input_video_path, output_path_raw, output_path_muxed):
        return f"""
set -euo pipefail

ENCODED_OUTPUT_PATH_RAW={shlex.quote(output_path_raw)}
ENCODED_OUTPUT_PATH_MUXED={shlex.quote(output_path_muxed)}
INPUT_VIDEO_PATH={shlex.quote(input_video_path)}
X265_CONFIG_STRING={shlex.quote(self.current_config_string)}

declare -a X265_CMD_ARGS
X265_CMD_ARGS=(--y4m -o "$ENCODED_OUTPUT_PATH_RAW")
read -ra CONFIG_ARGS_ARRAY <<< "$X265_CONFIG_STRING"
if [ ${{#CONFIG_ARGS_ARRAY[@]}} -gt 0 ]; then
    X265_CMD_ARGS+=("${{CONFIG_ARGS_ARRAY[@]}}")
fi
X265_CMD_ARGS+=("-")

ENCODER_OUTPUT_FILE="/tmp/encoder_output.log"
TIME_METRICS_FILE="/tmp/time_metrics.log"

ffmpeg -i "$INPUT_VIDEO_PATH" -f yuv4mpegpipe -strict -1 - < /dev/null \\
    | /usr/bin/time -p -o "$TIME_METRICS_FILE" x265 "${{X265_CMD_ARGS[@]}}" > "$ENCODER_OUTPUT_FILE" 2>&1

if ! grep -q 'encoded' "$ENCODER_OUTPUT_FILE"; then
    echo "ERROR: Encoding failed. Raw log:" >&2
    cat "$ENCODER_OUTPUT_FILE" >&2
    exit 1
fi

ffmpeg -i "$ENCODED_OUTPUT_PATH_RAW" -c:v copy -y "$ENCODED_OUTPUT_PATH_MUXED" > /dev/null 2>&1

FPS_RAW=$(grep -oE 'encoded [0-9]+ frames in [0-9]+\\.[0-9]+s \\([0-9]+\\.[0-9]+ fps\\)' "$ENCODER_OUTPUT_FILE" | grep -oE '[0-9]+\\.[0-9]+ fps' | awk '{{print $1}}' | tail -n 1)
FPS=${{FPS_RAW:-0}}
TIME_STATS=$(cat "$TIME_METRICS_FILE")
FILE_SIZE_BYTES=$(stat -c%s "$ENCODED_OUTPUT_PATH_MUXED")
DURATION_RAW=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$ENCODED_OUTPUT_PATH_MUXED")
DURATION=${{DURATION_RAW:-0}}

echo "FPS_RESULT:$FPS"
echo "TIME_RESULT:$TIME_STATS"
echo "SIZE_BYTES_RESULT:$FILE_SIZE_BYTES"
echo "DURATION_RESULT:$DURATION"

rm -f "$ENCODED_OUTPUT_PATH_RAW" "$ENCODED_OUTPUT_PATH_MUXED" "$ENCODER_OUTPUT_FILE" "$TIME_METRICS_FILE"
"""
