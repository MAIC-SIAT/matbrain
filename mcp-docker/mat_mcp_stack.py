#!/usr/bin/env python3
"""One-entry Mat-MCP Docker stack orchestrator and validator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user-facing environment check
    raise SystemExit(f"Missing PyYAML: {exc}") from exc


ROOT = Path(__file__).resolve().parents[1]
MCP_ROOT = Path(__file__).resolve().parent
DEFAULT_STACK = MCP_ROOT / "mat_mcp_stack.yaml"
CONFIG_ROOT = MCP_ROOT / "configs"
SCRIPTS_ROOT = MCP_ROOT / "scripts"
MAIN_CONFIG = CONFIG_ROOT / "structure_property_tools.yaml"
EVIDENCE_CONFIG = CONFIG_ROOT / "evidence_tools.yaml"
LEGACY_MAIN_CONFIG = ROOT / "train/rl/tool_configs/structure_property_tools.yaml"
LEGACY_EVIDENCE_CONFIG = ROOT / "train/rl/tool_configs/evidence_tools.yaml"
GENERATED_STACK_DIR = MCP_ROOT / ".generated_stack"
SHARED_NETWORK_NAME = "matgl-mcp-network"
SHARED_NETWORK_SUBNET = "172.30.0.0/16"
SHARED_NETWORK_GATEWAY = "172.30.0.1"


@dataclass(frozen=True)
class Service:
    name: str
    directory: Path
    port: int
    sse_path: str
    health_path: str
    enabled: bool
    gpu: bool = False
    depends_on: tuple[str, ...] = ()
    note: str = ""
    routerized: bool = True
    replicas_smoke: int = 4
    replicas_production32: int = 8
    replicas_production128: int = 16
    concurrency_cap_single: int = 1
    concurrency_cap_smoke: int = 4
    concurrency_cap_production32: int = 8
    concurrency_cap_production128: int = 16
    gpu_devices: tuple[str, ...] = ()

    @property
    def sse_url(self) -> str:
        return f"http://localhost:{self.port}{self.sse_path}"

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}{self.health_path}"

    def replicas_for_profile(self, profile: str) -> int:
        if profile == "single":
            return 1
        if profile == "smoke":
            return max(1, self.replicas_smoke)
        if profile == "production32":
            return max(1, self.replicas_production32)
        if profile == "production128":
            return max(1, self.replicas_production128)
        raise ValueError(f"Unknown profile: {profile}")

    def concurrency_cap_for_profile(self, profile: str) -> int:
        if profile == "single":
            return max(1, self.concurrency_cap_single)
        if profile == "smoke":
            return max(1, self.concurrency_cap_smoke)
        if profile == "production32":
            return max(1, self.concurrency_cap_production32)
        if profile == "production128":
            return max(1, self.concurrency_cap_production128)
        raise ValueError(f"Unknown profile: {profile}")

    def service_names(self, profile: str, backend_count: int | None = None) -> list[str]:
        if not self.routerized:
            return [self.name]
        desired = backend_count if backend_count is not None else self.replicas_for_profile(profile)
        if profile == "single":
            return [self.name, f"{self.name}-backend-1"]
        if profile in {"smoke", "production32", "production128"}:
            replica_count = max(1, desired)
            names = [self.name]
            names.extend(f"{self.name}-backend-{idx}" for idx in range(1, replica_count + 1))
            return names
        raise ValueError(f"Unknown profile: {profile}")

    def extra_backend_services(self, profile: str, backend_count: int | None = None) -> list[str]:
        if not self.routerized:
            return []
        desired = max(1, backend_count if backend_count is not None else self.replicas_for_profile(profile))
        max_replicas = max(8, self.replicas_smoke, self.replicas_production32, self.replicas_production128, desired)
        if desired >= max_replicas:
            return []
        return [f"{self.name}-backend-{idx}" for idx in range(desired + 1, max_replicas + 1)]


def load_stack(path: Path) -> list[Service]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    services: list[Service] = []
    for item in data.get("services", []):
        services.append(
            Service(
                name=item["name"],
                directory=MCP_ROOT / item["directory"],
                port=int(item["port"]),
                sse_path=item["sse_path"],
                health_path=item["health_path"],
                enabled=bool(item.get("enabled", True)),
                gpu=bool(item.get("gpu", False)),
                depends_on=tuple(item.get("depends_on", [])),
                note=item.get("note", ""),
                routerized=bool(item.get("routerized", True)),
                replicas_smoke=int(item.get("replicas_smoke", item.get("replicas_high", 4))),
                replicas_production32=int(item.get("replicas_production32", item.get("replicas_high", 8))),
                replicas_production128=int(item.get("replicas_production128", item.get("replicas_production32", item.get("replicas_high", 8)))),
                concurrency_cap_single=int(item.get("concurrency_cap_single", 1)),
                concurrency_cap_smoke=int(item.get("concurrency_cap_smoke", item.get("replicas_smoke", item.get("replicas_high", 4)))),
                concurrency_cap_production32=int(
                    item.get("concurrency_cap_production32", item.get("replicas_production32", item.get("replicas_high", 8)))
                ),
                concurrency_cap_production128=int(
                    item.get("concurrency_cap_production128", item.get("replicas_production128", item.get("replicas_production32", item.get("replicas_high", 8))))
                ),
                gpu_devices=tuple(str(device) for device in item.get("gpu_devices", [])),
            )
        )
    return services


def read_env(path: Path) -> dict[str, str]:
    env_file = path / ".env"
    values: dict[str, str] = {}
    if not env_file.exists():
        return values
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config_urls(paths: list[Path]) -> set[str]:
    urls: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in data.get("tools", []):
            urls.add(str(item["server_url"]))
    return urls


def url_to_port_path(url: str) -> tuple[int, str]:
    parsed = urlparse(url)
    return int(parsed.port or 80), parsed.path


def selected_services(services: list[Service], include_extra: bool, only_gpu: bool = False) -> list[Service]:
    result = [service for service in services if service.enabled or include_extra]
    if only_gpu:
        result = [service for service in result if service.gpu]
    return result


def selected_profile(args: argparse.Namespace) -> str:
    if getattr(args, "profile", None):
        return args.profile
    legacy_mode = getattr(args, "mode", None)
    if legacy_mode == "single":
        return "single"
    if legacy_mode == "high-concurrency":
        return "production32"
    return "single"


def desired_backend_count(service: Service, args: argparse.Namespace) -> int:
    if getattr(args, "replicas_override", None) is not None:
        return max(1, int(args.replicas_override))
    return service.replicas_for_profile(selected_profile(args))


def desired_concurrency_cap(service: Service, args: argparse.Namespace) -> int:
    if getattr(args, "concurrency_cap_override", None) is not None:
        return max(1, int(args.concurrency_cap_override))
    return service.concurrency_cap_for_profile(selected_profile(args))


def compose_files_for_service(service: Service, profile: str, backend_count: int) -> list[Path]:
    base = service.directory / "docker-compose.yml"
    if not service.routerized:
        return [base]
    override = write_mode_override(service, profile, backend_count)
    return [base, override]


def write_mode_override(service: Service, profile: str, backend_count: int) -> Path:
    base = yaml.safe_load((service.directory / "docker-compose.yml").read_text(encoding="utf-8")) or {}
    services = base.get("services", {})
    router_name = service.name
    router_service = services.get(router_name, {})
    template_name = f"{service.name}-backend-2" if f"{service.name}-backend-2" in services else f"{service.name}-backend-1"
    template_service = services.get(template_name)
    if not template_service:
        raise SystemExit(f"{service.name}: could not find backend template service in docker-compose.yml")

    backend_names = [f"{service.name}-backend-{idx}" for idx in range(1, backend_count + 1)]
    backend_urls = ",".join(f"http://{name}:${{SERVER_PORT}}" for name in backend_names)

    def env_to_map(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return copy.deepcopy(value)
        result: dict[str, Any] = {}
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "=" in item:
                    key, val = item.split("=", 1)
                    result[key] = val
        return result

    def deploy_with_device(service_def: dict[str, Any], device: str) -> dict[str, Any]:
        deploy = copy.deepcopy(service_def.get("deploy") or {})
        resources = deploy.setdefault("resources", {})
        reservations = resources.setdefault("reservations", {})
        reservations["devices"] = [
            {
                "driver": "nvidia",
                "device_ids": [device],
                "capabilities": ["gpu"],
            }
        ]
        return deploy

    override_services: dict[str, Any] = {
        router_name: {
            "depends_on": [f"{service.name}-backend-1"],
            "environment": {
                **env_to_map(router_service.get("environment")),
                "BACKEND_URLS": backend_urls,
            },
        }
    }

    def backend_override(base_service: dict[str, Any], idx: int) -> dict[str, Any]:
        override: dict[str, Any] = {}
        if service.gpu and service.gpu_devices:
            device = service.gpu_devices[(idx - 1) % len(service.gpu_devices)]
            override["environment"] = {
                **env_to_map(base_service.get("environment")),
                "NVIDIA_VISIBLE_DEVICES": device,
                # Docker maps the selected physical GPU to cuda:0 inside the
                # container. Keeping the physical id here can hide the device.
                "CUDA_VISIBLE_DEVICES": "0",
            }
            override["deploy"] = deploy_with_device(base_service, device)
        return override

    for idx in range(1, min(backend_count, 2) + 1):
        backend_name = f"{service.name}-backend-{idx}"
        base_backend = services.get(backend_name, template_service)
        override = backend_override(base_backend, idx)
        if override:
            override_services[backend_name] = override
    for idx in range(3, backend_count + 1):
        backend_name = f"{service.name}-backend-{idx}"
        override_services[backend_name] = copy.deepcopy(template_service)
        override = backend_override(template_service, idx)
        if override:
            override_services[backend_name]["environment"] = override["environment"]
            override_services[backend_name]["deploy"] = override["deploy"]

    override = {"services": override_services}
    profile_dir = GENERATED_STACK_DIR / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    override_path = profile_dir / f"{service.name}.override.yml"
    override_path.write_text(yaml.safe_dump(override, sort_keys=False), encoding="utf-8")
    return override_path


def docker_compose_base(sudo: bool) -> list[str]:
    if shutil.which("docker"):
        cmd = ["docker", "compose"]
    elif shutil.which("docker-compose"):
        cmd = ["docker-compose"]
    else:
        raise SystemExit("Neither docker compose nor docker-compose is available on PATH.")
    if sudo:
        return ["sudo", *cmd]
    return cmd


def docker_base(sudo: bool) -> list[str]:
    if not shutil.which("docker"):
        raise SystemExit("docker is not available on PATH.")
    if sudo:
        return ["sudo", "docker"]
    return ["docker"]


def shared_network_name() -> str:
    return os.environ.get("MAT_MCP_SHARED_NETWORK_NAME", SHARED_NETWORK_NAME)


def shared_network_subnet() -> str:
    return os.environ.get("MAT_MCP_SHARED_NETWORK_SUBNET", SHARED_NETWORK_SUBNET)


def shared_network_gateway() -> str:
    return os.environ.get("MAT_MCP_SHARED_NETWORK_GATEWAY", SHARED_NETWORK_GATEWAY)


def inspect_network(sudo: bool, name: str) -> dict[str, Any] | None:
    cmd = [*docker_base(sudo), "network", "inspect", name]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout or "[]")
    if not payload:
        return None
    return payload[0]


def network_subnets(network_info: dict[str, Any]) -> list[str]:
    ipam = network_info.get("IPAM") or {}
    configs = ipam.get("Config") or []
    subnets: list[str] = []
    for item in configs:
        subnet = item.get("Subnet")
        if subnet:
            subnets.append(str(subnet))
    return subnets


def create_shared_network(args: argparse.Namespace) -> None:
    cmd = [
        *docker_base(args.sudo),
        "network",
        "create",
        "--driver",
        "bridge",
        "--subnet",
        shared_network_subnet(),
        "--gateway",
        shared_network_gateway(),
        shared_network_name(),
    ]
    run(cmd, cwd=MCP_ROOT, env=command_env(args), dry_run=args.dry_run)


def ensure_shared_network(args: argparse.Namespace, *, recreate: bool = False) -> None:
    name = shared_network_name()
    desired_subnet = shared_network_subnet()
    info = inspect_network(args.sudo, name)
    if info is None:
        create_shared_network(args)
        return

    subnets = network_subnets(info)
    if desired_subnet in subnets:
        print(f"shared_network_ok name={name} subnet={desired_subnet}")
        return

    attached = info.get("Containers") or {}
    attached_names = sorted(
        str(item.get("Name"))
        for item in attached.values()
        if isinstance(item, dict) and item.get("Name")
    )
    if not recreate:
        raise SystemExit(
            "Shared Docker network exists with an incompatible subnet: "
            f"name={name} actual={subnets or ['<unknown>']} desired={desired_subnet}. "
            "Run `python3 mcp-docker/mat_mcp_stack.py recreate-shared-network` after stopping attached services."
        )
    if attached_names:
        raise SystemExit(
            "Cannot recreate shared Docker network while containers are attached: "
            f"{attached_names}. Run stack down first, then retry."
        )
    run([*docker_base(args.sudo), "network", "rm", name], cwd=MCP_ROOT, env=command_env(args), dry_run=args.dry_run)
    create_shared_network(args)


def command_env(args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if args is not None:
        env["MAT_MCP_RUNTIME_PROFILE"] = selected_profile(args)
        if getattr(args, "concurrency_cap_override", None) is not None:
            env["MAT_MCP_CONCURRENCY_CAP_OVERRIDE"] = str(int(args.concurrency_cap_override))
        if getattr(args, "session_pool_size", None) is not None:
            env["MAT_MCP_SESSION_POOL_SIZE"] = str(int(args.session_pool_size))
    if args is not None and getattr(args, "no_proxy", False):
        # docker compose resolves build args from the environment and the
        # service .env file. Clearing these here provides a per-run escape hatch
        # for hosts where the configured proxy is intermittently unavailable.
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env[key] = ""
        env["no_proxy"] = env.get("no_proxy") or "localhost,127.0.0.1,::1"
        env["NO_PROXY"] = env.get("NO_PROXY") or env["no_proxy"]
    return env


def run(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=cwd, check=False, env=env)
    if check and completed.returncode:
        raise SystemExit(completed.returncode)
    return completed.returncode


def python_with_mcp_client() -> str:
    candidates = [
        sys.executable,
        str(ROOT / ".venv" / "bin" / "python3"),
        str(ROOT.parent / "LYT" / ".venv" / "bin" / "python3"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen or not Path(candidate).exists():
            continue
        seen.add(candidate)
        probe = subprocess.run(
            [candidate, "-c", "import mcp"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return candidate
    raise SystemExit(
        "Could not find a Python interpreter with the MCP client installed. "
        "Tried current interpreter plus repo-local and sibling LYT virtualenvs."
    )


def check_config(services: list[Service], include_extra: bool) -> None:
    problems: list[str] = []
    active = selected_services(services, include_extra)
    names = {service.name for service in services}
    ports: dict[int, str] = {}

    for service in active:
        if not service.directory.exists():
            problems.append(f"{service.name}: missing directory {service.directory}")
            continue
        for filename in ("docker-compose.yml", ".env", "Dockerfile"):
            if not (service.directory / filename).exists():
                problems.append(f"{service.name}: missing {filename}")
        if service.port in ports:
            problems.append(f"{service.name}: duplicate port {service.port} also used by {ports[service.port]}")
        ports[service.port] = service.name
        env = read_env(service.directory)
        if env.get("HOST_PORT") and int(env["HOST_PORT"]) != service.port:
            problems.append(f"{service.name}: .env HOST_PORT={env['HOST_PORT']} but stack port={service.port}")
        if env.get("SERVER_PORT") and int(env["SERVER_PORT"]) != service.port:
            problems.append(f"{service.name}: .env SERVER_PORT={env['SERVER_PORT']} but stack port={service.port}")
        if service.sse_path.startswith("/pymatgen") and env.get("MCP_PATH_PREFIX") != "/pymatgen":
            problems.append(f"{service.name}: expected MCP_PATH_PREFIX=/pymatgen")
        for dependency in service.depends_on:
            if dependency not in names:
                problems.append(f"{service.name}: unknown dependency {dependency}")
        if service.routerized:
            compose_path = service.directory / "docker-compose.yml"
            compose_text = compose_path.read_text(encoding="utf-8") if compose_path.exists() else ""
            required_snippets = [
                f"{service.name}-backend-1",
                "mcp-sse-router",
                "BACKEND_URLS=",
            ]
            for snippet in required_snippets:
                if snippet not in compose_text:
                    problems.append(f"{service.name}: routerized stack missing '{snippet}' in docker-compose.yml")

    config_paths = [MAIN_CONFIG, EVIDENCE_CONFIG]
    if not MAIN_CONFIG.exists() and LEGACY_MAIN_CONFIG.exists():
        config_paths[0] = LEGACY_MAIN_CONFIG
    if not EVIDENCE_CONFIG.exists() and LEGACY_EVIDENCE_CONFIG.exists():
        config_paths[1] = LEGACY_EVIDENCE_CONFIG

    configured = load_config_urls(config_paths)
    if not configured:
        problems.append(
            "no tool config URLs loaded; expected configs under "
            f"{CONFIG_ROOT} or legacy train/rl/tool_configs"
        )
    active_endpoints = {url_to_port_path(service.sse_url) for service in services if service.enabled}
    missing_urls = sorted(url for url in configured if url_to_port_path(url) not in active_endpoints)
    if missing_urls:
        problems.append(f"tool config URLs not represented in enabled stack services: {missing_urls}")

    print(f"checked_services={len(active)} enabled_services={sum(1 for s in services if s.enabled)}")
    for service in active:
        env = read_env(service.directory)
        flags = []
        if service.gpu:
            flags.append("gpu")
        if not service.enabled:
            flags.append("extra")
        mp_key = env.get("MP_API_KEY")
        if mp_key:
            flags.append("mp_key=set")
        elif "MP_API_KEY" in env:
            flags.append("mp_key=empty")
        api_missing = [key for key in ("BING_API_KEY", "TAVILY_API_KEY") if key in env and not env[key]]
        if api_missing:
            flags.append("api_key_missing=" + ",".join(api_missing))
        print(f"{service.name:30} {service.sse_url:38} {' '.join(flags)}")

    if problems:
        print("\nConfig problems:")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("Mat-MCP static config check passed.")


def compose_action(services: list[Service], args: argparse.Namespace, action: str) -> None:
    if action == "up":
        ensure_shared_network(args)
    compose = docker_compose_base(args.sudo)
    env = command_env(args)
    for service in selected_services(services, args.include_extra, args.gpu_only):
        profile = selected_profile(args)
        backend_count = desired_backend_count(service, args)
        compose_files = compose_files_for_service(service, profile, backend_count)
        compose_cmd = [*compose]
        for compose_file in compose_files:
            compose_cmd.extend(["-f", str(compose_file)])
        if action == "up":
            cmd = [*compose_cmd, "up", "-d", "--remove-orphans"]
            if args.build:
                cmd.append("--build")
            cmd.extend(service.service_names(profile, backend_count=backend_count))
        elif action == "down":
            cmd = [*compose_cmd, "down"]
        elif action == "restart":
            cmd = [*compose_cmd, "restart"]
            cmd.extend(service.service_names(profile, backend_count=backend_count))
        elif action == "ps":
            cmd = [*compose_cmd, "ps"]
        else:
            raise AssertionError(action)
        run(cmd, cwd=service.directory, check=not args.keep_going, env=env, dry_run=args.dry_run)
        if action in {"restart"}:
            extras = service.extra_backend_services(profile, backend_count=backend_count)
            if extras:
                run(
                    [*compose_cmd, "stop", *extras],
                    cwd=service.directory,
                    check=not args.keep_going,
                    env=env,
                    dry_run=args.dry_run,
                )


def warmup_mace(args: argparse.Namespace) -> None:
    service_dir = MCP_ROOT / "uip-relax-mcp"
    compose = docker_compose_base(args.sudo)
    run([*compose, "--profile", "warmup", "run", "--rm", "mace-model-warmup"], cwd=service_dir, env=command_env(args))


def verify(args: argparse.Namespace) -> None:
    py = python_with_mcp_client()
    run([py, str(SCRIPTS_ROOT / "check_mat_mcp_stack.py"), "--timeout", str(args.timeout)], cwd=MCP_ROOT)
    run([py, str(SCRIPTS_ROOT / "check_mat_mcp_tool_calls.py"), "--coverage-only"], cwd=MCP_ROOT)
    for tags in args.tags:
        cmd = [
            py,
            str(SCRIPTS_ROOT / "check_mat_mcp_tool_calls.py"),
            "--tags",
            tags,
            "--timeout",
            str(args.timeout),
        ]
        if args.strict_optional:
            cmd.append("--strict-optional")
        run(cmd, cwd=MCP_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    def add_options(target: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
        default = argparse.SUPPRESS if suppress_defaults else None
        target.add_argument("--stack", type=Path, default=DEFAULT_STACK if not suppress_defaults else default)
        target.add_argument(
            "--sudo",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Run docker compose through sudo.",
        )
        target.add_argument(
            "--include-extra",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Include mat-query and mattergen services.",
        )
        target.add_argument(
            "--gpu-only",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Apply compose actions only to GPU services.",
        )
        target.add_argument(
            "--build",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Build images during up.",
        )
        target.add_argument(
            "--keep-going",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Do not stop compose loops after a service command fails.",
        )
        target.add_argument(
            "--dry-run",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Print docker compose commands without executing them.",
        )
        target.add_argument(
            "--profile",
            choices=("single", "smoke", "production32", "production128"),
            default="single" if not suppress_defaults else default,
            help="Stack sizing profile. single=1 backend, smoke=moderate concurrency, production32=high concurrency, production128=aggressive high concurrency.",
        )
        target.add_argument(
            "--replicas-override",
            type=int,
            default=default,
            help="Force every routerized service to this many backend replicas for the current command.",
        )
        target.add_argument(
            "--concurrency-cap-override",
            type=int,
            default=default,
            help="Force MAT_MCP_CONCURRENCY_CAP_OVERRIDE for downstream clients.",
        )
        target.add_argument(
            "--session-pool-size",
            type=int,
            default=default,
            help="Force MAT_MCP_SESSION_POOL_SIZE for downstream clients.",
        )
        target.add_argument(
            "--mode",
            choices=("single", "high-concurrency"),
            default=default,
            help=argparse.SUPPRESS,
        )
        target.add_argument(
            "--no-proxy",
            action="store_true",
            default=False if not suppress_defaults else default,
            help="Temporarily clear http_proxy/https_proxy for docker compose build/runtime.",
        )
        target.add_argument("--timeout", type=float, default=90.0 if not suppress_defaults else default)
        target.add_argument(
            "--tags",
            action="append",
            default=[] if not suppress_defaults else default,
            help="Tool-call tag group to verify. Can be repeated. Default for verify/all is quick.",
        )
        target.add_argument(
            "--strict-optional",
            action="store_true",
            default=False if not suppress_defaults else default,
        )

    add_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("check-config", "ensure-shared-network", "recreate-shared-network", "up", "down", "restart", "ps", "warmup-mace", "verify", "all"):
        subparser = subparsers.add_parser(command)
        add_options(subparser, suppress_defaults=True)
    args = parser.parse_args()
    for attr, value in {
        "stack": DEFAULT_STACK,
        "sudo": False,
        "include_extra": False,
        "gpu_only": False,
        "build": False,
        "keep_going": False,
        "dry_run": False,
        "profile": "single",
        "replicas_override": None,
        "concurrency_cap_override": None,
        "session_pool_size": None,
        "mode": None,
        "no_proxy": False,
        "timeout": 90.0,
        "tags": [],
        "strict_optional": False,
    }.items():
        if not hasattr(args, attr):
            setattr(args, attr, value)
    return args


def main() -> None:
    args = parse_args()
    services = load_stack(args.stack)
    if args.command == "check-config":
        check_config(services, args.include_extra)
    elif args.command == "ensure-shared-network":
        ensure_shared_network(args)
    elif args.command == "recreate-shared-network":
        ensure_shared_network(args, recreate=True)
    elif args.command in {"up", "down", "restart", "ps"}:
        compose_action(services, args, args.command)
    elif args.command == "warmup-mace":
        warmup_mace(args)
    elif args.command == "verify":
        if not args.tags:
            args.tags = ["quick"]
        verify(args)
    elif args.command == "all":
        if not args.tags:
            args.tags = ["quick"]
        check_config(services, args.include_extra)
        compose_action(services, args, "up")
        warmup_mace(args)
        verify(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
