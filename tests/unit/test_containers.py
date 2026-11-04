"""Unit tests for Phase 8: Containerization and Kubernetes manifests.

These tests validate container specifications, Kubernetes manifests, configuration
environment injection, security contexts, and resource bounds statically without
requiring a running Docker engine or Kubernetes cluster.
"""

import importlib
import re
from pathlib import Path

import yaml

import src.common.config as config_mod


def test_config_reads_env_vars_with_defaults(monkeypatch):
    """Verify Settings loads from environment variables and falls back cleanly."""
    monkeypatch.setenv("FRAUDSHIELD_ENV", "staging")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka.internal:9092")

    importlib.reload(config_mod)
    try:
        cfg = config_mod.Settings()
        assert cfg.ENV == "staging"
        assert cfg.API_PORT == 9000
        assert cfg.POSTGRES_HOST == "db.internal"
        assert cfg.KAFKA_BOOTSTRAP_SERVERS == "kafka.internal:9092"
        assert "postgresql://postgres:postgres@db.internal:5432/fraudshield" in cfg.postgres_uri
    finally:
        monkeypatch.undo()
        importlib.reload(config_mod)


def test_no_hardcoded_localhost_or_credentials_in_src():
    """Verify production src code does not hardcode unconfigurable database connection strings."""
    for py_file in Path("src").rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "postgresql://" in text:
            # Any database URI in src/ must be constructed dynamically from Settings/os.getenv
            assert any(term in text for term in ["Settings", "settings", "os.getenv", "postgres_uri", "config"]), (
                f"{py_file} contains raw postgresql connection string without referencing config"
            )


def test_dockerfiles_exist_and_use_non_root_user():
    """Verify all required Dockerfiles exist, pin base images, and enforce non-root users."""
    dockerfiles = [
        Path("docker/Dockerfile.serving"),
        Path("docker/Dockerfile.dashboard"),
        Path("docker/Dockerfile.stream"),
        Path("docker/Dockerfile.drift"),
    ]

    for df in dockerfiles:
        assert df.exists(), f"Missing required Dockerfile: {df}"
        content = df.read_text(encoding="utf-8")

        # Verify pinned base image
        assert re.search(r"FROM python:3\.\d+(-slim|-slim-bookworm)?", content), (
            f"{df} must specify a pinned Python base image"
        )

        # Verify dedicated non-root user declaration
        assert "USER " in content, f"{df} must specify an unprivileged USER"
        assert not re.search(r"USER\s+root", content, re.IGNORECASE), (
            f"{df} cannot run as root user"
        )

        # Verify multi-stage build
        assert "AS builder" in content and "AS runner" in content, (
            f"{df} must use multi-stage builds (builder and runner)"
        )


def test_docker_compose_valid_yaml_and_references():
    """Verify docker-compose.yml parses correctly and references existing Dockerfiles."""
    compose_path = Path("docker/docker-compose.yml")
    assert compose_path.exists(), "Missing docker/docker-compose.yml"

    with open(compose_path, encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data
    services = compose_data["services"]

    # Verify key services are present
    expected_services = [
        "postgres",
        "mlflow",
        "serving",
        "dashboard",
        "producer",
        "consumer",
        "drift-worker",
    ]
    for svc in expected_services:
        assert svc in services, f"Missing service in compose: {svc}"

    # Verify referenced dockerfiles exist
    for svc_name, svc_conf in services.items():
        if "build" in svc_conf:
            build_info = svc_conf["build"]
            df_rel = build_info.get("dockerfile")
            if df_rel:
                df_path = Path(df_rel)
                assert df_path.exists(), f"Service {svc_name} references missing dockerfile {df_path}"

    # Verify resource limits exist to protect 8GB environments
    for svc_name in ["serving", "dashboard", "consumer", "drift-worker"]:
        svc_conf = services[svc_name]
        assert "deploy" in svc_conf and "resources" in svc_conf["deploy"], (
            f"Service {svc_name} must declare resource limits"
        )


def test_kubernetes_manifests_valid_yaml_and_labels():
    """Verify Kubernetes manifests parse, match selectors, and configure resource limits."""
    k8s_dir = Path("k8s")
    manifest_files = sorted(k8s_dir.glob("*.yaml"))
    assert len(manifest_files) >= 5, "Expected at least 5 Kubernetes manifest files"

    all_docs = []
    for mf in manifest_files:
        with open(mf, encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            all_docs.extend(docs)

    # Collect deployments, statefulsets, services, and cronjobs
    services = {}
    workloads = {}

    for doc in all_docs:
        if not doc or not isinstance(doc, dict):
            continue
        kind = doc.get("kind")
        name = doc.get("metadata", {}).get("name")

        if kind == "Service":
            selector = doc.get("spec", {}).get("selector", {})
            services[name] = selector
        elif kind in ("Deployment", "StatefulSet"):
            match_labels = (
                doc.get("spec", {}).get("selector", {}).get("matchLabels", {})
            )
            template_labels = (
                doc.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
            )
            containers = doc.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
            workloads[name] = {
                "match_labels": match_labels,
                "template_labels": template_labels,
                "containers": containers,
            }

    # Verify each service selector matches its target workload labels
    for svc_name, selector in services.items():
        assert svc_name in workloads, f"Service {svc_name} has no matching workload"
        workload_labels = workloads[svc_name]["template_labels"]
        for k, v in selector.items():
            assert workload_labels.get(k) == v, (
                f"Selector mismatch for service {svc_name}: {k}={v} vs {workload_labels.get(k)}"
            )

    # Verify resource limits on workloads
    for wl_name, wl_data in workloads.items():
        for container in wl_data["containers"]:
            resources = container.get("resources", {})
            assert "limits" in resources, f"Workload {wl_name} container missing resource limits"
            assert "cpu" in resources["limits"], f"Workload {wl_name} missing cpu limit"
            assert "memory" in resources["limits"], f"Workload {wl_name} missing memory limit"


def test_kubernetes_probes_configured():
    """Verify health probes are present on serving and dashboard workloads."""
    serving_manifest = Path("k8s/05-serving.yaml").read_text(encoding="utf-8")
    dashboard_manifest = Path("k8s/06-dashboard.yaml").read_text(encoding="utf-8")

    assert "readinessProbe:" in serving_manifest
    assert "livenessProbe:" in serving_manifest
    assert "path: /health" in serving_manifest

    assert "readinessProbe:" in dashboard_manifest
    assert "livenessProbe:" in dashboard_manifest
    assert "path: /_stcore/health" in dashboard_manifest


def test_kubernetes_manifest_validation_workflow_configured():
    """Verify GitHub Actions CI workflow configures valid, pinned kubeconform manifest validation."""
    workflow_path = Path(".github/workflows/containers.yml")
    assert workflow_path.exists(), "containers.yml workflow file must exist"

    content = workflow_path.read_text(encoding="utf-8")

    # 1. Ensure the validation job exists
    assert "kubernetes-manifest-validation:" in content
    assert "k8s/*.yaml" in content

    # 2. Verify invalid GitHub Action reference was removed
    assert "kubeconform-action" not in content, (
        "containers.yml must not reference unmaintained or invalid kubeconform-action"
    )

    # 3. Verify pinned kubeconform binary installation and execution
    assert "KUBECONFORM_VERSION:" in content
    assert "https://github.com/yannh/kubeconform/releases/download/" in content
    assert "./kubeconform -summary" in content


def test_postgres_kubernetes_initialization_configured():
    """Verify k8s/03-postgres.yaml configures ConfigMap DDL mounting to /docker-entrypoint-initdb.d."""
    manifest_path = Path("k8s/03-postgres.yaml")
    assert manifest_path.exists(), "k8s/03-postgres.yaml must exist"

    with open(manifest_path, encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    configmaps = [d for d in docs if d.get("kind") == "ConfigMap"]
    statefulsets = [d for d in docs if d.get("kind") == "StatefulSet"]

    assert len(configmaps) >= 1, "k8s/03-postgres.yaml must declare ConfigMap for database initialization"
    cm = next(c for c in configmaps if c["metadata"]["name"] == "postgres-init-config")
    init_sql = cm["data"]["init.sql"]
    assert "CREATE TABLE IF NOT EXISTS transactions" in init_sql
    assert "CREATE TABLE IF NOT EXISTS scores" in init_sql
    assert "CREATE TABLE IF NOT EXISTS cases" in init_sql
    assert "CREATE TABLE IF NOT EXISTS labels" in init_sql

    assert len(statefulsets) >= 1, "k8s/03-postgres.yaml must declare PostgreSQL StatefulSet"
    ss = statefulsets[0]
    container = ss["spec"]["template"]["spec"]["containers"][0]
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert "postgres-init" in mounts, "StatefulSet must mount postgres-init volume"
    assert mounts["postgres-init"] == "/docker-entrypoint-initdb.d"

    volumes = {v["name"]: v for v in ss["spec"]["template"]["spec"]["volumes"]}
    assert "postgres-init" in volumes
    assert volumes["postgres-init"]["configMap"]["name"] == "postgres-init-config"


def test_admin_api_key_secret_injection_configured():
    """Verify ADMIN_API_KEY is configured in Kubernetes secrets, serving manifest, and compose."""
    # 1. k8s/02-secrets.yaml declares ADMIN_API_KEY
    secrets_path = Path("k8s/02-secrets.yaml")
    with open(secrets_path, encoding="utf-8") as f:
        secret_doc = yaml.safe_load(f)
    assert "ADMIN_API_KEY" in secret_doc["stringData"]

    # 2. k8s/05-serving.yaml injects ADMIN_API_KEY from secret
    serving_path = Path("k8s/05-serving.yaml")
    with open(serving_path, encoding="utf-8") as f:
        serving_docs = list(yaml.safe_load_all(f))
    deployment = next(d for d in serving_docs if d.get("kind") == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env_vars = {
        e["name"]: e.get("valueFrom", {}).get("secretKeyRef", {})
        for e in container.get("env", [])
    }
    assert "ADMIN_API_KEY" in env_vars
    assert env_vars["ADMIN_API_KEY"]["name"] == "fraudshield-secrets"
    assert env_vars["ADMIN_API_KEY"]["key"] == "ADMIN_API_KEY"

    # 3. docker/docker-compose.yml references ADMIN_API_KEY
    compose_path = Path("docker/docker-compose.yml")
    with open(compose_path, encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)
    serving_env = compose_data["services"]["serving"]["environment"]
    assert any("ADMIN_API_KEY=" in e for e in serving_env)
