# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import os
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SHMEM_COMMIT = "0132a6f504c2035d9c1a04100e0d7254fd2968e4"


def test_package_version_has_one_source_of_truth():
    version_file = (
        PROJECT_ROOT / "ops_transformer_catccos_udma" / "_version.py"
    )
    package_version = runpy.run_path(str(version_file))["__version__"]
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", package_version)

    setup_source = (PROJECT_ROOT / "setup.py").read_text()
    package_source = (
        PROJECT_ROOT / "ops_transformer_catccos_udma" / "__init__.py"
    ).read_text()
    cmake_source = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    assert 'PACKAGE_NAME, "_version.py"' in setup_source
    assert "from ._version import __version__" in package_source
    assert "ops_transformer_catccos_udma/_version.py" in cmake_source
    assert "VERSION \"${OPS_TRANSFORMER_CATCCOS_UDMA_VERSION}\"" in cmake_source
    assert "VERSION 0." not in cmake_source


def test_wheel_build_cleans_and_verifies_distribution_metadata():
    source = (PROJECT_ROOT / "scripts" / "build_wheel.sh").read_text()
    assert f"EXPECTED_SHMEM_COMMIT={SHMEM_COMMIT}" in source
    clean = source.index("python3 setup.py clean")
    build = source.index("python3 -m build --wheel --no-isolation")
    verify = source.index("Verified Wheel metadata:")
    assert clean < build < verify
    assert 'name.endswith(".dist-info/METADATA")' in source
    assert "if len(metadata_paths) != 1" in source
    assert 'metadata.get("Version") != expected_version' in source


def test_prepare_dependencies_pins_official_shmem_release_snapshot():
    source = (PROJECT_ROOT / "scripts" / "prepare_dependencies.sh").read_text()
    assert "SHMEM_REPOSITORY=https://gitcode.com/cann/shmem.git" in source
    assert f"SHMEM_COMMIT={SHMEM_COMMIT}" in source
    assert 'SHMEM_DIR=${SHMEM_SOURCE_PATH:-"${REPO_ROOT}/3rdparty/shmem"}' in source


def test_cmake_rejects_a_mismatched_shmem_checkout():
    source = (PROJECT_ROOT / "CMakeLists.txt").read_text()
    assert f'set(SHMEM_TAG_ID "{SHMEM_COMMIT}")' in source
    assert "SHMEM revision mismatch" in source
    assert ".ops_transformer_source_commit" in source
    assert "${SHMEM_SOURCE_PATH}/include" in source
    assert "${CATCCOS_SHMEM_SOURCE_DIR}/include" not in source


def test_wheel_records_bundled_shmem_revision():
    source = (
        PROJECT_ROOT
        / "ops_transformer_catccos_udma"
        / "_build_versions.py"
    ).read_text()
    assert 'SHMEM_VERSION = "1.6.0"' in source
    assert 'SHMEM_REF = "release/v1.6.0"' in source
    assert f'SHMEM_COMMIT = "{SHMEM_COMMIT}"' in source


def _git(repository, *args):
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _run_dependency_shell(source, **environment):
    bash = shutil.which("bash")
    if bash is None and os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).parents[1] / "bin" / "bash.exe"
            if candidate.is_file():
                bash = str(candidate)
    if bash is None:
        pytest.skip("dependency preparation requires Bash")
    return subprocess.run(
        [bash, "-c", "set -euo pipefail\n" + source],
        env={**os.environ, **{key: str(value) for key, value in environment.items()}},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )


@pytest.fixture
def shmem_checkout(tmp_path):
    repository = tmp_path / "shmem"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / ".gitignore").write_text("/build/\n/install/\n")
    source = repository / "header.h"
    source.write_text("old revision\n")
    _git(repository, "add", ".")
    _git(repository, "-c", "user.name=Dependency test", "-c",
         "user.email=dependency@example.invalid", "commit", "--quiet", "-m", "old")
    old_revision = _git(repository, "rev-parse", "HEAD")
    source.write_text("new revision\n")
    _git(repository, "add", ".")
    _git(repository, "-c", "user.name=Dependency test", "-c",
         "user.email=dependency@example.invalid", "commit", "--quiet", "-m", "new")
    new_revision = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--quiet", "--detach", old_revision)
    return repository, old_revision, new_revision


def _prepare_checkout(repository, revision, remote=None):
    # Execute the real checkout stage against local Git repositories; toolkit
    # and CatCCOS setup are outside this upgrade/dirty-worktree regression.
    source = (PROJECT_ROOT / "scripts" / "prepare_dependencies.sh").read_text()
    start = source.index('if [[ ! -e "${SHMEM_DIR}/.git" ]]; then')
    end = source.index('if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then')
    return _run_dependency_shell(
        source[start:end], SHMEM_DIR=repository.as_posix(),
        SHMEM_COMMIT=revision,
        SHMEM_REPOSITORY=(remote or repository).as_posix(),
    )


@pytest.mark.parametrize("downgrade", [False, True])
def test_prepare_switches_clean_checkout_and_preserves_ignored_build(shmem_checkout, downgrade):
    repository, old_revision, revision = shmem_checkout
    if downgrade:
        _git(repository, "checkout", "--quiet", "--detach", revision)
        revision = old_revision
    (repository / "build").mkdir()
    cached_object = repository / "build" / "keep.o"
    cached_object.write_text("ignored build cache")

    result = _prepare_checkout(repository, revision)

    assert result.returncode == 0, result.stdout
    assert _git(repository, "rev-parse", "HEAD") == revision
    expected_source = "old revision\n" if downgrade else "new revision\n"
    assert (repository / "header.h").read_text() == expected_source
    assert cached_object.read_text() == "ignored build cache"


@pytest.mark.parametrize("filename", ["header.h", "untracked.h"])
@pytest.mark.parametrize("downgrade", [False, True])
def test_prepare_preserves_dirty_checkout(shmem_checkout, filename, downgrade):
    repository, old_revision, revision = shmem_checkout
    if downgrade:
        _git(repository, "checkout", "--quiet", "--detach", revision)
        old_revision, revision = revision, old_revision
    user_file = repository / filename
    user_file.write_text("user changes\n")

    result = _prepare_checkout(repository, revision)

    assert result.returncode != 0
    assert "local changes" in result.stdout
    assert _git(repository, "rev-parse", "HEAD") == old_revision
    assert user_file.read_text() == "user changes\n"


@pytest.mark.parametrize("stale_part", ["source", "library", None])
def test_wheel_preflight_requires_matching_source_and_library(shmem_checkout, stale_part):
    repository, old_revision, revision = shmem_checkout
    source_revision = old_revision if stale_part == "source" else revision
    library_revision = old_revision if stale_part == "library" else revision
    _git(repository, "checkout", "--quiet", "--detach", source_revision)
    (repository / "include").mkdir()
    (repository / "include" / "shmem.h").touch()
    # The fake public header is test scaffolding, not a dirty source edit.
    (repository / ".git" / "info" / "exclude").write_text("/include/\n")
    install = repository / "install"
    libraries = install / "shmem" / "lib"
    libraries.mkdir(parents=True)
    for library in ("libshmem.so", "libshmem_utils.so"):
        (libraries / library).touch()
    marker = install / "shmem" / ".ops_transformer_source_commit"
    marker.write_text(library_revision + "\n")
    source = (PROJECT_ROOT / "scripts" / "build_wheel.sh").read_text()
    start = source.index('if [[ ! -f "${SHMEM_SOURCE_PATH}/include/shmem.h"')
    end = source.index('cd "${PROJECT_DIR}"')

    result = _run_dependency_shell(
        source[start:end], SHMEM_SOURCE_PATH=repository.as_posix(),
        SHMEM_HOME_PATH=install.as_posix(), EXPECTED_SHMEM_COMMIT=revision,
    )

    assert result.returncode == (0 if stale_part is None else 1), result.stdout
    if stale_part is not None:
        assert "source or library is not the pinned official release/v1.6.0 snapshot" in result.stdout
    assert _git(repository, "rev-parse", "HEAD") == source_revision
    assert marker.read_text().strip() == library_revision


def test_prepare_fetches_missing_pinned_commit(shmem_checkout, tmp_path):
    remote, old_revision, revision = shmem_checkout
    _git(remote, "branch", "legacy", old_revision)
    clone = tmp_path / "shmem-clone"
    subprocess.check_call([
        "git", "clone", "--quiet", "--no-local", "--single-branch",
        "--branch", "legacy", str(remote), str(clone),
    ])
    missing = subprocess.run([
        "git", "-C", str(clone), "cat-file", "-e", f"{revision}^{{commit}}",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert missing.returncode != 0

    result = _prepare_checkout(clone, revision, remote=remote)

    assert result.returncode == 0, result.stdout
    assert _git(clone, "rev-parse", "HEAD") == revision


@pytest.mark.parametrize("build_status", [0, 1])
def test_incomplete_build_cannot_keep_old_provenance(tmp_path, build_status):
    repository = tmp_path / "shmem"
    (repository / "scripts").mkdir(parents=True)
    (repository / "install" / "shmem").mkdir(parents=True)
    marker = repository / "install" / "shmem" / ".ops_transformer_source_commit"
    marker.write_text("old revision\n")
    (repository / "VERSION").write_text("1.6.0\n")
    (repository / "scripts" / "build.sh").write_text(f"exit {build_status}\n")
    source = (PROJECT_ROOT / "scripts" / "prepare_dependencies.sh").read_text()
    start = source.index('rm -f "${SHMEM_DIR}/install/shmem/.ops_transformer_source_commit"')
    end = source.index('echo "CatCCOS:')

    result = _run_dependency_shell(
        source[start:end], SHMEM_DIR=repository.as_posix(),
        SHMEM_COMMIT=SHMEM_COMMIT, ABI_ARG="-use_cxx11_abi0",
    )

    assert result.returncode != 0
    assert not marker.exists()
