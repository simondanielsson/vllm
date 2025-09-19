# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.transformers_utils.config import resolve_gguf_filename


def test_resolve_gguf_filename_local_suffix(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "model-Q4_K_M.gguf").touch()
    (repo_dir / "model-Q5_K_M.gguf").touch()

    resolved = resolve_gguf_filename(str(repo_dir), "Q4_K_M")

    assert resolved == "model-Q4_K_M.gguf"


def test_resolve_gguf_filename_requires_specificity(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "alpha-q4.gguf").touch()
    (repo_dir / "beta-q4.gguf").touch()

    with pytest.raises(ValueError):
        resolve_gguf_filename(str(repo_dir), "q4")
