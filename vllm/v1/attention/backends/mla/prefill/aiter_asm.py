# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER ASM FP8 backend for MLA prefill on AMD gfx950 (MI350).

Dispatches through aiter.mla_prefill_ps_asm_fwd -> aiter.mla_reduce_v1.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )

logger = init_logger(__name__)

# Q-side tile size baked into the gfx950 mla_prefill_ps_asm_fwd kernel.
_FP8_PREFILL_TILE_Q = 256
# K-side tiling granularity required by the PS scheduler.
_KVLEN_GRANULARITY = 128


def _is_fp8_cache_dtype(cache_dtype: str) -> bool:
    return cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")


class AiterAsmPrefillBackend(MLAPrefillBackend):
    """FP8 MLA prefill backend built on AITER persistent-scheduling ASM on gfx950.

    The PS metadata is built once per batch for the new-tokens chunk and once per
    context chunk, then reused inside the kernel dispatch.

    Requires:
        - gfx950 (``DeviceCapability`` ``major=9, minor=5``)
        - AITER built with ``mla_prefill_ps_asm_fwd``, ``mla_reduce_v1``,
          ``get_ps_metadata_v1``, ``get_ps_metadata_info_v1`` exported
        - DeepSeek R1 MLA dimensions (qk_nope=128, qk_rope=64, v_head_dim=128)
        - FP8 KV cache (the FP8 ASM kernel produces wrong results otherwise)
    """

    supported_dtypes = [torch.float16, torch.bfloat16]
    requires_r1_mla_dimensions = True
    # mla_prefill_ps_asm_fwd only accepts FP8 Q/K/V; force the cast in the
    # parent regardless of attention_config.use_prefill_query_quantization.
    requires_fp8_query_quantization = True

    @staticmethod
    def get_name() -> str:
        return "AITER_ASM"

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return device_capability.major == 9 and device_capability.minor == 5

    @classmethod
    def is_available(cls) -> bool:
        try:
            from vllm.platforms.rocm import on_gfx950
        except Exception:  # noqa: BLE001
            return False
        if not on_gfx950():
            return False
        try:
            from aiter import (  # noqa: F401
                get_ps_metadata_info_v1,
                get_ps_metadata_v1,
                mla_prefill_ps_asm_fwd,
                mla_reduce_v1,
            )
        except Exception:  # noqa: BLE001
            return False
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            device_capability, selector_config
        )
        # Otherwise produces incorrect results
        if selector_config.cache_dtype not in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            invalid_reasons.append(
                f"cache_dtype {selector_config.cache_dtype!r} is not FP8 "
                "(requires fp8/fp8_e4m3/fp8_e5m2)"
            )
        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        from aiter import (
            get_ps_metadata_info_v1,
            get_ps_metadata_v1,
            mla_prefill_ps_asm_fwd,
            mla_reduce_v1,
        )

        self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
        self._mla_reduce_v1 = mla_reduce_v1
        self._get_ps_metadata_v1 = get_ps_metadata_v1
        self._get_ps_metadata_info_v1 = get_ps_metadata_info_v1

        # Populated by prepare_metadata before each forward pass.
        self._new_tokens_ps: dict | None = None
        self._context_ps: list[dict] = []

        # Worst-case sizes used to pre-allocate persistent PS buffers for the
        # new-tokens chunk. Mirrors PR #42509: declaring max sizes up-front
        # changes the PS scheduler's K-split layout vs allocating with exact
        # per-batch sizes, which empirically affected gsm8k accuracy.
        self._ps_max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self._ps_max_qlen = min(
            vllm_config.model_config.max_model_len,
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        # Lazy init on first prepare_metadata call (need device).
        self._persistent_new_tokens_buffers: dict | None = None

    def _ensure_persistent_buffers(self, device: torch.device) -> None:
        if self._persistent_new_tokens_buffers is not None:
            return
        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = self._get_ps_metadata_info_v1(
            batch_size=self._ps_max_num_reqs,
            num_head_k=self.num_heads,
            max_qlen=self._ps_max_qlen,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
        )
        self._persistent_new_tokens_buffers = {
            "work_metadata": torch.empty(
                work_metadata_size, dtype=work_metadata_dtype, device=device
            ),
            "work_indptr": torch.empty(
                work_indptr_size, dtype=work_indptr_dtype, device=device
            ),
            "work_info": torch.empty(
                *work_info_size, dtype=work_info_dtype, device=device
            ),
            "reduce_indptr": torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
            ),
            "reduce_final_map": torch.empty(
                *reduce_final_map_size,
                dtype=reduce_final_map_dtype,
                device=device,
            ),
            "reduce_partial_map": torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_dtype,
                device=device,
            ),
        }

    def _build_ps_for_chunk(
        self,
        qo_indptr_cpu: torch.Tensor,
        kv_indptr_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        is_causal: bool,
        device: torch.device,
        persistent: bool = False,
    ) -> dict:
        """Allocate and populate persistent device buffers.

        `seq_lens_cpu` drives the PS scheduler's K-side work split. For the
        new-tokens chunk (Q == K) Q-side and K-side lengths are identical.
        For context chunks we pass K-side per-request lengths.
        """
        # max_qlen is Q-side; the kernel uses it to size per-tile workspace.
        max_qlen = int((qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).max().item())
        batch_size = seq_lens_cpu.numel()
        num_head_k = self.num_heads

        if persistent:
            self._ensure_persistent_buffers(device)
            assert self._persistent_new_tokens_buffers is not None
            buffers = self._persistent_new_tokens_buffers
            work_metadata = buffers["work_metadata"]
            work_indptr = buffers["work_indptr"]
            work_info = buffers["work_info"]
            reduce_indptr = buffers["reduce_indptr"]
            reduce_final_map = buffers["reduce_final_map"]
            reduce_partial_map = buffers["reduce_partial_map"]
        else:
            (
                (work_metadata_size, work_metadata_dtype),
                (work_indptr_size, work_indptr_dtype),
                (work_info_size, work_info_dtype),
                (reduce_indptr_size, reduce_indptr_dtype),
                (reduce_final_map_size, reduce_final_map_dtype),
                (reduce_partial_map_size, reduce_partial_map_dtype),
            ) = self._get_ps_metadata_info_v1(
                batch_size=batch_size,
                num_head_k=num_head_k,
                max_qlen=max_qlen,
                qlen_granularity=_FP8_PREFILL_TILE_Q,
            )

            work_metadata = torch.empty(
                work_metadata_size, dtype=work_metadata_dtype, device=device
            )
            work_indptr = torch.empty(
                work_indptr_size, dtype=work_indptr_dtype, device=device
            )
            work_info = torch.empty(
                *work_info_size, dtype=work_info_dtype, device=device
            )
            reduce_indptr = torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
            )
            reduce_final_map = torch.empty(
                *reduce_final_map_size,
                dtype=reduce_final_map_dtype,
                device=device,
            )
            reduce_partial_map = torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_dtype,
                device=device,
            )

        self._get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            1,  # gqa_ratio: K is decompressed to num_heads, matching Q.
            num_head_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=1,
            qlen_granularity=_FP8_PREFILL_TILE_Q,
            kvlen_granularity=_KVLEN_GRANULARITY,
            block_size=1,
            is_causal=is_causal,
        )

        # reduce_indptr[-1] counts the K-split partial tiles emitted by the
        # PS scheduler; it sizes the per-call scratch for the kernel pair.
        # Reading it here forces one host sync per chunk per batch — fine at
        # build time, would break CUDA Graph capture if done in forward.
        num_partial_tiles = int(reduce_indptr[-1].item())

        return {
            "work_indptr": work_indptr,
            "work_info": work_info,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
            "num_partial_tiles": num_partial_tiles,
            "max_q_len": max_qlen,
        }

    def prepare_metadata(self, prefill_metadata: "MLACommonPrefillMetadata") -> None:
        super().prepare_metadata(prefill_metadata)

        qo_indptr = prefill_metadata.query_start_loc  # device int32 [bs+1]
        device = qo_indptr.device
        # One host sync per build. Could be eliminated by surfacing
        # query_start_loc_cpu on MLACommonPrefillMetadata; deferred.
        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        q_seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)

        # 1. New-tokens chunk (causal): K == Q.
        ps = self._build_ps_for_chunk(
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=qo_indptr_cpu,
            seq_lens_cpu=q_seq_lens_cpu,
            is_causal=True,
            device=device,
            persistent=True,
        )
        total_q = int(qo_indptr_cpu[-1].item())
        ps["qo_indptr"] = qo_indptr
        ps["kv_indptr"] = qo_indptr
        ps["kv_indices"] = torch.arange(total_q, device=device, dtype=torch.int32)
        self._new_tokens_ps = ps

        # 2. Context chunks (non-causal): K is the chunk's gathered cache (flat,
        # indexed 0..chunk_total-1), Q is the same new-tokens Q.
        self._context_ps = []
        chunked = prefill_metadata.chunked_context
        if chunked is None:
            return
        n_chunks = len(chunked.cu_seq_lens)
        for i in range(n_chunks):
            kv_cu_seq_lens = chunked.cu_seq_lens[i]  # device int32 [bs+1]
            kv_indptr_cpu = kv_cu_seq_lens.to("cpu", dtype=torch.int32)
            kv_seq_lens_cpu = (kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).to(torch.int32)
            ps_i = self._build_ps_for_chunk(
                qo_indptr_cpu=qo_indptr_cpu,
                kv_indptr_cpu=kv_indptr_cpu,
                seq_lens_cpu=kv_seq_lens_cpu,
                is_causal=False,
                device=device,
            )
            chunk_total = int(kv_indptr_cpu[-1].item())
            ps_i["qo_indptr"] = qo_indptr
            ps_i["kv_indptr"] = kv_cu_seq_lens
            ps_i["kv_indices"] = torch.arange(
                chunk_total, device=device, dtype=torch.int32
            )
            self._context_ps.append(ps_i)

    def _run_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ps: dict,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the PS ASM kernel + reduce, returning `(out, lse)`.

        Output dtype matches `prefill_metadata.output_dtype` so the caller
        can feed it directly to `merge_attn_states` or copy it into the
        final `output` buffer.
        """
        from vllm.v1.worker.workspace import current_workspace_manager

        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q
        num_partial_tiles = ps["num_partial_tiles"]

        out_dtype = self._prefill_metadata.output_dtype
        assert out_dtype is not None
        out = torch.empty(
            (total_q, nhead, v_head_dim),
            dtype=out_dtype,
            device=q.device,
        )

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        logits, attn_lse, final_lse = current_workspace_manager().get_simultaneous(
            ((num_partial_tiles * tile_q, nhead, v_head_dim), torch.float32),
            ((num_partial_tiles * tile_q, nhead), torch.float32),
            ((total_q, nhead), torch.float32),
        )

        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            ps["qo_indptr"],
            ps["kv_indptr"],
            ps["kv_indices"],
            ps["work_indptr"],
            ps["work_info"],
            ps["max_q_len"],
            self.scale,
            is_causal,
            logits,
            attn_lse,
            out,
            one_scale,
            one_scale,
            one_scale,
        )

        self._mla_reduce_v1(
            logits,
            attn_lse,
            ps["reduce_indptr"],
            ps["reduce_final_map"],
            ps["reduce_partial_map"],
            tile_q,
            out,
            final_lse,
        )
        return out, final_lse

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self._new_tokens_ps is not None, (
            "prepare_metadata must be called before run_prefill_new_tokens"
        )
        out, lse = self._run_kernel(q, k, v, self._new_tokens_ps, is_causal=True)
        if return_softmax_lse:
            return out, lse
        return out

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 <= chunk_idx < len(self._context_ps), (
            f"chunk_idx={chunk_idx} out of range; "
            f"prepare_metadata built {len(self._context_ps)} context chunks"
        )
        return self._run_kernel(q, k, v, self._context_ps[chunk_idx], is_causal=False)
