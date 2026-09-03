# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _build_server_info(raw_request: Request) -> dict:
    """Build the data-parallel discovery payload for external routers.

    Args:
        raw_request: The incoming request, used to read the server's
            ``vllm_config`` from application state.

    Returns:
        A flat mapping with the fields required by external routers (such as
        the SGLang router) for data-parallel worker registration, including a
        top-level integer ``dp_size``.
    """
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    served_model_name = mc.served_model_name or mc.model
    return {
        "dp_size": pc.data_parallel_size,
        "tp_size": pc.tensor_parallel_size,
        "model_path": mc.model,
        "model_id": served_model_name,
        "served_model_name": served_model_name,
    }


@router.get("/server_info")
@router.get("/get_server_info")
async def show_server_info(raw_request: Request):
    return JSONResponse(content=_build_server_info(raw_request))


def attach_router(app: FastAPI):
    app.include_router(router)
