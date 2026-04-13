import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)


def load_onnx_model(model_path: str):
    """
    Load an ONNX model using ONNX Runtime with optimizations.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    # Try GPU first, fallback to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
    except Exception:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

    logger.info(f"ONNX model loaded from {model_path}")
    logger.info(f"Using providers: {session.get_providers()}")
    return session