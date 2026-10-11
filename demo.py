import argparse
import atexit
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Iterable, Optional, Tuple

import gradio as gr
import requests
import torch
from torch import Tensor
from tqdm import tqdm

os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"
gr.TEMP_DIR = "tmp_gradio"

from src.data.dataset import DatasetConfig, RigDatasetModule
from src.data.transform import Transform
from src.model.tokenrig import TokenRigResult
from src.tokenizer.parse import get_tokenizer
from src.server.spec import (
    BPY_SERVER,
    default_device,
    default_precision,
    get_model,
    object_to_bytes,
    bytes_to_object,
)
from src.data.vertex_group import voxel_skin

MODEL_CKPTS = [
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
]

HF_PATHS = [
    "None",
]


def start_bpy_server():
    # SKINTOKENS_BPY_PYTHON: interpreter for the Blender (bpy) server, if bpy is not installable in this one
    # (e.g. Windows + AMD: PyTorch ROCm wheels need Python 3.12, bpy wheels need Python 3.11)
    python = os.environ.get("SKINTOKENS_BPY_PYTHON", sys.executable)
    popen_kwargs = dict(
        args=[python, "bpy_server.py"],
        stdout=None,
        stderr=None,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(**popen_kwargs)
    print(f"[Main] bpy_server.py started (pid={proc.pid})")

    def cleanup():
        print(f"[Main] Terminating bpy_server.py (pid={proc.pid})")
        try:
            if proc.poll() is not None:
                return
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    atexit.register(cleanup)
    return proc


model = None
tokenizer = None
transform = None
CURRENT_MODEL_CKPT: Optional[str] = None
CURRENT_HF_PATH: Optional[str] = None
DEVICE: Optional[str] = None  # None: cuda if available (NVIDIA or AMD/ROCm), else cpu
PRECISION: Optional[str] = None  # None: bf16 on NVIDIA, mixed (fp32 weights + bf16 autocast) on AMD/ROCm and CPU


def load_model(model_ckpt: str, hf_path: Optional[str]) -> Tuple[str, str]:
    global model, tokenizer, transform, CURRENT_MODEL_CKPT, CURRENT_HF_PATH
    if hf_path == "None":
        hf_path = None
    if model is not None and model_ckpt == CURRENT_MODEL_CKPT and hf_path == CURRENT_HF_PATH:
        return ("Model already loaded.", model_ckpt)

    if not model_ckpt:
        raise RuntimeError("model_ckpt is empty. Please select a checkpoint.")

    print(f"Loading model: {model_ckpt}, hf_path={hf_path}")
    model = get_model(model_ckpt, hf_path=hf_path, device=DEVICE, precision=PRECISION)
    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])
    CURRENT_MODEL_CKPT = model_ckpt
    CURRENT_HF_PATH = hf_path
    return ("Model loaded.", model_ckpt)


SUPPORTED_EXT = {".obj", ".fbx", ".glb"}


def collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    files = []
    for p in input_path.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXT:
            files.append(p)
    return files


def map_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    rel = in_path.relative_to(input_root)
    return (output_root / rel).with_suffix(".glb")


def post_bpy_payload(endpoint: str, payload):
    payload_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"skintokens_{endpoint}_", suffix=".pt", delete=False) as f:
            f.write(object_to_bytes(payload))
            payload_path = f.name
        request_payload = {"payload_path": payload_path}
        response = requests.post(
            f"{BPY_SERVER}/{endpoint}",
            data=object_to_bytes(request_payload),
        )
        response.raise_for_status()
        result = bytes_to_object(response.content)
        if isinstance(result, dict) and result.get("error") is not None:
            raise RuntimeError(result.get("traceback") or result["error"])
        return result
    finally:
        if payload_path is not None:
            try:
                os.remove(payload_path)
            except OSError:
                pass


def run_rig(
    filepaths: List[Path],
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    num_beams: int,
    use_skeleton: bool,
    use_transfer: bool,
    use_postprocess: bool,
    output_paths: List[Path],
    model_ckpt: str,
    hf_path: Optional[str],
):
    assert len(filepaths) == len(output_paths)

    load_model(model_ckpt, hf_path)

    datapath = {
        "data_name": None,
        "loader": "bpy_server",
        "filepaths": {"articulation": [str(p) for p in filepaths]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=transform,
        tokenizer=tokenizer,
        process_fn=model._process_fn,
    )

    dataloader = module.predict_dataloader()["articulation"]

    results_out = []

    device = DEVICE or default_device()
    precision = PRECISION or default_precision(device)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        if not use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = dict(
            max_length=2048,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=1,
            num_beams=int(num_beams),
            do_sample=True,
        )

        if "skeleton_tokens" in batch and "skeleton_mask" in batch:
            mask = batch["skeleton_mask"][0] == 1
            skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
        else:
            skeleton_tokens = None

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=precision == "mixed"):
            preds: List[TokenRigResult] = model.predict_step(
                batch,
                skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
                make_asset=True,
            )["results"]

        asset = preds[0].asset
        assert asset is not None

        if use_postprocess:
            voxel = asset.voxel(resolution=196)
            asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=asset.joints,
                vertices=asset.vertices,
                faces=asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            asset.normalize_skin()

        out_path = output_paths[i]
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if use_transfer:
            payload = dict(
                source_asset=asset,
                target_path=asset.path,
                export_path=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("transfer", payload)
        else:
            payload = dict(
                asset=asset,
                filepath=str(out_path),
                group_per_vertex=4,
            )
            res = post_bpy_payload("export", payload)

        if res != "ok":
            print(f"[Error] {res}")
        else:
            print(f"[OK] Exported: {out_path}")

        results_out.append(out_path)

    return results_out


def run_cli(args):
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    files = collect_files(input_path)
    if not files:
        raise RuntimeError("No valid 3D files found.")

    if len(files) == 1 and output_path.suffix:
        outputs = [output_path]
    else:
        outputs = [
            map_output_path(f, input_path, output_path)
            for f in files
        ]

    run_rig(
        files,
        args.top_k,
        args.top_p,
        args.temperature,
        args.repetition_penalty,
        args.num_beams,
        args.use_skeleton,
        args.use_transfer,
        args.use_postprocess,
        outputs,
        args.model_ckpt,
        args.hf_path,
    )


TOT = 0
def run_gradio(
    files,
    top_k,
    top_p,
    temperature,
    repetition_penalty,
    num_beams,
    use_skeleton,
    use_transfer,
    use_postprocess,
    model_ckpt,
    hf_path,
):
    if not files:
        return "Please upload at least one 3D model.", None

    tmp_out = Path(tempfile.mkdtemp(prefix="tokenrig_"))
    filepaths = [Path(f.name) for f in files]
    global TOT
    outputs = []
    for filepath in filepaths:
        TOT += 1
        outputs.append(tmp_out / f"res_{TOT}.glb")

    run_rig(
        filepaths,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        num_beams,
        use_skeleton,
        use_transfer,
        use_postprocess,
        outputs,
        model_ckpt,
        hf_path,
    )

    return f"Processed {len(outputs)} models.", [str(p) for p in outputs]


def launch_gradio():
    model_ckpts = MODEL_CKPTS
    hf_paths = HF_PATHS
    default_ckpt = model_ckpts[0] if model_ckpts else ""
    default_hf = hf_paths[0] if hf_paths else "None"

    with gr.Blocks(title="TokenRig Demo") as demo:
        gr.Markdown("## TokenRig Demo")
        gr.Markdown("Upload 3D assets, configure parameters, generate rigged GLB")

        files = gr.File(
            label="3D Models",
            file_count="multiple",
            file_types=[".obj", ".fbx", ".glb"],
        )

        with gr.Accordion("Generation Parameters", open=True):
            model_ckpt = gr.Dropdown(
                choices=model_ckpts,
                value=default_ckpt,
                label="Model checkpoint",
                interactive=True,
            )
            hf_path = gr.Dropdown(
                choices=hf_paths,
                value=default_hf,
                label="HF path",
                interactive=True,
            )
            top_k = gr.Slider(1, 200, value=5, step=1, label="top_k")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="temperature")
            repetition_penalty = gr.Slider(0.5, 3.0, value=2.0, step=0.1, label="repetition_penalty")
            num_beams = gr.Slider(1, 20, value=10, step=1, label="num_beams")
            use_skeleton = gr.Checkbox(False, label="Use skeleton (only generate skin if skeleton exists)")
            use_transfer = gr.Checkbox(False, label="Use transfer (maintain texture)")
            use_postprocess = gr.Checkbox(False, label="Use postprocess (voxel skin)")

        run_btn = gr.Button("Run", variant="primary")
        load_btn = gr.Button("Load Model")
        log = gr.Textbox(label="Status")
        output = gr.File(label="Generated GLB", interactive=False)

        load_btn.click(
            lambda ckpt, hf: load_model(ckpt, hf)[0],
            inputs=[model_ckpt, hf_path],
            outputs=[log],
        )

        run_btn.click(
            run_gradio,
            inputs=[
                files,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                num_beams,
                use_skeleton,
                use_transfer,
                use_postprocess,
                model_ckpt,
                hf_path,
            ],
            outputs=[log, output],
        )

    demo.launch(server_port=1024)

def wait_for_bpy_server(timeout=30):
    t0 = time.time()
    while True:
        try:
            requests.get(f"{BPY_SERVER}/ping", timeout=1)
            print("[Main] bpy_server is ready")
            return
        except Exception:
            if time.time() - t0 > timeout:
                raise RuntimeError("bpy_server failed to start")
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TokenRig Demo")
    parser.add_argument("--input", help="Input file or directory")
    parser.add_argument("--output", help="Output file or directory")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=10)

    parser.add_argument("--use_skeleton", action="store_true")
    parser.add_argument("--use_transfer", action="store_true")
    parser.add_argument("--use_postprocess", action="store_true")

    parser.add_argument("--model_ckpt", default=MODEL_CKPTS[0] if MODEL_CKPTS else "")
    parser.add_argument("--hf_path", default=None)

    parser.add_argument("--gradio", action="store_true")

    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="default: cuda if available (NVIDIA, or AMD with a ROCm build of PyTorch), else cpu")
    parser.add_argument("--precision", choices=["bf16", "mixed"], default=None,
                        help="default: bf16 on NVIDIA, mixed (fp32 weights + bf16 autocast) on AMD/ROCm and CPU")

    args = parser.parse_args()
    DEVICE = args.device
    PRECISION = args.precision

    server_proc = start_bpy_server()
    wait_for_bpy_server()

    if args.gradio or not args.input:
        launch_gradio()
    else:
        run_cli(args)
