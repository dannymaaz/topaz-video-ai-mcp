import os
import sys
import json
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

def find_topaz_app_dir() -> Path:
    """Dynamically locates Topaz Video AI installation across standard OS paths."""
    # 1. Check custom environment variable
    custom_path = os.environ.get("TOPAZ_APP_DIR") or os.environ.get("TOPAZ_PATH")
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
        
    # 2. Windows standard paths
    candidates = [
        Path(r"C:\Program Files\Topaz Labs LLC\Topaz Video AI"),
        Path(r"C:\Program Files (x86)\Topaz Labs LLC\Topaz Video AI"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Topaz Labs LLC" / "Topaz Video AI",
        # macOS standard path
        Path("/Applications/Topaz Video AI.app/Contents/MacOS")
    ]
    
    for candidate in candidates:
        if candidate.exists() and ((candidate / "ffmpeg.exe").exists() or (candidate / "ffmpeg").exists()):
            return candidate
            
    # Default fallback to standard Windows path
    return Path(r"C:\Program Files\Topaz Labs LLC\Topaz Video AI")

def find_topaz_models_dir() -> Path:
    """Locates Topaz Video AI models directory."""
    custom_models = os.environ.get("TVAI_MODEL_DIR")
    if custom_models and Path(custom_models).exists():
        return Path(custom_models)
        
    candidates = [
        Path(r"C:\ProgramData\Topaz Labs LLC\Topaz Video AI\models"),
        Path(os.environ.get("APPDATA", "")) / "Topaz Labs LLC" / "Topaz Video AI" / "models",
        Path.home() / "Library" / "Application Support" / "Topaz Labs LLC" / "Topaz Video AI" / "models"
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(r"C:\ProgramData\Topaz Labs LLC\Topaz Video AI\models")

TOPAZ_APP_DIR = find_topaz_app_dir()
TOPAZ_FFMPEG = TOPAZ_APP_DIR / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
TOPAZ_FFPROBE = TOPAZ_APP_DIR / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
TOPAZ_MODELS_DIR = find_topaz_models_dir()

# Measured GPU Benchmark Profiles on Intel Iris Xe Graphics (fps during inference)
SPEED_PROFILES = {
    "iris-2": {"x1": 1.25, "x2": 0.82},
    "iris-3": {"x1": 1.30, "x2": 0.85},
    "prob-4": {"x1": 2.20, "x2": 1.65},
    "nyx-2":  {"x1": 2.40, "x2": 1.70},
    "ahq-12": {"x1": 3.00, "x2": 2.20},
    "ghq-5":  {"x1": 2.00, "x2": 1.50}
}

def get_topaz_env() -> Dict[str, str]:
    """Returns environment variables required by Topaz FFmpeg engine."""
    env = os.environ.copy()
    env["TVAI_MODEL_DIR"] = str(TOPAZ_MODELS_DIR)
    env["TVAI_MODEL_DATA_DIR"] = str(TOPAZ_MODELS_DIR)
    env["PATH"] = f"{str(TOPAZ_APP_DIR)}{os.pathsep}{env.get('PATH', '')}"
    return env

def make_even(n: int, multiple: int = 8) -> int:
    """Snaps resolution dimension to a multiple of 8 or 16 for optimal GPU encoding."""
    return (n // multiple) * multiple

def probe_video(video_path: str) -> Dict[str, Any]:
    """Probes video using Topaz ffprobe."""
    if not TOPAZ_FFPROBE.exists():
        raise FileNotFoundError(
            f"Topaz ffprobe not found at {TOPAZ_FFPROBE}. "
            "Please ensure Topaz Video AI is installed or set TOPAZ_APP_DIR."
        )
    
    cmd = [
        str(TOPAZ_FFPROBE),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True, env=get_topaz_env())
    if res.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {res.stderr}")
    
    data = json.loads(res.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")
    
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    
    fps_raw = video_stream.get("r_frame_rate", "30/1")
    try:
        num, den = map(float, fps_raw.split("/"))
        fps = num / den if den != 0 else 30.0
    except Exception:
        fps = 30.0
        
    duration = float(data.get("format", {}).get("duration", video_stream.get("duration", 0)))
    total_frames = int(video_stream.get("nb_frames", int(duration * fps) if duration > 0 else 0))
    bitrate = int(data.get("format", {}).get("bit_rate", video_stream.get("bit_rate", 0)))
    
    return {
        "path": os.path.abspath(video_path),
        "filename": os.path.basename(video_path),
        "width": width,
        "height": height,
        "aspect_ratio": f"{width}:{height}",
        "fps": round(fps, 3),
        "duration_sec": round(duration, 2),
        "duration_minutes": round(duration / 60, 2),
        "total_frames": total_frames,
        "video_codec": video_stream.get("codec_name"),
        "pixel_format": video_stream.get("pix_fmt"),
        "bitrate_kbps": round(bitrate / 1000, 1) if bitrate else 0,
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None
    }

def analyze_frames(video_path: str, num_samples: int = 5, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts representative frames and analyzes them for:
    - Faces / people (using multi-cascade detection)
    - Blur score (Laplacian variance)
    - Noise score
    """
    info = probe_video(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = info["total_frames"]
    if total_frames <= 0:
        total_frames = 100
        
    indices = [int(total_frames * (i + 1) / (num_samples + 1)) for i in range(num_samples)]
    
    cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    
    face_detections = 0
    blur_scores = []
    noise_scores = []
    saved_frames = []
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-cascade detection for low-res faces
        faces = cascade_alt2.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
        if len(faces) == 0:
            faces = cascade_default.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
        if len(faces) == 0:
            faces = cascade_profile.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
            
        if len(faces) > 0:
            face_detections += len(faces)
            
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(lap_var)
        
        median = cv2.medianBlur(gray, 3)
        diff = cv2.absdiff(gray, median)
        noise_score = float(np.mean(diff))
        noise_scores.append(noise_score)
        
        if save_dir:
            frame_path = os.path.join(save_dir, f"frame_{i+1}_idx_{idx}.jpg")
            thumb = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(thumb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(frame_path, thumb)
            saved_frames.append(frame_path)
            
    cap.release()
    
    has_faces = face_detections > 0
    avg_blur = float(np.mean(blur_scores)) if blur_scores else 0.0
    avg_noise = float(np.mean(noise_scores)) if noise_scores else 0.0
    
    return {
        "video_info": info,
        "sampled_frames_count": len(blur_scores),
        "total_faces_detected": face_detections,
        "has_faces": has_faces,
        "sharpness_score": round(avg_blur, 2),
        "noise_score": round(avg_noise, 2),
        "saved_frame_paths": saved_frames
    }

def estimate_render_time(total_frames: int, model: str, scale_factor: int) -> Dict[str, Any]:
    """Estimates time to complete render based on measured benchmarks."""
    scale_key = "x2" if scale_factor >= 2 else "x1"
    model_profile = SPEED_PROFILES.get(model, SPEED_PROFILES["prob-4"])
    expected_fps = model_profile.get(scale_key, 1.0)
    
    total_seconds = total_frames / expected_fps if expected_fps > 0 else total_frames
    minutes = round(total_seconds / 60, 1)
    hours = round(total_seconds / 3600, 2)
    
    if hours >= 1.0:
        display_time = f"{int(hours)}h {int((total_seconds % 3600) // 60)}m"
    else:
        display_time = f"{int(minutes)}m {int(total_seconds % 60)}s"
        
    return {
        "expected_render_fps": expected_fps,
        "estimated_seconds": int(total_seconds),
        "estimated_minutes": minutes,
        "estimated_hours": hours,
        "display_time": display_time
    }

def get_optimal_recipe(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies user's golden rules with optimized filter structure:
    1. Resolution rule:
       - Height < 1080 -> 2x upscale (target width/height snapped to multiple of 16)
       - Height >= 1080 -> 1x enhance (only de-noise/sharpen, no upscale)
    2. Model selection:
       - If faces/people present -> 'iris-2' (Face Restoration)
       - If high noise and no faces -> 'prob-4' or 'nyx-2'
       - General scenery -> 'prob-4'
    3. Anti-plastic & Realism tuning:
       - blend=0.25 (preserves original organic skin texture)
       - grain=0.018 (micrograin prevents plastic look)
       - noise=0.03 - 0.08 (conservative de-noise)
       - details=0.25 (balanced facial recovery)
    4. Hardware optimization:
       - instances=0 (avoids memory contention on integrated Intel GPU)
       - vram=1.0
       - scale=2 direct filter (removes redundant second ffmpeg scale filter)
    """
    info = analysis["video_info"]
    w = info["width"]
    h = info["height"]
    total_frames = info["total_frames"]
    has_faces = analysis.get("has_faces", False)
    noise_score = analysis.get("noise_score", 0.0)
    sharpness = analysis.get("sharpness_score", 0.0)
    
    is_sub_1080 = h < 1080 and w < 1920
    if is_sub_1080:
        scale = 2
        target_w = make_even(w * 2, multiple=16)
        target_h = make_even(h * 2, multiple=16)
        scale_reason = f"Resolución de entrada ({w}x{h}) es menor a 1080p -> Escalado x2 a {target_w}x{target_h} (alineado a mod 16 para aceleración QSV)."
    else:
        scale = 1
        target_w = make_even(w, multiple=16)
        target_h = make_even(h, multiple=16)
        scale_reason = f"Resolución de entrada ({w}x{h}) ya es 1080p o superior -> Modo Mejora 1x (sin reescalado innecesario)."

    if has_faces:
        model = "iris-2"
        model_name = "Iris v2 (Face Restoration)"
        model_reason = "Se detectaron rostros/personas en el metraje. Iris restaura ojos, expresiones y piel sin distorsión plástica."
    elif noise_score > 4.5:
        model = "prob-4"
        model_name = "Proteus v4 (High Noise Reduction)"
        model_reason = "Metraje con alto ruido/grano sin rostros. Proteus limpia ruido conservando bordes nítidos."
    else:
        model = "prob-4"
        model_name = "Proteus v4 (Fine-Tune Quality)"
        model_reason = "Metraje general sin rostros. Proteus v4 ofrece la mayor nitidez y velocidad de render."

    if model.startswith("iris"):
        preblur = -0.08 if sharpness < 120 else -0.04
        noise = 0.03 if noise_score < 2.5 else 0.07
        details = 0.25
        halo = 0.08
        blur = 0.25
        compression = 0.52
        blend = 0.25
        grain = 0.018
        gsize = 1.0
    else:
        preblur = -0.06
        noise = 0.04 if noise_score < 3.0 else 0.10
        details = 0.28
        halo = 0.08
        blur = 0.24
        compression = 0.48
        blend = 0.20
        grain = 0.015
        gsize = 1.0
        
    filter_string = (
        f"tvai_up=model={model}:scale={scale}:w={target_w}:h={target_h}:"
        f"preblur={preblur:.4f}:noise={noise:.4f}:details={details:.4f}:"
        f"halo={halo:.4f}:blur={blur:.4f}:compression={compression:.4f}:"
        f"blend={blend:.4f}:grain={grain:.4f}:gsize={gsize:.2f}:"
        f"device=-2:vram=1:instances=0"
    )
    
    est_selected = estimate_render_time(total_frames, model, scale)
    est_proteus = estimate_render_time(total_frames, "prob-4", scale)
    
    return {
        "model": model,
        "model_display_name": model_name,
        "model_selection_reason": model_reason,
        "scale_mode": f"x{scale}" if scale > 1 else "1x Enhance",
        "scale_reason": scale_reason,
        "input_resolution": f"{w}x{h}",
        "target_resolution": f"{target_w}x{target_h}",
        "target_width": target_w,
        "target_height": target_h,
        "parameters": {
            "preblur": preblur,
            "noise": noise,
            "details": details,
            "halo": halo,
            "blur": blur,
            "compression": compression,
            "blend": blend,
            "grain": grain,
            "gsize": gsize
        },
        "anti_plastic_guarantee": [
            f"Ruido muy sutil ({noise}) para evitar el efecto piel de cera/plástico.",
            f"Mezcla con original ({int(blend*100)}%) para preservar microporos y textura real del sensor.",
            f"Desenfocado y dehalo moderados para evitar halos o líneas dobles artificiales.",
            f"Micrograno cinematográfico ({grain}) que rompe la apariencia digital de IA."
        ],
        "render_time_estimate": {
            "selected_model": model,
            "estimated_time": est_selected["display_time"],
            "expected_fps": est_selected["expected_render_fps"],
            "faster_alternative": {
                "model": "prob-4",
                "estimated_time": est_proteus["display_time"],
                "speedup_percentage": f"{int((est_proteus['expected_render_fps'] / est_selected['expected_render_fps'] - 1) * 100)}% más rápido"
            } if model != "prob-4" else None
        },
        "ffmpeg_filter_complex": filter_string
    }

def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    filter_string: str,
    is_preview: bool = False,
    start_time: float = 0.0,
    duration: Optional[float] = None
) -> List[str]:
    """Builds optimized FFmpeg command compatible with Topaz Video AI."""
    cmd = [
        str(TOPAZ_FFMPEG),
        "-hide_banner",
        "-nostdin",
        "-y",
    ]
    
    if start_time > 0:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
        
    cmd.extend([
        "-i", os.path.abspath(input_path),
        "-sws_flags", "spline+accurate_rnd+full_chroma_int",
        "-filter_complex", filter_string,
        "-c:v", "h264_qsv",
        "-profile:v", "high",
        "-preset", "medium",
        "-pix_fmt", "nv12",
        "-g", "30",
        "-b:v", "50M"
    ])
    
    if not is_preview:
        cmd.extend([
            "-map", "0:a?",
            "-c:a", "copy"
        ])
    else:
        cmd.extend(["-an"])
        
    cmd.extend([
        "-fps_mode:v", "passthrough",
        "-movflags", "+faststart",
        os.path.abspath(output_path)
    ])
    
    return cmd
