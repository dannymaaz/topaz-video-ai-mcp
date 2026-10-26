import os
import sys
import re
import json
import uuid
import threading
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP

# Add local directory to import topaz_engine
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import topaz_engine

mcp = FastMCP("topaz-video-ai")

# Default final output path
# Default final output path: Uses environment variable TOPAZ_OUTPUT_DIR or falls back to ~/Videos/Topaz_Enhanced
DEFAULT_OUTPUT_DIR = Path(os.environ.get("TOPAZ_OUTPUT_DIR", Path.home() / "Videos" / "Topaz_Enhanced"))

# In-memory job tracker for async enhancement tasks
JOBS: Dict[str, Dict[str, Any]] = {}

@mcp.tool()
def topaz_analyze_video(video_path: str, num_samples: int = 5) -> str:
    """
    Analiza a fondo un archivo de video:
    - Extrae resolución, duración, tasa de fotogramas, codecs y bitrate.
    - Muestrea fotogramas a lo largo del metraje y los analiza para detectar presencia de rostros humanos.
    - Evalúa niveles de nitidez (blur score) y nivel de ruido/compresión.
    
    Args:
        video_path: Ruta absoluta o relativa al video en disco.
        num_samples: Cantidad de fotogramas a muestrear para el análisis (por defecto 5).
    """
    try:
        norm_path = os.path.abspath(video_path)
        if not os.path.exists(norm_path):
            return json.dumps({"error": f"El archivo de video no existe: {video_path}"}, indent=2, ensure_ascii=False)
            
        preview_dir = BASE_DIR / "analysis_frames" / Path(norm_path).stem
        analysis = topaz_engine.analyze_frames(norm_path, num_samples=num_samples, save_dir=str(preview_dir))
        return json.dumps(analysis, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_recommend_settings(video_path: str) -> str:
    """
    Analiza el video y determina automáticamente la configuración óptima para Topaz Video AI:
    - Regla de resolución: Menor a 1080p -> Upscale x2 (alineado a mod 16). Igual o mayor a 1080p -> Modo Mejora 1x.
    - Regla de modelo: Si detecta rostros/personas -> Utiliza Iris (iris-2). Si es metraje general -> Proteus v4 (prob-4).
    - Garantía Anti-Plástico: Ajusta desenfoque, reducción de ruido sutil, blend con original y micrograno para evitar la piel de cera.
    - Estimación de tiempo de renderizado: Calcula el tiempo estimado según tu GPU Intel Iris Xe.
    
    Args:
        video_path: Ruta al archivo de video.
    """
    try:
        norm_path = os.path.abspath(video_path)
        if not os.path.exists(norm_path):
            return json.dumps({"error": f"El archivo de video no existe: {video_path}"}, indent=2, ensure_ascii=False)
            
        analysis = topaz_engine.analyze_frames(norm_path, num_samples=5)
        recipe = topaz_engine.get_optimal_recipe(analysis)
        return json.dumps(recipe, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_estimate_render_time(
    duration_minutes: float,
    fps: float = 30.0,
    model: str = "iris-2",
    scale_factor: int = 2
) -> str:
    """
    Calcula exactamente cuánto tardará un video en procesarse en tu hardware (Intel Iris Xe)
    comparando el modelo seleccionado contra alternativas más rápidas.
    
    Args:
        duration_minutes: Duración del video en minutos (ej: 40.0 para 40 minutos).
        fps: Fotogramas por segundo del video (por defecto 30.0).
        model: Modelo a cotizar ('iris-2', 'prob-4', 'nyx-2', etc.).
        scale_factor: Escala (1 para 1x enhance, 2 para 2x upscale).
    """
    try:
        total_frames = int(duration_minutes * 60.0 * fps)
        est = topaz_engine.estimate_render_time(total_frames, model, scale_factor)
        est_proteus = topaz_engine.estimate_render_time(total_frames, "prob-4", scale_factor)
        
        return json.dumps({
            "video_duration_minutes": duration_minutes,
            "fps": fps,
            "total_frames": total_frames,
            "selected_model": {
                "model": model,
                "scale": f"x{scale_factor}",
                "expected_fps": est["expected_render_fps"],
                "estimated_time": est["display_time"],
                "estimated_hours": est["estimated_hours"]
            },
            "faster_alternative": {
                "model": "prob-4 (Proteus v4)",
                "expected_fps": est_proteus["expected_render_fps"],
                "estimated_time": est_proteus["display_time"],
                "estimated_hours": est_proteus["estimated_hours"],
                "speedup_percentage": f"{int((est_proteus['expected_render_fps'] / est['expected_render_fps'] - 1) * 100)}% más rápido"
            } if model != "prob-4" else None
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_preview_enhance(
    video_path: str,
    start_time_sec: float = 0.0,
    duration_sec: float = 3.0,
    model: Optional[str] = None,
    scale_factor: Optional[int] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Genera un clip de previsualización rápido (de 2 a 5 segundos) con Topaz Video AI.
    Permite verificar la calidad visual y texturas realistas antes de renderizar el video completo.
    
    Args:
        video_path: Ruta al archivo de video de entrada.
        start_time_sec: Segundo exacto donde comenzar la muestra (por defecto 0.0).
        duration_sec: Duración de la muestra en segundos (por defecto 3.0).
        model: Modelo Topaz a forzar ('iris-2', 'prob-4', etc.), o dejar en blanco para autodetección.
        scale_factor: Escala (1 para 1x enhance, 2 para x2 upscale), o blanco para autodetección.
        output_path: Ruta de salida opcional del video de muestra MP4.
    """
    try:
        norm_path = os.path.abspath(video_path)
        if not os.path.exists(norm_path):
            return json.dumps({"error": f"El archivo de video no existe: {video_path}"}, indent=2, ensure_ascii=False)
            
        analysis = topaz_engine.analyze_frames(norm_path, num_samples=3)
        recipe = topaz_engine.get_optimal_recipe(analysis)
        
        if model:
            recipe["model"] = model
        if scale_factor:
            orig_w = analysis["video_info"]["width"]
            orig_h = analysis["video_info"]["height"]
            scale = scale_factor
            target_w = topaz_engine.make_even(orig_w * scale, multiple=16) if scale > 1 else topaz_engine.make_even(orig_w, multiple=16)
            target_h = topaz_engine.make_even(orig_h * scale, multiple=16) if scale > 1 else topaz_engine.make_even(orig_h, multiple=16)
            recipe["target_width"] = target_w
            recipe["target_height"] = target_h
            p = recipe["parameters"]
            recipe["ffmpeg_filter_complex"] = (
                f"tvai_up=model={recipe['model']}:scale={scale}:w={target_w}:h={target_h}:"
                f"preblur={p['preblur']:.4f}:noise={p['noise']:.4f}:details={p['details']:.4f}:"
                f"halo={p['halo']:.4f}:blur={p['blur']:.4f}:compression={p['compression']:.4f}:"
                f"blend={p['blend']:.4f}:grain={p['grain']:.4f}:gsize={p['gsize']:.2f}:"
                f"device=-2:vram=1:instances=0"
            )
            
        if not output_path:
            previews_dir = BASE_DIR / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(previews_dir / f"preview_{Path(norm_path).stem}_{recipe['model']}.mp4")
            
        cmd = topaz_engine.build_ffmpeg_command(
            input_path=norm_path,
            output_path=output_path,
            filter_string=recipe["ffmpeg_filter_complex"],
            is_preview=True,
            start_time=start_time_sec,
            duration=duration_sec
        )
        
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=topaz_engine.get_topaz_env()
        )
        
        if proc.returncode != 0:
            return json.dumps({
                "success": False,
                "error": "Error durante el render de previsualización en Topaz FFmpeg",
                "details": proc.stderr[-1000:]
            }, indent=2, ensure_ascii=False)
            
        return json.dumps({
            "success": True,
            "preview_file": output_path,
            "start_time": start_time_sec,
            "duration": duration_sec,
            "applied_model": recipe["model"],
            "resolution": f"{recipe['target_width']}x{recipe['target_height']}",
            "note": "Previsualización generada exitosamente. Revise el archivo para confirmar la textura orgánica antes de procesar el video completo."
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_compare_frames(
    video_orig: str,
    video_enhanced: str,
    timestamp_sec: float = 1.0,
    output_image_path: Optional[str] = None
) -> str:
    """
    Genera una imagen comparativa lado a lado (Antes vs Después) entre el video original
    y el video mejorado en un segundo determinado.
    
    Args:
        video_orig: Ruta al video original sin procesar.
        video_enhanced: Ruta al video procesado/mejorado.
        timestamp_sec: Segundo donde extraer el fotograma a comparar (por defecto 1.0).
        output_image_path: Ruta destino de la imagen JPG comparativa (opcional).
    """
    try:
        norm_orig = os.path.abspath(video_orig)
        norm_enh = os.path.abspath(video_enhanced)
        
        if not os.path.exists(norm_orig) or not os.path.exists(norm_enh):
            return json.dumps({"error": "Uno de los archivos de video no existe."}, indent=2, ensure_ascii=False)
            
        cap_orig = cv2.VideoCapture(norm_orig)
        cap_enh = cv2.VideoCapture(norm_enh)
        
        fps = cap_orig.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = int(timestamp_sec * fps)
        
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_enh.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, f_orig = cap_orig.read()
        ret2, f_enh = cap_enh.read()
        
        cap_orig.release()
        cap_enh.release()
        
        if not ret1 or not ret2:
            return json.dumps({"error": "No se pudieron extraer los fotogramas en ese segundo."}, indent=2, ensure_ascii=False)
            
        # Match height for horizontal side-by-side
        target_h = f_enh.shape[0]
        scale_ratio = target_h / f_orig.shape[0]
        target_w = int(f_orig.shape[1] * scale_ratio)
        f_orig_res = cv2.resize(f_orig, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Add labels
        cv2.putText(f_orig_res, f"ORIGINAL ({f_orig.shape[1]}x{f_orig.shape[0]})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(f_enh, f"TOPAZ ENHANCED ({f_enh.shape[1]}x{f_enh.shape[0]})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        combined = np.hstack([f_orig_res, f_enh])
        
        if not output_image_path:
            previews_dir = BASE_DIR / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            output_image_path = str(previews_dir / f"compare_{Path(norm_orig).stem}_t{int(timestamp_sec)}.jpg")
            
        cv2.imwrite(output_image_path, combined)
        return json.dumps({
            "success": True,
            "comparison_image": output_image_path,
            "timestamp_sec": timestamp_sec
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_enhance_video(
    video_path: str,
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    scale_factor: Optional[int] = None,
    start_time_sec: float = 0.0,
    duration_sec: Optional[float] = None
) -> str:
    """
    Ejecuta el procesamiento y mejora completa o por segmento de un video con Topaz Video AI:
    - Analiza el video para aplicar las reglas (Iris para rostros, Proteus para general).
    - Aplica la regla de resolución (<1080p -> x2, >=1080p -> 1x).
    - Configura parámetros anti-plástico (piel realista, microtexturas, cero cera).
    - Guarda automáticamente en la carpeta de salida configurada (TOPAZ_OUTPUT_DIR o ~/Videos/Topaz_Enhanced).
    - Preserva el audio intacto sin recodificar (-c:a copy).
    - Permite segmentar (start_time_sec / duration_sec) para videos largos.
    
    Args:
        video_path: Ruta al archivo de video a procesar.
        output_path: Ruta destino del video mejorado (opcional).
        model: Forzar un modelo ('iris-2', 'prob-4') o dejar automático.
        scale_factor: Forzar escala (1 para 1x, 2 para 2x) o dejar automático.
        start_time_sec: Segundo inicial opcional para procesar un fragmento.
        duration_sec: Duración opcional en segundos para procesar solo ese tramo.
    """
    try:
        norm_path = os.path.abspath(video_path)
        if not os.path.exists(norm_path):
            return json.dumps({"error": f"El archivo de video no existe: {video_path}"}, indent=2, ensure_ascii=False)
            
        analysis = topaz_engine.analyze_frames(norm_path, num_samples=5)
        recipe = topaz_engine.get_optimal_recipe(analysis)
        
        if model:
            recipe["model"] = model
        if scale_factor:
            orig_w = analysis["video_info"]["width"]
            orig_h = analysis["video_info"]["height"]
            scale = scale_factor
            target_w = topaz_engine.make_even(orig_w * scale, multiple=16) if scale > 1 else topaz_engine.make_even(orig_w, multiple=16)
            target_h = topaz_engine.make_even(orig_h * scale, multiple=16) if scale > 1 else topaz_engine.make_even(orig_h, multiple=16)
            recipe["target_width"] = target_w
            recipe["target_height"] = target_h
            p = recipe["parameters"]
            recipe["ffmpeg_filter_complex"] = (
                f"tvai_up=model={recipe['model']}:scale={scale}:w={target_w}:h={target_h}:"
                f"preblur={p['preblur']:.4f}:noise={p['noise']:.4f}:details={p['details']:.4f}:"
                f"halo={p['halo']:.4f}:blur={p['blur']:.4f}:compression={p['compression']:.4f}:"
                f"blend={p['blend']:.4f}:grain={p['grain']:.4f}:gsize={p['gsize']:.2f}:"
                f"device=-2:vram=1:instances=0"
            )
            
        if not output_path:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(DEFAULT_OUTPUT_DIR / f"{Path(norm_path).stem}_enhanced_{recipe['model']}.mp4")
            
        job_id = str(uuid.uuid4())[:8]
        cmd = topaz_engine.build_ffmpeg_command(
            input_path=norm_path,
            output_path=output_path,
            filter_string=recipe["ffmpeg_filter_complex"],
            is_preview=False,
            start_time=start_time_sec,
            duration=duration_sec
        )
        
        log_path = BASE_DIR / "logs" / f"job_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_frames = analysis["video_info"]["total_frames"]
        if duration_sec:
            total_frames = int(duration_sec * analysis["video_info"]["fps"])
            
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "RUNNING",
            "input_file": norm_path,
            "output_file": output_path,
            "recipe": recipe,
            "total_frames": total_frames,
            "log_path": str(log_path),
            "proc": None
        }
        
        def run_job():
            with open(log_path, "w", encoding="utf-8") as f_log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    env=topaz_engine.get_topaz_env()
                )
                JOBS[job_id]["proc"] = proc
                proc.wait()
                if proc.returncode == 0:
                    JOBS[job_id]["status"] = "COMPLETED"
                elif JOBS[job_id].get("status") == "CANCELLED":
                    pass
                else:
                    JOBS[job_id]["status"] = "FAILED"
                    JOBS[job_id]["returncode"] = proc.returncode
                    
        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()
        
        return json.dumps({
            "success": True,
            "job_id": job_id,
            "status": "RUNNING",
            "input_file": norm_path,
            "output_file": output_path,
            "target_resolution": f"{recipe['target_width']}x{recipe['target_height']}",
            "applied_model": recipe["model"],
            "estimated_time": recipe["render_time_estimate"]["estimated_time"],
            "anti_plastic_guarantee": recipe["anti_plastic_guarantee"],
            "check_status_command": f"topaz_get_job_status(job_id='{job_id}')"
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_get_job_status(job_id: str) -> str:
    """
    Consulta el estado y progreso en tiempo real de un trabajo de renderizado de Topaz Video AI.
    Calcula el porcentaje completado, FPS actuales y tiempo restante estimado (ETA).
    
    Args:
        job_id: Identificador del trabajo devuelto por topaz_enhance_video.
    """
    job = JOBS.get(job_id)
    if not job:
        return json.dumps({"error": f"No se encontró el trabajo {job_id}"}, indent=2, ensure_ascii=False)
        
    recent_logs = ""
    current_frame = 0
    current_fps = 0.0
    progress_pct = 0.0
    eta_sec = 0
    
    log_path = job.get("log_path")
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                recent_logs = "".join(lines[-15:])
                
                # Parse current frame and fps from ffmpeg output
                for line in reversed(lines):
                    m = re.search(r"frame=\s*(\d+)\s+fps=\s*([\d\.]+)", line)
                    if m:
                        current_frame = int(m.group(1))
                        current_fps = float(m.group(2))
                        break
                        
                total_frames = job.get("total_frames", 1)
                if total_frames > 0:
                    progress_pct = round(min(100.0, (current_frame / total_frames) * 100.0), 1)
                    if current_fps > 0:
                        remaining_frames = max(0, total_frames - current_frame)
                        eta_sec = int(remaining_frames / current_fps)
        except Exception:
            pass
            
    eta_str = f"{eta_sec // 60}m {eta_sec % 60}s" if eta_sec < 3600 else f"{eta_sec // 3600}h {(eta_sec % 3600) // 60}m"
    
    return json.dumps({
        "job_id": job_id,
        "status": job["status"],
        "progress_percentage": f"{progress_pct}%",
        "current_frame": current_frame,
        "total_frames": job.get("total_frames", 0),
        "current_render_fps": current_fps,
        "eta": eta_str if job["status"] == "RUNNING" else "N/A",
        "input_file": job["input_file"],
        "output_file": job["output_file"],
        "recent_logs": recent_logs
    }, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_cancel_job(job_id: str) -> str:
    """
    Cancela un trabajo de renderizado en curso y libera recursos de GPU/CPU.
    
    Args:
        job_id: Identificador del trabajo devuelto por topaz_enhance_video.
    """
    job = JOBS.get(job_id)
    if not job:
        return json.dumps({"error": f"No se encontró el trabajo {job_id}"}, indent=2, ensure_ascii=False)
        
    proc = job.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            job["status"] = "CANCELLED"
            return json.dumps({"success": True, "message": f"Trabajo {job_id} cancelado con éxito."}, indent=2, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Error al cancelar proceso: {str(e)}"}, indent=2, ensure_ascii=False)
            
    return json.dumps({"message": f"El trabajo {job_id} no está corriendo actualmente."}, indent=2, ensure_ascii=False)

@mcp.tool()
def topaz_list_models() -> str:
    """
    Lista los modelos de inteligencia artificial disponibles en la instalación local de Topaz Video AI,
    su velocidad relativa de procesamiento y para qué tipo de contenido está diseñado cada uno.
    """
    models_info = [
        {
            "id": "iris-2",
            "name": "Iris v2 (Medium Quality / Face Restoration)",
            "speed_rating": "Normal (~0.8 fps en x2)",
            "category": "Rostros / Personas / Redes Sociales",
            "recommended_for": "Videos donde aparecen personas o rostros (TikTok, OnlyFans, vlogs, primeros planos). Recupera ojos, dientes, expresiones y piel sin distorsiones plásticas."
        },
        {
            "id": "prob-4",
            "name": "Proteus v4 (Fine-Tune Quality)",
            "speed_rating": "Rápido (~1.65 fps en x2 - Doble de rápido que Iris)",
            "category": "Metraje General / Paisajes / Cine",
            "recommended_for": "Videos estándar sin rostros dominantes o metrajes largos. Ofrece control milimétrico sobre antialias, ruido, nitidez y recuperación de textura original."
        },
        {
            "id": "nyx-2",
            "name": "Nyx v2 (Fast Denoise)",
            "speed_rating": "Rápido (~1.7 fps en x2)",
            "category": "Escenas Nocturnas / Ruido Extremo ISO",
            "recommended_for": "Videos con grano muy alto de sensor en baja iluminación sin barrer los detalles reales."
        },
        {
            "id": "ahq-12",
            "name": "Artemis High Quality v12",
            "speed_rating": "Muy Rápido (~2.2 fps en x2)",
            "category": "Metraje Limpio de Alta Calidad",
            "recommended_for": "Videos que ya tienen buena calidad pero se desea un acabado nítido y libre de artefactos."
        },
        {
            "id": "ghq-5",
            "name": "Gaia High Quality v5",
            "speed_rating": "Normal (~1.5 fps en x2)",
            "category": "Animación / CGI / Gráficos",
            "recommended_for": "Animación, renders 3D o metraje con líneas vectoriales definidas."
        }
    ]
    return json.dumps(models_info, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")
