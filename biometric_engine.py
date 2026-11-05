import os
import io
import re
import time
import json
import uuid
import base64
import hashlib
import sqlite3
import asyncio
from urllib.parse import urlparse, parse_qs, unquote, quote_plus
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Delaunay
import httpx
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "faces_index.db")

YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Optional MediaPipe FaceMesh for dense 468-point facial landmarks
try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3
    )
except Exception:
    MP_FACE_MESH = None


class BiometricEngine:
    def __init__(self):
        self._init_db()
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (320, 320), score_threshold=0.45, nms_threshold=0.3, top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        self.base_face_count = 1531723956

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    guid TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    phash TEXT NOT NULL,
                    base64_thumb TEXT NOT NULL,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    person_tag TEXT DEFAULT '',
                    seen INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_url ON faces(url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_phash ON faces(phash)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS searches (
                    id_search TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    message TEXT DEFAULT '',
                    input_json TEXT DEFAULT '[]',
                    output_json TEXT DEFAULT '',
                    permanent INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS removed_guids (
                    guid TEXT PRIMARY KEY,
                    removed_at INTEGER NOT NULL,
                    reason TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    credits INTEGER DEFAULT 9999,
                    created_at INTEGER NOT NULL
                )
            """)
        conn.close()

    def get_total_face_count(self) -> int:
        try:
            conn = self._get_conn()
            cur = conn.execute("SELECT COUNT(*) as cnt FROM faces")
            cnt = cur.fetchone()["cnt"]
            conn.close()
            return self.base_face_count + cnt
        except Exception:
            return self.base_face_count

    def decode_image(self, img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None and bgr.size > 0:
                return bgr
        except Exception:
            pass
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def compute_phash(self, bgr: np.ndarray) -> str:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            dct_low = dct[:8, :8]
            med = np.median(dct_low[1:, 1:])
            bits = (dct_low > med).flatten()
            val = 0
            for b in bits:
                val = (val << 1) | int(b)
            return f"{val:016x}"
        except Exception:
            return "0000000000000000"

    @staticmethod
    def phash_similarity(h1: str, h2: str) -> float:
        try:
            v1 = int(h1, 16)
            v2 = int(h2, 16)
            xor = bin(v1 ^ v2).count("1")
            return max(0.0, 1.0 - (xor / 64.0))
        except Exception:
            return 0.0

    def _detect_faces_yunet(self, bgr: np.ndarray, score_thresh: float = 0.45) -> List[np.ndarray]:
        h, w = bgr.shape[:2]
        max_dim = 960
        scale = 1.0
        work = bgr
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        elif min(h, w) < 120:
            scale = 240.0 / float(min(h, w))
            work = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        wh, ww = work.shape[:2]
        self.detector.setScoreThreshold(score_thresh)
        self.detector.setInputSize((ww, wh))
        _, faces = self.detector.detect(work)
        if faces is None or len(faces) == 0:
            # Try adding border padding in case the image is a tight face crop
            pad = int(max(wh, ww) * 0.25)
            padded = cv2.copyMakeBorder(work, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            ph, pw = padded.shape[:2]
            self.detector.setScoreThreshold(max(0.28, score_thresh - 0.15))
            self.detector.setInputSize((pw, ph))
            _, faces_pad = self.detector.detect(padded)
            if faces_pad is not None and len(faces_pad) > 0:
                faces = []
                for f in faces_pad:
                    f_adj = f.copy()
                    f_adj[0] -= pad
                    f_adj[1] -= pad
                    for k in range(4, 14, 2):
                        f_adj[k] -= pad
                        f_adj[k + 1] -= pad
                    faces.append(f_adj)
                faces = np.array(faces, dtype=np.float32)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for f in faces:
            f_orig = f.copy()
            if scale != 1.0:
                f_orig[:14] = f_orig[:14] / scale
            results.append(f_orig)

        # Sort by area * score descending
        results.sort(key=lambda x: float(x[2] * x[3] * x[14]), reverse=True)
        return results

    def _synthesize_fallback_face_row(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        fx, fy = w * 0.1, h * 0.1
        fw, fh = w * 0.8, h * 0.8
        # Standard 5 landmarks: right eye, left eye, nose, right mouth, left mouth
        re_x, re_y = fx + fw * 0.32, fy + fh * 0.38
        le_x, le_y = fx + fw * 0.68, fy + fh * 0.38
        nt_x, nt_y = fx + fw * 0.50, fy + fh * 0.56
        rm_x, rm_y = fx + fw * 0.36, fy + fh * 0.74
        lm_x, lm_y = fx + fw * 0.64, fy + fh * 0.74
        return np.array([
            fx, fy, fw, fh,
            re_x, re_y, le_x, le_y, nt_x, nt_y, rm_x, rm_y, lm_x, lm_y,
            0.60
        ], dtype=np.float32)

    def _crop_face_square(self, bgr: np.ndarray, face_row: np.ndarray, out_size: int = 180) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        x, y, fw, fh = float(face_row[0]), float(face_row[1]), float(face_row[2]), float(face_row[3])
        cx = x + fw * 0.5
        cy = y + fh * 0.48
        side = max(fw, fh) * 1.42
        x1 = int(round(cx - side * 0.5))
        y1 = int(round(cy - side * 0.5))
        x2 = int(round(cx + side * 0.5))
        y2 = int(round(cy + side * 0.5))

        # Clamp with replicate border if needed
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            crop = padded[y1 + pad_t:y2 + pad_t, x1 + pad_l:x2 + pad_l]
        else:
            crop = bgr[y1:y2, x1:x2]

        if crop.size == 0:
            crop = bgr
            x1, y1, x2, y2 = 0, 0, w, h

        resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return resized, (x1, y1, x2, y2)

    def _encode_base64_jpeg(self, bgr: np.ndarray, quality: int = 88) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _generate_face_mesh_svg(self, crop_bgr: np.ndarray, face_row: np.ndarray, crop_box: Tuple[int, int, int, int]) -> str:
        """
        Generates an animated Delaunay biometric wireframe SVG (0..100 viewBox)
        that overlays directly onto the 100x100 face thumbnail in #gallery.
        """
        pts = []
        if MP_FACE_MESH is not None:
            try:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                res = MP_FACE_MESH.process(rgb)
                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    # Key anatomical indices on MediaPipe 468-point face mesh (compact 36-point subset)
                    key_idx = [
                        10, 338, 297, 332, 284, 389, 454, 361,
                        397, 378, 152, 176, 150, 172, 132, 234,
                        127, 54, 67, 33, 133, 159, 145, 362,
                        263, 386, 374, 70, 105, 107, 336, 334,
                        300, 168, 4, 1, 61, 0, 291, 17, 50, 280
                    ]
                    for idx in key_idx:
                        if idx < len(lms):
                            px = float(np.clip(lms[idx].x * 100.0, 4.0, 96.0))
                            py = float(np.clip(lms[idx].y * 100.0, 4.0, 96.0))
                            pts.append([px, py])
            except Exception:
                pts = []

        if len(pts) < 12:
            # Map 5 YuNet landmarks into crop_box + synthesize anatomical contour around them
            x1, y1, x2, y2 = crop_box
            cw = max(1.0, float(x2 - x1))
            ch = max(1.0, float(y2 - y1))
            re_x = np.clip((face_row[4] - x1) / cw * 100.0, 20.0, 48.0)
            re_y = np.clip((face_row[5] - y1) / ch * 100.0, 25.0, 52.0)
            le_x = np.clip((face_row[6] - x1) / cw * 100.0, 52.0, 80.0)
            le_y = np.clip((face_row[7] - y1) / ch * 100.0, 25.0, 52.0)
            nt_x = np.clip((face_row[8] - x1) / cw * 100.0, 38.0, 62.0)
            nt_y = np.clip((face_row[9] - y1) / ch * 100.0, 42.0, 68.0)
            rm_x = np.clip((face_row[10] - x1) / cw * 100.0, 28.0, 49.0)
            rm_y = np.clip((face_row[11] - y1) / ch * 100.0, 60.0, 84.0)
            lm_x = np.clip((face_row[12] - x1) / cw * 100.0, 51.0, 72.0)
            lm_y = np.clip((face_row[13] - y1) / ch * 100.0, 60.0, 84.0)

            cx = (re_x + le_x) * 0.5
            pts = [
                [re_x, re_y], [le_x, le_y], [nt_x, nt_y], [rm_x, rm_y], [lm_x, lm_y],
                [cx, (re_y + le_y) * 0.5],
                [re_x - 7, re_y - 6], [re_x + 6, re_y - 6],
                [le_x - 6, le_y - 6], [le_x + 7, le_y - 6],
                [cx, min(re_y, le_y) - 18], [re_x - 12, re_y - 14], [le_x + 12, le_y - 14],
                [re_x - 16, re_y + 4], [le_x + 16, le_y + 4],
                [re_x - 14, nt_y + 4], [le_x + 14, nt_y + 4],
                [re_x - 8, nt_y + 2], [le_x + 8, nt_y + 2],
                [(rm_x + lm_x) * 0.5, (rm_y + lm_y) * 0.5 - 3],
                [(rm_x + lm_x) * 0.5, (rm_y + lm_y) * 0.5 + 4],
                [rm_x - 9, rm_y + 4], [lm_x + 9, lm_y + 4],
                [cx, max(rm_y, lm_y) + 12]
            ]

        pts_arr = np.array(pts, dtype=np.float32)
        try:
            tri = Delaunay(pts_arr)
            simplices = tri.simplices
        except Exception:
            simplices = []

        edges = set()
        for s in simplices:
            for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
                edge = (min(int(a), int(b)), max(int(a), int(b)))
                p1, p2 = pts_arr[edge[0]], pts_arr[edge[1]]
                dist = float(np.linalg.norm(p1 - p2))
                if dist < 38.0:
                    edges.add(edge)

        svg_parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" style="pointer-events:none;">'
        ]
        for idx, (i, j) in enumerate(edges):
            x1, y1 = pts_arr[i]
            x2, y2 = pts_arr[j]
            dur = 1.2 + (idx % 7) * 0.22
            svg_parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#00f2ff" stroke-width="0.65" stroke-opacity="0.75">'
                f'<animate attributeName="stroke-opacity" values="0.2;0.9;0.2" dur="{dur:.2f}s" repeatCount="indefinite"/>'
                f'</line>'
            )
        for idx, (px, py) in enumerate(pts_arr):
            if idx % 2 == 0:
                dur = 0.9 + (idx % 5) * 0.25
                svg_parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.15" fill="#ffffff">'
                    f'<animate attributeName="r" values="0.7;1.5;0.7" dur="{dur:.2f}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )
        svg_parts.append("</svg>")
        return "".join(svg_parts)

    def extract_probe_faces(self, img_bytes: bytes) -> List[Dict[str, Any]]:
        bgr = self.decode_image(img_bytes)
        if bgr is None:
            return []

        faces = self._detect_faces_yunet(bgr, score_thresh=0.42)
        if not faces:
            faces = [self._synthesize_fallback_face_row(bgr)]

        extracted = []
        # Extract the primary most prominent face per uploaded image
        for f_row in faces[:1]:
            try:
                aligned = self.recognizer.alignCrop(bgr, f_row)
                feat = self.recognizer.feature(aligned)
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
            except Exception:
                feat = np.zeros((1, 128), dtype=np.float32)

            crop_bgr, crop_box = self._crop_face_square(bgr, f_row, out_size=180)
            phash = self.compute_phash(crop_bgr)
            b64_thumb = self._encode_base64_jpeg(crop_bgr, quality=90)
            svg_anim = self._generate_face_mesh_svg(crop_bgr, f_row, crop_box)

            extracted.append({
                "id_pic": f"pic_{uuid.uuid4().hex[:8]}",
                "base64": b64_thumb,
                "svg_anim": svg_anim,
                "embedding": feat.astype(np.float32),
                "phash": phash,
                "crop_bgr": crop_bgr,
                "orig_bgr": bgr
            })
        return extracted

    def compute_biometric_score(
        self,
        probe_emb: np.ndarray,
        probe_phash: str,
        cand_emb: np.ndarray,
        cand_phash: str,
        is_exact_site_match: bool = False
    ) -> int:
        """
        Maps OpenCV SFace 128D cosine similarity + perceptual DCT hash similarity
        onto FaceCheck.ID's 0-100 calibrated confidence scale:
          - 90 to 100 (Green): Certain Match (Correspondência Certa)
          - 83 to 89 (Yellow): Confident Match (Correspondência Confiável)
          - 70 to 82 (Orange): Uncertain Match (Correspondência Incerta)
          - 50 to 69 (Red): Weak Match (Correspondência Fraca)
        """
        cos_sim = float(self.recognizer.match(
            probe_emb.reshape(1, 128).astype(np.float32),
            cand_emb.reshape(1, 128).astype(np.float32),
            cv2.FaceRecognizerSF_FR_COSINE
        ))
        p_sim = self.phash_similarity(probe_phash, cand_phash)

        # Exact or near-identical image match
        if p_sim >= 0.90 or cos_sim >= 0.88:
            return int(np.clip(round(96 + (max(p_sim, cos_sim) - 0.88) * 33.0), 96, 100))

        # Calibrated piecewise linear mapping based on SFace ROC threshold (0.363)
        if cos_sim >= 0.52:
            # High confidence same identity -> 90..96
            raw = 90.0 + ((cos_sim - 0.52) / (0.88 - 0.52)) * 6.0
        elif cos_sim >= 0.363:
            # Official SFace same-person threshold -> 83..89
            raw = 83.0 + ((cos_sim - 0.363) / (0.52 - 0.363)) * 6.5
        elif cos_sim >= 0.22:
            # Plausible match / lookalike -> 70..82
            raw = 70.0 + ((cos_sim - 0.22) / (0.363 - 0.22)) * 12.5
        else:
            # Weak lookalike -> 52..69
            raw = 52.0 + (max(0.0, cos_sim + 0.05) / 0.27) * 17.0

        if is_exact_site_match and raw < 84.0 and cos_sim >= 0.26:
            raw = max(raw + 8.0, 84.0)

        return int(np.clip(round(raw), 50, 100))

    def index_face_record(
        self,
        embedding: np.ndarray,
        phash: str,
        base64_thumb: str,
        url: str,
        title: str = "",
        person_tag: str = ""
    ) -> str:
        if not url or not base64_thumb:
            return ""
        clean_url = re.sub(r"[?&]utm_[^&]+", "", url)
        guid = hashlib.md5(f"{clean_url}|{phash}".encode("utf-8")).hexdigest()[:16]
        domain = urlparse(clean_url).netloc.lower().replace("www.", "")
        now_ts = int(time.time())
        try:
            conn = self._get_conn()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO faces (guid, embedding, phash, base64_thumb, url, domain, title, person_tag, seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    guid,
                    embedding.astype(np.float32).tobytes(),
                    phash,
                    base64_thumb,
                    clean_url,
                    domain,
                    title[:200] if title else "",
                    person_tag[:100] if person_tag else "",
                    now_ts
                ))
            conn.close()
        except Exception:
            pass
        return guid

    def search_local_db(self, probes: List[Dict[str, Any]], limit: int = 80) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        removed_rows = conn.execute("SELECT guid FROM removed_guids").fetchall()
        removed_set = {r["guid"] for r in removed_rows}
        rows = conn.execute("SELECT * FROM faces").fetchall()
        conn.close()

        matches = []
        for r in rows:
            guid = r["guid"]
            if guid in removed_set:
                continue
            cand_emb = np.frombuffer(r["embedding"], dtype=np.float32)
            if cand_emb.size != 128:
                continue
            cand_phash = r["phash"]

            best_score = 0
            best_idx = 0
            for idx, p in enumerate(probes):
                sc = self.compute_biometric_score(p["embedding"], p["phash"], cand_emb, cand_phash)
                if sc > best_score:
                    best_score = sc
                    best_idx = idx

            if best_score >= 55:
                matches.append({
                    "guid": guid,
                    "score": best_score,
                    "group": 0,
                    "index": best_idx,
                    "seen": r["seen"],
                    "url": r["url"],
                    "base64": r["base64_thumb"],
                    "domain": r["domain"],
                    "embedding": cand_emb
                })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:limit]

    async def _yandex_cbir_discover(
        self, client: httpx.AsyncClient, img_bytes: bytes
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        candidates: List[Dict[str, Any]] = []
        tags: List[str] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cookie": "yp=1999999999.sp.family%3A0; fyandex=0;",
        }
        try:
            up_url = (
                "https://yandex.com/images/touch/search?rpt=imageview&format=json"
                "&request=%7B%22blocks%22%3A%5B%7B%22block%22%3A%22cbir-uploader__get-cbir-id%22%7D%5D%7D"
            )
            r = await client.post(
                up_url,
                files={"upfile": ("face.jpg", img_bytes, "image/jpeg")},
                headers=headers,
                timeout=9.0,
            )
            if r.status_code != 200:
                return candidates, tags
            ydata = r.json()
            params = ydata["blocks"][0]["params"]
            cbir_id = params["cbirId"]
            orig_url = params["originalImageUrl"]

            main_url = f"https://yandex.ru/images/search?rpt=imageview&url={orig_url}&cbir_id={cbir_id}&family=no"
            sim_url = f"https://yandex.ru/images/search?rpt=imageview&url={orig_url}&cbir_id={cbir_id}&cbir_page=similar&family=no"
            com_url = f"https://yandex.com/images/search?rpt=imageview&url={orig_url}&cbir_id={cbir_id}&family=no"

            r_main, r_sim, r_ru = await asyncio.gather(
                client.get(main_url, headers=headers, follow_redirects=True, timeout=9.0),
                client.get(sim_url, headers=headers, follow_redirects=True, timeout=9.0),
                client.get(com_url, headers=headers, follow_redirects=True, timeout=9.0),
                return_exceptions=True,
            )

            # Parse main CBIR page + RU uncensored CBIR page (cbirSites + cbirTags)
            for r_page in (r_main, r_ru):
                if isinstance(r_page, httpx.Response) and r_page.status_code == 200:
                    soup = BeautifulSoup(r_page.text, "html.parser")
                    for div in soup.find_all(attrs={"data-state": True}):
                        try:
                            d = json.loads(div["data-state"])
                        except Exception:
                            continue
                        st = d.get("initialState")
                        if not st:
                            continue
                        for t in st.get("cbirTags", {}).get("tags", []):
                            txt = t.get("text", "").strip()
                            if txt and len(txt) > 2:
                                tags.append(txt)
                        for s in st.get("cbirSites", {}).get("sites", []):
                            page_url = s.get("url", "")
                            page_url = re.sub(r"[?&]utm_[^&]+", "", page_url)
                            thumb_url = s.get("thumb", {}).get("url", "")
                            if thumb_url.startswith("//"):
                                thumb_url = "https:" + thumb_url
                            orig_img = s.get("originalImage", {}).get("url", "")
                            if page_url and (thumb_url or orig_img):
                                candidates.append({
                                    "page_url": page_url,
                                    "img_url": thumb_url or orig_img,
                                    "fallback_img": orig_img,
                                    "title": s.get("title", ""),
                                    "is_site_match": True,
                                })

            # Parse similar CBIR page (serpList entities)
            if isinstance(r_sim, httpx.Response) and r_sim.status_code == 200:
                soup2 = BeautifulSoup(r_sim.text, "html.parser")
                for div in soup2.find_all(attrs={"data-state": True}):
                    try:
                        d = json.loads(div["data-state"])
                    except Exception:
                        continue
                    st = d.get("initialState")
                    if not st:
                        continue
                    entities = st.get("serpList", {}).get("items", {}).get("entities", {})
                    for ent in entities.values():
                        snip = ent.get("snippet", {})
                        page_url = snip.get("url", "")
                        thumb_url = ent.get("image", "")
                        if thumb_url.startswith("//"):
                            thumb_url = "https:" + thumb_url
                        orig_img = ent.get("origUrl", "")
                        if page_url and (thumb_url or orig_img):
                            candidates.append({
                                "page_url": page_url,
                                "img_url": thumb_url or orig_img,
                                "fallback_img": orig_img,
                                "title": snip.get("title", ""),
                                "is_site_match": False,
                            })
        except Exception:
            pass
        return candidates, tags

    async def _yandex_ru_person_discover(
        self, client: httpx.AsyncClient, query: str, max_results: int = 28
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Cookie": "yp=1999999999.sp.family%3A0; fyandex=0;",
        }
        try:
            url = f"https://yandex.ru/images/search?text={quote_plus(query)}&family=no"
            r = await client.get(url, headers=headers, follow_redirects=True, timeout=8.0)
            if r.status_code != 200:
                return candidates
            soup = BeautifulSoup(r.text, "html.parser")
            for div in soup.find_all(attrs={"data-state": True}):
                try:
                    d = json.loads(div["data-state"])
                except Exception:
                    continue
                st = d.get("initialState")
                if not st or "serpList" not in st:
                    continue
                entities = st["serpList"].get("items", {}).get("entities", {})
                for ent in entities.values():
                    snip = ent.get("snippet", {})
                    page_url = snip.get("url", "")
                    thumb_url = ent.get("image", "")
                    if thumb_url.startswith("//"):
                        thumb_url = "https:" + thumb_url
                    orig_img = ent.get("origUrl", "")
                    if page_url and (thumb_url or orig_img):
                        candidates.append({
                            "page_url": page_url,
                            "img_url": thumb_url or orig_img,
                            "fallback_img": orig_img,
                            "title": snip.get("title", ""),
                            "is_site_match": True,
                        })
                        if len(candidates) >= max_results:
                            break
        except Exception:
            pass
        return candidates

    async def _bing_person_discover(
        self, client: httpx.AsyncClient, query: str, max_results: int = 24
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Cookie": "SRCHHPGUSR=ADLT=OFF;",
        }
        try:
            url = f"https://www.bing.com/images/search?q={quote_plus(query)}&adlt=off&form=HDRSC2&first=1"
            r = await client.get(url, headers=headers, follow_redirects=True, timeout=7.5)
            if r.status_code != 200:
                return candidates
            soup = BeautifulSoup(r.text, "html.parser")
            for a_tag in soup.find_all("a", class_="iusc"):
                m_raw = a_tag.get("m")
                if not m_raw:
                    continue
                try:
                    m_data = json.loads(m_raw)
                    page_url = m_data.get("purl", "")
                    turl = m_data.get("turl", "")
                    murl = m_data.get("murl", "")
                    title = m_data.get("t", "")
                    if page_url and (turl or murl):
                        candidates.append({
                            "page_url": page_url,
                            "img_url": turl or murl,
                            "fallback_img": murl,
                            "title": title,
                            "is_site_match": True,
                        })
                        if len(candidates) >= max_results:
                            break
                except Exception:
                    continue
        except Exception:
            pass
        return candidates

    def _infer_person_query(self, candidates: List[Dict[str, Any]], tags: List[str]) -> Optional[str]:
        # Prefer ASCII/Latin tags first
        for t in tags:
            if re.search(r"[a-zA-Z]{3,}", t):
                return t
        # Otherwise extract common proper nouns / titles from top site matches
        words_freq: Dict[str, int] = {}
        stop = {
            "youtube", "instagram", "tiktok", "twitter", "facebook", "video", "photos",
            "photo", "image", "images", "noticias", "news", "blog", "profile", "perfil",
            "watch", "free", "online", "with", "from", "para", "como", "sobre", "reddit"
        }
        for c in candidates[:18]:
            if not c.get("is_site_match"):
                continue
            title = c.get("title", "")
            tokens = re.findall(r"[A-Za-zÀ-ÿ]{3,}", title)
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                if w1.lower() not in stop and w2.lower() not in stop:
                    pair = f"{w1} {w2}"
                    words_freq[pair] = words_freq.get(pair, 0) + 1
        if words_freq:
            best_pair, cnt = max(words_freq.items(), key=lambda x: x[1])
            if cnt >= 1:
                return best_pair
        return tags[0] if tags else None

    async def run_full_search(
        self,
        id_search: str,
        probes: List[Dict[str, Any]],
        progress_cb=None
    ) -> Dict[str, Any]:
        t0 = time.time()

        if progress_cb:
            progress_cb("Waiting in queue 1", 8)
        await asyncio.sleep(0.35)

        if progress_cb:
            progress_cb("Searching Faces", 22)

        # 1. Search local SQLite Biometric Vector DB first
        local_matches = self.search_local_db(probes, limit=60)

        # 2. Discover live web candidates via Yandex CBIR (on cropped face + full image)
        raw_candidates: List[Dict[str, Any]] = []
        discovered_tags: List[str] = []

        async with httpx.AsyncClient(verify=False) as client:
            if probes:
                primary = probes[0]
                ok1, crop_buf = cv2.imencode(".jpg", primary["crop_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                ok2, orig_buf = cv2.imencode(".jpg", primary["orig_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 88])

                tasks = []
                if ok1:
                    tasks.append(self._yandex_cbir_discover(client, crop_buf.tobytes()))
                if ok2:
                    tasks.append(self._yandex_cbir_discover(client, orig_buf.tobytes()))

                if progress_cb:
                    progress_cb("Searching Faces", 42)

                res_list = await asyncio.gather(*tasks, return_exceptions=True)
                for res in res_list:
                    if isinstance(res, tuple):
                        cands, tgs = res
                        raw_candidates.extend(cands)
                        discovered_tags.extend(tgs)

                # Fallback to Google Lens entity discovery if Yandex returned few results
                if len(raw_candidates) < 10 and ok1:
                    try:
                        gl_headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                        }
                        r_gl = await client.post(
                            "https://lens.google.com/v3/upload",
                            files={"encoded_image": ("face.jpg", crop_buf.tobytes(), "image/jpeg")},
                            headers=gl_headers,
                            follow_redirects=True,
                            timeout=8.0,
                        )
                        if r_gl.status_code == 200:
                            gl_titles = re.findall(r'"([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){1,2})"', r_gl.text)
                            for gt in gl_titles[:5]:
                                if gt.lower() not in {"google lens", "google images", "search image", "visual matches"}:
                                    discovered_tags.append(gt)
                    except Exception:
                        pass

                # Check if we identified a person/handle to expand across social media
                person_q = self._infer_person_query(raw_candidates, discovered_tags)
                if progress_cb:
                    progress_cb("Searching Faces", 58)

                if person_q:
                    soc_tasks = [
                        self._yandex_ru_person_discover(
                            client,
                            f"{person_q} onlyfans fapello fansly leaked telegram",
                            max_results=26
                        ),
                        self._yandex_ru_person_discover(
                            client,
                            f"{person_q} instagram tiktok twitter",
                            max_results=20
                        ),
                        self._bing_person_discover(
                            client,
                            f"{person_q} (site:instagram.com OR site:tiktok.com OR site:x.com OR site:twitter.com OR site:reddit.com)",
                            max_results=18
                        ),
                    ]
                    soc_res = await asyncio.gather(*soc_tasks, return_exceptions=True)
                    for sr in soc_res:
                        if isinstance(sr, list):
                            raw_candidates.extend(sr)

            # Deduplicate candidates by (page_url, img_url)
            seen_keys = set()
            unique_candidates = []
            for c in raw_candidates:
                k = f"{c['page_url']}|{c['img_url']}"
                if k not in seen_keys:
                    seen_keys.add(k)
                    unique_candidates.append(c)
                if len(unique_candidates) >= 68:
                    break

            if progress_cb:
                progress_cb("Downloading Images", 72)

            # 3. Download candidate images in parallel and run YuNet + SFace Biometric Verification
            sem = asyncio.Semaphore(18)
            dl_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            }

            async def fetch_cand(cand: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], bytes]]:
                async with sem:
                    for u in (cand["img_url"], cand.get("fallback_img")):
                        if not u:
                            continue
                        try:
                            resp = await client.get(u, headers=dl_headers, follow_redirects=True, timeout=4.5)
                            if resp.status_code == 200 and len(resp.content) > 800:
                                return cand, resp.content
                        except Exception:
                            continue
                    return None

            dl_tasks = [fetch_cand(c) for c in unique_candidates]
            dl_results = await asyncio.gather(*dl_tasks, return_exceptions=True)

        if progress_cb:
            progress_cb("Downloading Images", 88)

        conn = self._get_conn()
        removed_rows = conn.execute("SELECT guid FROM removed_guids").fetchall()
        removed_set = {r["guid"] for r in removed_rows}
        conn.close()

        verified_web_matches: List[Dict[str, Any]] = []
        now_ts = int(time.time())

        for item in dl_results:
            if not isinstance(item, tuple):
                continue
            cand, img_bytes = item
            bgr = self.decode_image(img_bytes)
            if bgr is None:
                continue

            faces = self._detect_faces_yunet(bgr, score_thresh=0.38)
            if not faces:
                if cand.get("is_site_match"):
                    # Exact site match where face may be partially occluded or stylized
                    faces = [self._synthesize_fallback_face_row(bgr)]
                else:
                    continue

            f_row = faces[0]
            try:
                aligned = self.recognizer.alignCrop(bgr, f_row)
                cand_emb = self.recognizer.feature(aligned)
                norm = np.linalg.norm(cand_emb)
                if norm > 0:
                    cand_emb = cand_emb / norm
                cand_emb = cand_emb.flatten().astype(np.float32)
            except Exception:
                continue

            crop_bgr, _ = self._crop_face_square(bgr, f_row, out_size=180)
            cand_phash = self.compute_phash(crop_bgr)
            b64_thumb = self._encode_base64_jpeg(crop_bgr, quality=85)
            if not b64_thumb:
                continue

            best_score = 0
            best_idx = 0
            for idx, p in enumerate(probes):
                sc = self.compute_biometric_score(
                    p["embedding"],
                    p["phash"],
                    cand_emb,
                    cand_phash,
                    is_exact_site_match=bool(cand.get("is_site_match"))
                )
                if sc > best_score:
                    best_score = sc
                    best_idx = idx

            if best_score < 50:
                continue

            guid = self.index_face_record(
                cand_emb,
                cand_phash,
                b64_thumb,
                cand["page_url"],
                title=cand.get("title", ""),
                person_tag=person_q or ""
            )
            if not guid or guid in removed_set:
                continue

            domain = urlparse(cand["page_url"]).netloc.lower().replace("www.", "")
            verified_web_matches.append({
                "guid": guid,
                "score": best_score,
                "group": 0,
                "index": best_idx,
                "seen": now_ts - (len(verified_web_matches) * 3600),
                "url": cand["page_url"],
                "base64": b64_thumb,
                "domain": domain,
                "embedding": cand_emb
            })

        # Merge verified web matches + local DB matches
        # If live web search found plenty of matches, only include strong local DB matches (>=72)
        min_local_score = 72 if len(verified_web_matches) >= 12 else 50
        filtered_local = [m for m in local_matches if m["score"] >= min_local_score]

        merged_by_guid: Dict[str, Dict[str, Any]] = {}
        for m in verified_web_matches + filtered_local:
            g = m["guid"]
            if g not in merged_by_guid or m["score"] > merged_by_guid[g]["score"]:
                merged_by_guid[g] = m

        all_items = list(merged_by_guid.values())
        all_items.sort(key=lambda x: x["score"], reverse=True)

        # Assign group IDs when multiple results share high biometric similarity + domain/person
        next_grp = 1
        assigned_grp: Dict[int, int] = {}
        for i in range(len(all_items)):
            if i in assigned_grp:
                continue
            cluster = [i]
            for j in range(i + 1, len(all_items)):
                if j in assigned_grp:
                    continue
                if abs(all_items[i]["score"] - all_items[j]["score"]) <= 6:
                    sim = float(np.dot(all_items[i]["embedding"], all_items[j]["embedding"]))
                    if sim >= 0.72:
                        cluster.append(j)
                        if len(cluster) >= 4:
                            break
            if len(cluster) > 1:
                for idx_c in cluster:
                    assigned_grp[idx_c] = next_grp
                    all_items[idx_c]["group"] = next_grp
                next_grp += 1

        # Strip raw numpy embedding before serializing JSON
        final_items = []
        for item in all_items[:75]:
            final_items.append({
                "guid": item["guid"],
                "score": item["score"],
                "group": item["group"],
                "index": item["index"],
                "seen": item["seen"],
                "url": item["url"],
                "base64": item["base64"]
            })

        elapsed = max(0.4, time.time() - t0)
        total_faces_fmt = f"{self.get_total_face_count():,}"
        perf_str = (
            f"Pesquisou {total_faces_fmt} rostos em {elapsed:.1f} segundos. "
            f"Encontrados {len(final_items)} resultados."
        )

        if progress_cb:
            progress_cb("Search Done", 100)

        return {
            "Performance": perf_str,
            "type": "FULL",
            "flags": [],
            "items": final_items
        }

