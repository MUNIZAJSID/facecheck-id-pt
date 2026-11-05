import os
import io
import re
import time
import json
import uuid
import base64
import asyncio
from urllib.parse import unquote
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from biometric_engine import BiometricEngine
from osint_engine import OsintEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
IMG_DIR = os.path.join(BASE_DIR, "img")
ICON_CACHE_DIR = os.path.join(IMG_DIR, "icon")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(ICON_CACHE_DIR, exist_ok=True)

app = FastAPI(title="FaceCheck.ID - Motor de Busca por Reconhecimento Facial")
engine = BiometricEngine()
osint = OsintEngine(biometric_engine=engine)

# In-memory active search sessions (backed by SQLite for persistence/permalinks)
SEARCH_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _load_template(name: str) -> str:
    path = os.path.join(TEMPLATES_DIR, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Page not found</h1>"


def _render_index_html(request: Request) -> str:
    html = _load_template("index_pt.html")
    face_count = engine.get_total_face_count()
    html = re.sub(r'"facecount":\d+', f'"facecount":{face_count}', html)
    # Set TrialSearches and Credits high so user never hits trial limits
    html = html.replace('"Credits":0', '"Credits":9999')
    html = html.replace('"TrialSearches":200', '"TrialSearches":9999')
    return html


@app.get("/", response_class=HTMLResponse)
@app.get("/pt", response_class=HTMLResponse)
@app.get("/pt/", response_class=HTMLResponse)
async def home_page(request: Request):
    resp = HTMLResponse(_render_index_html(request))
    if not request.cookies.get("LID"):
        resp.set_cookie("LID", "VIP-USER-PT", max_age=365 * 86400)
    return resp


@app.get("/status")
async def get_status():
    return JSONResponse({
        "facecount": engine.get_total_face_count(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S.0000000Z", time.gmtime()),
        "online": True,
        "updating": False,
        "state": "online",
        "uptime": 99999,
        "js_hash": "JBJJSCDSFSDSL_670702D1859DD3E8",
        "js_hash_time": 1792525476,
        "Credits": 9999,
        "CreditsDes": "VIP Ilimitado",
        "TrialSearches": 9999,
        "LoggedIn": True,
        "message": None,
        "announcement": None
    })


@app.post("/captcha.png")
@app.get("/captcha.png")
async def get_captcha_png():
    id_cap = uuid.uuid4().hex
    target_x = int(np.random.randint(115, 195))
    target_y = int(np.random.randint(32, 85))
    pw, ph = 46, 46

    arr = np.zeros((155, 340, 4), dtype=np.uint8)
    for y in range(155):
        for x in range(278):
            arr[y, x] = [
                int(28 + (x / 278.0) * 110),
                int(45 + (y / 155.0) * 120),
                int(115 + ((x * 2 + y) % 90)),
                255
            ]
    img = Image.fromarray(arr, "RGBA")
    draw = ImageDraw.Draw(img)

    # Decorative geometric shapes so the puzzle alignment is visually obvious
    draw.ellipse([25, 18, 115, 108], fill=(224, 9, 76, 190), outline=(255, 255, 255, 210), width=2)
    draw.ellipse([105, 30, 200, 125], fill=(0, 180, 216, 190), outline=(255, 255, 255, 210), width=2)
    draw.rectangle([175, 22, 255, 118], fill=(255, 190, 11, 185), outline=(255, 255, 255, 210), width=2)

    # Extract puzzle piece from background
    piece = img.crop((target_x, target_y, target_x + pw, target_y + ph))
    # Draw darkened target slot on left background
    draw.rounded_rectangle(
        [target_x, target_y, target_x + pw, target_y + ph],
        radius=6,
        fill=(12, 14, 22, 220),
        outline=(255, 255, 255, 200),
        width=2
    )
    # Paste draggable puzzle piece onto right 62x155 strip
    img.paste(piece, (278 + 6, target_y))
    draw.rounded_rectangle(
        [278 + 6, target_y, 278 + 6 + pw, target_y + ph],
        radius=6,
        outline=(255, 255, 255, 255),
        width=2
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    js_eval = f"window.encrypt=function(b){{return b;}};window._gp=null;this.x={target_x};"
    trailer = f"IDCA {id_cap}|{js_eval}\x00".encode("ascii")
    raw = bytearray(png_bytes + trailer)

    w = b"FUCKOFF!!!"
    b = 3
    f = 257
    n = len(raw)
    while f < n:
        raw[f] ^= w[f % len(w)]
        if f > n - 500:
            b = 1
        f += b

    return Response(content=bytes(raw), media_type="image/png")


@app.post("/chaptcha_verify")
async def verify_captcha(request: Request):
    return JSONResponse({"result": "OK"})


def _get_or_create_session(id_search: str) -> Dict[str, Any]:
    if not id_search or id_search == "NEW" or len(id_search) < 8:
        id_search = uuid.uuid4().hex[:12]

    if id_search in SEARCH_SESSIONS:
        return SEARCH_SESSIONS[id_search]

    # Check SQLite persistence
    conn = engine._get_conn()
    row = conn.execute("SELECT * FROM searches WHERE id_search = ?", (id_search,)).fetchone()
    conn.close()
    if row:
        inp = json.loads(row["input_json"]) if row["input_json"] else []
        out = json.loads(row["output_json"]) if row["output_json"] else None
        sess = {
            "id_search": id_search,
            "created_at": row["created_at"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "probes": [],
            "input_public": inp,
            "output": out,
            "task": None
        }
        SEARCH_SESSIONS[id_search] = sess
        return sess

    sess = {
        "id_search": id_search,
        "created_at": int(time.time()),
        "status": "drag",
        "progress": 0,
        "message": "",
        "probes": [],
        "input_public": [],
        "output": None,
        "task": None
    }
    SEARCH_SESSIONS[id_search] = sess
    return sess


def _save_session_to_db(sess: Dict[str, Any]):
    try:
        conn = engine._get_conn()
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO searches (id_search, created_at, status, progress, message, input_json, output_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sess["id_search"],
                sess["created_at"],
                sess["status"],
                sess["progress"],
                sess["message"],
                json.dumps(sess["input_public"]),
                json.dumps(sess["output"]) if sess["output"] else ""
            ))
        conn.close()
    except Exception:
        pass


@app.post("/api/upload_pic")
async def api_upload_pic(
    images: List[UploadFile] = File(...),
    id_search: str = Form("NEW"),
    reset_prev_images: str = Form("false")
):
    sess = _get_or_create_session(id_search)
    if reset_prev_images.lower() == "true":
        sess["probes"] = []
        sess["input_public"] = []
        sess["output"] = None
        sess["status"] = "drag"

    new_inputs = []
    for up in images[:3]:
        img_bytes = await up.read()
        if not img_bytes:
            continue
        extracted = engine.extract_probe_faces(img_bytes)
        for item in extracted:
            if len(sess["probes"]) >= 3:
                break
            sess["probes"].append(item)
            pub_item = {
                "id_pic": item["id_pic"],
                "base64": item["base64"],
                "svg_anim": item["svg_anim"],
                "url_source": None
            }
            sess["input_public"].append(pub_item)
            new_inputs.append(pub_item)

    if not new_inputs and not sess["input_public"]:
        return JSONResponse({
            "id_search": sess["id_search"],
            "error": "[Nenhum Rosto Detectado] Não foi possível detectar um rosto nítido nesta imagem. Tente enviar outra foto.",
            "code": "NO_FACE"
        })

    _save_session_to_db(sess)
    return JSONResponse({
        "id_search": sess["id_search"],
        "input": new_inputs,
        "error": None,
        "code": None
    })


@app.post("/api/delete_pic")
async def api_delete_pic(id_search: str = "", id_pic: str = ""):
    if id_search in SEARCH_SESSIONS:
        sess = SEARCH_SESSIONS[id_search]
        sess["probes"] = [p for p in sess["probes"] if p["id_pic"] != id_pic]
        sess["input_public"] = [p for p in sess["input_public"] if p["id_pic"] != id_pic]
        _save_session_to_db(sess)
    return JSONResponse({"code": "DELETED", "error": None})


@app.post("/api/search")
async def api_search(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    id_search = body.get("id_search") or "NEW"
    sess = _get_or_create_session(id_search)

    # If search already finished, return final output immediately
    if sess.get("output") is not None:
        resp = JSONResponse({
            "id_search": sess["id_search"],
            "new_from": sess["created_at"] - 86400,
            "input": sess["input_public"],
            "output": sess["output"],
            "error": None
        })
        resp.headers["credits"] = "9999"
        return resp

    # If background search task hasn't started yet, start it
    if sess.get("task") is None:
        if not sess.get("probes") and sess.get("input_public"):
            # Reconstruct probe from base64 if loaded from DB
            for pub in sess["input_public"]:
                b64 = pub.get("base64", "")
                if "," in b64:
                    raw_bytes = base64.b64decode(b64.split(",", 1)[1])
                    sess["probes"].extend(engine.extract_probe_faces(raw_bytes))

        if not sess.get("probes"):
            return JSONResponse({
                "id_search": sess["id_search"],
                "error": "[Sem Foto] Por favor, envie uma foto primeiro.",
                "code": "SOMETHIN_WONG"
            })

        sess["status"] = "searching"
        sess["progress"] = 10
        sess["message"] = "Waiting in queue 1"

        def update_prog(msg: str, prog: int):
            sess["message"] = msg
            sess["progress"] = prog

        async def _worker():
            try:
                out = await engine.run_full_search(sess["id_search"], sess["probes"], update_prog)
                sess["output"] = out
                sess["status"] = "done"
                sess["progress"] = 100
                sess["message"] = "Search Done"
                _save_session_to_db(sess)
            except Exception as e:
                sess["status"] = "error"
                sess["message"] = str(e)

        sess["task"] = asyncio.create_task(_worker())

    # Wait briefly so client polling sees smooth progress updates without hammering
    for _ in range(6):
        if sess.get("output") is not None:
            break
        await asyncio.sleep(0.18)

    if sess.get("output") is not None:
        resp = JSONResponse({
            "id_search": sess["id_search"],
            "new_from": sess["created_at"] - 86400,
            "input": sess["input_public"],
            "output": sess["output"],
            "error": None
        })
        resp.headers["credits"] = "9999"
        return resp

    return JSONResponse({
        "id_search": sess["id_search"],
        "message": sess["message"] or "Searching Faces",
        "progress": sess["progress"] or 25,
        "input": sess["input_public"],
        "output": None,
        "error": None
    })


@app.post("/api/get_permalink")
async def api_get_permalink(request: Request, id_search: str = ""):
    if not id_search:
        id_search = request.query_params.get("id_search", "")
    conn = engine._get_conn()
    with conn:
        conn.execute("UPDATE searches SET permanent = 1 WHERE id_search = ?", (id_search,))
    conn.close()
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse({
        "url": f"{base_url}/pt#{id_search}",
        "error": None
    })


@app.post("/api/export/{fmt}/{id_search}")
async def api_export_init(fmt: str, id_search: str):
    return JSONResponse({
        "code": "REDIR",
        "message": f"/api/download_export/{fmt}/{id_search}",
        "error": None
    })


@app.get("/api/download_export/{fmt}/{id_search}")
async def api_download_export(fmt: str, id_search: str):
    sess = _get_or_create_session(id_search)
    items = (sess.get("output") or {}).get("items", [])

    if fmt.lower() == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, h - 50, f"FaceCheck.ID - Relatorio de Busca Facial (#{id_search})")
        c.setFont("Helvetica", 10)
        c.drawString(40, h - 68, f"Total de correspondencias encontradas: {len(items)}")
        y = h - 95
        for idx, it in enumerate(items, 1):
            if y < 60:
                c.showPage()
                y = h - 50
                c.setFont("Helvetica", 10)
            score = it.get("score", 0)
            url = it.get("url", "")
            tier = "Certa" if score >= 90 else "Confiavel" if score >= 83 else "Incerta" if score >= 70 else "Fraca"
            c.setFont("Helvetica-Bold", 10)
            c.drawString(40, y, f"{idx}. Score: {score}/100 ({tier})")
            c.setFont("Helvetica", 9)
            c.drawString(185, y, url[:85])
            y -= 18
        c.save()
        return Response(
            content=buf.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="FaceCheck_{id_search}.pdf"'}
        )
    else:
        lines = ["Posicao,Pontuacao (0-100),Classificacao,URL da Fonte"]
        for idx, it in enumerate(items, 1):
            score = it.get("score", 0)
            tier = "Correspondencia Certa" if score >= 90 else "Correspondencia Confiavel" if score >= 83 else "Correspondencia Incerta" if score >= 70 else "Correspondencia Fraca"
            url = it.get("url", "").replace('"', '""')
            lines.append(f'{idx},{score},"{tier}","{url}"')
        csv_bytes = "\ufeff".join([""]) + "\n".join(lines)
        return Response(
            content=csv_bytes.encode("utf-8-sig"),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="FaceCheck_{id_search}.csv"'}
        )


@app.get("/indirect_upload/")
@app.get("/indirect_upload")
async def indirect_upload(request: Request):
    target_url = unquote(request.url.query)
    if not target_url.startswith("http"):
        return Response(status_code=400, content=b"Invalid URL")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(verify=False) as client:
            r = await client.get(target_url, headers=headers, follow_redirects=True, timeout=10.0)
            if r.status_code == 200:
                ct = r.headers.get("content-type", "image/jpeg")
                return Response(content=r.content, media_type=ct)
    except Exception:
        pass
    return Response(status_code=404, content=b"Image not found")


# Domain Favicon Generator & Cache for /img/icon/{encoded}.webp
BRAND_COLORS = {
    "instagram.com": (225, 48, 108),
    "tiktok.com": (25, 25, 25),
    "youtube.com": (255, 0, 0),
    "twitter.com": (29, 161, 242),
    "x.com": (20, 20, 20),
    "facebook.com": (24, 119, 242),
    "linkedin.com": (10, 102, 194),
    "pinterest.com": (230, 0, 35),
    "reddit.com": (255, 69, 0),
    "t.me": (0, 136, 204),
    "onlyfans.com": (0, 175, 240),
    "vk.com": (0, 119, 255),
    "steamcommunity.com": (23, 26, 33),
}


@app.get("/img/icon/{encoded_name}.webp")
async def get_domain_icon(encoded_name: str):
    cache_file = os.path.join(ICON_CACHE_DIR, f"{encoded_name}.webp")
    if os.path.exists(cache_file):
        return FileResponse(cache_file, media_type="image/webp")

    # Decode domain from Bundle.js get_icon format: 2 checksum chars + base64(domain)
    domain = "web.com"
    try:
        b64_part = unquote(encoded_name)[2:]
        # Add base64 padding if needed
        pad = (4 - len(b64_part) % 4) % 4
        domain = base64.b64decode(b64_part + ("=" * pad)).decode("utf-8", errors="ignore").strip().lower()
    except Exception:
        pass

    # Try fetching real favicon via Google S2 Favicon service
    try:
        async with httpx.AsyncClient(verify=False) as client:
            r = await client.get(
                f"https://www.google.com/s2/favicons?domain={domain}&sz=64",
                timeout=3.5,
                follow_redirects=True
            )
            if r.status_code == 200 and len(r.content) > 100:
                pil_ico = Image.open(io.BytesIO(r.content)).convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
                pil_ico.save(cache_file, format="WEBP", quality=90)
                return FileResponse(cache_file, media_type="image/webp")
    except Exception:
        pass

    # Fallback: generate crisp brand badge WebP
    color = BRAND_COLORS.get(domain, (224, 9, 76))
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([4, 4, 60, 60], radius=14, fill=color + (255,))
    letter = (domain[0] if domain else "W").upper()
    draw.text((22, 14), letter, fill=(255, 255, 255, 255))
    img.save(cache_file, format="WEBP", quality=90)
    return FileResponse(cache_file, media_type="image/webp")


# Account, Credits, Export, and Photo Removal Endpoints
@app.post("/api/login_account")
async def api_login_account(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    if body.get("Account_ID") == "NEW":
        secret_id = uuid.uuid4().hex[:14].upper()
        conn = engine._get_conn()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO accounts (account_id, credits, created_at) VALUES (?, ?, ?)",
                (secret_id, 9999, int(time.time()))
            )
        conn.close()
        resp = JSONResponse({"result": "OK", "plain_text": secret_id, "error": None})
        resp.set_cookie("LID", secret_id, max_age=365 * 86400)
        return resp

    plain_text = (body.get("plain_text") or "").strip().upper()
    if not plain_text:
        return JSONResponse({"result": "ERR", "error": "[ID Inválido] Digite seu ID Secreto da conta."})

    conn = engine._get_conn()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO accounts (account_id, credits, created_at) VALUES (?, ?, ?)",
            (plain_text, 9999, int(time.time()))
        )
    conn.close()
    resp = JSONResponse({"result": "OK", "message": "LOGGED_IN", "error": None})
    resp.set_cookie("LID", plain_text, max_age=365 * 86400)
    return resp


@app.post("/api/buy_package")
async def api_buy_package(package: str = "", paymethod: str = ""):
    return JSONResponse({
        "result": "OK",
        "message": "REDIR",
        "code": "/dashboard?purchased=" + package
    })


@app.post("/api/aff_set_promo_code")
async def api_promo_code(request: Request):
    return JSONResponse({"result": "OK", "code": "OK"})


@app.post("/api/id-verification/submit")
async def api_id_verification_submit(
    ids: str = Form(""),
    id_captcha: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    guid_list = [g.strip() for g in ids.split(",") if g.strip()]
    now_ts = int(time.time())
    conn = engine._get_conn()
    with conn:
        for g in guid_list:
            conn.execute(
                "INSERT OR REPLACE INTO removed_guids (guid, removed_at, reason) VALUES (?, ?, ?)",
                (g, now_ts, "ID Verification Removal")
            )
            conn.execute("DELETE FROM faces WHERE guid = ?", (g,))
    conn.close()
    session_id = uuid.uuid4().hex[:12]
    return JSONResponse({"sessionId": session_id, "status": "APPROVED"})


@app.post("/SelfieGate", response_class=HTMLResponse)
async def selfie_gate(ids: str = Form(""), id_captcha: str = Form("")):
    guid_list = [g.strip() for g in ids.split(",") if g.strip()]
    now_ts = int(time.time())
    conn = engine._get_conn()
    with conn:
        for g in guid_list:
            conn.execute(
                "INSERT OR REPLACE INTO removed_guids (guid, removed_at, reason) VALUES (?, ?, ?)",
                (g, now_ts, "Selfie Check Removal")
            )
            conn.execute("DELETE FROM faces WHERE guid = ?", (g,))
    conn.close()
    return RedirectResponse(url=f"/SelfieStatus/{uuid.uuid4().hex[:12]}?removed={len(guid_list)}", status_code=303)


@app.get("/SelfieStatus/{session_id}", response_class=HTMLResponse)
async def selfie_status(session_id: str, removed: int = 1):
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="utf-8">
<title>FaceCheck - Solicitação de Remoção Concluída</title>
<link rel="stylesheet" href="/static/Bundle_FBSSJDL_5674728C717A4A99.css">
</head>
<body class="light">
<div class="container" style="max-width:720px;margin:60px auto;text-align:center;">
  <a href="/pt"><img src="/img/FaceCheck_logo-639029188200000000-FRESH.svg" style="max-width:340px;margin-bottom:30px;"></a>
  <div style="background:#1e2430;color:#fff;padding:36px;border-radius:14px;box-shadow:0 8px 28px rgba(0,0,0,0.25);">
    <h2 style="color:#2ecc71;margin-bottom:16px;">✓ Solicitação de Remoção Aprovada (#{session_id})</h2>
    <p style="font-size:18px;line-height:1.6;">As fotos selecionadas foram permanentemente removidas e bloqueadas do índice biométrico do FaceCheck.ID.</p>
    <a href="/pt" class="button-24" style="margin-top:24px;font-size:20px;padding:12px 28px;color:#fff;">Voltar para Nova Pesquisa</a>
  </div>
</div>
</body>
</html>""")


@app.post("/jserror")
async def js_error():
    return JSONResponse({"result": "OK"})


# Subpages routing
@app.get("/VerificationWidget", response_class=HTMLResponse)
async def verification_widget(lang: str = "pt"):
    return HTMLResponse(_load_template("verification_widget_pt.html"))


@app.get("/pt/Face-Search/FAQ", response_class=HTMLResponse)
@app.get("/Face-Search/FAQ", response_class=HTMLResponse)
async def faq_page():
    return HTMLResponse(_load_template("faq.html"))


@app.get("/pt/buy", response_class=HTMLResponse)
@app.get("/buy", response_class=HTMLResponse)
async def buy_page():
    return HTMLResponse(_load_template("buy.html"))


@app.get("/pt/RemoveMyPhotos", response_class=HTMLResponse)
@app.get("/RemoveMyPhotos", response_class=HTMLResponse)
async def remove_photos_page():
    return HTMLResponse(_load_template("remove.html"))


@app.get("/pt/Face-Search/ReverseImageSearchTips", response_class=HTMLResponse)
@app.get("/Face-Search/ReverseImageSearchTips", response_class=HTMLResponse)
async def tips_page():
    return HTMLResponse(_load_template("tips.html"))


@app.get("/pt/Face-Search/API", response_class=HTMLResponse)
@app.get("/Face-Search/API", response_class=HTMLResponse)
async def api_docs_page():
    return HTMLResponse(_load_template("api.html"))


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(_load_template("login.html"))


@app.get("/Face-Search/Privacy", response_class=HTMLResponse)
@app.get("/pt/Face-Search/Privacy", response_class=HTMLResponse)
async def privacy_page():
    return HTMLResponse(_load_template("privacy.html"))


@app.get("/Face-Search/Terms", response_class=HTMLResponse)
@app.get("/pt/Face-Search/Terms", response_class=HTMLResponse)
async def terms_page():
    return HTMLResponse(_load_template("terms.html"))


@app.get("/pt/compare-reverse-image-search-engines/", response_class=HTMLResponse)
@app.get("/compare-reverse-image-search-engines/", response_class=HTMLResponse)
async def compare_page():
    return HTMLResponse(_load_template("compare.html"))


@app.get("/pt/Procura-de-Rosto-em-Imagem-Reversa", response_class=HTMLResponse)
async def procura_page():
    return HTMLResponse(_load_template("procura.html"))


@app.get("/osint", response_class=HTMLResponse)
@app.get("/pt/osint", response_class=HTMLResponse)
async def osint_page():
    return HTMLResponse(_load_template("osint.html"))


@app.post("/api/osint/search")
async def api_osint_search(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    query = (body.get("query") or "").strip()
    data = await osint.search_identity(query)
    return JSONResponse(data)


@app.post("/api/osint/photo")
async def api_osint_photo(image: UploadFile = File(...)):
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"error": "Imagem vazia", "profiles": []})
    data = await osint.identify_from_photo(img_bytes)
    return JSONResponse(data)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    lid = request.cookies.get("LID", "VIP-ADMIN-ACCOUNT")
    conn = engine._get_conn()
    searches = conn.execute(
        "SELECT id_search, created_at, status, input_json, output_json FROM searches ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    local_face_cnt = conn.execute("SELECT COUNT(*) as c FROM faces").fetchone()["c"]
    conn.close()

    cards_html = []
    for s in searches:
        sid = s["id_search"]
        inp = json.loads(s["input_json"]) if s["input_json"] else []
        out = json.loads(s["output_json"]) if s["output_json"] else {}
        items_cnt = len(out.get("items", [])) if isinstance(out, dict) else 0
        thumb = inp[0]["base64"] if inp else "/img/favicon.svg"
        dt_str = time.strftime("%d/%m/%Y %H:%M", time.localtime(s["created_at"]))
        cards_html.append(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:12px 18px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:14px;">
            <img src="{thumb}" style="width:64px;height:64px;border-radius:8px;object-fit:cover;">
            <div style="text-align:left;">
              <div style="font-weight:bold;font-size:16px;">Pesquisa #{sid}</div>
              <div style="font-size:13px;opacity:0.75;">Data: {dt_str} • {items_cnt} correspondências faciais</div>
            </div>
          </div>
          <a href="/pt#{sid}" class="button-24" style="font-size:15px;padding:10px 18px;color:#fff;">Ver Resultados</a>
        </div>
        """)

    history_block = "".join(cards_html) if cards_html else "<p>Nenhuma pesquisa recente ainda.</p>"
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="utf-8">
<title>FaceCheck.ID - Painel da Conta & Histórico</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/static/Bundle_FBSSJDL_5674728C717A4A99.css">
</head>
<body class="dark">
<header>
  <nav class="navbar navbar-expand-sm navbar-light border-bottom box-shadow mb-3">
    <div id="navbar" style="display:flex;align-items:center;justify-content:space-between;">
      <a class="navbar-brand logolink" href="/pt"><img src="/img/FaceCheck_logo-639029188200000000-FRESH.svg" class="logo" alt="FaceCheck Home" style="max-width:240px;"></a>
      <div>
        <a href="/pt" class="button-24" style="font-size:16px;padding:10px 20px;color:#fff;">Nova Pesquisa Facial</a>
      </div>
    </div>
  </nav>
</header>
<div class="container" style="max-width:920px;margin:30px auto;">
  <div style="background:#1b2028;border-radius:14px;padding:28px;margin-bottom:24px;border:1px solid #2e3646;">
    <h2 style="margin-top:0;">Painel da Conta FaceCheck.ID</h2>
    <p><b>ID Secreto da Conta:</b> <code style="font-size:18px;color:#00f2ff;">{lid}</code></p>
    <p><b>Saldo de Créditos:</b> <span style="color:#2ecc71;font-weight:bold;">9.999 Créditos (VIP Desbloqueado — Todos os links liberados)</span></p>
    <p><b>Índice Neural Local (SFace 128D + YuNet):</b> {engine.get_total_face_count():,} rostos online ({local_face_cnt} rostos indexados localmente em cache)</p>
  </div>
  <div style="background:#1b2028;border-radius:14px;padding:28px;border:1px solid #2e3646;">
    <h3 style="margin-top:0;margin-bottom:18px;">Histórico de Pesquisas Recentes</h3>
    {history_block}
  </div>
</div>
</body>
</html>""")


# Dynamic proxy/cache fallback for any missing /img/ or /static/ asset
@app.get("/img/{file_path:path}")
async def serve_img(file_path: str):
    local_path = os.path.join(IMG_DIR, file_path.replace("/", os.sep))
    if os.path.exists(local_path) and os.path.isfile(local_path):
        return FileResponse(local_path)

    # Attempt on-the-fly fetch & cache from facecheck.id
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        async with httpx.AsyncClient(verify=False) as client:
            r = await client.get(
                f"https://facecheck.id/img/{file_path}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6.0,
                follow_redirects=True
            )
            if r.status_code == 200 and len(r.content) > 50:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                return FileResponse(local_path)
    except Exception:
        pass
    return FileResponse(os.path.join(IMG_DIR, "favicon.svg"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Language root routes (/en, /es, /fr, etc.)
@app.get("/{lang_code}", response_class=HTMLResponse)
async def lang_home(request: Request, lang_code: str):
    if len(lang_code) == 2:
        return HTMLResponse(_render_index_html(request))
    return RedirectResponse("/pt")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
