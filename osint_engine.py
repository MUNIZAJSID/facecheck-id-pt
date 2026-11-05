import re
import json
import hashlib
import asyncio
from urllib.parse import urlparse, quote_plus, unquote
from typing import Dict, Any, List, Optional, Tuple

import cv2
import httpx
from bs4 import BeautifulSoup

# Platform definitions for categorization and direct username checks
PLATFORM_CATEGORIES = {
    "instagram.com": ("Instagram", "Redes Sociais", "#E1306C"),
    "tiktok.com": ("TikTok", "Redes Sociais", "#00F2FE"),
    "x.com": ("X (Twitter)", "Redes Sociais", "#1DA1F2"),
    "twitter.com": ("Twitter / X", "Redes Sociais", "#1DA1F2"),
    "threads.net": ("Threads", "Redes Sociais", "#FFFFFF"),
    "youtube.com": ("YouTube", "Vídeo & Streaming", "#FF0000"),
    "facebook.com": ("Facebook", "Redes Sociais", "#1877F2"),
    "twitch.tv": ("Twitch", "Vídeo & Streaming", "#9146FF"),
    "kick.com": ("Kick", "Vídeo & Streaming", "#53FC18"),
    "reddit.com": ("Reddit", "Comunidades & Fóruns", "#FF4500"),
    "t.me": ("Telegram", "Comunidades & Canais", "#0088CC"),
    "telegram.me": ("Telegram", "Comunidades & Canais", "#0088CC"),
    "vk.com": ("VK", "Redes Sociais", "#0077FF"),
    "pinterest.com": ("Pinterest", "Redes Sociais", "#E60023"),
    "linkedin.com": ("LinkedIn", "Profissional", "#0A66C2"),
    "onlyfans.com": ("OnlyFans", "Plataformas de Criadores", "#00AFF0"),
    "fansly.com": ("Fansly", "Plataformas de Criadores", "#1DA1F2"),
    "fapello.com": ("Fapello", "Plataformas de Criadores", "#FF2E93"),
    "fancentro.com": ("FanCentro", "Plataformas de Criadores", "#FF5A5F"),
    "loyalfans.com": ("LoyalFans", "Plataformas de Criadores", "#D4AF37"),
    "manyvids.com": ("ManyVids", "Plataformas de Criadores", "#9B51E0"),
    "patreon.com": ("Patreon", "Plataformas de Criadores", "#FF424D"),
    "throne.com": ("Throne Wishlist", "Plataformas de Criadores", "#8B5CF6"),
    "ko-fi.com": ("Ko-fi", "Plataformas de Criadores", "#29ABE0"),
    "linktr.ee": ("Linktree", "Agregadores de Bio", "#43E660"),
    "allmylinks.com": ("AllMyLinks", "Agregadores de Bio", "#FF0066"),
    "beacons.ai": ("Beacons.ai", "Agregadores de Bio", "#FF5722"),
    "solo.to": ("Solo.to", "Agregadores de Bio", "#3B82F6"),
    "carrd.co": ("Carrd", "Agregadores de Bio", "#6366F1"),
    "imdb.com": ("IMDb", "Catálogos & Elenco", "#F5C518"),
    "iafd.com": ("IAFD Database", "Catálogos & Elenco", "#EAB308"),
    "babepedia.com": ("Babepedia", "Catálogos & Elenco", "#EC4899"),
    "indexxx.com": ("Indexxx", "Catálogos & Elenco", "#F43F5E"),
    "freeones.com": ("FreeOnes", "Catálogos & Elenco", "#A855F7"),
    "famousbirthdays.com": ("FamousBirthdays", "Catálogos & Elenco", "#10B981"),
    "github.com": ("GitHub", "Outras Plataformas", "#9CA3AF"),
    "steamcommunity.com": ("Steam", "Outras Plataformas", "#38BDF8"),
    "open.spotify.com": ("Spotify", "Outras Plataformas", "#1DB954"),
    "soundcloud.com": ("SoundCloud", "Outras Plataformas", "#FF5500"),
}

# Direct URL patterns for fast @username existence probing
DIRECT_USERNAME_PROBES = [
    ("Instagram", "Redes Sociais", "https://www.instagram.com/{u}/", "#E1306C"),
    ("TikTok", "Redes Sociais", "https://www.tiktok.com/@{u}", "#00F2FE"),
    ("X (Twitter)", "Redes Sociais", "https://x.com/{u}", "#1DA1F2"),
    ("YouTube", "Vídeo & Streaming", "https://www.youtube.com/@{u}", "#FF0000"),
    ("Twitch", "Vídeo & Streaming", "https://www.twitch.tv/{u}", "#9146FF"),
    ("Reddit", "Comunidades & Fóruns", "https://www.reddit.com/user/{u}", "#FF4500"),
    ("Telegram", "Comunidades & Canais", "https://t.me/{u}", "#0088CC"),
    ("Linktree", "Agregadores de Bio", "https://linktr.ee/{u}", "#43E660"),
    ("AllMyLinks", "Agregadores de Bio", "https://allmylinks.com/{u}", "#FF0066"),
    ("Beacons.ai", "Agregadores de Bio", "https://beacons.ai/{u}", "#FF5722"),
    ("Patreon", "Plataformas de Criadores", "https://www.patreon.com/{u}", "#FF424D"),
    ("Pinterest", "Redes Sociais", "https://www.pinterest.com/{u}/", "#E60023"),
    ("GitHub", "Outras Plataformas", "https://github.com/{u}", "#9CA3AF"),
    ("Steam", "Outras Plataformas", "https://steamcommunity.com/id/{u}", "#38BDF8"),
    ("SoundCloud", "Outras Plataformas", "https://soundcloud.com/{u}", "#FF5500"),
    ("VK", "Redes Sociais", "https://vk.com/{u}", "#0077FF"),
]

HANDLE_REGEXES = [
    (r"instagram\.com/(?:p/|reel/|stories/)?([A-Za-z0-9_.]{2,30})", "Instagram"),
    (r"tiktok\.com/@([A-Za-z0-9_.]{2,32})", "TikTok"),
    (r"(?:x|twitter)\.com/([A-Za-z0-9_]{2,25})", "X / Twitter"),
    (r"threads\.net/@([A-Za-z0-9_.]{2,30})", "Threads"),
    (r"onlyfans\.com/([A-Za-z0-9_.-]{2,35})", "OnlyFans"),
    (r"fansly\.com/([A-Za-z0-9_.-]{2,35})", "Fansly"),
    (r"fapello\.com/([A-Za-z0-9_.-]{2,35})", "Fapello"),
    (r"linktr\.ee/([A-Za-z0-9_.-]{2,35})", "Linktree"),
    (r"allmylinks\.com/([A-Za-z0-9_.-]{2,35})", "AllMyLinks"),
    (r"t\.me/(?:s/)?([A-Za-z0-9_]{3,35})", "Telegram"),
    (r"reddit\.com/(?:user|u|r)/([A-Za-z0-9_-]{2,30})", "Reddit"),
    (r"youtube\.com/@([A-Za-z0-9_.-]{2,35})", "YouTube"),
    (r"twitch\.tv/([A-Za-z0-9_]{2,30})", "Twitch"),
    (r"babepedia\.com/babe/([A-Za-z0-9_]{2,45})", "Babepedia"),
    (r"indexxx\.com/m/([A-Za-z0-9_-]{2,45})", "Indexxx"),
    (r"freeones\.com/([A-Za-z0-9_-]{2,45})", "FreeOnes"),
]

IGNORED_HANDLES = {
    "popular", "explore", "p", "reel", "reels", "stories", "share", "login",
    "intent", "hashtag", "search", "status", "watch", "channel", "results",
    "user", "r", "u", "s", "home", "about", "privacy", "terms"
}


class OsintEngine:
    def __init__(self, biometric_engine=None):
        self.bio = biometric_engine
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cookie": "yp=1999999999.sp.family%3A0; fyandex=0;",
        }

    def _classify_url(self, url: str, title: str = "") -> Optional[Dict[str, Any]]:
        if not url or not url.startswith("http"):
            return None
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        # Normalize mobile/regional subdomains
        for base_dom in PLATFORM_CATEGORIES:
            if domain == base_dom or domain.endswith("." + base_dom):
                domain = base_dom
                break

        plat_name, category, color = PLATFORM_CATEGORIES.get(
            domain, (domain.split(".")[0].capitalize(), "Web & Menções", "#64748B")
        )

        # Extract @handle if present
        handle = ""
        for pattern, _ in HANDLE_REGEXES:
            m = re.search(pattern, url, re.IGNORECASE)
            if m:
                cand = unquote(m.group(1)).strip("/")
                if cand.lower() not in IGNORED_HANDLES:
                    handle = "@" + cand
                    break

        return {
            "url": url,
            "domain": domain,
            "platform": plat_name,
            "category": category,
            "color": color,
            "handle": handle,
            "title": title or plat_name,
        }

    async def _ddg_search(self, client: httpx.AsyncClient, query: str) -> List[Dict[str, str]]:
        results = []
        try:
            r = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kp": "-2"},
                headers=self.headers,
                follow_redirects=True,
                timeout=8.0,
            )
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                for item in soup.select(".result"):
                    a = item.select_one("a.result__a")
                    snip = item.select_one(".result__snippet")
                    if a and a.get("href"):
                        href = a["href"]
                        if href.startswith("http"):
                            results.append({
                                "title": a.get_text(strip=True),
                                "url": href,
                                "snippet": snip.get_text(strip=True) if snip else "",
                            })
        except Exception:
            pass
        return results

    async def _yandex_ru_search(self, client: httpx.AsyncClient, query: str) -> Tuple[List[Dict[str, str]], List[str]]:
        results = []
        thumbs = []
        try:
            url = f"https://yandex.ru/images/search?text={quote_plus(query)}&family=no"
            r = await client.get(url, headers=self.headers, follow_redirects=True, timeout=8.0)
            if r.status_code == 200:
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
                        title = snip.get("title", "")
                        img = ent.get("image", "")
                        if img.startswith("//"):
                            img = "https:" + img
                        if img and len(thumbs) < 12:
                            thumbs.append(img)
                        if page_url:
                            results.append({
                                "title": title,
                                "url": page_url,
                                "snippet": snip.get("domain", ""),
                                "thumb": img,
                            })
        except Exception:
            pass
        return results, thumbs

    async def _probe_direct_username(self, client: httpx.AsyncClient, username: str) -> List[Dict[str, Any]]:
        clean_u = username.lstrip("@").strip().replace(" ", "")
        if not clean_u or len(clean_u) < 2 or not re.match(r"^[A-Za-z0-9_.-]+$", clean_u):
            return []

        sem = asyncio.Semaphore(10)

        async def check_one(plat_name: str, category: str, url_tpl: str, color: str):
            url = url_tpl.format(u=clean_u)
            async with sem:
                try:
                    r = await client.get(url, headers=self.headers, follow_redirects=True, timeout=4.5)
                    if r.status_code == 200:
                        txt_low = r.text[:4000].lower()
                        # Filter common soft-404 phrases
                        if any(p in txt_low for p in [
                            "page not found", "esta página não está disponível",
                            "nobody on reddit goes by that name", "sorry, this page isn't available",
                            "this account doesn't exist"
                        ]):
                            return None
                        return {
                            "url": url,
                            "domain": urlparse(url).netloc.lower().replace("www.", ""),
                            "platform": plat_name,
                            "category": category,
                            "color": color,
                            "handle": "@" + clean_u,
                            "title": f"{plat_name} (@{clean_u})",
                            "snippet": "Perfil verificado diretamente por @username",
                        }
                except Exception:
                    pass
                return None

        tasks = [check_one(*p) for p in DIRECT_USERNAME_PROBES]
        res = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in res if isinstance(r, dict)]

    async def identify_from_photo(self, img_bytes: bytes) -> Dict[str, Any]:
        """
        Detects the face in the uploaded photo, runs reverse visual entity discovery,
        identifies the person's name/stage name/handles, and then launches full social discovery.
        """
        detected_names: List[str] = []
        raw_site_matches: List[Dict[str, Any]] = []
        face_thumb_b64 = ""

        if self.bio is not None:
            probes = self.bio.extract_probe_faces(img_bytes)
            if probes:
                face_thumb_b64 = probes[0]["base64"]
                ok, crop_buf = cv2.imencode(".jpg", probes[0]["crop_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok:
                    img_bytes_crop = crop_buf.tobytes()
                else:
                    img_bytes_crop = img_bytes
            else:
                img_bytes_crop = img_bytes
        else:
            img_bytes_crop = img_bytes

        async with httpx.AsyncClient(verify=False) as client:
            if self.bio is not None:
                cands1, tags1 = await self.bio._yandex_cbir_discover(client, img_bytes)
                cands2, tags2 = await self.bio._yandex_cbir_discover(client, img_bytes_crop)
                all_cands = cands1 + cands2
                all_tags = tags1 + tags2

                # Also query Google Lens for Latin name recognition
                try:
                    r_gl = await client.post(
                        "https://lens.google.com/v3/upload",
                        files={"encoded_image": ("face.jpg", img_bytes, "image/jpeg")},
                        headers=self.headers,
                        follow_redirects=True,
                        timeout=8.0,
                    )
                    if r_gl.status_code == 200:
                        gl_names = re.findall(r'"([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){1,2})"', r_gl.text)
                        for gn in gl_names[:6]:
                            if gn.lower() not in {"google lens", "google images", "search image", "visual matches"}:
                                detected_names.append(gn)
                except Exception:
                    pass

                inferred = self.bio._infer_person_query(all_cands, all_tags)
                if inferred:
                    detected_names.insert(0, inferred)

                for t in all_tags:
                    if t not in detected_names:
                        detected_names.append(t)

                for c in all_cands:
                    raw_site_matches.append({
                        "title": c.get("title", ""),
                        "url": c.get("page_url", ""),
                        "snippet": "Encontrado via reconhecimento visual da foto",
                        "thumb": c.get("img_url", ""),
                    })

        # Also extract any @handle or model name directly from matched URLs
        for m in raw_site_matches:
            u = m.get("url", "")
            for pattern, _ in HANDLE_REGEXES:
                match = re.search(pattern, u, re.IGNORECASE)
                if match:
                    cand_h = unquote(match.group(1)).replace("_", " ").replace("-", " ").strip()
                    if cand_h.lower() not in IGNORED_HANDLES and len(cand_h) >= 3:
                        detected_names.append(cand_h)

        primary_query = detected_names[0] if detected_names else ""
        search_data = await self.search_identity(primary_query, extra_raw_items=raw_site_matches)
        search_data["face_thumb"] = face_thumb_b64
        search_data["detected_names"] = list(dict.fromkeys(detected_names))[:8]
        return search_data

    async def search_identity(
        self, query: str, extra_raw_items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        extra_raw_items = extra_raw_items or []

        # Detect query type: Email, Phone, @Username, or Person/Actress Name
        is_email = bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", query))
        digits_only = re.sub(r"\D", "", query)
        is_phone = len(digits_only) >= 8 and bool(re.match(r"^[\d\s\+\-\(\)]+$", query))
        is_handle = query.startswith("@") or (" " not in query and len(query) >= 3 and not is_email and not is_phone)

        discovered_items: List[Dict[str, Any]] = []
        gallery_thumbs: List[str] = []
        aliases: List[str] = []
        bio_summary = ""

        async with httpx.AsyncClient(verify=False) as client:
            tasks = []

            if query:
                if is_email:
                    user_prefix = query.split("@")[0]
                    aliases.append(f"@{user_prefix}")
                    # Gravatar check
                    md5_email = hashlib.md5(query.lower().encode("utf-8")).hexdigest()
                    grav_img = f"https://www.gravatar.com/avatar/{md5_email}?s=240&d=404"
                    try:
                        rg = await client.get(f"https://en.gravatar.com/{md5_email}.json", timeout=4.0)
                        if rg.status_code == 200:
                            g_entry = rg.json().get("entry", [{}])[0]
                            disp = g_entry.get("displayName") or user_prefix
                            bio_summary = g_entry.get("aboutMe", "")
                            gallery_thumbs.append(grav_img)
                            for acc in g_entry.get("accounts", []):
                                if acc.get("url"):
                                    extra_raw_items.append({
                                        "title": f"{acc.get('shortname', 'Conta')} ({disp})",
                                        "url": acc["url"],
                                        "snippet": "Conta vinculada ao Gravatar do e-mail",
                                    })
                    except Exception:
                        pass

                    tasks.append(self._ddg_search(client, f'"{query}" OR "{user_prefix}"'))
                    tasks.append(self._probe_direct_username(client, user_prefix))

                elif is_phone:
                    # Generate direct messaging & OSINT search links for phone
                    extra_raw_items.append({
                        "title": f"WhatsApp Direto (+{digits_only})",
                        "url": f"https://wa.me/{digits_only}",
                        "snippet": "Link direto para verificar perfil/foto no WhatsApp",
                    })
                    extra_raw_items.append({
                        "title": f"Telegram por Número (+{digits_only})",
                        "url": f"https://t.me/+{digits_only}",
                        "snippet": "Link direto para localizar conta do Telegram pelo telefone",
                    })
                    tasks.append(self._ddg_search(client, f'"{query}" OR "{digits_only}"'))

                else:
                    clean_q = query.lstrip("@")
                    # 4 parallel uncensored DuckDuckGo queries covering all social, creator, and database platforms
                    q_social = (
                        f'{clean_q} (site:instagram.com OR site:tiktok.com OR site:x.com '
                        f'OR site:twitter.com OR site:youtube.com OR site:threads.net OR site:twitch.tv OR site:facebook.com)'
                    )
                    q_creator = (
                        f'{clean_q} (site:onlyfans.com OR site:fansly.com OR site:fapello.com '
                        f'OR site:manyvids.com OR site:patreon.com OR site:fancentro.com OR site:loyalfans.com OR site:throne.com)'
                    )
                    q_catalog = (
                        f'{clean_q} (site:imdb.com OR site:iafd.com OR site:babepedia.com '
                        f'OR site:indexxx.com OR site:freeones.com OR site:linktr.ee OR site:allmylinks.com OR site:beacons.ai)'
                    )
                    q_comm = (
                        f'{clean_q} (site:t.me OR site:reddit.com OR site:vk.com OR site:pinterest.com OR site:steamcommunity.com)'
                    )
                    tasks.extend([
                        self._ddg_search(client, q_social),
                        self._ddg_search(client, q_creator),
                        self._ddg_search(client, q_catalog),
                        self._ddg_search(client, q_comm),
                        self._yandex_ru_search(client, f"{clean_q} instagram tiktok onlyfans"),
                    ])
                    if is_handle or " " not in clean_q:
                        tasks.append(self._probe_direct_username(client, clean_q))

            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            raw_collected: List[Dict[str, Any]] = list(extra_raw_items)
            for tr in task_results:
                if isinstance(tr, tuple):
                    y_items, y_thumbs = tr
                    raw_collected.extend(y_items)
                    gallery_thumbs.extend(y_thumbs)
                elif isinstance(tr, list):
                    raw_collected.extend(tr)

            # Classify, deduplicate, and extract handles
            seen_urls = set()
            handles_counter: Dict[str, int] = {}

            for item in raw_collected:
                u = item.get("url", "")
                if not u:
                    continue
                u_clean = re.sub(r"[?&]utm_[^&]+", "", u).rstrip("/")
                if u_clean in seen_urls:
                    continue
                seen_urls.add(u_clean)

                # If already classified (from direct probe), keep it
                if "platform" in item and "category" in item:
                    classified = item
                else:
                    classified = self._classify_url(u, item.get("title", ""))
                    if not classified:
                        continue
                    classified["snippet"] = item.get("snippet", "")
                    if item.get("thumb"):
                        classified["thumb"] = item["thumb"]
                        if len(gallery_thumbs) < 16 and item["thumb"] not in gallery_thumbs:
                            gallery_thumbs.append(item["thumb"])

                h = classified.get("handle", "")
                if h:
                    handles_counter[h] = handles_counter.get(h, 0) + 1
                if not bio_summary and classified.get("snippet") and len(classified["snippet"]) > 35:
                    bio_summary = classified["snippet"]

                discovered_items.append(classified)

            # If we discovered a primary @handle from social results and haven't probed it directly yet, probe it!
            if handles_counter and not is_handle and not is_email and not is_phone:
                top_handle = max(handles_counter.items(), key=lambda x: x[1])[0]
                probed = await self._probe_direct_username(client, top_handle)
                for p_item in probed:
                    u_clean = p_item["url"].rstrip("/")
                    if u_clean not in seen_urls:
                        seen_urls.add(u_clean)
                        discovered_items.append(p_item)

        aliases.extend([h for h, _ in sorted(handles_counter.items(), key=lambda x: x[1], reverse=True)[:8]])
        aliases = list(dict.fromkeys(aliases))

        # Sort items so major social/creator platforms appear first
        cat_order = {
            "Redes Sociais": 1,
            "Plataformas de Criadores": 2,
            "Agregadores de Bio": 3,
            "Catálogos & Elenco": 4,
            "Vídeo & Streaming": 5,
            "Comunidades & Canais": 6,
            "Comunidades & Fóruns": 7,
            "Profissional": 8,
            "Outras Plataformas": 9,
            "Web & Menções": 10,
        }
        discovered_items.sort(key=lambda x: (cat_order.get(x["category"], 99), x["platform"]))

        return {
            "query": query,
            "query_type": "email" if is_email else "phone" if is_phone else "handle" if is_handle else "name",
            "aliases": aliases,
            "bio_summary": bio_summary,
            "gallery_thumbs": gallery_thumbs[:12],
            "total_found": len(discovered_items),
            "profiles": discovered_items,
        }
