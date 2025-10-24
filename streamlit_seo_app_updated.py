import io
import re
import json
import time
import zipfile
import typing as t
import unicodedata
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# =========================================
# Config / Scoring Weights (tweak as needed)
# =========================================
SM_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Paths that imply non-main pages for the reference/audit lists
FILTER_OUT_PATHS = [
    "/wp-", "/wp-json", "/feed", "/rss", "/tag/", "/category/",
    "/search", "/cart", "/account", "/login", "/privacy", "/terms",
    "/sitemap.xml"
]

# Item-like paths (eligible for dish keyword extraction)
ITEM_PATH_HINTS = ["/items/", "/item/", "/menu-item/", "/product/"]

WEIGHTS = {
    "missing_title": 20,
    "missing_meta": 15,
    "missing_h1": 15,
    "thin_content": 10,    # word_count < 150
    "no_schema": 10,       # no Restaurant/LocalBusiness
    "dup_title": 6,
    "dup_meta": 6
}

THRESHOLDS = {
    "thin_word_count": 150
}

LOCAL_JSONLD_TYPES = {"Restaurant", "LocalBusiness", "FoodEstablishment"}

# ============================
# Normalization Maps & Helpers
# ============================
NORMALIZE_MAP = [
    (r"^/$", "Homepage"),
    (r"^/(about|our-story|story|who-we-are)/?$", "Our Story"),
    (r"^/(press|news)/?$", "Press"),
    (r"^/(blog|press-and-blog|journal)/?$", "Blog"),
    (r"^/(contact|contact-us)/?$", "Contact"),
    (r"^/(gift-cards|giftcards)/?$", "Gift Cards"),
    (r"^/(careers|jobs)/?$", "Careers"),

    (r"^/(hours|hours-and-location|location|locations|visit|find-us)/?$", "Hours and Location"),
    (r"^/(reservations|reserve|book)/?$", "Reservations"),
    (r"^/(order-online|order|take-out-and-delivery|takeout|delivery)/?$", "Take Out and Delivery"),

    (r"^/(menu|menus)/?$", "Menu"),

    (r"^/(catering|catering-menu|catering-packages)/?$", "Catering Page"),
    (r"^/(private-events|events/private|events)/?$", "Private Events"),
    (r"^/(wedding-packages|weddings)/?$", "Wedding Packages"),
    (r"^/(private-dining)/?$", "Private Dining"),
    (r"^/(happy-hour|happyhour)/?$", "Happy Hour"),
    (r"^/(events|happenings|what-s-on)/?$", "Events"),

    (r"^/(gallery|photos)/?$", "Gallery"),
    (r"^/(faq|faqs)/?$", "FAQs"),
]

# Pages that should NOT appear in the SEO audit list
AUDIT_EXCLUDE = {
    "Contact", "Gift Cards", "Reservations", "Gallery",
    "Careers", "FAQs", "Hours and Location", "Take Out and Delivery"
}

# Media/asset file extensions to exclude from URL lists
ASSET_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf")

# --- NEW: Bucket mapping (for YAML url_map), kept separate to avoid conflicts with NORMALIZE_MAP above
NORMALIZE_MAP_BUCKETS = [
    (r"^/$", "homepage"),
    (r"^/(about|our-story|story|who-we-are)/?$", "about"),
    (r"^/(press|news)/?$", "blog_or_press"),
    (r"^/(blog|press-and-blog|journal)/?$", "blog_or_press"),
    (r"^/(contact|contact-us)/?$", "contact"),
    (r"^/(gift-cards|giftcards)/?$", "gift_cards"),
    (r"^/(careers|jobs)/?$", "careers"),
    (r"^/(hours|hours-and-location|location|locations|visit|find-us)/?$", "locations"),
    (r"^/(reservations|reserve|book)/?$", "reservations"),
    (r"^/(order-online|order|take-out-and-delivery|takeout|delivery)/?$", "order_online"),
    (r"^/(menu|menus)/?$", "menu"),
    (r"^/(catering|catering-menu|catering-packages)/?$", "catering"),
    (r"^/(private-events|events/private|events)/?$", "private_events"),
    (r"^/(wedding-packages|weddings)/?$", "private_dining"),
    (r"^/(private-dining)/?$", "private_dining"),
    (r"^/(happy-hour|happyhour)/?$", "happy_hour"),
    (r"^/(events|happenings|what-s-on)/?$", "events"),
    (r"^/(gallery|photos)/?$", "gallery"),
]

# Generic food nouns to prioritize when collapsing dish names to search-behavior keywords
FOOD_PRIORITY = [
    "steak", "burger", "pizza", "pasta", "noodles", "taco", "burrito", "sandwich",
    "chicken", "beef", "pork", "lamb", "duck", "turkey",
    "shrimp", "prawn", "lobster", "crab", "oyster", "oysters", "clam", "mussel", "fish",
    "salmon", "tuna", "halibut", "cod", "seafood",
    "salad", "soup", "dessert", "cake", "pie", "brownie", "ice cream", "gelato",
    "risotto", "gnocchi", "lasagna", "ravioli", "sushi", "ramen", "udon", "pho"
]

STOPWORDS = set(("with and the a an to for of in on from by at our house style fresh freshly daily "
                 "day aged bone in out crispy grilled roasted braised seared baked pan fried wood oven "
                 "served over topped slow wild organic local free range creamy spicy sweet savory half "
                 "shell white mexican black truffle sauce gluten free vegan vegetarian option jumbo "
                 "large small mini classic special chef signature new york la philadelphia san francisco "
                 "texas carolina japanese italian french thai indian mexican peruvian german hungarian "
                 "viennese hawaiian greek").split())

# =========================================
# Utilities
# =========================================

def strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def load_zip_htmls(zf_bytes: bytes) -> t.List[t.Tuple[str, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes)) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith(".html") or low.endswith(".htm"):
                out.append((name, z.read(name)))
    return out


def parse_sitemap_urls(xml_bytes: bytes, max_urls: int = 5000) -> pd.DataFrame:
    rows = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return pd.DataFrame(columns=["url", "lastmod"])

    idx = root.findall("sm:sitemap", SM_NS)
    if idx:
        for sm in idx:
            loc = sm.find("sm:loc", SM_NS)
            if loc is not None and loc.text:
                rows.append({"url": loc.text.strip(), "lastmod": None})
        return pd.DataFrame(rows)[:max_urls]

    for u in root.findall("sm:url", SM_NS):
        loc = u.find("sm:loc", SM_NS)
        lm = u.find("sm:lastmod", SM_NS)
        if loc is not None and loc.text:
            url = loc.text.strip()
            if not url:
                continue
            rows.append({
                "url": url,
                "lastmod": (lm.text.strip() if lm is not None and lm.text else None)
            })
            if len(rows) >= max_urls:
                break
    return pd.DataFrame(rows)


def parse_url_list_text(raw_text: str, base_domain: str | None = None) -> pd.DataFrame:
    """Parse newline/CSV URL list into DataFrame with columns url, lastmod(None).
    Optionally filter to base_domain if provided."""
    if not raw_text.strip():
        return pd.DataFrame(columns=["url", "lastmod"])
    tokens = re.split(r"[\n,]+", raw_text)
    urls = []
    for tkn in tokens:
        u = tkn.strip().strip('"\'')
        if not u:
            continue
        if not re.match(r"^https?://", u):
            if u.startswith("//"):
                u = "https:" + u
            else:
                if base_domain:
                    u = base_domain.rstrip("/") + ("/" + u.lstrip("/"))
                else:
                    continue
        urls.append(u)
    df = pd.DataFrame({"url": urls, "lastmod": None})
    if base_domain:
        host = urlparse(base_domain).netloc
        if host:
            df = df[df["url"].apply(lambda x: urlparse(x).netloc == host)]
    return df.reset_index(drop=True)


def is_asset_url(u: str) -> bool:
    p = urlparse(u).path.lower()
    return any(p.endswith(ext) for ext in ASSET_EXTS)


def exclude_utility(u: str) -> bool:
    p = urlparse(u).path.lower()
    if any(seg in p for seg in FILTER_OUT_PATHS):
        return True
    if is_asset_url(u):
        return True
    return False


def canonicalize_url(u: str) -> str:
    parts = urlparse(u)
    scheme = 'https' if parts.scheme in ('http', 'https') else parts.scheme or 'https'
    path = parts.path.rstrip('/') if parts.path not in ('/', '') else ''
    netloc = parts.netloc
    return f"{scheme}://{netloc}{'/' + path.lstrip('/') if path else ''}"


def normalize_path_to_name(path: str) -> str | None:
    path = '/' + path.strip('/')
    if path == '/':
        return "Homepage"
    for patt, name in NORMALIZE_MAP:
        if re.match(patt, path, flags=re.IGNORECASE):
            return name
    if re.match(r"^/(menu|menus)/[a-z0-9\-_/]+$", path, flags=re.IGNORECASE):
        seg = path.split('/')[-1]
        seg = re.sub(r"[-_]+", " ", seg).title()
        return seg
    return None


def filter_main_pages(df_urls: pd.DataFrame) -> pd.DataFrame:
    if df_urls.empty:
        return pd.DataFrame(columns=["name", "url", "path"])
    df = df_urls[~df_urls["url"].apply(exclude_utility)].copy()
    if df.empty:
        return pd.DataFrame(columns=["name", "url", "path"])

    df["canon_url"] = df["url"].apply(canonicalize_url)
    df["path"] = df["canon_url"].apply(lambda u: urlparse(u).path or '/')
    df["name"] = df["path"].apply(normalize_path_to_name)
    df = df[df["name"].notna()].copy()

    df["path_len"] = df["path"].apply(len)
    df = df.sort_values(["name", "path_len"]).drop_duplicates(subset=["name"], keep="first")

    return df[["name", "canon_url", "path"]].copy()


def build_reference_and_audit(df_urls: pd.DataFrame) -> tuple[list[str], list[str]]:
    main_pages = filter_main_pages(df_urls)
    if main_pages.empty:
        return [], []

    ref_lines = [f"{row.name} – {row.canon_url}" for row in main_pages.itertuples(index=False, name="Row")]

    audit_names = []
    for nm in main_pages["name"].tolist():
        if nm in AUDIT_EXCLUDE:
            continue
        if nm.lower() in {
            "menu", "homepage", "our story", "press", "private events",
            "wedding packages", "blog", "private dining", "happy hour",
            "events", "catering page"
        }:
            audit_names.append(nm)
    seen = set()
    audit_lines = []
    for n in audit_names:
        if n not in seen:
            seen.add(n)
            audit_lines.append(n)

    return ref_lines, audit_lines


# ==============================
# Item/Dish Keyword Extraction
# ==============================

def looks_like_item_url(u: str) -> bool:
    p = urlparse(u).path.lower()
    if any(h in p for h in ITEM_PATH_HINTS):
        return True
    if re.match(r"^/(menus?|dinner|lunch|brunch|desserts?)/[a-z0-9\-]{3,}(/[a-z0-9\-]{3,})?$", p):
        leaf = p.rstrip('/').split('/')[-1]
        if leaf not in {"menu", "menus", "dinner", "lunch", "brunch", "dessert", "desserts"}:
            return True
    return False


def slug_to_title(slug: str) -> str:
    slug = slug.strip('/').split('/')[-1]
    slug = re.sub(r"[-_]+", " ", slug)
    slug = strip_accents(slug)
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug.title()


def normalize_dish_to_generic(name: str) -> str:
    raw = strip_accents(name.lower())
    tokens = re.findall(r"[a-z]+", raw)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return name.lower().strip()
    for pri in FOOD_PRIORITY:
        if pri in tokens or (pri.replace(" ", "") in tokens):
            return pri
    return tokens[-1]


def extract_item_keywords(df_urls: pd.DataFrame, sample_n: int | None = None) -> dict:
    if df_urls.empty:
        return {"item_urls": [], "dish_names": [], "generic_dish_names": [], "search_keywords": []}

    urls = [u for u in df_urls["url"].tolist() if looks_like_item_url(u)]
    if not urls:
        return {"item_urls": [], "dish_names": [], "generic_dish_names": [], "search_keywords": []}

    if sample_n is not None and sample_n > 0:
        urls = urls[:sample_n]

    dish_names = [slug_to_title(urlparse(u).path) for u in urls]
    generic = [normalize_dish_to_generic(n) for n in dish_names]

    def to_keyword(tok: str) -> str:
        t = tok.strip().lower()
        if t.endswith('s') and t[:-1] in FOOD_PRIORITY:
            t = t[:-1]
        if t == 'oyster':
            return 'oysters'
        return t

    keywords = [to_keyword(t) for t in generic]

    return {
        "item_urls": urls,
        "dish_names": dish_names,
        "generic_dish_names": generic,
        "search_keywords": keywords,
    }


# =========================================
# HTML Extraction & Scoring
# =========================================

def get_headings(soup: BeautifulSoup) -> t.List[str]:
    heads = []
    for level in ["h1","h2","h3","h4","h5","h6"]:
        for h in soup.find_all(level):
            txt = h.get_text(" ", strip=True)
            if txt:
                heads.append(f"{level.upper()}: {txt}")
    return heads


def get_sections(soup: BeautifulSoup) -> t.List[str]:
    out = []
    for tag in ["header", "nav", "main", "section", "article", "aside", "footer"]:
        for s in soup.find_all(tag):
            h = s.find(["h1","h2","h3","h4","h5","h6"])
            title = h.get_text(" ", strip=True) if h else ""
            if title:
                out.append(f"{tag}: {title}")
            else:
                text = s.get_text(" ", strip=True)
                snippet = " ".join(text.split()[:8]) if text else ""
                out.append(f"{tag}: {snippet}")
    return out


def extract_jsonld_types(soup: BeautifulSoup) -> t.List[str]:
    types = set()
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.get_text(strip=True))
            items = data if isinstance(data, list) else [data]
            for item in items:
                ttype = item.get("@type")
                if isinstance(ttype, list):
                    for t_ in ttype:
                        types.add(str(t_))
                elif isinstance(ttype, str):
                    types.add(ttype)
        except Exception:
            continue
    return list(types)


def extract_from_html(html_bytes: bytes, filename: str) -> dict:
    try:
        html_text = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        html_text = str(html_bytes)

    soup = BeautifulSoup(html_text, "html.parser")

    row = {
        "source_file": filename,
        "url": None,
        "title": None, "title_length": None,
        "meta_description": None, "meta_description_length": None,
        "h1": None,
        "headings_all": None,
        "sections_all": None,
        "word_count": None,
        "jsonld_types": None, "has_localbusiness_schema": None
    }

    canon = soup.find("link", rel=lambda x: x and "canonical" in x)
    if canon and canon.get("href"):
        row["url"] = canon["href"].strip()
    else:
        ogu = soup.find("meta", attrs={"property": "og:url"})
        if ogu and ogu.get("content"):
            row["url"] = ogu["content"].strip()
        else:
            base = soup.find("base", href=True)
            row["url"] = base["href"].strip() if base else f"uploaded://{filename}"

    if soup.title and soup.title.string:
        row["title"] = soup.title.string.strip()
        row["title_length"] = len(row["title"])
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        row["meta_description"] = md["content"].strip()
        row["meta_description_length"] = len(row["meta_description"])

    h1 = soup.find("h1")
    row["h1"] = h1.get_text(" ", strip=True) if h1 else None
    row["headings_all"] = get_headings(soup)
    row["sections_all"] = get_sections(soup)

    text = soup.get_text(" ", strip=True)
    words = re.findall(r"\b[^\W\d_]{3,}\b", text.lower())
    row["word_count"] = len(words)

    jl_types = extract_jsonld_types(soup)
    row["jsonld_types"] = ", ".join(sorted(jl_types)) if jl_types else None
    row["has_localbusiness_schema"] = any(t in LOCAL_JSONLD_TYPES for t in jl_types)

    return row


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dup_title"] = False
    df["dup_meta"] = False

    if "title" in df.columns:
        dup_t = df["title"].fillna("").duplicated(keep=False) & df["title"].notna()
        df.loc[dup_t, "dup_title"] = True
    if "meta_description" in df.columns:
        dup_m = df["meta_description"].fillna("").duplicated(keep=False) & df["meta_description"].notna()
        df.loc[dup_m, "dup_meta"] = True

    scores = []
    for _, r in df.iterrows():
        score = 100
        if pd.isna(r.get("title")) or r.get("title") in [None, ""]:
            score -= WEIGHTS["missing_title"]
        if pd.isna(r.get("meta_description")) or r.get("meta_description") in [None, ""]:
            score -= WEIGHTS["missing_meta"]
        if pd.isna(r.get("h1")) or r.get("h1") in [None, ""]:
            score -= WEIGHTS["missing_h1"]
        if (r.get("word_count") or 0) < THRESHOLDS["thin_word_count"]:
            score -= WEIGHTS["thin_content"]
        if not bool(r.get("has_localbusiness_schema")):
            score -= WEIGHTS["no_schema"]
        if r.get("dup_title", False):
            score -= WEIGHTS["dup_title"]
        if r.get("dup_meta", False):
            score -= WEIGHTS["dup_meta"]
        scores.append(max(0, min(100, score)))
    df["health_score"] = scores
    return df


def site_score(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(round(df["health_score"].mean()))


# =========================================
# --- NEW: YAML helpers (safe dumper, name/desc, URL bucketer, HTML text/meta)
# =========================================

# Safe YAML dump with fallback to JSON if PyYAML isn't present
try:
    import yaml
    def dump_yaml(obj): return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
except Exception:
    dump_yaml = lambda obj: json.dumps(obj, indent=2, ensure_ascii=False)

def choose_restaurant_name(candidates: list[str]) -> str | None:
    clean = []
    for c in candidates or []:
        c0 = re.sub(r"\s*[\|\-–—].*$", "", str(c)).strip()
        if c0:
            clean.append(c0)
    if not clean:
        return None
    clean.sort(key=len)
    return clean[0]

def best_description(desc_candidates: list[str]) -> str | None:
    if not desc_candidates:
        return None
    return sorted(desc_candidates, key=lambda s: abs(len(s) - 140))[0]

def normalize_path_to_bucket(path: str) -> str | None:
    p = '/' + path.strip('/')
    if p == '/':
        return "homepage"
    for patt, bucket in NORMALIZE_MAP_BUCKETS:
        if re.match(patt, p, flags=re.IGNORECASE):
            return bucket
    return None

def classify_urls_for_yaml(urls: list[str]) -> dict:
    out = {
        "homepage": "",
        "menu": "",
        "order_online": "",
        "catering": "",
        "private_events": "",
        "private_dining": "",
        "events": "",
        "happy_hour": "",
        "locations": "",
        "about": "",
        "contact": "",
        "reservations": "",
        "gift_cards": "",
        "gallery": "",
        "blog_or_press": "",
        "other_relevant": []
    }
    best: dict[str, tuple[str,int]] = {}
    for u in urls or []:
        if exclude_utility(u):
            continue
        cu = canonicalize_url(u)
        path = urlparse(cu).path or '/'
        b = normalize_path_to_bucket(path)
        if b:
            plen = len(path)
            if b not in best or plen < best[b][1]:
                best[b] = (cu, plen)
        else:
            out["other_relevant"].append(cu)
    for k,(val,_) in best.items():
        out[k] = val
    out["other_relevant"] = sorted(list(dict.fromkeys(out["other_relevant"])))
    return out

def extract_name_desc_offers_addresses(html_bytes: bytes) -> dict:
    try:
        html_text = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        html_text = str(html_bytes)
    soup = BeautifulSoup(html_text, "html.parser")

    name_cands = []
    if soup.title and soup.title.string:
        name_cands.append(soup.title.string.strip())
    og_site = soup.find("meta", property="og:site_name")
    if og_site and og_site.get("content"):
        name_cands.append(og_site["content"].strip())
    app_name = soup.find("meta", attrs={"name": "application-name"})
    if app_name and app_name.get("content"):
        name_cands.append(app_name["content"].strip())

    desc_cands = []
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        desc_cands.append(md["content"].strip())
    main = soup.find(["main","article","section"])
    if main:
        p = main.find("p")
        if p:
            txt = p.get_text(" ", strip=True)
            if txt and len(txt.split()) >= 8:
                desc_cands.append(txt)

    text = soup.get_text(" ", strip=True)
    addr_pat = re.compile(r"\b\d{2,5}\s+[A-Za-z0-9\.\-'\s]+,\s*[A-Za-z\.\-'\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b")
    addresses = addr_pat.findall(text)

    offerings = []
    for phrase in ["happy hour", "weekend brunch", "brunch", "catering", "private dining", "private events"]:
        if phrase in text.lower():
            offerings.append(phrase.title())

    return {
        "name_candidates": name_cands,
        "desc_candidates": desc_cands,
        "addresses": sorted(list(dict.fromkeys(addresses))),
        "offerings": sorted(list(dict.fromkeys(offerings))),
    }

def make_initial_yaml_packet(
    restaurant_name: str | None,
    url_map: dict,
    cuisine_terms: list[str],
    signature_dishes: list[str],
    offerings: list[str],
    addresses: list[str],
    description: str | None
):
    # Very light structuring of addresses
    locs = []
    for a in addresses:
        m = re.match(r"^(\d{2,5}\s+[^,]+),\s*([^,]+),\s*([A-Z]{2})\s*(\d{5}(?:-\d{4})?)$", a)
        if m:
            street, city, state, postal = m.groups()
            locs.append({
                "label": f"{city}, {state}",
                "street": street, "city": city, "state": state, "postal_code": postal, "country": "USA"
            })
        else:
            locs.append({"label": a, "street": "", "city": "", "state": "", "postal_code": "", "country": ""})

    packet = {
        "restaurant_name": restaurant_name or "",
        "urls": url_map,
        "dishes_cuisine_offerings": {
            "cuisine": cuisine_terms,
            "signature_dishes": signature_dishes,
            "offerings": offerings
        },
        "locations": locs,
        "short_description": description or ""
    }
    return packet

# =========================================
# --- NEW: Ahrefs CSV analyzers (helpers + packet)
# =========================================
def read_keywords_csv(path_or_bytes) -> pd.DataFrame:
    # Ahrefs keywords TSVs are often UTF-16 tab-delimited; handle flexibly
    if isinstance(path_or_bytes, (bytes, bytearray)):
        bio = io.BytesIO(path_or_bytes)
        try:
            return pd.read_csv(bio, encoding="utf-16", sep="\t")
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio)
    else:
        try:
            return pd.read_csv(path_or_bytes, encoding="utf-16", sep="\t")
        except Exception:
            return pd.read_csv(path_or_bytes)

def categorize_keyword(k: str, brand_tokens: list[str], location_tokens: list[str]) -> list[str]:
    k_lower = str(k).lower()
    cats = []
    if any(bt.lower() in k_lower for bt in brand_tokens if bt):
        cats.append("branded")
    if any(lt.lower() in k_lower for lt in location_tokens if lt):
        cats.append("local")
    for token in ["restaurant", "menu", "reservations", "reservation", "happy hour", "gift card", "hours"]:
        if token in k_lower:
            cats.append("generic_restaurant"); break
    cuisine_terms = set(FOOD_PRIORITY + ["alfredo","reuben","nachos","brisket","pasta","sushi","tacos","wings","pizza","burger"])
    if any(ct in k_lower for ct in cuisine_terms):
        cats.append("cuisine")
    return list(dict.fromkeys(cats))

def quick_win(row, vol_floor=200):
    try:
        pos = float(row.get("Current position", float("nan")))
    except Exception:
        return False
    vol = row.get("Volume")
    traf = row.get("Current organic traffic")
    return (5 <= pos <= 15) and (pd.notna(vol) and vol >= vol_floor) and (pd.notna(traf) and traf < 30)

def analyze_keywords(df_kw: pd.DataFrame, brand_tokens: list[str], location_tokens: list[str], top_k: int = 10) -> dict:
    if df_kw is None or df_kw.empty:
        return {"top_keywords": [], "notes": ["No keywords CSV loaded."]}

    df = df_kw.copy()
    rename_map = {
        "Keyword":"Keyword","Country":"Country","Location":"Location","Volume":"Volume",
        "Previous organic traffic":"Previous organic traffic","Current organic traffic":"Current organic traffic",
        "Organic traffic change":"Organic traffic change","Previous position":"Previous position",
        "Current position":"Current position","Position change":"Position change",
        "Previous URL":"Previous URL","Current URL":"Current URL"
    }
    df = df.rename(columns=rename_map)
    for c in ["Volume","Previous organic traffic","Current organic traffic","Previous position","Current position","Position change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["categories"] = df["Keyword"].apply(lambda k: categorize_keyword(k, brand_tokens, location_tokens))
    df["quick_win"] = df.apply(quick_win, axis=1)

    cols_out = ["Keyword","Volume","Current organic traffic","Current position","Current URL","categories","quick_win"]
    present = [c for c in cols_out if c in df.columns]
    top = df.sort_values(by="Current organic traffic", ascending=False).head(top_k)[present]

    return {"top_keywords": top.to_dict(orient="records")}

def analyze_performance(df_perf: pd.DataFrame) -> dict:
    if df_perf is None or df_perf.empty:
        return {"traffic_trend": {}, "position_distribution": {}}
    date_pat = re.compile(r"^\d{4}-\d{2}$")
    if "Metric" not in df_perf.columns:
        return {"traffic_trend": {}, "position_distribution": {}}
    perf_months = df_perf[df_perf["Metric"].astype(str).str.match(date_pat)].copy()
    if perf_months.empty:
        return {"traffic_trend": {}, "position_distribution": {}}

    def to_num(s):
        if pd.isna(s): return float("nan")
        if isinstance(s, (int,float)): return float(s)
        s = str(s).replace(",","").strip()
        try: return float(s)
        except: return float("nan")

    traffic_col = " Avg. organic traffic"
    bucket_cols = [
        " Organic positions: 1–3",
        " Organic positions: 4–10",
        " Organic positions: 11–20",
        " Organic positions: 21–50",
        " Organic positions: 51+",
    ]
    for c in [traffic_col] + bucket_cols:
        if c in perf_months.columns:
            perf_months[c] = perf_months[c].apply(to_num)

    perf_months["date"] = pd.to_datetime(perf_months["Metric"], format="%Y-%m")
    trend = perf_months[["date", traffic_col] + bucket_cols].sort_values("date")
    vals = trend[traffic_col].astype(float).values
    months = trend["date"].dt.strftime("%Y-%m").tolist()
    if len(vals) == 0:
        return {"traffic_trend": {}, "position_distribution": {}}

    import numpy as np

    has_valid = np.any(~np.isnan(vals))
    if has_valid:
        peak_idx = int(np.nanargmax(vals))
        trough_idx = int(np.nanargmin(vals))
        peak_month = months[peak_idx] if months else None
        trough_month = months[trough_idx] if months else None
    else:
        peak_idx = trough_idx = None
        peak_month = trough_month = None

    def _first_non_nan(sequence):
        for value in sequence:
            if not pd.isna(value):
                return value
        return None

    first = _first_non_nan(vals)
    last = _first_non_nan(vals[::-1])
    if first is None or last is None:
        direction = "flat"
    else:
        direction = "up" if last > first else ("down" if last < first else "flat")

    latest = trend.sort_values("date").tail(1)
    buckets = {}
    if not latest.empty:
        row = latest.iloc[0]
        for c in bucket_cols:
            if c in row.index:
                buckets[c.replace(" Organic positions: ","")] = (None if pd.isna(row[c]) else float(row[c]))

    # rough phases (heuristic)
    phases = []
    if len(vals) >= 4:
        q = max(1, len(vals)//4)
        chunks = [vals[i:i+q] for i in range(0, len(vals), q)]
        labels = []
        for i,ch in enumerate(chunks):
            if len(ch)==0: continue
            ch = [x for x in ch if not pd.isna(x)]
            if not ch: continue
            labels.append((i, np.nanmean(ch)))
        for i in range(1,len(labels)):
            d = labels[i][1] - labels[i-1][1]
            if d > 0: phases.append("rise")
            elif d < 0: phases.append("decline")
            else: phases.append("flat")

    def summarize_phases(phs):
        out = []
        if not phs: return out
        mapping = {0:"early",1:"mid",2:"late",3:"very late"}
        for idx,ph in enumerate(phs[:4]):
            if ph=="rise": out.append(f"{mapping.get(idx,'phase')} increase")
            elif ph=="decline": out.append(f"{mapping.get(idx,'phase')} decline")
        return out

    return {
        "traffic_trend": {
            "direction": "mixed" if len(set([p for p in phases if p!='flat']))>1 else direction,
            "key_phases": summarize_phases(phases),
            "approx_peak_month": peak_month,
            "approx_trough_month": trough_month,
            "rough_change_percent": "qualitative",
            "confidence": "medium" if len(months)>=6 else "low",
        },
        "position_distribution": {
            "buckets": {
                "1-3":   {"trend_direction": "n/a", "note": ""} if "1–3" not in buckets else {"trend_direction":"n/a","note":""},
                "4-10":  {"trend_direction": "n/a", "note": ""},
                "11-20": {"trend_direction": "n/a", "note": ""},
                "21-50": {"trend_direction": "n/a", "note": ""},
                "51+":   {"trend_direction": "n/a", "note": ""},
            },
            "latest_counts": buckets,
            "overall_visibility_note": "",
            "confidence": "medium"
        }
    }

def make_seo_preanalysis_packet(site_name: str, perf_summary: dict, kw_summary: dict):
    return {
        "origin": "streamlit_preanalysis",
        "source_system": "Ahrefs CSV export",
        "client_facing": False,
        "site_name": site_name,
        "time_range": "",
        "traffic_trend": perf_summary.get("traffic_trend", {}),
        "position_distribution": perf_summary.get("position_distribution", {}),
        "top_keywords": kw_summary.get("top_keywords", []),
        "insights": [],
        "handoff_instructions": {
            "note_for_gpt_seo_audit_assistant": "Initial data derived from CSV exports; convert to client-facing language and merge with the full audit. Keep keyword text EXACT.",
            "suggested_focus_areas": ["Non-branded cuisine keyword recovery","URL/content alignment for lost menu items"],
            "assumptions_or_gaps": ["Exact bucket counts depend on export schema"]
        }
    }

# =========================================
# Streamlit App (OFFLINE uploads only)
# =========================================

st.set_page_config(page_title="Gabby's SEO Data Collector", layout="wide")
st.title("Gabby's SEO Data Collector")
st.caption("Upload sitemap.xml and HTML (.html/.htm) files or one .zip with HTML.")

with st.sidebar:
    st.header("Uploads")
    sitemap_file = st.file_uploader("Sitemap (sitemap.xml)", type=["xml"])
    raw_list = st.text_area("Or paste a raw list of URLs (newline/CSV)")
    base_for_raw = st.text_input("Base domain (only for relative paths in pasted list)", value="")
    html_files = st.file_uploader(
        "HTML files (or a single .zip)", type=["html", "htm", "zip"], accept_multiple_files=True
    )
    # NEW: Ahrefs CSV inputs (optional; processed by the same button)
    perf_csv = st.file_uploader("Performance CSV (Ahrefs)", type=["csv"], key="perf_csv")
    kw_csv = st.file_uploader("Top Keywords Bringing Traffic CSV (Ahrefs)", type=["csv", "tsv", "txt"], key="kw_csv")

    page_limit = st.slider("Limit pages processed (for very large exports)", 10, 10000, 2000, step=10)
    sample_items = st.number_input("Dish keyword sample size (0 = all)", min_value=0, max_value=200, value=5, step=1)
    run_btn = st.button("Process & Score", type="primary", use_container_width=True)

if run_btn:
    # Collect HTML bytes
    html_sources: t.List[t.Tuple[str, bytes]] = []
    zips = [f for f in html_files if f.name.lower().endswith(".zip")] if html_files else []
    singles = [f for f in html_files if not f.name.lower().endswith(".zip")] if html_files else []

    for z in zips:
        try:
            html_sources.extend(load_zip_htmls(z.read()))
        except Exception as e:
            st.error(f"Failed to read zip '{z.name}': {e}")
    for f in singles:
        try:
            html_sources.append((f.name, f.read()))
        except Exception as e:
            st.error(f"Failed to read file '{f.name}': {e}")

    if not html_sources:
        st.error("Please upload HTML files (.html/.htm) or a .zip containing them.")
        st.stop()

    # Parse URLs from inputs
    sitemap_urls = pd.DataFrame(columns=["url", "lastmod"])

    if sitemap_file is not None:
        sm_df = parse_sitemap_urls(sitemap_file.read(), max_urls=page_limit)
        if not sm_df.empty:
            sitemap_urls = pd.concat([sitemap_urls, sm_df], ignore_index=True)
        else:
            st.warning("Sitemap provided but no URLs found (index-only or malformed).")

    if raw_list.strip():
        raw_df = parse_url_list_text(raw_list, base_domain=base_for_raw or None)
        if not raw_df.empty:
            sitemap_urls = pd.concat([sitemap_urls, raw_df], ignore_index=True)

    if not sitemap_urls.empty:
        sitemap_urls["url"] = sitemap_urls["url"].apply(canonicalize_url)
        sitemap_urls = sitemap_urls.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # Build reference & audit lists from sitemap URLs
    ref_lines, audit_lines = build_reference_and_audit(sitemap_urls)

    # Extract fields from HTML
    st.subheader("Extracting page data…")
    rows = []
    progress = st.progress(0.0)
    cap = min(len(html_sources), page_limit)
    for i, (name, content) in enumerate(html_sources[:cap], start=1):
        rows.append(extract_from_html(content, name))
        progress.progress(i / cap)
        time.sleep(0.005)

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No HTML parsed.")
        st.stop()

    # Compute scores (+ duplicate checks)
    df = compute_scores(df)

    # UI: Reference / Audit output (STRICT FORMAT)
    st.subheader("📜 Sitemap → Pages Lists (strict format)")

    if ref_lines or audit_lines:
        ref_text = "Pages for restaurant profile:\n" + ("\n".join(ref_lines) if ref_lines else "")
        audit_text = "Pages for SEO audit:\n" + ("\n".join(audit_lines) if audit_lines else "")
        st.code(ref_text + "\n\n" + audit_text, language=None)
        st.download_button(
            "Download Pages Lists (.txt)",
            (ref_text + "\n\n" + audit_text).encode("utf-8"),
            "pages_lists.txt",
            "text/plain",
        )
    else:
        st.info("No eligible main pages detected from the provided URLs. Check filters or paste a raw list.")

    # Item/Dish keyword extraction
    st.subheader("🍽️ Dish Keywords (from item URLs)")

    kw = extract_item_keywords(sitemap_urls, sample_n=(None if sample_items == 0 else int(sample_items)))

    if kw["item_urls"]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Item/Dish URLs**")
            st.code("\n".join(kw["item_urls"]))
        with c2:
            st.markdown("**Dish Names (pretty)**")
            st.code("\n".join(kw["dish_names"]))
        with c3:
            st.markdown("**Generalized Dish Names**")
            st.code("\n".join(kw["generic_dish_names"]))
        st.markdown("**Search-Behavior Keyword List**")
        st.code("\n".join(kw["search_keywords"]))

        out = {
            "item_urls": kw["item_urls"],
            "dish_names": kw["dish_names"],
            "generic_dish_names": kw["generic_dish_names"],
            "search_keywords": kw["search_keywords"],
        }
        st.download_button(
            "Download Dish Keywords (JSON)",
            json.dumps(out, indent=2).encode("utf-8"),
            "dish_keywords.json",
            "application/json",
        )
    else:
        st.info("No item/dish URLs detected based on common patterns (e.g., /items/, /menu-item/).")

    # Quick metrics
    st.subheader("🔎 Quick Findings")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Pages", len(df))
    c2.metric("Avg Score", site_score(df))
    c3.metric("Missing Titles", int(df["title"].isna().sum()))
    c4.metric("Missing Meta", int(df["meta_description"].isna().sum()))
    c5.metric("Missing H1", int(df["h1"].isna().sum()))
    c6.metric(
        f"Thin (<{THRESHOLDS['thin_word_count']})",
        int((df["word_count"].fillna(0) < THRESHOLDS['thin_word_count']).sum()),
    )
    c7.metric("No Schema", int((~df["has_localbusiness_schema"].fillna(False)).sum()))
    c8.metric("Dup Titles", int(df["dup_title"].sum()))

    # Table
    st.subheader("📋 Pages (Filter & Inspect)")
    with st.expander("Column help", expanded=False):
        st.write("- headings_all: all H1–H6 found, order of appearance")
        st.write("- sections_all: header/nav/main/section/article/footer labels with snippet/first heading")
    issue_filter = st.multiselect(
        "Quick filters",
        [
            "Missing Title",
            "Missing Meta",
            "Missing H1",
            f"Thin (<{THRESHOLDS['thin_word_count']} words))",
            "No Schema",
            "Duplicate Title",
            "Duplicate Meta",
            "Low Score (<70)",
        ],
        default=[],
    )
    df_view = df.copy()
    for fopt in issue_filter:
        if fopt == "Missing Title":
            df_view = df_view[df_view["title"].isna()]
        elif fopt == "Missing Meta":
            df_view = df_view[df_view["meta_description"].isna()]
        elif fopt == "Missing H1":
            df_view = df_view[df_view["h1"].isna()]
        elif fopt.startswith("Thin"):
            df_view = df_view[df_view["word_count"].fillna(0) < THRESHOLDS["thin_word_count"]]
        elif fopt == "No Schema":
            df_view = df_view[~df_view["has_localbusiness_schema"].fillna(False)]
        elif fopt == "Duplicate Title":
            df_view = df_view[df_view["dup_title"]]
        elif fopt == "Duplicate Meta":
            df_view = df_view[df_view["dup_meta"]]
        elif fopt == "Low Score (<70)":
            df_view = df_view[df_view["health_score"] < 70]

    search_kw = st.text_input("Search Title / H1 / URL")
    if search_kw:
        mask = (
            df_view["title"].fillna("").str.contains(search_kw, case=False)
            | df_view["h1"].fillna("").str.contains(search_kw, case=False)
            | df_view["url"].fillna("").str.contains(search_kw, case=False)
        )
        df_view = df_view[mask]

    st.dataframe(df_view, use_container_width=True, hide_index=True)

    # Exports
    st.subheader("⬇️ Exports")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (page-level current data + score)",
        csv_bytes,
        "seo_audit_current_data_scored.csv",
        "text/csv",
    )

    main_pages_df = filter_main_pages(sitemap_urls)
    if not main_pages_df.empty:
        mp_csv = (
            main_pages_df.rename(columns={"name": "page_name", "canon_url": "url"})[["page_name", "url"]]
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            "Download CSV (Main Pages from sitemap)", mp_csv, "main_pages_from_sitemap.csv", "text/csv"
        )

    # =========================================
    # NEW: YAML Restaurant Packet (built from current run)
    # =========================================
    st.subheader("🧾 YAML: Restaurant Packet")

    # Gather name/desc/addresses/offerings from uploaded HTML
    name_cands, desc_cands, all_addresses, all_offers = [], [], [], []
    for _, content in html_sources[:cap]:
        meta = extract_name_desc_offers_addresses(content)
        name_cands.extend(meta["name_candidates"])
        desc_cands.extend(meta["desc_candidates"])
        all_addresses.extend(meta["addresses"])
        all_offers.extend(meta["offerings"])

    restaurant_name = choose_restaurant_name(name_cands)
    description = best_description(desc_cands)
    offerings = sorted(list(dict.fromkeys(all_offers)))
    addresses = sorted(list(dict.fromkeys(all_addresses)))

    # Build URL map from sitemap URLs if present, otherwise from parsed page URLs
    if not sitemap_urls.empty:
        url_list_for_map = sitemap_urls["url"].dropna().tolist()
    else:
        url_list_for_map = df["url"].dropna().tolist()
    url_map = classify_urls_for_yaml(url_list_for_map)

    # Cuisine terms & signature dishes from extracted item keywords
    cuisine_terms = sorted(list(dict.fromkeys([w.title() for w in kw.get("search_keywords", [])])))
    signature_dishes = kw.get("dish_names", [])[:10] if kw.get("dish_names") else []

    initial_packet = make_initial_yaml_packet(
        restaurant_name=restaurant_name,
        url_map=url_map,
        cuisine_terms=cuisine_terms,
        signature_dishes=signature_dishes,
        offerings=offerings,
        addresses=addresses,
        description=description
    )

    yaml_text = dump_yaml(initial_packet)
    st.code(yaml_text, language="yaml")
    st.download_button(
        "Download initial_packet.yaml",
        yaml_text.encode("utf-8"),
        "initial_packet.yaml",
        "text/yaml",
    )

    # =========================================
    # NEW: Ahrefs CSVs → Pre-analysis YAML (optional, same button)
    # =========================================
    if (perf_csv is not None) or (kw_csv is not None):
        st.subheader("🧪 YAML: Pre-analysis (Ahrefs)")

        # Read performance CSV
        df_perf, df_kw = None, None
        if perf_csv is not None:
            try:
                df_perf = pd.read_csv(perf_csv)
            except Exception as e:
                st.error(f"Failed to read Performance CSV: {e}")

        # Read keywords CSV (UTF-16 TSV-friendly)
        if kw_csv is not None:
            try:
                df_kw = read_keywords_csv(kw_csv)
            except Exception as e:
                st.error(f"Failed to read Keywords CSV: {e}")

        # Auto-derive brand/location tokens if possible (no extra inputs needed)
        brand_tokens = []
        if restaurant_name:
            brand_tokens = [t for t in re.findall(r"[A-Za-z]{3,}", restaurant_name) if t]

        location_tokens = []
        for a in addresses:
            parts = [p.strip() for p in re.split(r"[,\s]+", a) if p.strip()]
            # prefer alphabetic tokens; keep short state codes too
            for p in parts:
                if re.match(r"^[A-Za-z]{2,}$", p):
                    location_tokens.append(p)
        # de-dupe
        brand_tokens = list(dict.fromkeys(brand_tokens))
        location_tokens = list(dict.fromkeys(location_tokens))

        # Analyses (guard if files missing)
        kw_summary = analyze_keywords(df_kw, brand_tokens, location_tokens) if df_kw is not None else {"top_keywords": []}
        perf_summary = analyze_performance(df_perf) if df_perf is not None else {"traffic_trend": {}, "position_distribution": {}}

        pre_packet = make_seo_preanalysis_packet(site_name=(restaurant_name or "Site"), perf_summary=perf_summary, kw_summary=kw_summary)
        pre_yaml_text = dump_yaml(pre_packet)
        st.code(pre_yaml_text, language="yaml")
        st.download_button(
            "Download seo_preanalysis.yaml",
            pre_yaml_text.encode("utf-8"),
            "seo_preanalysis.yaml",
            "text/yaml",
        )
