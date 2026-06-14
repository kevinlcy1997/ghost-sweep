"""
================================================================================
  走鬼 GHOST ALERT LISTENER — API Scraper & Activity Recorder
================================================================================

  A Python tool that taps into the 走鬼APP ("Run Ghost") real-time alert
  network — the Hong Kong community-driven enforcement-officer tracker —
  and silently records every sighting to a local JSON archive.

  Built entirely from reverse engineering.  No official API docs exist.

================================================================================
  HOW THIS WAS BUILT — The Reverse Engineering Story
================================================================================

  1. WEBSITE RECON (run.echk.com.hk)
     The public website is a dead-end — just a static splash page with a
     single <img> tag. robots.txt blocks everything. No JS, no SPA, no
     discoverable endpoints.

  2. APK DECOMPILATION (jadx)
     Downloaded 走鬼_1.64_APKPure.apk (26 MB). Decompiled with jadx into
     9,311 Java source files. Native Java/Kotlin app — not Flutter/RN.

  3. ARCHITECTURE MAPPING
     Package: com.echk.run
     Key classes found:
       - AppManager.java      → server URLs, session state, data parsing
       - BaseHttpService.java  → request construction, auto-params
       - HttpManager.java      → AES encrypt → POST → AES decrypt
       - GeneralAPI.java       → all API calls funnel through here
       - AES.java              → hardcoded encryption key material

  4. TRANSPORT ENCRYPTION CRACKED
     Every request/response is AES-256-CBC encrypted:
       Passphrase : echk.com.hk
       Salt       : [10,32,17,15,17,14,12,18,50,65,11,39,48,18,25,13]
       IV         : [36,19,11,18,10,15,23,54,41,3,17,38,24,12,13,64]
       KDF        : PBKDF2-HMAC-SHA1, 10 000 iterations
     POST body format: php=AES(action_name) & vars=AES(url_encoded_params)
     Response: AES-encrypted JSON blob, same key.

  5. SERVER INFRASTRUCTURE DISCOVERED
     Production API : http://parking.echk.com.hk/api/interface.php
     Dev API        : http://run.echk.com.hk/api/interface.php
     Image CDN      : http://ghostimage.echk.com.hk/image/
     All 61 PHP endpoints catalogued from grep across decompiled source.

  6. ANONYMOUS SESSION FLOW FOUND
     The app does NOT require Firebase login for core features!
     On every launch SplashActivity calls doLoginByToken.php with a random
     device UUID. The server returns a fresh session (uuid + token). This
     script does the same — no account, no email, no password needed.

  7. GRID SWEEP STRATEGY
     Alert endpoints are location-scoped (lat/lng + radius). This script
     sweeps a grid of ~115 cells (0.05° ≈ 5 km step) across the entire
     Hong Kong bounding box to capture all active alerts territory-wide.

================================================================================
  61 API ENDPOINTS DISCOVERED (all via POST /api/interface.php)
================================================================================

  AUTH & SESSION        ALERTS (CORE)            COMMUNITY
  ─────────────        ─────────────            ─────────
  appChecking           doAlert                  communityCreatePost
  doLogin               doAlertWithType          communityEditPost
  doLoginByToken        doUpdateAlertRecord      communityDeletePost
  doLogout              doVoteToAlertRecord      communityReplyPost
  doCheckEmailIsVer..   getNearByAlertLocation   communityDeletePostReply
  updateUserProfile     getNearByAlertLoc..24h   communityGetPostByID
  updateUserInfo        getUserAlertRecord       communityGetPostByAlert..
                        getSponsorAlertRecord    communityGetPostReply..
  PARKING               getTrafficAlert          communityGetHottestPost..
  ─────────             getNotificationRecord    communitySearchHottestPost
  doCheckin             getPersonalNotiRecord    communityGroupPostUpDown..
  doCheckOut                                     communityGetGroupList
  parkingBnbCheck..    NEWS & COUPONS            communitySearchGroup
                       ─────────────             communityGetGroupDetail
  SHOP / HELPER        getNews                   communityGetGroupPost..
  ─────────────        getUserCoupon             communityGetGroupMember
  helperGetShopUser    scanCoupon                communityChangeGroupMem..
  helperAddOrUpdate..  redeemCoupon              communityJoinGroup
  helperGetUserDisp..  checkScanCode             communityQuitGroup
  editShopDetail       checkQrCode
                       checkInvitationCode      SOS / ACCIDENT
  FORMS & MISC         doMemberLinkUpQrcode     ──────────────
  ─────────────                                  submitAccidentInfo
  doSubmitFeedback                               submitAccidentStatus
  doSubmitOpenApiForm                            getAllAccidentInfo
  pushTest
  recordWatchVideo
  submitProductEnquiry

================================================================================
  USAGE
================================================================================

  # One-shot — grab current state and exit:
  python ghost_listener.py --once

  # Continuous monitor — poll every 60 seconds (default):
  python ghost_listener.py

  # Custom interval and output:
  python ghost_listener.py --interval 300 -o my_data.json

================================================================================
  OUTPUT — ghost_alerts.json
================================================================================

  {
    "alerts": {                          ← de-duplicated by alert_record_id
      "5233453": {
        "alert_record_id": 5233453,
        "lat": 22.315428,
        "lng": 114.169763,
        "address": "37 Dundas St, Mong Kok, Kowloon",
        "alert_type": "alert",
        "create_dt": "2026-06-13 11:37:50",
        "upvote": "3", "downvote": "0",
        "_source": "nearby24h(22.3000,114.1500)",
        "_first_seen": "2026-06-13T15:43:14Z"
      }, ...
    },
    "news": [ ... ],                     ← 166 articles
    "sponsors": [ ... ],                 ← 31 partner shops with lat/lng
    "config": { ... },                   ← app strings, SOS contacts
    "meta": {
      "first_run": "...",
      "last_poll": "...",
      "total_alerts": 90
    }
  }

================================================================================
  DEPENDENCIES — pip install cryptography requests
================================================================================
"""

import argparse
import base64
import json
import logging
import os
import time
import urllib.parse
import uuid as uuid_mod
from datetime import datetime, timezone

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

try:
    from ghost_db import GhostDB
    _HAS_DB = True
except ImportError:
    _HAS_DB = False

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 1 — Configuration (all values extracted from decompiled APK)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# From AppManager.java line 277: isProductionServer ? "parking.echk.com.hk" : "run.echk.com.hk"
API_URL = "http://parking.echk.com.hk/api/interface.php"
DEFAULT_OUTPUT = "ghost_alerts.json"
DEFAULT_POLL_SEC = 60

# From com.echk.api.AES — hardcoded key material (yes, really)
PASSPHRASE = b"echk.com.hk"
SALT = bytes([10, 32, 17, 15, 17, 14, 12, 18, 50, 65, 11, 39, 48, 18, 25, 13])
IV = bytes([36, 19, 11, 18, 10, 15, 23, 54, 41, 3, 17, 38, 24, 12, 13, 64])
KDF_ITERATIONS = 10000
KEY_LENGTH = 256  # bits

# Hong Kong bounding box — sweep grid to capture all alerts territory-wide
HK_LAT_MIN, HK_LAT_MAX = 22.15, 22.56
HK_LNG_MIN, HK_LNG_MAX = 113.83, 114.41
GRID_STEP = 0.05  # ~5 km cells  →  ~70 cells covering HK

# Fake device fingerprint (BaseHttpService.java appends these to every request)
PLATFORM = "android"
DEVICE_NAME = "Python Listener"
OS_VERSION = "13"
APP_VERSION = "1.64"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ghost")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 2 — AES Transport Encryption (mirrors com.echk.api.AES)          ║
# ║                                                                         ║
# ║  Every HTTP request and response is AES-256-CBC encrypted.              ║
# ║  The key is derived via PBKDF2 from a passphrase baked into the APK.    ║
# ║  POST body: php=AES(action) & vars=AES(urlencoded_params)              ║
# ║  Response:  AES(JSON)                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class GhostAES:
    """AES-256-CBC encrypt / decrypt using the app's hardcoded key material."""

    def __init__(self):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA1(),
            length=KEY_LENGTH // 8,
            salt=SALT,
            iterations=KDF_ITERATIONS,
        )
        self._key = kdf.derive(PASSPHRASE)

    # -- Custom Base64 encode (matches com.echk.api.Base64Encoder) ----------
    @staticmethod
    def _b64_encode(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _b64_decode(text: str) -> bytes:
        return base64.b64decode(text)

    # -- Encrypt (returns base64 string) ------------------------------------
    def encrypt(self, plaintext: bytes) -> str:
        padder = sym_padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(IV))
        enc = cipher.encryptor()
        ct = enc.update(padded) + enc.finalize()
        return self._b64_encode(ct)

    # -- Decrypt (accepts base64 string, returns utf-8 text) ----------------
    def decrypt(self, ciphertext_b64: str) -> str:
        raw = self._b64_decode(ciphertext_b64)
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(IV))
        dec = cipher.decryptor()
        padded = dec.update(raw) + dec.finalize()
        unpadder = sym_padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded) + unpadder.finalize()
        return plaintext.decode("utf-8")


aes = GhostAES()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 3 — Anonymous Session (the key discovery)                        ║
# ║                                                                         ║
# ║  The app has Firebase Auth (Google/Facebook/Email login), but it turns  ║
# ║  out you DON'T need any of it. On every launch, SplashActivity calls    ║
# ║  doLoginByToken.php with just a random UUID. The server returns a       ║
# ║  fresh session token — no account required. We do the same.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Session:
    """Manages the anonymous API session (uuid + token from doLoginByToken)."""

    def __init__(self):
        self.uuid = ""
        self.token = ""
        self.login_token_id = None

    def login(self) -> bool:
        """Call doLoginByToken.php with a random device UUID to get a session."""
        log.info("Obtaining anonymous session via doLoginByToken.php ...")
        seed_uuid = str(uuid_mod.uuid4())
        params = [
            ("uuid", seed_uuid),
            ("push_key", "listener"),
            ("platform", PLATFORM),
            ("device", DEVICE_NAME),
            ("os_version", OS_VERSION),
            ("version", APP_VERSION),
            ("token", ""),
            ("uid", ""),
        ]
        data = _raw_call(params, "doLoginByToken.php")
        if not data:
            log.error("  login failed – no response")
            return False
        aui = data.get("app_user_info", {})
        self.uuid = aui.get("uuid", "")
        self.token = aui.get("token", "")
        self.login_token_id = aui.get("login_token_id")
        if self.uuid and self.token:
            log.info("  session OK – login_token_id=%s", self.login_token_id)
            return True
        log.error("  login failed – no uuid/token in response")
        return False

    def base_params(self) -> list[tuple[str, str]]:
        return [
            ("uuid", self.uuid),
            ("push_key", "listener"),
            ("platform", PLATFORM),
            ("device", DEVICE_NAME),
            ("os_version", OS_VERSION),
            ("version", APP_VERSION),
            ("token", self.token),
        ]


session = Session()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 4 — API Caller (mirrors HttpManager.startConnection)             ║
# ║                                                                         ║
# ║  Encrypts params, POSTs to the single endpoint, decrypts response.     ║
# ║  Auto re-authenticates if the server returns result code 103.           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _raw_call(params: list[tuple[str, str]], action: str) -> dict | None:
    """Low-level API call – builds encrypted POST from raw param list."""
    encoded_pairs = "&".join(
        f"{k}={urllib.parse.quote(v, safe='')}" for k, v in params
    )
    vars_encoded = urllib.parse.quote(encoded_pairs, safe="")

    payload = {
        "php": aes.encrypt(action.encode("utf-8")),
        "vars": aes.encrypt(vars_encoded.encode("utf-8")),
    }

    try:
        resp = requests.post(API_URL, data=payload, timeout=15)
        resp.raise_for_status()
        decrypted = aes.decrypt(resp.text)
        return json.loads(decrypted)
    except requests.RequestException as e:
        log.warning("HTTP error calling %s: %s", action, e)
    except Exception as e:
        log.warning("Decrypt/parse error for %s: %s", action, e)
    return None


def call_api(
    action: str,
    extra_params: list[tuple[str, str]] | None = None,
    auth: dict | None = None,
) -> dict | None:
    """
    Call the 走鬼 API.
    - action: the PHP file name, e.g. "getNearByAlertLocation.php"
    - extra_params: additional key-value pairs
    - auth: ignored (kept for compat) – session credentials used automatically
    Returns the parsed JSON response dict, or None on failure.

    If the server returns result 103 (session expired), auto-re-login once.
    """
    params = session.base_params()
    if extra_params:
        params.extend(extra_params)

    data = _raw_call(params, action)

    # Auto re-login on session expiry
    if data and data.get("result") == 103:
        log.info("Session expired – re-authenticating ...")
        if session.login():
            params = session.base_params()
            if extra_params:
                params.extend(extra_params)
            data = _raw_call(params, action)

    return data


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 5 — Local JSON Store (de-duplicated alert archive)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": {}, "news": [], "sponsors": [], "meta": {"first_run": _now()}}


def save_store(store: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def merge_alerts(store: dict, records: list[dict], source: str):
    """De-duplicate alerts by alert_record_id and merge into the store."""
    new_count = 0
    for rec in records:
        rid = rec.get("alert_record_id", "")
        if not rid:
            continue
        if rid not in store["alerts"]:
            rec["_source"] = source
            rec["_first_seen"] = _now()
            store["alerts"][rid] = rec
            new_count += 1
        else:
            # Update mutable fields (votes can change)
            existing = store["alerts"][rid]
            for field in ("upvote", "downvote", "lastupdate_dt", "user_up", "user_down"):
                if rec.get(field):
                    existing[field] = rec[field]
            existing["_last_seen"] = _now()
    return new_count


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 6 — Polling Engine (5 data sources per cycle)                    ║
# ║                                                                         ║
# ║  Each cycle calls:                                                      ║
# ║    1. appChecking.php    → sponsors, config          (0 params)         ║
# ║    2. getNews.php        → news articles             (0 params)         ║
# ║    3. getNotificationRecord.php → recent alerts      (0 params)         ║
# ║    4. getNearByAlertLocationIn24Hours.php × 115 cells (lat/lng grid)     ║
# ║    5. getNearByAlertLocation.php × 115 cells          (lat/lng grid)     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def poll_app_checking(store: dict):
    """appChecking.php – sponsors + startup info."""
    log.info("Polling appChecking.php ...")
    data = call_api("appChecking.php")
    if not data:
        return
    # Sponsors
    sponsors = data.get("sponsor_list", [])
    if sponsors:
        store["sponsors"] = sponsors
        log.info("  sponsors: %d", len(sponsors))
    # Startup message
    sm = data.get("startupMessage")
    if sm:
        store["startup_message"] = sm
    # Other config fields
    for key in ("gang_title", "gang_content", "gang_btn_text",
                "sponsor_intro", "tire_popped_contact",
                "location_message_title", "location_message_content"):
        if data.get(key):
            store.setdefault("config", {})[key] = data[key]


def poll_news(store: dict):
    """getNews.php – news articles."""
    log.info("Polling getNews.php ...")
    data = call_api("getNews.php")
    if not data:
        return
    news = data.get("news", data.get("news_list", []))
    if isinstance(news, list) and news:
        store["news"] = news
        log.info("  news items: %d", len(news))


def poll_notification_record(store: dict):
    """getNotificationRecord.php – recent notification alerts."""
    log.info("Polling getNotificationRecord.php ...")
    data = call_api("getNotificationRecord.php")
    if not data:
        return
    records = data.get("alert_record", [])
    if records:
        n = merge_alerts(store, records, "notificationRecord")
        log.info("  notification alerts: %d total, %d new", len(records), n)


def poll_nearby_alerts(store: dict):
    """Sweep a lat/lng grid across HK for getNearByAlertLocationIn24Hours.php."""
    log.info("Sweeping HK grid for 24h alerts ...")
    total_new = 0
    lat = HK_LAT_MIN
    while lat <= HK_LAT_MAX:
        lng = HK_LNG_MIN
        while lng <= HK_LNG_MAX:
            data = call_api(
                "getNearByAlertLocationIn24Hours.php",
                [("lat", f"{lat:.4f}"), ("lng", f"{lng:.4f}")],
            )
            if data:
                records = data.get("alert_record", [])
                if records:
                    n = merge_alerts(store, records, f"nearby24h({lat:.4f},{lng:.4f})")
                    total_new += n
            lng += GRID_STEP
        lat += GRID_STEP
    log.info("  grid sweep done – %d new alerts", total_new)


def poll_nearby_active(store: dict):
    """Sweep grid for getNearByAlertLocation.php (currently-active alerts)."""
    log.info("Sweeping HK grid for active alerts ...")
    total_new = 0
    lat = HK_LAT_MIN
    while lat <= HK_LAT_MAX:
        lng = HK_LNG_MIN
        while lng <= HK_LNG_MAX:
            data = call_api(
                "getNearByAlertLocation.php",
                [("lat", f"{lat:.4f}"), ("lng", f"{lng:.4f}")],
            )
            if data:
                records = data.get("alert_record", [])
                if records:
                    n = merge_alerts(store, records, f"nearbyActive({lat:.4f},{lng:.4f})")
                    total_new += n
            lng += GRID_STEP
        lat += GRID_STEP
    log.info("  active sweep done – %d new alerts", total_new)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LAYER 7 — Main Loop                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_once(store: dict):
    """Execute one full polling cycle."""
    poll_app_checking(store)
    poll_news(store)
    poll_notification_record(store)
    poll_nearby_alerts(store)
    poll_nearby_active(store)
    store["meta"]["last_poll"] = _now()
    store["meta"]["total_alerts"] = len(store["alerts"])


def main():
    parser = argparse.ArgumentParser(description="走鬼 Ghost Alert Listener")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output JSON file")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_SEC, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()

    log.info("=== 走鬼 Ghost Alert Listener ===")
    log.info("API:    %s", API_URL)
    log.info("Output: %s", os.path.abspath(args.output))
    log.info("Grid:   %.2f° step (~%d cells)", GRID_STEP,
             int(((HK_LAT_MAX - HK_LAT_MIN) / GRID_STEP + 1) * ((HK_LNG_MAX - HK_LNG_MIN) / GRID_STEP + 1)))

    # Obtain anonymous session (no Firebase login needed)
    if not session.login():
        log.error("Failed to obtain session – exiting")
        return

    store = load_store(args.output)

    # SQLite integration for ML pipeline — store DB next to the JSON output
    if _HAS_DB:
        db_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "ghost_alerts.db")
        db = GhostDB(db_path)
    else:
        db = None
    _inserted_ids: set[str] = set()

    while True:
        try:
            run_once(store)
            save_store(store, args.output)
            log.info(
                "Saved %d unique alerts, %d sponsors, %d news to %s",
                len(store["alerts"]),
                len(store.get("sponsors", [])),
                len(store.get("news", [])),
                args.output,
            )
            # Write only new alerts to SQLite
            if db:
                new_alerts = [
                    rec for rid, rec in store.get("alerts", {}).items()
                    if str(rid) not in _inserted_ids
                ]
                if new_alerts:
                    db.insert_sightings(new_alerts)
                    _inserted_ids.update(str(r.get("alert_record_id", "")) for r in new_alerts)
                db.insert_poll_cycle(
                    timestamp=store["meta"]["last_poll"],
                    total_alerts=len(store["alerts"]),
                    new_alerts=len(new_alerts),
                    duration_sec=0,
                )
        except KeyboardInterrupt:
            save_store(store, args.output)
            log.info("Interrupted – data saved.")
            raise
        except Exception:
            log.exception("Error during poll cycle")

        if args.once:
            break

        log.info("Sleeping %ds until next cycle ...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
