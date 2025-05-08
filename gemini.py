"""
assistant_ai.py

🎯 目的
    ・音声入力 → Whisper 文字起こし
    ・Gemini-2.0-Flash で回答生成
    ・AIVISpeech で音声合成 → 再生
    ・覚えて／忘れて／ブラウザ要約 など既存機能はそのまま
    ・ESC で終了、F2 で録音トグル

💾 依存ライブラリ  (例: requirements.txt)
    faster-whisper==1.0.1
    google-generativeai==0.5.2
    duckduckgo_search==5.3.0
    feedparser==6.0.11
    sounddevice==0.4.7
    soundfile==0.12.1
    python-dotenv==1.0.1
    Flask==3.0.2
    flask-cors==4.0.0
    beautifulsoup4==4.12.3
"""

# ======= 📦 標準／外部モジュール =======
import os
import time
import json
import re
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
import keyboard
import feedparser
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from faster_whisper import WhisperModel
import google.generativeai as genai                     # ✅ Gemini SDK
from duckduckgo_search import DDGS

from flask import Flask, request
from flask_cors import CORS

# ======= 🔧 環境変数ロード =======
load_dotenv()
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ======= 🔧 Gemini 初期化 =======
genai.configure(api_key=GENAI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")  # ← ここでモデル指定

# ======= 🔧 Whisper 初期化 =======
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

# ======= 🧠 記憶管理ファイル =======
MEMORY_FILE = Path("gpt_memory.json")
MEMORY_LOCK = threading.Lock()   # 同時書き込み対策

# ======= 🎚️ 録音パラメータ =======
THRESHOLD_START = 0.02   # 認識開始音量
THRESHOLD_STOP  = 0.01   # 無音判定音量
SILENCE_DURATION = 1.0   # 無音継続秒
SAMPLE_RATE = 44_100

# ======= 🌐 Flask（ブラウザ共有） =======
app = Flask(__name__)
CORS(app)
browser_data: dict = {}

# ======= 💬 会話履歴（deque で上限 30） =======
from collections import deque
messages = deque(maxlen=30)  # {"role": "...", "content": "..."}

# ======= 🏁 アプリ稼働フラグ =======
is_running = True   # ESC で False に

# ---------------------------------------------------------------------
# 🧠 記憶ヘルパ
# ---------------------------------------------------------------------
def load_persona() -> str:
    """JSON からユーザー記憶を読み取り、システムプロンプト用文字列に変換"""
    if not MEMORY_FILE.exists():
        return ""
    with open(MEMORY_FILE, encoding="utf-8") as f:
        memory_data = json.load(f)
    memory_lines = [f"{k}：{v}" for k, v in memory_data.items()]
    return "これは覚えておくべきユーザー情報です。\n" + "\n".join(memory_lines)

def save_persona(new_data: dict):
    """記憶 JSON に key:value を追加／更新（排他制御付き）"""
    with MEMORY_LOCK:
        memory_data = {}
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, encoding="utf-8") as f:
                memory_data = json.load(f)
        memory_data.update(new_data)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

def handle_memory_command(user_text: str):
    """『覚えて～』『これは忘れて～』『～って覚えてる？』を処理"""
    try:
        if user_text.startswith("覚えて"):
            info = user_text.replace("覚えて", "").strip()
            if "は" in info:
                key, value = info.split("は", 1)
                save_persona({key.strip(): value.strip()})
                return f"うん、{key.strip()}は『{value.strip()}』って覚えたよ！"
            return "うーん、なんて覚えればいいか分かんなかった..."

        if user_text.startswith("これは忘れて"):
            key = user_text.replace("これは忘れて", "").strip()
            with MEMORY_LOCK:
                if MEMORY_FILE.exists():
                    with open(MEMORY_FILE, encoding="utf-8") as f:
                        memory_data = json.load(f)
                    if key in memory_data:
                        del memory_data[key]
                        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(memory_data, f, indent=2, ensure_ascii=False)
                        return f"『{key}』って記憶は消したよ"
            return f"『{key}』って記憶はなかったみたい"

        if user_text.endswith("って覚えてる？"):
            key = user_text.replace("って覚えてる？", "").strip()
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, encoding="utf-8") as f:
                    memory_data = json.load(f)
                if key in memory_data:
                    return f"うん、『{key}』は『{memory_data[key]}』って覚えてるよ！"
            return f"ごめん、『{key}』は覚えてないみたい…"
    except Exception as e:
        return f"⚠️ 記憶処理エラー: {e}"
    return None

# ---------------------------------------------------------------------
# 🔍 ニュース／天気／ブラウザ要約
# ---------------------------------------------------------------------
def get_latest_news(limit=5):
    feed_url = "https://news.yahoo.co.jp/rss/topics/top-picks.xml"
    feed = feedparser.parse(feed_url)
    if not feed.entries:
        return "ごめんね、ニュースを取得できなかったみたい。"
    items = [entry.title for entry in feed.entries[:limit]]
    return "📢 最新ニュースだよ！\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))

def get_lat_lon(city):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY}
    try:
        res = requests.get(geo_url, params=params, timeout=10).json()
        if res:
            return res[0]["lat"], res[0]["lon"]
    except Exception as e:
        print("⚠️ 緯度経度取得エラー:", e)
    return None, None

def get_daily_weather_by_day(city="Tokyo", offset=0, lang="ja"):
    lat, lon = get_lat_lon(city)
    if lat is None:
        return "都市名から緯度経度が取得できなかったよ"
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon, "exclude": "current,minutely,hourly,alerts",
        "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": lang
    }
    try:
        daily = requests.get(url, params=params, timeout=10).json().get("daily", [])
        if len(daily) <= offset:
            return "その日の天気データが見つからなかったよ"
        day = daily[offset]
        dt = time.strftime("%m/%d", time.gmtime(day["dt"]))
        weather = day["weather"][0]["description"]
        Tmin, Tmax = day["temp"]["min"], day["temp"]["max"]
        label = ["今日", "明日", "明後日"][offset] if offset < 3 else f"{offset}日後"
        return f"{label}（{dt}）の{city}の天気は「{weather}」、最低{Tmin:.1f}℃、最高{Tmax:.1f}℃だよ☀️"
    except Exception as e:
        return f"⚠️ 天気取得エラー: {e}"

def get_daily_weather(city="Tokyo", lang="ja"):
    lat, lon = get_lat_lon(city)
    if lat is None:
        return "都市名から緯度経度が取得できなかったよ"
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat, "lon": lon, "exclude": "current,minutely,hourly,alerts",
        "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": lang
    }
    try:
        daily = requests.get(url, params=params, timeout=10).json().get("daily", [])[:7]
        if not daily:
            return "週間天気が取得できなかったよ"
        lines = []
        for d in daily:
            dt = time.strftime("%m/%d", time.gmtime(d["dt"]))
            desc = d["weather"][0]["description"]
            Tmin, Tmax = d["temp"]["min"], d["temp"]["max"]
            lines.append(f"{dt}：{desc}（{Tmin:.1f}〜{Tmax:.1f}℃）")
        return "📅 週間天気だよ！\n" + "\n".join(lines)
    except Exception as e:
        return f"⚠️ 天気取得エラー: {e}"

def handle_search_command(text):
    if "ニュース" in text:
        return get_latest_news()
    if "天気" in text:
        if re.search(r"(明後日|あさって)", text):
            return get_daily_weather_by_day(offset=2)
        if re.search(r"(明日|あした)", text):
            return get_daily_weather_by_day(offset=1)
        if re.search(r"(今日|きょう)", text):
            return get_daily_weather_by_day(offset=0)
        return get_daily_weather()
    return None

# ---------------------------------------------------------------------
# 🤖 Gemini 応答生成
# ---------------------------------------------------------------------
SYSTEM_PROMPT = (
    "あなたはユーザーのアシスタントです。"
    "プロとしての自覚をもってサポートしてください。"
    "ユーザーの問いに的確に答えたり、困っていそうな事柄に積極的に手助けする。"
    "数字で箇条書きで説明はしない。口調は女の子で、明るく知的に。"
    "敬語は使わずにキミと話す口調で返してね。"
)

def convert_history(msgs):
    """OpenAI 形式 → Gemini 形式に変換"""
    return [{"role": m["role"], "parts": [m["content"]]} for m in msgs]

# ---------------------------------------------
# 🤖 Gemini 応答生成（修正版）
# ---------------------------------------------
SYSTEM_PROMPT = (
    "あなたはユーザーのアシスタントです。"
    "プロとしての自覚をもってサポートしてください。"
    "ユーザーの問いに的確に答えたり、困っていそうな事柄に積極的に手助けする。"
    "数字で箇条書きで説明はしない。口調は女の子で、明るく知的に。"
    "敬語は使わずにキミと話す口調で返してね。"
)

# -------------------------------------------------
# 🤖 Gemini 応答生成（system ロール禁止版）
# -------------------------------------------------
SYSTEM_PROMPT = (
    "あなたはユーザーのアシスタントです。"
    "プロとしての自覚をもってサポートしてください。"
    "ユーザーの問いに的確に答えたり、困っていそうな事柄に積極的に手助けする。"
    "数字で箇条書きで説明はしない。口調は女の子で、明るく知的に。"
    "敬語は使わずにキミと話す口調で返してね。"
)

def to_chat_history(msgs):
    """deque → Gemini 用 history（user / model だけ）"""
    chat = []
    for m in msgs:
        role = "user" if m["role"] == "user" else "model"
        chat.append({"role": role, "parts": [m["content"]]})
    return chat

def get_gpt_reply(user_input: str) -> str:
    # 🧠 記憶をロード
    memory = load_persona()          # 空なら ""

    # ① 1 本目の user メッセージにシステム指示+記憶を詰め込む
    preamble = SYSTEM_PROMPT + ("\n" + memory if memory else "")

    # ② 直近履歴を user/model 形式で用意
    chat_history = [{"role": "user", "parts": [preamble]}]
    chat_history += to_chat_history(messages)
    chat_history.append({"role": "user", "parts": [user_input]})

    try:
        response = GEMINI_MODEL.generate_content(chat_history)
        reply = response.text.strip()

        # ③ 履歴を更新（deque なので自動で古い分は捨てる）
        messages.append({"role": "user",      "content": user_input})
        messages.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        return f"⚠️ Gemini 応答生成エラー: {e}"



# ---------------------------------------------------------------------
# 🎙️ 録音 → Whisper 文字起こし
# ---------------------------------------------------------------------
def smart_record(max_duration=8):
    """無音自動停止＋F2 で手動停止できる録音"""
    buffer, is_recording, silence_start = [], False, None
    stop_requested = False

    def monitor_stop_key():
        nonlocal stop_requested
        while True:
            if keyboard.is_pressed("F2"):
                stop_requested = True
                break
            time.sleep(0.1)
    threading.Thread(target=monitor_stop_key, daemon=True).start()

    def callback(indata, frames, time_info, status):
        nonlocal is_recording, silence_start, buffer
        volume = np.linalg.norm(indata)
        if not is_recording and volume > THRESHOLD_START:
            is_recording = True
            buffer.append(indata.copy())
        elif is_recording:
            buffer.append(indata.copy())
            if volume < THRESHOLD_STOP:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    raise sd.CallbackStop()
            else:
                silence_start = None
        if stop_requested:
            raise sd.CallbackStop()

    with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1):
        try:
            sd.sleep(int(max_duration * 1000))
        except sd.CallbackStop:
            pass

    if not buffer:
        return None
    audio_data = np.concatenate(buffer, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio_data, SAMPLE_RATE)
    return tmp.name

def transcribe_audio(path: str) -> str:
    """Whisper で文字起こし"""
    segments, _ = whisper_model.transcribe(path)
    return " ".join(s.text.strip() for s in segments)

# ---------------------------------------------------------------------
# 🗣️ AIVISpeech 音声合成 → 再生
# ---------------------------------------------------------------------
def synthesize_voice(text: str, speaker=1325133120, speed=1.2, volume=0.3):
    """AIVISpeech エンジンで WAV を生成しパスを返す"""
    try:
        query = requests.post(
            "http://127.0.0.1:10101/audio_query",
            params={"text": text, "speaker": speaker}
        ).json()
        query.update(speedScale=speed, volumeScale=volume)
        audio = requests.post(
            "http://127.0.0.1:10101/synthesis",
            params={"speaker": speaker}, json=query
        )
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio.content)
        return tmp.name
    except Exception as e:
        print("⚠️ 音声合成エラー:", e)
        return None

def play_voice(path: str):
    """WAV を再生（F2 でスキップ可）"""
    if not path or not os.path.exists(path):
        print("⚠️ 再生ファイルなし")
        return
    stop_playback = False
    def monitor():
        nonlocal stop_playback
        while is_running:
            if keyboard.is_pressed("F2"):
                stop_playback = True
                break
            time.sleep(0.1)
    threading.Thread(target=monitor, daemon=True).start()
    data, fs = sf.read(path)
    sd.play(data, fs)
    while sd.get_stream().active:
        if stop_playback or not is_running:
            sd.stop()
            break
        time.sleep(0.1)
    sd.wait()
    try:
        os.remove(path)
    except Exception:
        pass

# ---------------------------------------------------------------------
# 🌐 Flask 受信エンドポイント
# ---------------------------------------------------------------------
@app.route("/browser-data", methods=["POST"])
def browser_data_endpoint():
    global browser_data
    browser_data = request.json or {}
    print("📂 受信ブラウザデータ:", browser_data)
    return "OK", 200

def run_flask_server():
    app.run(port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_flask_server, daemon=True).start()

def handle_browser_command():
    """最新ブラウザページを要約（Gemini-Flash 仕様準拠版）"""
    if not browser_data:
        return "🌐 ブラウザの情報がまだ受信されていないよ。"

    url   = browser_data.get("url")
    title = browser_data.get("title", "タイトルなし")

    try:
        html  = requests.get(url, timeout=10).text
        soup  = BeautifulSoup(html, "html.parser")
        text  = soup.get_text()[:3000]        # 3000 文字に切り詰め

        # ── Gemini 形式メッセージ ──
        first_user = (
            "以下のページ内容を日本語で分かりやすく簡潔に要約し、"
            "独自の観点で評価して雑談口調で話してね。\n"
            "タイトル: " + title +
            "\n内容:\n" + text
        )

        chat = [
            {"role": "user",  "parts": [first_user]}
        ]

        response = GEMINI_MODEL.generate_content(chat)
        return response.text.strip()

    except Exception as e:
        return f"要約生成中にエラーが発生: {e}"



# ---------------------------------------------------------------------
# 🎛️ 音声入力→応答 主処理
# ---------------------------------------------------------------------
def process_audio_and_generate_reply(audio_path):
    user_text = transcribe_audio(audio_path)
    print(f"👤 ユーザー: {user_text}")

    # ① 記憶系
    if mem := handle_memory_command(user_text):
        print("🧠", mem)
        return synthesize_voice(mem)

    # ② 検索系
    if result := handle_search_command(user_text):
        print("🔍", result)
        return synthesize_voice(result)

    # ③ ブラウザ要約
    if "ページの情報を教えて" in user_text:
        summary = handle_browser_command()
        print("🌐", summary)
        return synthesize_voice(summary)

    # ④ 通常対話
    reply = get_gpt_reply(user_text)
    print("🤖", reply)
    return synthesize_voice(reply)

# ---------------------------------------------------------------------
# ⌨️ キー監視 / ループ
# ---------------------------------------------------------------------
def monitor_keys():
    global is_running
    while is_running:
        if keyboard.is_pressed("esc"):
            is_running = False
            print("👋 ESC で終了")
        time.sleep(0.1)

def main():
    global is_running
    print("🔁 F2 で録音開始 / 終了 | ESC でアプリ終了")
    threading.Thread(target=monitor_keys, daemon=True).start()
    recording = False
    while is_running:
        if keyboard.is_pressed("F2"):
            time.sleep(0.2)               # チャタリング防止
            if not recording:
                recording = True
                try:
                    audio = smart_record()
                    if not audio:
                        print("⚠️ 録音失敗")
                        recording = False
                        continue
                    import concurrent.futures as cf
                    with cf.ThreadPoolExecutor() as ex:
                        voice_path = ex.submit(process_audio_and_generate_reply, audio).result()
                    play_voice(voice_path)
                except Exception as e:
                    print("⚠️ エラー:", e)
                finally:
                    recording = False
        time.sleep(0.1)

if __name__ == "__main__":
    main()
