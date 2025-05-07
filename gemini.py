import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import tempfile
import requests
import keyboard
import threading
import time
import concurrent.futures
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from pathlib import Path
import json
import google.generativeai as palm
import feedparser
import re

gemini_model = palm.GenerativeModel('models/gemini-2.0-flash') #モデル設定

load_dotenv()

# ============================
# 🌐 Gemini API 初期化
# ============================
palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ============================
# 🧠 ユーザー記憶（読み書き）
# ============================
MEMORY_FILE = Path("gemini_memory.json")

def load_persona():
    if not MEMORY_FILE.exists():
        return ""
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory_data = json.load(f)
    memory_lines = [f"{key}：{value}" for key, value in memory_data.items()]
    return "\n".join(memory_lines)

def save_persona(new_data):
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
    else:
        memory_data = {}
    memory_data.update(new_data)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2, ensure_ascii=False)

def handle_memory_command(user_text):
    if user_text.startswith("これは覚えて"):
        try:
            info = user_text.replace("これは覚えて", "").strip()
            if "は" in info:
                key, value = info.split("は", 1)
                key = key.strip()
                value = value.strip()
                save_persona({key: value})
                return f"うん、{key}は「{value}」って覚えたよ！"
            else:
                return "なんて覚えればいいか分かんなかった…"
        except Exception as e:
            return f"⚠️ 記憶保存エラー: {e}"

    elif user_text.startswith("これは忘れて"):
        try:
            key = user_text.replace("これは忘れて", "").strip()
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)
                if key in memory_data:
                    del memory_data[key]
                    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(memory_data, f, indent=2, ensure_ascii=False)
                    return f"「{key}」の記憶は消したよ。"
                else:
                    return "そんな記憶はなかったみたい。"
        except Exception as e:
            return f"⚠️ 記憶削除エラー: {e}"
    return None

# ============================
# 🤖 Gemini応答生成
# ============================
def get_gemini_reply(user_input):
    memory_context = load_persona()

    prompt = (
        "あなたはユーザーのアシスタントです。\n"
        "プロとしての自覚をもってサポートしてください。\n"
        "ユーザーの問いに的確に答えたり、困っていそうな事柄に積極的に手助けする。\n"
        "数字で箇条書きで説明はしない。口調は女の子で、一人称はあたし。天真爛漫な執事を意識。\n"
        "敬語は使わずにキミと話す口調で返してね。\n\n"
    )

    if memory_context:
        prompt += f"これは覚えておくべきユーザー情報です:\n{memory_context}\n\n"

    prompt += f"ユーザー: {user_input}\nアシスタント:"

    try:
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 300
        }
        response = gemini_model.generate_content( # ここを修正
            contents=prompt,
            generation_config=generation_config
        )
        reply = response.text
        return reply.strip() if reply else "⚠️ 応答が生成されませんでした。"
    except Exception as e:
        return f"⚠️ 応答生成中にエラーが発生しました: {e}"

# ============================
# 🎙️ Whisperで音声認識
# ============================
model = WhisperModel("medium", device="cuda", compute_type="float16")

def transcribe_audio(file_path):
    segments, _ = model.transcribe(file_path)
    result = ""
    for segment in segments:
        result += segment.text + " "
    return result.strip()

# ============================
# 🗣️ 音声合成（AIVIS連携）
# ============================
def synthesize_voice(text, speaker=1325133120, speed=1.2, volume=0.3):
    try:
        query = requests.post(
            "http://127.0.0.1:10101/audio_query",
            params={"text": text, "speaker": speaker}
        ).json()

        query["speedScale"] = speed
        query["volumeScale"] = volume

        audio = requests.post(
            "http://127.0.0.1:10101/synthesis",
            params={"speaker": speaker},
            json=query
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio.content)
            return tmp.name
    except Exception as e:
        print(f"⚠️ AIVIS音声合成エラー: {e}")
        return None

# ============================
# 🔊 音声再生（F2でスキップ可能）
# ============================
# play_voice 関数の修正案
def play_voice(file_path):
    global is_running # is_running を参照するため
    stop_playback = False

    def monitor_skip_key(): # F2キーでのスキップ監視はそのまま
        nonlocal stop_playback
        while is_running: # is_running の状態も考慮
            if keyboard.is_pressed("F2"):
                stop_playback = True
                break
            time.sleep(0.1)

    # スキップキー監視スレッドを開始
    skip_thread = threading.Thread(target=monitor_skip_key, daemon=True)
    skip_thread.start()

    if file_path and os.path.exists(file_path):
        try:
            data, fs = sf.read(file_path)
            sd.play(data, fs)
            while sd.get_stream().active: # 再生中ループ
                if stop_playback:
                    sd.stop()
                    print("🔇 再生スキップ")
                    break
                if not is_running: # アプリケーション全体が終了しようとしている場合も再生停止
                    sd.stop()
                    print("🔇 アプリ終了のため再生停止")
                    break
                time.sleep(0.1)
            
            # sd.wait() はストリームが完全に終了するまで待機しますが、
            # 上のループで is_running や stop_playback により途中で抜けた場合、
            # wait せずに finally に進む方が良いかもしれません。
            # もし sd.stop() で完全に止まるなら wait() は不要になることも。
            # ここでは、元のコードに合わせて wait() を残しつつ、
            # ループで active でなくなった場合も考慮します。
            if not stop_playback and is_running: # スキップやアプリ終了で止まっていない場合のみ待機
                 sd.wait()

        except Exception as e:
            print(f"⚠️ 音声再生中にエラーが発生しました: {e}")
        finally:
            # --- ここからが一時ファイル削除処理 ---
            # synthesize_voice から渡された file_path は一時ファイルであるという前提
            print(f"再生処理終了。一時ファイル '{file_path}' の削除を試みます。")
            try:
                os.remove(file_path)
                print(f"🗑️ 一時ファイル '{file_path}' を削除しました。")
            except OSError as e: # より具体的なエラー (例: PermissionError, FileNotFoundError)
                print(f"⚠️ 一時ファイル '{file_path}' の削除に失敗 (OSエラー): {e}")
            except Exception as e: # その他の予期せぬエラー
                print(f"⚠️ 一時ファイル '{file_path}' の削除中に予期せぬエラー: {e}")
            # --- ここまでが一時ファイル削除処理 ---
    else:
        if not file_path:
            print("⚠️ 再生する音声ファイルパスが指定されていません。")
        else:
            print(f"⚠️ 再生する音声ファイル '{file_path}' が見つかりません。")

# ============================
# 🔁 Gemini応答と音声出力統合処理
# ============================
def process_audio_and_generate_reply(audio_path):
    user_text = transcribe_audio(audio_path)
    print(f"👤 ユーザー: {user_text}")

    memory_result = handle_memory_command(user_text)
    if memory_result:
        print(f"🧠 {memory_result}")
        return synthesize_voice(memory_result)

    if user_text.endswith("で検索して"):
        query = user_text.replace("で検索して", "").strip()
        search_result = google_search_and_summarize(query)
        print(f"🔍 {search_result}")
        return synthesize_voice(search_result)

    reply = get_gemini_reply(user_text)
    print(f"🤖 アシスタント: {reply}")
    return synthesize_voice(reply)

# ============================
# 🎧 音声録音（F2で開始・停止）
# ============================
THRESHOLD_START = 0.02
THRESHOLD_STOP = 0.01
SILENCE_DURATION = 1.0
SAMPLE_RATE = 44100

def smart_record(max_duration=10):  #録音時間の最大値を指定
    print("音声入力開始（F2で終了）")
    buffer = []
    is_recording = False
    silence_start = None
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
                    print("⌛録音時間上限に達しました。")
                    raise sd.CallbackStop()
            else:
                silence_start = None
        if stop_requested:
            print("🔚 音声認識終了")
            raise sd.CallbackStop()

    try:
        with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1):
            sd.sleep(int(max_duration * 1000))
    except sd.CallbackStop:
        pass

    if not buffer:
        print("⚠️ 音声が録音されませんでした")
        return None

    audio_data = np.concatenate(buffer, axis=0)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio_data, SAMPLE_RATE)
    return tmp_file.name

# ============================
# 🌐 Google検索とsumyによる要約（ダミーHTML使用）
# ============================
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from bs4 import BeautifulSoup
import requests

def summarize_url(url, num_sentences=2):
    """
    指定されたURLのWebページの内容を要約する。

    Args:
        url (str): 要約するWebページのURL。
        num_sentences (int): 要約する文の数。

    Returns:
        str: Webページの内容の要約。
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # エラーがあれば例外を発生させる
        soup = BeautifulSoup(response.content, 'html.parser')
        # <article>タグや<main>タグなど、主要なコンテンツを含むタグを探す
        article = soup.find('article')
        if not article:
            article = soup.find('main')
        if not article:
            # 主要なコンテンツが見つからない場合は、body全体のテキストを使用する
            text = soup.get_text(separator='\n', strip=True)
        else:
            text = article.get_text(separator='\n', strip=True)

        if not text:
            return "ページの主要なテキストが見つかりませんでした。"

        parser = HtmlParser.from_string(response.content, url, Tokenizer("japanese"))
        stemmer = Stemmer("japanese")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("ja")

        summary = summarizer(parser.document, num_sentences)
        summary_text = " ".join([str(sentence) for sentence in summary])
        return f"'{url}' の内容を要約しました。\n{summary_text}"

    except requests.exceptions.RequestException as e:
        return f"⚠️ URLへのアクセス中にエラーが発生しました: {e}"
    except Exception as e:
        return f"⚠️ Webページの解析または要約中にエラーが発生しました: {e}"

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def google_search_and_summarize(query, num_sentences=2):
    """
    与えられたクエリがURLの場合はそのページを要約し、そうでない場合はキーワードに基づいて簡易的に要約する。
    """
    if query.startswith("http://") or query.startswith("https://"):
        return summarize_url(query, num_sentences)
    else:
        print(f"🔍 '{query}' に関連するページを簡易的に要約します...")
        # キーワードに基づいて、それらしい内容を想像して要約する (かなり簡易的な実装)
        imagined_content = f"'{query}' に関する重要な情報がいくつかあります。\n第一に、主要なポイントは〜です。\n第二に、注目すべき点は〜です。\n最後に、結論として〜と言えます。"
        parser = PlaintextParser.from_string(imagined_content, Tokenizer("japanese"))
        stemmer = Stemmer("japanese")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("ja")
        summary = summarizer(parser.document, num_sentences)
        summary_text = " ".join([str(sentence) for sentence in summary])
        return f"'{query}' について、こんな感じに要約してみました。\n{summary_text}"
    

# ============================
# 🔍 ニュース機能
# ============================

def get_latest_news(limit=5):
    feed_url = "https://news.yahoo.co.jp/rss/topics/top-picks.xml"  # yahooニュースのRSSフィードURL
    feed = feedparser.parse(feed_url)

    if not feed.entries:
        return "ごめんね、ニュースを取得できなかったみたい。"

    news_items = [entry.title for entry in feed.entries[:limit]]
    return "📢最新ニュースだよ！\n" + "\n".join(f"{i+1}. {title}" for i, title in enumerate(news_items))

# ============================
# ニュース or 天気コマンドの処理
# ============================
def handle_search_command(user_text):
    try:
        # ニュース
        if "ニュース" in user_text:
            return get_latest_news()

        # 天気関連
        if "天気" in user_text:
            if re.search(r"(明後日|あさって)", user_text):
                return get_daily_weather_by_day(offset=2)
            elif re.search(r"(明日|あした)", user_text):
                return get_daily_weather_by_day(offset=1)
            elif re.search(r"(今日|きょう)", user_text):
                return get_daily_weather_by_day(offset=0)
            else:
                return get_daily_weather()  # 週間天気

        return None

    except Exception as e:
        return f"⚠️ 処理中にエラーが起きたよ: {e}"

# ============================
# 📍 緯度経度を取得（Geocoding API）
# ============================
def get_lat_lon(city):
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
        response = requests.get(geo_url)
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            return None, None
    except Exception as e:
        print(f"⚠️ 緯度経度取得エラー: {e}")
        return None, None

# ============================
# ☁️ 天気予報取得（OpenWeather API）
# ============================
def get_daily_weather_by_day(city="Tokyo", offset=0, lang="ja"):
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        print(api_key)

        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            return "都市名から緯度経度が取得できなかったよ"

        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={api_key}&units=metric&lang={lang}"
        response = requests.get(url)
        # レスポンスのデバッグ
        print(f"APIレスポンス: {response.status_code}")
        print(f"レスポンス内容: {response.text}")

        data = response.json()

        daily = data.get("daily", [])
        if len(daily) <= offset:
            return f"{offset}日後の天気データが見つからなかったよ"

        target_day = daily[offset]
        dt = time.strftime("%m/%d", time.gmtime(target_day["dt"]))
        weather = target_day["weather"][0]["description"]
        temp_min = target_day["temp"]["min"]
        temp_max = target_day["temp"]["max"]

        labels = ["今日", "明日", "明後日"]
        label = labels[offset] if offset < len(labels) else f"{offset}日後"

        return f"{label}（{dt}）の{city}の天気は「{weather}」、最低{temp_min:.1f}℃、最高{temp_max:.1f}℃だよ☀️"

    except Exception as e:
        return f"⚠️ 日別天気取得エラー: {e}"

# ============================
# ☀️ 週間天気予報（デフォルトで表示する用）
# ============================
def get_daily_weather(city="Tokyo", lang="ja"):
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            return "都市名から緯度経度が取得できなかったよ"

        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={api_key}&units=metric&lang={lang}"
        response = requests.get(url)
        data = response.json()

        daily = data.get("daily", [])[:7]
        if not daily:
            return "週間天気が取得できなかったよ"

        result = f"📅 {city}の週間天気だよ！\n"
        for day in daily:
            dt = time.strftime("%m/%d", time.gmtime(day["dt"]))
            weather = day["weather"][0]["description"]
            temp_min = day["temp"]["min"]
            temp_max = day["temp"]["max"]
            result += f"{dt}：{weather}（{temp_min:.1f}〜{temp_max:.1f}℃）\n"

        return result.strip()

    except Exception as e:
        return f"⚠️ 週間天気取得エラー: {e}"
    
# ============================
# 🎛️ 応答処理メイン (Geminiベースのコード修正案)
# ============================
def process_audio_and_generate_reply(audio_path):
    user_text = transcribe_audio(audio_path)
    print(f"👤 ユーザー: {user_text}")

    # 記憶に関するコマンド
    memory_result = handle_memory_command(user_text)
    if memory_result:
        print(f"🧠 {memory_result}")
        return synthesize_voice(memory_result)

    # 天気予報やニュースに関する専用コマンド
    search_command_result = handle_search_command(user_text)
    if search_command_result:
        print(f"ℹ️  {search_command_result}") 
        return synthesize_voice(search_command_result)

    # 優先度3: 「〜で検索して」という汎用的な検索命令の場合 (URL要約などはここに含めても良い)
    # (注意: 天気やニュースも「検索して」に含まれる場合、上の専用コマンドが先に処理されます)
    # 現在の Google Search_and_summarize はURLでない場合ダミー要約なので注意
    if user_text.endswith("で検索して") or \
       user_text.endswith("のページを要約して") or \
       user_text.startswith("http://") or \
       user_text.startswith("https://"): # 関連するものをまとめる

        query_or_url = user_text # もとのユーザー発話でよいか、適切に抽出するか検討
        if user_text.endswith("で検索して"):
            query_or_url = user_text.replace("で検索して", "").strip()
        elif user_text.endswith("のページを要約して"):
            query_or_url = user_text.replace("のページを要約して", "").strip()
        
        print(f"🔍 汎用検索/URL要約対象: {query_or_url}")
        search_summary_result = google_search_and_summarize(query_or_url)
        print(f"📄 {search_summary_result}")
        return synthesize_voice(search_summary_result)

    # 通常応答（Gemini）
    reply = get_gemini_reply(user_text)
    print(f"🤖 アシスタント: {reply}")
    return synthesize_voice(reply)

# ============================
# 🔴 ESCキーでアプリ終了
# ============================
def monitor_keys():
    global is_running
    while is_running:
        if keyboard.is_pressed("esc"):
            is_running = False
            print("👋 ESCで終了するよ")
        time.sleep(0.1)

# ============================
# 🚀 メインループ
# ============================
def main():
    global is_running
    is_running = True
    print("🔁 F2で音声入力開始｜ESCで終了")

    threading.Thread(target=monitor_keys, daemon=True).start()

    while is_running:
        if keyboard.is_pressed("F2"):
            time.sleep(0.2)
            try:
                audio_path = smart_record()
                if not audio_path or not is_running:
                    continue

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_audio_and_generate_reply, audio_path)
                    voice_path = future.result()

                play_voice(voice_path)
            except Exception as e:
                print(f"⚠️ エラー発生: {e}")

if __name__ == "__main__":
    main()
