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
from openai import OpenAI

# ============================
# 🎮 AI設定と初期化
# ============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

THRESHOLD_START = 0.02
THRESHOLD_STOP = 0.01
SILENCE_DURATION = 1.0import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import tempfile
import requests
import keyboard
import threading
import time
import concurrent.futures
import feedparser
from faster_whisper import WhisperModel
from openai import OpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from pathlib import Path
import re
from datetime import datetime, timedelta
import json

load_dotenv()

# ============================
# 🧠 記憶管理（読み書き機能）
# ============================
MEMORY_FILE = Path("tsuyoshi_memory.json")

def load_persona():
    if not MEMORY_FILE.exists():
        return ""
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory_data = json.load(f)
    memory_lines = [f"{key}：{value}" for key, value in memory_data.items()]
    return "これは覚えておくべきユーザー情報です。\n" + "\n".join(memory_lines)

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
                key = key.strip("、 。. ") 
                value = value.strip("、 。. ")
                save_persona({key: value})
                return f"うん、{key}は『{value}』って覚えたよ！"
            else:
                return "うーん、なんて覚えればいいか分かんなかった..."
        except Exception as e:
            return f"⚠️ 記憶処理でエラーが起きたよ: {e}"

    elif user_text.startswith("これは忘れて"):
        try:
            key = user_text.replace("これは忘れて", "").strip("、 。. ")
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    memory_data = json.load(f)

                if key in memory_data:
                    del memory_data[key]
                    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(memory_data, f, indent=2, ensure_ascii=False)
                    return f"『{key}』って記憶は消したよ"
                else:
                    return f"『{key}』って記憶はなかったみたい"
            else:
                return "まだ何も覚えてないよ"
        except Exception as e:
            return f"⚠️ 記憶削除でエラーが起きたよ: {e}"

    return None


# ============================
# 🎮 AI設定と初期化
# ============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

THRESHOLD_START = 0.02
THRESHOLD_STOP = 0.01
SILENCE_DURATION = 1.0
SAMPLE_RATE = 44100
is_running = True

messages = [
    {
        "role": "system",
        "content": "あなたはユーザーのアシスタントです。プロとしての自覚をもってサポートをしてください。ユーザーの問いに的確に答えたり、ユーザーが困っていそうな事柄について積極的に手助けをする。回答は分かりやすく短めにし、あくまで会話であることを意識。数字で箇条書きで説明はしない。口調は女の子、性格は明るく知的。敬語を使わず、キミと話す口調で返してね。"
    }
]

# ============================
# 🎧 音声録音（スペースキーで停止）
# ============================
def smart_record(max_duration=8):
    print("音声認識開始（スペースキーで終了）")
    buffer = []
    is_recording = False
    silence_start = None
    stop_requested = False

    def monitor_stop_key():
        nonlocal stop_requested
        while True:
            if keyboard.is_pressed("space"):  # F13キーからスペースキーに変更
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
                    print("無音検出で録音終了")
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
# 🔍 検索機能
# ============================
def web_search_duckduckgo(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        summaries = [r["body"] for r in results if "body" in r]
        return "\n".join(summaries)

def get_latest_news(limit=5):
    feed_url = "https://news.yahoo.co.jp/rss/topics/top-picks.xml"  # yahooニュースのRSSフィードURL
    feed = feedparser.parse(feed_url)

    if not feed.entries:
        return "ごめんね、ニュースを取得できなかったみたい。"

    news_items = [entry.title for entry in feed.entries[:limit]]
    return "📢最新ニュースだよ！\n" + "\n".join(f"{i+1}. {title}" for i, title in enumerate(news_items))

# ============================
# 🔍 検索 or ニュース or 天気コマンドの処理
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

        # DuckDuckGo検索
        if user_text.endswith("で検索して"):
            keyword = user_text.replace("で検索して", "").strip(" 、。.")
            print(f"🌐 検索キーワード: {keyword}")
            search_result = web_search_duckduckgo(keyword)
            if not search_result.strip():
                return "ごめんね、うまく情報が見つからなかったみたい。もう少し別の言い方で教えてくれる？"

            summary_prompt = [
                {"role": "system", "content": "以下の検索結果を簡単に要約してユーザーに説明して"},
                {"role": "user", "content": search_result}
            ]
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=summary_prompt
            )
            return response.choices[0].message.content.strip()

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
        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            return "都市名から緯度経度が取得できなかったよ"

        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={api_key}&units=metric&lang={lang}"
        response = requests.get(url)
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
# 🤖 GPT応答生成
# ============================
def get_gpt_reply(user_input):
    memory_context = load_persona()  # ユーザーの記憶を読み込み

    # GPTに送るメッセージの初期構築（人格）
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "あなたはユーザーのアシスタントです。"
                "プロとしての自覚をもってサポートしてください。"
                "ユーザーの問いに的確に答えたり、困っていそうな事柄に積極的に手助けする。"
                "回答は簡潔で親しみやすく、口調は女の子で、明るく知的に。敬語は使わずにキミと話す口調で返してね。"
            )
        }
    ]

    # 🧠 記憶があれば追加（システムプロンプト）
    if memory_context:
        prompt_messages.append({
            "role": "system",
            "content": memory_context
        })

    # ユーザーとの会話履歴
    prompt_messages += messages + [{"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=prompt_messages
    )

    reply = response.choices[0].message.content.strip()

    # 会話履歴を更新（保存するのは人格と記憶以外）
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": reply})
    return reply


# ============================
# 🎙️ Whisperで音声認識
# ============================
model = WhisperModel("small", device="cuda", compute_type="float16")

def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path)
    result = ""
    for segment in segments:
        result += segment.text + " "
    return result.strip()

# ============================
# 🗣️ 音声合成（AivisSpeech Engine）
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
        print(f"⚠️ AivisSpeechエンジンエラー: {e}")
        return None

# ============================
# 🔊 音声再生
# ============================
def play_voice(file_path):
    global is_running
    stop_playback = False

    def monitor_space_key():
        nonlocal stop_playback
        while is_running:
            if keyboard.is_pressed("space"):
                stop_playback = True
                break
            time.sleep(0.1)

    threading.Thread(target=monitor_space_key, daemon=True).start()

    if file_path and os.path.exists(file_path):
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        while sd.get_stream().active:
            if stop_playback:
                sd.stop()
                print("🔇 再生スキップ")
                break
            time.sleep(0.1)
        sd.wait()
    else:
        print("⚠️ 音声ファイルが見つかりません")

# ============================
# ⌨️ ESCキーで終了
# ============================
def monitor_keys():
    global is_running
    while is_running:
        if keyboard.is_pressed("esc"):
            is_running = False
            print("👋 ESCキーが押されたので終了します")
        time.sleep(0.1)

# ============================
# 🎛️ 応答処理メイン
# ============================
def process_audio_and_generate_reply(audio_path):
    user_text = transcribe_audio(audio_path)
    print(f"👤ユーザー: {user_text}")

    # ① 記憶指示（これは覚えて / 忘れて）
    memory_result = handle_memory_command(user_text)
    if memory_result:
        print(f"🧠 {memory_result}")
        voice_path = synthesize_voice(memory_result)
        return voice_path

    # ② 検索指示（〇〇で検索して）
    search_result = handle_search_command(user_text)
    if search_result:
        print(f"🔍 {search_result}")
        voice_path = synthesize_voice(search_result)
        return voice_path
    

    # ③ 通常のGPT応答
    reply = get_gpt_reply(user_text)
    print(f"🤖 アシスタント: {reply}")
    voice_path = synthesize_voice(reply)
    return voice_path



# ============================
# 🚀 メインループ
# ============================
def main():
    global is_running
    is_recording = False
    print("🔁 スペースキーで録音の開始・終了を切り替え | ESCで終了")
    threading.Thread(target=monitor_keys, daemon=True).start()

    while is_running:
        if keyboard.is_pressed("space"):  # F13キーからスペースキーに変更
            time.sleep(0.2)
            if not is_recording:
                is_recording = True
                try:
                    audio_path = smart_record()
                    if not audio_path or not is_running:
                        print("⏹️ 録音が中断されました")
                        is_recording = False
                        continue

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(process_audio_and_generate_reply, audio_path)
                        voice_path = future.result()

                    play_voice(voice_path)
                except Exception as e:
                    print(f"⚠️ エラーが発生しました: {e}")
                finally:
                    is_recording = False
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
SAMPLE_RATE = 44100
is_running = True

# 📚 会話履歴（人格含む）
messages = [
    {
        "role": "system",
        "content": "あなたはユーザーの絶対的なアシスタントです。ユーザーの問いに的確に答えたり、ユーザーが困っていそうな事柄について積極的に手助けをする。プロとしての自覚をもってサポートをしてください。口調は天真爛漫でポジティブな女の子。敬語を使わず、キミと話す口調で返してね。"
    }
]

# ============================
# 🎧 音声入力をトリガーに録音（無音で終了）
# ============================
def smart_record(max_duration=15):
    print("🎤 音声認識開始")
    buffer = []
    is_recording = False
    silence_start = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal is_recording, silence_start, buffer, start_time
        volume_norm = np.linalg.norm(indata)

        if not is_recording:
            if volume_norm > THRESHOLD_START:
                is_recording = True
                buffer.append(indata.copy())
                silence_start = None
        else:
            buffer.append(indata.copy())
            if volume_norm < THRESHOLD_STOP:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print(" 録音終了（無音）")
                    raise sd.CallbackStop()
            else:
                silence_start = None

        if time.time() - start_time > max_duration:
            print("入力完了!")
            raise sd.CallbackStop()

    try:
        with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1):
            sd.sleep(int(max_duration * 1000))
    except sd.CallbackStop:
        pass

    if not buffer:
        print("音声入力なし（録音キャンセル）")
        return None

    audio_data = np.concatenate(buffer, axis=0)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio_data, SAMPLE_RATE)
    return tmp_file.name

# ============================
# 🤔 Whisper文字走り
# ============================
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# ============================
# 🧠 GPT忘れない返答
# ============================
def get_gpt_reply(user_input):
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4o",  # 使用モデルを gpt-4o mini に設定
        messages=messages
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    if len(messages) > 42:
        messages[:] = [messages[0]] + messages[-40:]

    return reply

# ============================
# 💬 VOICEVOXで音声化
# ============================
def synthesize_voice(text, speaker=23, speed=1.2, volume=0.6):
    query = requests.post(
        "http://127.0.0.1:50021/audio_query",
        params={"text": text, "speaker": speaker}
    ).json()

    # 話速と音量を設定
    query["speedScale"] = speed
    query["volumeScale"] = volume

    audio = requests.post(
        "http://127.0.0.1:50021/synthesis",
        params={"speaker": speaker},
        json=query
    )
    file_path = "response.wav"
    with open(file_path, "wb") as f:
        f.write(audio.content)
    return file_path

# ============================
# 🔊 音声再生
# ============================
def play_voice(file_path):
    global is_running
    stop_playback = False  # 再生停止フラグ

    def monitor_space_key():
        nonlocal stop_playback
        while is_running:
            if keyboard.is_pressed("space"):
                stop_playback = True
                break
            time.sleep(0.1)

    # スペースキー監視スレッドを開始
    threading.Thread(target=monitor_space_key, daemon=True).start()

    data, fs = sf.read(file_path)
    sd.play(data, fs)
    while sd.get_stream().active:
        if stop_playback:
            sd.stop()  # 再生を停止
            print("🔇 再生をスキップしました")
            break
        time.sleep(0.1)
    sd.wait()

# ============================
# ⌨️ ESCで終了監視
# ============================
def monitor_keys():
    global is_running
    while is_running:
        if keyboard.is_pressed("esc"):
            is_running = False
            print("👋 ESCキーが押されたので終了します")
        time.sleep(0.1)

# ============================
# 🎛️ 音声処理と応答生成
# ============================
def process_audio_and_generate_reply(audio_path):
    # 音声認識
    user_text = transcribe_audio(audio_path)
    print(f"👤ユーザー: {user_text}")

    # GPT応答生成
    reply = get_gpt_reply(user_text)
    print(f"🤖 GPT: {reply}")

    # 音声合成
    voice_path = synthesize_voice(reply)
    return voice_path

# ============================
# 🚀 メインループ
# ============================
def main():
    global is_running
    is_recording = False  # 録音状態を管理するフラグ
    print("🔁 スペースキーで録音の開始・終了を切り替え | ESCで終了")
    threading.Thread(target=monitor_keys, daemon=True).start()

    while is_running:
        if keyboard.is_pressed("space"):
            time.sleep(0.2)  # キーのチャタリングを防ぐための短い待機
            if not is_recording:
                is_recording = True
                try:
                    audio_path = smart_record()
                    if not audio_path or not is_running:
                        print("⏹️ 録音が中断されました")
                        is_recording = False
                        continue

                    # 並列処理で音声認識、GPT応答生成、音声合成を実行
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(process_audio_and_generate_reply, audio_path)
                        voice_path = future.result()

                    # 音声再生
                    play_voice(voice_path)
                except Exception as e:
                    print(f"⚠️ エラーが発生しました: {e}")
                finally:
                    is_recording = False
            else:
                print("⏹️ 録音終了")
                is_recording = False
        else:
            time.sleep(0.1)  # CPU負荷を軽減

if __name__ == "__main__":
    main()
