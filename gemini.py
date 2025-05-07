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
gemini_model = palm.GenerativeModel('models/gemini-2.0-flash')

load_dotenv()

# ============================
# 🌐 Gemini API 初期化
# ============================
palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ============================
# 🧠 ユーザー記憶（読み書き）
# ============================
MEMORY_FILE = Path("memory.json")

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
        "数字で箇条書きで説明はしない。口調は女の子で、明るく知的に。\n"
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
# 🔊 音声再生
# ============================
def play_voice(file_path):
    if file_path and os.path.exists(file_path):
        try:
            data, fs = sf.read(file_path)
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"⚠️ 音声再生中にエラーが発生しました: {e}")
    else:
        print("⚠️ 再生する音声ファイルが見つかりません")


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

def smart_record(max_duration=8):
    print("🎙️ 録音を開始するよ（F2で終了）")
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
                    print("🔇 無音で停止")
                    raise sd.CallbackStop()
            else:
                silence_start = None
        if stop_requested:
            print("🛑 手動で停止")
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

def google_search_and_summarize(query, num_sentences=2):
    """
    与えられたクエリでGoogle検索を行い（ダミーHTMLを使用）、sumyで要約する。

    Args:
        query (str): 検索クエリ。
        num_sentences (int): 要約する文の数。

    Returns:
        str: 検索結果の要約。
    """
    print(f"🔍 '{query}' で検索を実行し、sumyで要約します...")
    # ここに実際のGoogle検索とHTML取得のロジックが入ります
    # ダミーのHTMLコンテンツ
    dummy_html = """
    <html>
    <head><title>ダミー検索結果</title></head>
    <body>
        <p>これはクエリ '{query}' に関連する最初のダミーコンテンツです。重要な情報が含まれています。</p>
        <p>こちらは2番目の段落です。最初の段落を補足する詳細が書かれています。</p>
        <p>3番目の段落では、少し異なる視点からの情報を提供しています。これも重要かもしれません。</p>
        <p>最後に、結論となる4番目の段落です。全体の要点をまとめています。</p>
    </body>
    </html>
    """

    try:
        parser = HtmlParser.from_string(dummy_html, "dummy_url", Tokenizer("japanese"))
        stemmer = Stemmer("japanese")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("ja")

        summary = summarizer(parser.document, num_sentences)
        summary_text = " ".join([str(sentence) for sentence in summary])
        return f"'{query}' に関する検索結果の要約です。\n{summary_text}"

    except Exception as e:
        return f"⚠️ sumyによる要約中にエラーが発生しました: {e}"
    
# ============================
# 🎛️ 応答処理メイン
# ============================
def process_audio_and_generate_reply(audio_path):
    user_text = transcribe_audio(audio_path)
    print(f"👤 ユーザー: {user_text}")

    # 記憶操作
    memory_result = handle_memory_command(user_text)
    if memory_result:
        print(f"🧠 {memory_result}")
        return synthesize_voice(memory_result)

    # 検索コマンド
    if user_text.endswith("で検索して"):
        query = user_text.replace("で検索して", "").strip()
        search_result = google_search_and_summarize(query)
        print(f"🔍 {search_result}")
        return synthesize_voice(search_result)

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
    print("🔁 F2で録音 → Gemini応答 → AIVIS音声出力｜ESCで終了")

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

