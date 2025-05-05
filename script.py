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
SILENCE_DURATION = 1.0
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
