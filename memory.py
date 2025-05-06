from pathlib import Path
import json

MEMORY_FILE = Path("memory.json")

def save_persona(new_data):
    """
    💾 ユーザーの記憶をファイルに保存（追記・更新も可）
    new_data: 辞書型で {"好きな食べ物": "カレー"} のように渡す
    """
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
    else:
        memory_data = {}

    memory_data.update(new_data)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2, ensure_ascii=False)
    print("📝 記憶を保存したよ")
 