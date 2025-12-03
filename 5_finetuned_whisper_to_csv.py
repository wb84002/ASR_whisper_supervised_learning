from pathlib import Path
import csv

import torch
from transformers import pipeline


def transcribe_to_csv(
    audio_path: Path,
    model_dir: Path,
    output_csv: Path,
    language: str = "chinese", # 要與微調時一致
    task: str = "transcribe", # "transcribe"：聽中文 → 輸出中文, 聽外語 → 翻成英文
    chunk_length_s: int = 30,
):
    """
    使用「微調後的 Whisper」對長音檔做自動分段，直接輸出 csv 字幕。

    csv 欄位格式：
        audio_path, start, end, text

    :param audio_path: 要辨識的音檔路徑
    :param model_dir: 你微調後模型的資料夾
    :param output_csv: 輸出 csv 路徑
    :param language: Whisper 語言設定（跟訓練時一樣，用 'chinese'）
    :param task: 'transcribe' 或 'translate'
    :param chunk_length_s: pipeline 內部每塊處理秒數
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音檔：{audio_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"找不到模型資料夾：{model_dir}")

    print("音檔：", audio_path)
    print("使用模型：", model_dir)

    # 0 = 第一張 GPU, -1 = CPU
    device = 0 if torch.cuda.is_available() else -1
    print("使用裝置：", "cuda" if device == 0 else "cpu")

    # 建立 ASR pipeline（會沿用你微調後的 tokenizer / feature_extractor）
    asr = pipeline(
        task="automatic-speech-recognition",
        model=str(model_dir),
        tokenizer=str(model_dir),
        feature_extractor=str(model_dir),
        device=device,
        return_timestamps=True,          # 要求回傳時間戳
        chunk_length_s=chunk_length_s,   # 長音檔分塊處理
        stride_length_s=(5, 0),          # 前面保留 5 秒做重疊，避免句子被硬切斷
    )

    # Whisper 的語言 / 任務用 generate_kwargs 傳進去
    generate_kwargs = {
        "task": task,
        "language": language,
    }

    print("開始辨識並自動分段...")
    result = asr(str(audio_path), generate_kwargs=generate_kwargs)

    # pipeline 會回傳：
    # {
    #   "text": ".....",
    #   "chunks": [
    #       {"text": "...", "timestamp": (start, end)},
    #       ...
    #   ]
    # }
    chunks = result.get("chunks", None)
    if not chunks:
        # 理論上不太會發生，備用：整段當一塊
        print("沒有 chunks，改用整段輸出一列。")
        chunks = [
            {
                "text": result.get("text", "").strip(),
                "timestamp": (0.0, 0.0),  # 沒有時間資訊
            }
        ]

    print("總共取得", len(chunks), "個片段。")
    print("輸出 CSV：", output_csv)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "start", "end", "text"])

        for ch in chunks:
            text = (ch.get("text") or "").strip()
            ts = ch.get("timestamp") or (0.0, 0.0)
            start, end = ts

            writer.writerow(
                [
                    str(audio_path),
                    f"{start:.3f}",
                    f"{end:.3f}",
                    text,
                ]
            )

    print("完成！已寫入 CSV。")


def main():
    base_dir = Path(__file__).resolve().parent

    # 1. 微調後模型的位置
    model_dir = base_dir / "whisper-small-medical-zh"

    # 2. 要轉字幕的長音檔
    audio_path = (
        base_dir / "Data" / "input" / "【智慧健康與醫療技術商談媒合會】奇美醫院 智慧醫療中心 急診AI病情監控儀表板.mp3"
    )

    # 3. 輸出的 csv 路徑
    output_path = (
        base_dir / "Data" / "output" / "【智慧健康與醫療技術商談媒合會】奇美醫院 智慧醫療中心 急診AI病情監控儀表板.mp3"
    )
    output_csv = output_path.with_suffix(".finetuned.csv")

    # 🔹確保 output 目錄存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    transcribe_to_csv(
        audio_path=audio_path,
        model_dir=model_dir,
        output_csv=output_csv,
        language="chinese",
        task="transcribe",
        chunk_length_s=30,
    )


if __name__ == "__main__":
    main()
