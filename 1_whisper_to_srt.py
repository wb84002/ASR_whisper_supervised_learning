import torch
import whisper
from pathlib import Path

def seconds_to_srt_timestamp(seconds: float) -> str:
    # 將秒數轉成 SRT 格式時間戳：HH:MM:SS,mmm
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds  // (3600*1000)
    milliseconds %= 3600*1000

    minutes = milliseconds  // (60*1000)
    milliseconds %= 60*1000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def transcribe_to_srt(
        audio_path: Path,
        model_name: str = "small",
        language: str = "zh",
):
    """
    使用 Whisper 將音檔轉成 SRT 字幕檔。

    param audio_path: 音檔路徑（Path 物件）
    param model_name: Whisper 模型名稱（tiny/base/small/medium/large）
    param language: 語言代碼，中文建議用 "zh"
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音訊檔:{audio_path}")
    print("音訊檔:", audio_path)

    #載入模型
    print("開始載入whisper模型")
    model = whisper.load_model(model_name, device="cuda")

    print("開始轉寫並自動分段")
    result = model.transcribe(
        str(audio_path),
        language=language, # zh
        task = "transcribe",
        verbose = False,    #不在終端機列印每一段
    )

    segments = result.get("segments", [])
    if not segments:
        print("沒有得到任何分段結果，請確認音檔內容是否正常。")
        return

    # SRT 檔案路徑：與音檔同名，副檔名改為 .srt
    srt_path = audio_path.with_suffix(".srt")
    print("輸出字幕檔：", srt_path)

    with srt_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start_ts = seconds_to_srt_timestamp(seg["start"])

            end_ts = seconds_to_srt_timestamp(seg["end"])
            text = seg["text"].strip()

            # SRT 格式：
            # 序號
            # 開始時間 --> 結束時間
            # 文字
            # （空行）
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{text}\n\n")

    print("完成！可以用任何字幕編輯工具或文字編輯器開啟 .srt 來校稿。")


def main():
    # TODO：把這裡換成實際的音檔路徑
    audio_path = (
        Path(__file__).resolve().parent
        / "Data"
        / "input"
        / "【智慧健康與醫療技術商談媒合會】福寶科技股份有限公司 仿生外骨骼智慧健康照護應用.mp3"
    )

    transcribe_to_srt(audio_path, model_name="small", language="zh")


if __name__ == "__main__":
    main()

