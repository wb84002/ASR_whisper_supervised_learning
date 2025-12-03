from pathlib import Path
import csv
import subprocess


def srt_timestamp_to_seconds(ts: str) -> float:
    """
    把 'HH:MM:SS,mmm' 轉成秒數（float）
    例如：'00:00:05,900' -> 5.9
    """
    hms, ms = ts.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def cut_audio_from_csv(
    csv_path: Path,
    src_audio: Path,          # 新增：原始音檔路徑從參數傳進來
    output_wav_dir: Path,
    sample_rate: int = 16000,
):
    """
    讀取 csv，依照 start/end 切出很多短音檔，
    並輸出新的 metadata.csv（欄位：file, text）
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 檔：{csv_path}")

    if not src_audio.exists():
        raise FileNotFoundError(f"找不到原始音檔：{src_audio}")

    output_wav_dir.mkdir(parents=True, exist_ok=True)
    print("輸出音檔資料夾：", output_wav_dir)

    rows = []
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print("CSV 欄位名稱：", reader.fieldnames)
        for row in reader:
            rows.append(row)

    metadata_rows = []

    for idx, row in enumerate(rows, start=1):
        # 這裡不要再用 row["audio_path"] 了，統一用 src_audio
        start = srt_timestamp_to_seconds(row["start"])
        end = srt_timestamp_to_seconds(row["end"])
        text = row["text"]

        out_name = f"{idx:06d}.wav"
        out_path = output_wav_dir / out_name

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-i", str(src_audio),
            "-ss", str(start),
            "-to", str(end),
            "-ar", str(sample_rate),
            "-ac", "1",
            str(out_path),
        ]

        subprocess.run(cmd, check=True)

        metadata_rows.append(
            {
                "file": str(out_path),
                "text": text,
            }
        )

    metadata_csv = output_wav_dir.parent / "metadata.csv"
    print("輸出 metadata：", metadata_csv)

    with metadata_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("完成，總共切出", len(metadata_rows), "段音檔。")


def main():
    base_dir = Path(__file__).resolve().parent

    # srt_to_csv 產生的 csv
    csv_path = (
        base_dir
        / "Data"
        / "input"
        / "【智慧健康與醫療技術商談媒合會】奇美醫院 智慧醫療中心 急診AI病情監控儀表板.csv"
    )

    # 原始長音檔路徑（請改成實際檔名）
    src_audio = (
        base_dir
        / "Data"
        / "input"
        / "【智慧健康與醫療技術商談媒合會】奇美醫院 智慧醫療中心 急診AI病情監控儀表板.mp3"
    )

    # 短音檔輸出位置
    output_wav_dir = base_dir / "Data" / "segments"

    cut_audio_from_csv(csv_path, src_audio, output_wav_dir)


if __name__ == "__main__":
    main()
