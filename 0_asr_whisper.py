import whisper
from pathlib import Path

def main():
    # 1.設定音訊檔路徑
    audio_path = (
        Path(__file__).resolve().parent/"Data"/"input"/"【智慧健康與醫療技術商談媒合會】奇美醫院 智慧醫療中心 急診AI病情監控儀表板.mp3"
    )

    # 檢查檔案是否存在
    if not audio_path.exists():
        print("找不到音訊檔:", audio_path)
        return
    print("音訊檔路徑:", audio_path)

    # 2. 載入 Whisper 模型
    # 可選 tiny, base, small, medium, large
    # 模型越大越準，但速度越慢、吃的記憶體越多
    print("正在載入模型 small")
    model = whisper.load_model("small")

    # 3. 執行語音辨識
    # language="zh" 固定為中文。如果想讓模型自動判斷語言，可以拿掉這個參數。
    print("開始辨識音訊")
    result = model.transcribe(
        str(audio_path), language = "zh", # 音檔主要是中文
        task="transcribe" # 轉寫，不做翻譯
    )

    text = result.get("text","")
    print("\n========語音辨識結果========")
    print(text)

    # 4. 存成文字檔，放在同一個資料夾
    output_text = audio_path.with_suffix(".txt")
    output_text.write_text(text, encoding = "utf-8")
    print("\n辨識結果已存成文字檔:", output_text)

if __name__ == "__main__":
    main()