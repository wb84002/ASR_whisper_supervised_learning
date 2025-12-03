from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import librosa
import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate


# ====== 基本設定 ======
MODEL_NAME = "openai/whisper-small"   # 之後可以改成 tiny / base / medium 等
LANGUAGE = "chinese"                  # Whisper 語言名稱（對應 <|zh|>）
TASK = "transcribe"                   # 我們做語音轉寫
SAMPLE_RATE = 16000                   # Whisper 要 16kHz 音訊


# ====== DataCollator：把一個 batch 打包成 tensor ======
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 1. 處理 audio 特徵：input_features 已經是固定長度的 log-mel
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 2. 處理 labels（文字）
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 如果最前面是 BOS token，就剪掉，因為 model 會自己加
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    print("=== 開始 train_whisper_finetune ===")

    # ========= 路徑設定 =========
    base_dir = Path(__file__).resolve().parent
    metadata_path = base_dir / "Data" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 metadata.csv：{metadata_path}")

    output_dir = base_dir / "whisper-small-medical-zh"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("使用的 metadata 檔：", metadata_path)
    print("模型輸出路徑：", output_dir)

    # ========= 讀取本地 CSV 資料集 =========
    # metadata.csv 格式：file,text
    dataset_dict = load_dataset(
        "csv",
        data_files={"train": str(metadata_path)},
        encoding="utf-8"
    )

    full_dataset = dataset_dict["train"].train_test_split(
        test_size=0.1, seed=42
    )
    print("Dataset 分割結果：", full_dataset)

    # ========= 載入 Whisper Processor & Model =========
    print(f"載入 Whisper 模型：{MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK,
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # 生成相關設定
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None  # 訓練時關掉 forced decoding
    model.config.use_cache = False

    # ========= 資料前處理：自己讀音檔 =========
    def prepare_dataset(batch):
        audio_path = batch["file"]

        # 1. 讀入音訊
        audio_array, sr = sf.read(audio_path)

        # 多聲道轉單聲道
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # 取樣率不符合就重取樣
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(
                y=audio_array,
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )
            sr = SAMPLE_RATE

        # 2. 抽取 log-Mel 特徵
        batch["input_features"] = processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
        ).input_features[0]

        # 3. 文字標註轉成 token ids
        batch["labels"] = processor.tokenizer(
            batch["text"]
        ).input_ids

        return batch

    print("開始對 train / eval 做特徵前處理（map）...")
    processed_dataset = full_dataset.map(
        #map()：對資料集裡的每一筆資料跑 prepare_dataset
        prepare_dataset,
        remove_columns=full_dataset["train"].column_names,
    )

    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["test"]

    # ========= DataCollator & Metric =========
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 中文建議用 CER（字元錯誤率）
    metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # 把 -100 換回 pad_token_id 才能 decode
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        cer = 100 * metric.compute(
            predictions=pred_str,
            references=label_str,
        )
        return {"cer": cer}

    # ========= TrainingArguments =========
    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,   # 如果 OOM 再往下調
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,   # 等效 batch_size = 2*4 = 8
        learning_rate=1e-5,              # 學習率，不要太大，微調用小一點比較穩定。
        warmup_steps=50,
        num_train_epochs=3,              # 先跑3個 epoch 試試，所以理論上會有3個 checkpoint
        gradient_checkpointing=False,
        fp16=use_fp16,                   # 有 GPU 的話用半精度浮點（float16），省記憶體、加速。
        eval_strategy="epoch",           # 每個 epoch 結束做一次驗證，同時存一次 checkpoint
        save_strategy="epoch",           # 每個 epoch 結束做一次驗證，同時存一次 checkpoint
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        report_to=[],                    # 不用 tensorboard 就留空
        load_best_model_at_end=True,     # 訓練結束後，自動載入「驗證集 CER 最小」的那個 checkpoint
        metric_for_best_model="cer",     # 告訴 Trainer：「cer 越小越好」
        greater_is_better=False,         # 告訴 Trainer：「cer 越小越好」
    )

    # ========= Trainer =========
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,        # 準備好的資料集
        eval_dataset=eval_dataset,          # 準備好的資料集
        data_collator=data_collator,        # 剛剛寫的那個 class，負責把一批 sample 組成 batch
        compute_metrics=compute_metrics,    # 負責計算 CER，用來顯示訓練效果
        tokenizer=processor.feature_extractor,
    )

    # ========= 開始訓練 =========
    # 讀資料 → 組 batch → 前向傳播 → 算 loss → 反向傳播 → 更新參數 → 反覆 N 個 epoch
    print("開始訓練 Whisper 微調模型...")
    trainer.train()

    # ========= 存模型 =========
    print("訓練結束，儲存模型與 processor...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("=== 訓練程式結束 ===")


if __name__ == "__main__":
    main()
