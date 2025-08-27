import logging
from argparse import ArgumentParser
from pathlib import Path

import evaluate
import librosa
import numpy as np
import torch
import wandb
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from torch.cuda.amp import GradScaler, autocast


class ReadSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir: str,
        do_awgn: bool = False,
        awgn_snr: float = 0.05,
        do_masking: bool = False,
        max_mask_len: int = 3200,
    ) -> None:
        self.dir = dir
        self.do_awgn = do_awgn
        self.awgn_snr = awgn_snr
        self.do_masking = do_masking
        self.max_mask_len = max_mask_len

        self.text = {}
        self.names = []
        # Assuming text.txt is in the same directory as the wav folder
        text_file_path = Path(dir) / "text.txt"
        if not text_file_path.exists():
             # If text.txt is not in the specified directory, check the parent directory
            text_file_path = Path(dir).parent / "text.txt"
            if not text_file_path.exists():
                raise FileNotFoundError(f"text.txt not found in {dir} or {Path(dir).parent}")


        with open(text_file_path) as fid:
            for line in fid:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                # Remove BOM character if present
                if line.startswith('\ufeff'):
                    line = line[1:]
                parts = line.split(maxsplit=1) # Split into at most 2 parts
                if len(parts) < 2: # Skip lines with only filename or empty sentence
                     continue
                name = parts[0]
                sentence = parts[1]

                self.text[name] = sentence
                self.names.append(name)

    def _awgn(self, audio: np.ndarray) -> np.ndarray:
        noise = np.random.randn(len(audio))
        audio_power = np.sum(audio**2)
        noise_power = np.sum(noise**2)
        scale = np.sqrt(audio_power / noise_power * 10 ** -(self.awgn_snr / 10))
        audio = audio + scale * noise
        audio = np.clip(audio, -1, 1)
        return audio

    def _masking(self, audio: np.ndarray) -> np.ndarray:
        mask_len = np.random.randint(0, self.max_mask_len)
        mask_start = np.random.randint(0, len(audio) - mask_len)
        audio[mask_start : mask_start + mask_len] = 0
        return audio

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> "dict[str, np.ndarray]":
        name = self.names[index]
        # Remove BOM character from the name if present
        if name.startswith('\ufeff'):
            name = name[1:]
        # Construct the full path to the .wav file
        # Ensure no double .wav extension
        if name.endswith(".wav"):
            audio_path = Path(self.dir) / "wav" / name
        else:
            audio_path = Path(self.dir) / "wav" / f"{name}.wav"


        audio, _ = librosa.load(
            audio_path, sr=16000, mono=True
        )
        audio = audio.squeeze()
        if self.do_awgn:
            audio = self._awgn(audio)
        if self.do_masking:
            audio = self._masking(audio)
        text = self.text[self.names[index]]
        return {"audio": audio, "text": text}


class Collator:
    def __init__(self, processor: Wav2Vec2Processor) -> None:
        self.processor = processor

    def __call__(self, features: "list[dict[str, np.ndarray]]"):
        # features = [{"audio": np.array, "text": str}, ...]
        features = [
            self.processor(audio=x["audio"], sampling_rate=16000, text=x["text"])
            for x in features
        ]
        # features = [{"input_values": np.array, "labels": np.array}, ...]
        input_features = [{"input_values": x["input_values"][0]} for x in features]
        labels = [{"input_ids": x["labels"]} for x in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.pad(
            labels=labels, padding=True, return_tensors="pt"
        )
        labels = labels_batch.input_ids.masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return {"input_values": batch["input_values"], "labels": batch["labels"]}


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred: "EvalPrediction", tokenizer: Wav2Vec2CTCTokenizer):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# Custom Trainer class to incorporate gradient scaling and clipping
class CustomTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast(enabled=self.args.fp16):
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Check for NaNs in loss
        if torch.isnan(loss):
            print("NaN loss detected, skipping backward pass")
            return loss.detach()


        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Use scaler for backward pass if fp16 is enabled
        if self.args.fp16:
            scaled_loss = self.scaler.scale(loss)
            if scaled_loss is not None: # Check if scaling was successful
                scaled_loss.backward()
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.accelerator.backward(loss)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()


        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss.detach()


    # Override create_optimizer_and_scheduler to initialize scaler and add max_grad_norm argument
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        if self.args.fp16:
            self.scaler = GradScaler()


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--wandb_proj", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument(
        "--base_model", type=str, default="facebook/wav2vec2-base-960h"
        #facebook/wav2vec2-large-xlsr-53
    )
    parser.add_argument("--do_awgn", action="store_true")
    parser.add_argument("--awgn_snr", type=float, default=0.05)
    parser.add_argument("--do_masking", action="store_true")
    parser.add_argument("--max_mask_len", type=int, default=3200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0) # Added gradient clipping argument
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) # Added gradient accumulation argument
    args = parser.parse_args()

    if args.wandb_proj:
        wandb.init(
            project=args.wandb_proj, config=args, dir=args.output_dir, save_code=True, mode='disabled' # Added mode='disabled'
        )

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file="vocab.json", word_delimiter_token=" ")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base_model)
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model, vocab_size=tokenizer.vocab_size, ctc_loss_reduction="mean" ,   ignore_mismatched_sizes=True
    )
    # model.freeze_feature_encoder()
    print(model)

    train_dataset = ReadSpeechDataset(
        f"{args.data_dir}/train",
        do_awgn=args.do_awgn,
        awgn_snr=args.awgn_snr,
        do_masking=args.do_masking,
        max_mask_len=args.max_mask_len,
    )
    eval_dataset = ReadSpeechDataset(f"{args.data_dir}/test")

    logging.basicConfig(level=logging.INFO, force=True)
    torch.autograd.set_detect_anomaly(True)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to=["tensorboard"] + ["wandb"] if args.wandb_proj else [],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        dataloader_num_workers=4,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        disable_tqdm=True,
        metric_for_best_model="wer",
        load_best_model_at_end=True,
        greater_is_better=False,
        max_grad_norm=args.max_grad_norm, # Added max_grad_norm to training_args
        gradient_accumulation_steps=args.gradient_accumulation_steps # Added gradient_accumulation_steps to training_args
    )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(processor),
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    processor.save_pretrained(args.output_dir)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.evaluate()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()