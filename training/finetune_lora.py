# -*- coding: utf-8 -*-
"""
SUST Academic Assistant - LoRA Fine-Tuning Script
==================================================

يقوم هذا السكربت بتكييف نموذج Qwen3-8B على البيانات الأكاديمية الخاصة
بجامعة السودان للعلوم والتكنولوجيا، باستخدام تقنية LoRA (Low-Rank Adaptation)
لتقليل عدد المعاملات القابلة للتدريب والحفاظ على كفاءة الذاكرة.

ملاحظة: هذا الملف هو النسخة النظيفة والنهائية من عملية التدريب. تم حذف عدة
تجارب استكشافية أولية (مثل تجربة أنظمة استرجاع بديلة باستخدام BM25/FAISS)
لأنها لم تكن جزءاً من التصميم النهائي للنظام، والذي يعتمد على ChromaDB
(انظر backend/main.py).

الاستخدام:
    python finetune_lora.py --data-path ./data/question_fixed.json

المتطلبات مذكورة في requirements.txt
"""

import argparse
import json
import warnings

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split  # noqa: F401 (kept for reproducibility notes)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B with LoRA on SUST academic Q&A data")
    parser.add_argument("--data-path", type=str, required=True,
                         help="مسار ملف JSON الذي يحتوي على أزواج الأسئلة والأجوبة")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B",
                         help="اسم/مسار النموذج الأساسي على Hugging Face")
    parser.add_argument("--output-dir", type=str, default="./qwen3-lora-output",
                         help="مسار حفظ أوزان LoRA الناتجة عن التدريب")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_prepare_dataset(data_path, eval_split, seed):
    """
    يقرأ ملف الأسئلة والأجوبة. يدعم الصيغتين المستخدمتين في قاعدة المعرفة:
      - {"instruction_variants": [...], "output": "..."}  (عدة صياغات لنفس السؤال)
      - {"instruction": "...", "output": "..."}            (سؤال واحد)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"File loaded. Number of raw records: {len(raw_data)}")

    clean_data = []
    for item in raw_data:
        if "instruction_variants" in item and "output" in item:
            for instruction in item["instruction_variants"]:
                clean_data.append({"instruction": instruction, "output": item["output"]})
        elif "instruction" in item and "output" in item:
            clean_data.append({"instruction": item["instruction"], "output": item["output"]})

    print(f"Recovered training samples: {len(clean_data)}")
    if clean_data:
        print(f"Sample: {clean_data[0]}")

    dataset = Dataset.from_list(clean_data)
    split_dataset = dataset.train_test_split(test_size=eval_split, seed=seed)
    return split_dataset["train"], split_dataset["test"]


def build_tokenize_fn(tokenizer, max_length):
    def tokenize_function(examples):
        texts = [
            f"### Instruction:\n{inp}\n\n### Response:\n{out}"
            for inp, out in zip(examples["instruction"], examples["output"])
        ]
        tokenized = tokenizer(
            texts, truncation=True, max_length=max_length, padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return tokenize_function


def main():
    args = parse_args()

    # -----------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------
    train_dataset, eval_dataset = load_and_prepare_dataset(
        args.data_path, args.eval_split, args.seed
    )
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # -----------------------------------------------------------------
    # 2. Tokenizer + Quantized Base Model (4-bit)
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    base_model.config.use_cache = False
    print("✓ Base model and tokenizer loaded")

    # -----------------------------------------------------------------
    # 3. Tokenize dataset
    # -----------------------------------------------------------------
    tokenize_fn = build_tokenize_fn(tokenizer, args.max_length)
    train_tokenized = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=train_dataset.column_names
    )
    eval_tokenized = eval_dataset.map(
        tokenize_fn, batched=True, remove_columns=eval_dataset.column_names
    )
    print(f"✓ Tokenized - Train: {len(train_tokenized)}, Eval: {len(eval_tokenized)}")

    # -----------------------------------------------------------------
    # 4. LoRA configuration
    # -----------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    print("✓ LoRA configured")

    # -----------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=1,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
    )

    print("=" * 50)
    print("Training...")
    print("=" * 50)
    trainer.train()
    print("✓ Training complete")

    # -----------------------------------------------------------------
    # 6. Evaluation + Save
    # -----------------------------------------------------------------
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = np.exp(eval_loss)
    print(f"Eval Loss: {eval_loss:.6f}")
    print(f"Perplexity: {perplexity:.6f}")

    lora_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✓ LoRA adapter saved to {args.output_dir}")
    print(
        "ملاحظة: لدمج الأوزان في نموذج مستقل جاهز للنشر، استخدم "
        "merge_lora_weights.py"
    )


if __name__ == "__main__":
    main()
