# -*- coding: utf-8 -*-
"""
Merge LoRA Adapter Weights into Base Model
===========================================

بعد التدريب باستخدام finetune_lora.py، يقوم هذا السكربت بدمج أوزان LoRA
مع النموذج الأساسي لإنتاج نموذج واحد مستقل جاهز للنشر (بدون الحاجة لتحميل
adapters بشكل منفصل عند الاستدلال).

الاستخدام:
    python merge_lora_weights.py \
        --base-model Qwen/Qwen3-8B \
        --lora-path ./qwen3-lora-output \
        --output-dir ./qwen3-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-path", type=str, required=True,
                         help="مسار أوزان LoRA الناتجة من finetune_lora.py")
    parser.add_argument("--output-dir", type=str, default="./qwen3-merged")
    return parser.parse_args()


def main():
    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {args.lora_path}")
    merged_model = PeftModel.from_pretrained(base_model, args.lora_path, device_map="auto")
    merged_model = merged_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✓ Merge complete")


if __name__ == "__main__":
    main()
