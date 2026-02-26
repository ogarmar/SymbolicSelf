# train_qlora.py — QLoRA fine-tuning de LLaVA sobre VQA-v2
"""
Fine-tuning con QLoRA (4-bit + LoRA adapters) sobre VQA-v2.

Usa las anotaciones VQA-v2 val2014 divididas 80/20 train/eval,
con imágenes COCO val2017 locales.

Restricciones RTX 4050 (6GB):
  - batch_size=1 + gradient_accumulation=8
  - gradient_checkpointing=True
  - max_seq_len=256
  - LoRA rank=8, targets: q_proj, v_proj

Uso:
  python train_qlora.py                          # 1000 muestras, 1 epoch
  python train_qlora.py --max_samples 5000       # más muestras
  python train_qlora.py --epochs 3 --lr 1e-4     # más épocas
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import MODEL_ID, QUANTIZATION, TORCH_DTYPE, MAX_MEMORY

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Dataset para training ────────────────────────────────────────────────


class VQATrainDataset(torch.utils.data.Dataset):
    """Dataset para QLoRA training: convierte VQA samples en input_ids + labels.

    IMPORTANTE: NO usar padding/truncation porque LLaVA-Next expande <image>
    en múltiples tokens de imagen. Truncar rompe la correspondencia
    num_image_tokens vs num_images.
    """

    def __init__(self, vqa_dataset, processor):
        self.samples = vqa_dataset.samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        from PIL import Image

        image = Image.open(sample["image_path"]).convert("RGB")
        # Redimensionar a 336x336 para forzar 1 patch (~576 image tokens)
        # Sin esto, AnyRes crea 2000-3000 tokens → 23 min/step en 6GB
        image = image.resize((336, 336), Image.LANCZOS)
        question = sample["question"]
        answer = sample["answer"]

        # Formato conversacional LLaVA
        prompt = f"USER: <image>\n{question}\nAnswer with a single word or short phrase. ASSISTANT: {answer}"

        # Procesar SIN padding ni truncation para no romper image tokens
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Squeezar dimensión batch (DataLoader la añade)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        # Labels: copiar input_ids pero enmascarar todo excepto la respuesta
        labels = input_ids.clone()

        # Encontrar dónde empieza "ASSISTANT:" para enmascarar todo lo anterior
        assistant_token = self.processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )
        ids_list = input_ids.tolist()
        assistant_start = -1
        for i in range(len(ids_list) - len(assistant_token) + 1):
            if ids_list[i:i + len(assistant_token)] == assistant_token:
                assistant_start = i + len(assistant_token)
                break

        if assistant_start > 0:
            labels[:assistant_start] = -100  # Solo supervisar la respuesta

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

        if "image_sizes" in inputs:
            result["image_sizes"] = inputs["image_sizes"].squeeze(0)

        return result


# ── Collator personalizado ───────────────────────────────────────────────


class VQACollator:
    """Data collator para batch_size=1 (sin padding necesario)."""

    def __call__(self, features):
        # batch_size=1: simplemente añadir dimensión batch
        feature = features[0]
        batch = {}
        for key, val in feature.items():
            batch[key] = val.unsqueeze(0)
        return batch


# ── Training ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning LLaVA VQA-v2")
    parser.add_argument("--max_samples", type=int, default=1000, help="Muestras de training")
    parser.add_argument("--epochs", type=int, default=1, help="Épocas")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=LORA_RANK, help="LoRA rank")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_adapter", help="Directorio para guardar adapter")
    parser.add_argument("--eval_samples", type=int, default=200, help="Muestras de evaluación")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  QLoRA Fine-tuning LLaVA → VQA-v2")
    print(f"  Samples: {args.max_samples} train, {args.eval_samples} eval")
    print(f"  LoRA rank: {args.lora_r}, LR: {args.lr}, Epochs: {args.epochs}")
    print(f"{'='*70}\n")

    # ── 1. Cargar modelo en 4-bit ──────────────────────────────────────
    print("🔧 Cargando modelo en 4-bit...")
    bnb_config = BitsAndBytesConfig(**QUANTIZATION)
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        max_memory=MAX_MEMORY,
        low_cpu_mem_usage=True,
    )

    # Configurar processor para training
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"  # Necesario para training

    # Configurar patch_size para evitar deprecation warning
    if hasattr(model.config, "vision_config"):
        processor.patch_size = getattr(model.config.vision_config, "patch_size", 14)
        processor.vision_feature_select_strategy = getattr(
            model.config, "vision_feature_select_strategy", "default"
        )

    # ── 2. Preparar para QLoRA ─────────────────────────────────────────
    print("🔧 Preparando modelo para QLoRA...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Configurar LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,  # Escala estándar: alpha = 2*r
        target_modules=["q_proj", "v_proj"],  # Capas de atención del LM
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"   Parámetros entrenables: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ── 3. Cargar datasets ─────────────────────────────────────────────
    print("📦 Cargando datasets...")
    from data.vqa_loader import VQADataset

    train_vqa = VQADataset(
        split="val",
        max_samples=args.max_samples,
        train_ratio=0.8,
        is_train_split=True,
    )
    eval_vqa = VQADataset(
        split="val",
        max_samples=args.eval_samples,
        train_ratio=0.8,
        is_train_split=False,
    )

    train_dataset = VQATrainDataset(train_vqa, processor)
    eval_dataset = VQATrainDataset(eval_vqa, processor)

    print(f"   Train: {len(train_dataset)} muestras")
    print(f"   Eval:  {len(eval_dataset)} muestras")

    # ── 4. Training args ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Reducir uso de memoria
        remove_unused_columns=False,  # Mantener pixel_values
        report_to="none",
        optim="paged_adamw_8bit",  # Optimizador 8-bit para ahorrar VRAM
    )

    # ── 5. Trainer ─────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VQACollator(),
    )

    # ── 6. Entrenar ────────────────────────────────────────────────────
    print("\n🚀 Empezando training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print(f"\n✅ Training completado en {elapsed/60:.1f} min")

    # ── 7. Guardar adapter ─────────────────────────────────────────────
    print(f"💾 Guardando adapter en {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"\n🎉 LoRA adapter guardado. Para evaluar:")
    print(f"   python benchmark_vqa.py --n_samples 20 --adapter {args.output_dir}")


if __name__ == "__main__":
    main()
