# 07 · Training Stage 1 (doar proiectorul audio)

**Scop:** antrenezi doar proiectorul audio, cu LLM-ul și encoderul Whisper înghețate, folosind TRL `SFTTrainer` și un data collator propriu.
**În notebook:** Download Dataset, Clean up, Model finetuning: Stage 1 (freeze, `SFTConfig`, W&B, data collator, trainer, push).
**Rulează pe:** Colab, recomandat A100 40 GB; pe 24 GB, Stage 1 e la limită. Exercițiile mici merg și pe Mac.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [TRL · SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) | `SFTConfig`, data collator propriu, loss doar pe completion | Obligatoriu |
| 2 | [Cookbook · Fine-tuning VLM cu TRL](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) | `collate_fn` și configurarea antrenării | Obligatoriu |
| 3 | [Audio course · Fine-tuning ASR](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning) | collator-ul cu `-100` | Obligatoriu |
| 4 | [Transformers · Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) | cum se calculează loss-ul pe `labels` | Obligatoriu |
| 5 | [PyTorch · Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) | `requires_grad`: cum îngheți parametri | Obligatoriu |
| 6 | [TRL · Reducing memory usage](https://huggingface.co/docs/trl/reducing_memory_usage) | trucuri pentru 24 GB VRAM | Obligatoriu |
| 7 | [W&B · Hugging Face integration](https://docs.coreweave.com/products/wandb/integrations/huggingface) | `wandb.init`, `report_to="wandb"` | Obligatoriu |
| 8 | [LLaVA paper](https://arxiv.org/abs/2304.08485) | Stage 1 = „feature alignment”: de ce se antrenează întâi doar proiectorul | Obligatoriu (recapitulare) |
| 9 | [Transformers · Data collator](https://huggingface.co/docs/transformers/main_classes/data_collator) | collator-ele existente | Referință |
| 10 | [Accelerate · Gradient accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation) | batch efectiv mai mare cu memorie puțină | Opțional |
| 11 | [Transformers · Instantiate a big model](https://huggingface.co/docs/transformers/main/en/big_models) | checkpoint-uri sharded (pentru întrebarea mentorului) | Opțional |
| 12 | [Transformers · Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) | `save_steps`, `hub_strategy`, `resume_from_checkpoint` (vezi și [COLAB.md](../00_setup/COLAB.md#checkpoint-uri-cum-salvăm-și-cum-reluăm)) | Obligatoriu |

## De exersat

1. Pe un model mic (ex. `Qwen/Qwen2.5-0.5B-Instruct`), îngheață tot în afară de un singur layer și numără parametrii antrenabili.
2. Scrie un collator pentru conversații text: `labels = -100` peste tot, mai puțin pe răspunsul asistentului. Verifică decodând doar tokenii care au label.
3. Rulează câțiva pași de SFT cu TRL pe modelul mic, cu un dataset mic, și urmărește loss-ul în W&B.

## Verifică-te

- De ce antrenăm întâi doar proiectorul, înainte de restul modelului?
- De ce se pune `-100` în `labels` și pe ce tokeni?
- Care e batch size-ul efectiv cu `per_device_train_batch_size=2` și `gradient_accumulation_steps=8`?
- *Întrebarea mentorului:* de ce, după push, s-a schimbat doar 1 din cele 4 shard-uri?
