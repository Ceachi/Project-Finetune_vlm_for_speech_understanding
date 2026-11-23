# 08 · Training Stage 2 (QLoRA, end-to-end)

**Scop:** după ce proiectorul e antrenat, faci fine-tuning pe întregul model cu QLoRA: greutăți cuantizate în 4-bit plus adaptere LoRA antrenabile.
**În notebook:** Model finetuning: Stage 2 (configurările LoRA, BitsAndBytes și SFT, antrenarea, save & push).
**Rulează pe:** Colab (L4). `bitsandbytes` nu e fiabil pe Mac.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [LoRA paper](https://arxiv.org/abs/2106.09685) | ideea de bază: matrici low-rank, `r`, `alpha` | Obligatoriu (abstract + metoda) |
| 2 | [Blog · 4-bit, bitsandbytes și QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) | NF4, double quantization, QLoRA în practică | Obligatoriu |
| 3 | [PEFT · Quicktour](https://huggingface.co/docs/peft/quicktour) | `LoraConfig`, `get_peft_model`, `print_trainable_parameters` | Obligatoriu |
| 4 | [LLM Course · LoRA](https://huggingface.co/learn/llm-course/chapter11/4) | cum funcționează LoRA, explicat pas cu pas | Obligatoriu |
| 5 | [PEFT · Quantization](https://huggingface.co/docs/peft/developer_guides/quantization) | LoRA peste un model cuantizat | Obligatoriu |
| 6 | [Transformers · bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) | `BitsAndBytesConfig` | Obligatoriu |
| 7 | [TRL · PEFT integration](https://huggingface.co/docs/trl/peft_integration) | `peft_config` în `SFTTrainer` | Obligatoriu |
| 8 | [QLoRA paper](https://arxiv.org/abs/2305.14314) | detaliile metodei | Opțional |
| 9 | [Transformers · PEFT](https://huggingface.co/docs/transformers/peft) | încărcarea adapterelor | Referință |

## De exersat

1. Aplică LoRA pe un model mic cu `get_peft_model` și rulează `print_trainable_parameters()`. Variază `r` și `target_modules` și urmărește cum se schimbă numărul de parametri.
2. Pe Colab: încarcă același model o dată în bf16 și o dată în 4-bit, apoi compară `torch.cuda.memory_allocated()`.
3. Salvează doar adapterul și compară cât ocupă pe disk față de model.

## Verifică-te

- Ce înseamnă `r` și `lora_alpha`?
- Ce e NF4 și de ce funcționează bine pentru greutățile rețelelor neuronale?
- De ce folosim QLoRA în Stage 2, dar nu și în Stage 1?
- Pe ce module ar trebui aplicat LoRA în modelul nostru (LLM, proiector, encoderul audio)?
