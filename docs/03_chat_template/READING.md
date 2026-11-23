# 03 · Format conversație și chat template (pașii 1–2 din diagramă)

**Scop:** aduci fiecare eșantion în format de conversație (system/user/assistant) și extinzi chat template-ul Qwen2-VL cu tokenii audio `<|audio_start|>`, `<|audio_pad|>`, `<|audio_end|>`.
**În notebook:** „TASK: Create a new repo on HF and upload your new tokenizer”, Step #1 Format Dataset, Step #2 Create chat template, push processor, întrebarea despre redimensionarea embedding-urilor, încărcarea processor-ului din repo.
**Rulează pe:** Mac. Ai nevoie doar de processor/tokenizer, nu de greutățile modelului.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [Transformers · Chat templates](https://huggingface.co/docs/transformers/chat_templating) | `apply_chat_template`, `add_generation_prompt` | Obligatoriu |
| 2 | [Transformers · Multimodal chat templates](https://huggingface.co/docs/transformers/chat_templating_multimodal) | mesaje cu `{"type": "image", ...}` | Obligatoriu |
| 3 | [Transformers · Writing a chat template](https://huggingface.co/docs/transformers/chat_templating_writing) | sintaxa Jinja a template-ului | Obligatoriu |
| 4 | [Transformers · Qwen2-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_vl) | formatul mesajelor, exemple de utilizare | Obligatoriu |
| 5 | [Model card · Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | exemplele de conversații | Obligatoriu |
| 6 | [Cookbook · Fine-tuning VLM cu TRL](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) | funcția `format_data` (aceeași idee ca Step #1) | Obligatoriu |
| 7 | [LLM Course · Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4) | ce face un tokenizer | Obligatoriu dacă e nou |
| 8 | [Transformers · Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) | `add_special_tokens`, `additional_special_tokens` | Referință |
| 9 | [huggingface_hub · Repositories](https://huggingface.co/docs/huggingface_hub/guides/repository) | `create_repo` | Obligatoriu |
| 10 | [huggingface_hub · Upload files](https://huggingface.co/docs/huggingface_hub/guides/upload) | `upload_folder` | Obligatoriu |

## De exersat

1. Încarcă doar processor-ul `Qwen/Qwen2-VL-2B-Instruct`, fără model. Printează `processor.chat_template` și citește-l.
2. Aplică `apply_chat_template(messages, tokenize=False)` pe o conversație cu o imagine și găsește în text `<|vision_start|><|image_pad|><|vision_end|>`.
3. Pe un tokenizer mic (ex. `gpt2`), adaugă un token special. Compară `len(tokenizer)` înainte și după cu `vocab_size` din config-ul modelului.
4. Creează un repo de test pe HF și urcă un folder cu `upload_folder`.

## Verifică-te

- Ce rol are fiecare dintre `<|audio_start|>`, `<|audio_pad|>` și `<|audio_end|>`?
- De ce îi adăugăm ca tokeni **speciali** și nu ca text obișnuit?
- *Întrebarea mentorului:* de ce nu trebuie redimensionate embedding-urile modelului? Indiciu: compară `len(processor.tokenizer)` cu `vocab_size` din config-ul Qwen2-VL.
- Ce face `add_generation_prompt=True` și când îl folosești (training vs inferență)?
