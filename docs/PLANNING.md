# Plan de învățare și implementare

**Scopul proiectului este să înveți.** Notebook-ul de la mentor ([`templates/fine_tuning_vlm_for_speech_understanding_trl.ipynb`](../templates/fine_tuning_vlm_for_speech_understanding_trl.ipynb)) e curriculumul. Nu ne grăbim să-l terminăm: întâi înțelegi conceptele, apoi le implementăm împreună.

## Cum lucrăm la fiecare etapă

1. Deschizi `READING.md` din folderul etapei.
2. Citești resursele marcate **Obligatoriu**. Cele **Opțional** sunt pentru aprofundare.
3. Faci exercițiile de la **De exersat** în caietul etapei din [`practice/`](../practice/README.md). Sunt mici, separate de proiect, și au verificări automate.
4. Răspunzi la întrebările de la **Verifică-te**. Le putem discuta împreună.
5. Scriem împreună spec-ul componentei în [`specs/`](../specs/README.md) (requirements, design, criterii de acceptare). Abia apoi implementăm TASK-ul din notebook.
6. Bifezi progresul de mai jos.

## Etapele proiectului

Diagrama pipeline-ului: [VLA.png](VLA.png). Pașii 1–5 din diagramă corespund etapelor 03–05.

| # | Etapă | Secțiune în notebook | Ce înveți | Unde rulează | Resurse principale |
|---|---|---|---|---|---|
| 00 | [Setup](00_setup/READING.md) · [pașii concreți](00_setup/SETUP.md) | §1 Install Dependencies, §2 HF Login | conturi (HF, Colab, W&B), fork-uri, checkpoint-uri | Mac + Colab | [HF tokens](https://huggingface.co/docs/hub/security-tokens), [Colab în VSCode](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) |
| 01 | [Audio data](01_audio_data/READING.md) | §2 Understand Dataset: streaming, un sample, vizualizare | audio digital, sampling rate, spectrogramă | Mac | [Curs Unit 1](https://huggingface.co/learn/audio-course/en/chapter1/audio_data), [Streaming](https://huggingface.co/learn/audio-course/en/chapter1/streaming) |
| 02 | [Whisper](02_whisper/READING.md) | TASK: whisper library → audio encodings | cum devine audio-ul vectori | Mac | [Whisper în Transformers](https://huggingface.co/docs/transformers/model_doc/whisper), [blog fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper) |
| 03 | [Chat template](03_chat_template/READING.md) | Step #1 Format Dataset, Step #2 Create chat template (diagramă: 1, 2) | chat templates, tokenizer, tokeni speciali | Mac | [Chat templates](https://huggingface.co/docs/transformers/chat_templating), [Qwen2-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_vl) |
| 04 | [Processor](04_processor/READING.md) | Step #3 Load audio, Step #4 Extend the Processor (diagramă: 3, 4) | cum procesează Qwen2-VL inputurile vizuale și cum facem la fel pentru audio | Mac | [qwen-vl-utils](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils) |
| 05 | [Model](05_model/READING.md) | Step #5 Create the Model, Initialize & Push, Update config (diagramă: 5) | encoder + projector + LLM | Mac (config mic) | [Qwen2-Audio](https://huggingface.co/docs/transformers/model_doc/qwen2_audio) (arhitectură foarte apropiată) |
| 06 | [Inferență](06_inference/READING.md) | Inference: funcția de inferență, testul VL, audio fără fine-tuning | generare, verificarea că partea vision a rămas intactă | Colab (L4) | [Text generation](https://huggingface.co/docs/transformers/llm_tutorial) |
| 07 | [Training Stage 1](07_training_stage1/READING.md) | Download Dataset, Clean up, Stage 1 | SFT, data collator, înghețarea parametrilor | Colab (A100 40 GB) | [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) |
| 08 | [Training Stage 2](08_training_stage2/READING.md) | Stage 2 (QLoRA) | LoRA, cuantizare 4-bit | Colab (L4) | [PEFT](https://huggingface.co/docs/peft/index), [blog QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) |
| 09 | [Evaluare și testare](09_evaluation/READING.md) | §5 Testing the Fine-Tuned Model | WER, testul pe vocea ta | Colab (exercițiile pe Mac) | [Curs Unit 5 · Evaluation](https://huggingface.co/learn/audio-course/en/chapter5/evaluation) |

Template-ul mentorului pare construit pe baza acestui cookbook: [Fine-tuning a VLM (Qwen2-VL-7B) cu TRL](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl). Merită parcurs o dată în întregime, chiar de la început.

## Fundamente (înainte de etapa 01, dacă e nevoie)

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [LLM Course · Ch. 1](https://huggingface.co/learn/llm-course/chapter1/1) | ce e un Transformer, encoder vs decoder | Obligatoriu dacă e nou |
| 2 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | atenția, vizual | Opțional |
| 3 | [LLM Course · Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4) | ce e un token, vocabular | Obligatoriu dacă e nou |
| 4 | [Vision Language Models Explained](https://huggingface.co/blog/vlms) | cum se combină un encoder de imagine cu un LLM (la noi: audio în loc de imagine) | Obligatoriu |
| 5 | [Audio course · Unit 1 intro](https://huggingface.co/learn/audio-course/en/chapter1/introduction) | punctul de plecare recomandat de mentor | Obligatoriu |

## Materiale din repo

- [VLA.png](VLA.png) / [VLA.svg](VLA.svg): diagrama pipeline-ului.
- [audio_course_notes.md](audio_course_notes.md): rezumatul cursului audio și snippet-urile de cod legate de TASK-uri.

## Unde rulăm și cât costă

- **Mac (gratuit):** etapele 00–05.
- **Colab Pay-As-You-Go din VSCode:** etapele 06–09. Pornim cu un pachet de 100 de unități la $9.99, plătit dinainte, deci nu se poate depăși.
  - L4 costă ~$0,17/h, A100 40 GB ~$0,53/h. Sunt rate măsurate de alți utilizatori; Google nu le publică.
  - Pentru faza de studiu și teste, pachetul ar trebui să ajungă cu marjă.
- **Rulări lungi cu laptopul închis (opțional, mai târziu):** Runpod cu credite preplătite sau PC-ul cu 3090. Training-ul rulează ca script în `tmux`.
- Checkpoint-urile se salvează pe HF Hub, deci un training întrerupt se poate relua. Detalii în [00_setup/SETUP.md](00_setup/SETUP.md#6-checkpoint-uri-cum-salvăm-și-cum-reluăm).

## Primul pas concret

**Setup:** urmează [00_setup/SETUP.md](00_setup/SETUP.md): Hugging Face, Colab în VSCode, W&B, fork-uri, checkpoint-uri.

Lucrăm într-o **copie** a notebook-ului (ex. `notebooks/speech_vlm.ipynb`), ca template-ul mentorului să rămână neatins.

**Apoi primul TASK (streaming).** Dataset-ul are ~25.000 de ore, adică terabytes de date. Cu `streaming=True` citești eșantioanele pe rând, din rețea, fără să descarci tot.

## Progres

- [ ] Fundamente
- [ ] 00 · Setup
- [ ] 01 · Audio data
- [ ] 02 · Whisper
- [ ] 03 · Chat template
- [ ] 04 · Processor
- [ ] 05 · Model
- [ ] 06 · Inferență (Colab)
- [ ] 07 · Training Stage 1 (Colab)
- [ ] 08 · Training Stage 2 (Colab)
- [ ] 09 · Evaluare și testare

## Note și întrebări pentru mentor

- **Versiunea Transformers:** notebook-ul fixează `transformers==4.47.0`, dar pomenește și un flag din Transformers v5 (`replace_extra_special_tokens`). Pe ce versiune vrea să faci fork-ul? Pentru primii pași nu contează.
- **Dataset redenumit:** `speechbrain/LargeScaleASR` se numește acum [`speechbrain/LoquaciousSet`](https://huggingface.co/datasets/speechbrain/LoquaciousSet). URL-ul vechi redirecționează. La prima rulare verificăm dacă `load_dataset` merge cu numele vechi; dacă nu, folosim numele nou.
- **Instrument lipsă în §5:** în textul „Convert to WAV: Use  to convert the file” lipsește numele instrumentului. Probabil e `ffmpeg`.
