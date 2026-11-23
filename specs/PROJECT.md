# Proiect: context, decizii, întrebări deschise

> Sursa unică de adevăr pentru context. O citim la începutul fiecărei sesiuni și o actualizăm când luăm o decizie sau descoperim un fapt nou.
> Ultima actualizare: 2026-11-23

## Scop

Fine-tuning pentru **Qwen2-VL-7B**, ca să înțeleagă vorbirea (ASR, speech-to-text). Îi adăugăm un **encoder audio Whisper turbo** și un **proiector audio**, după notebook-ul mentorului: [`templates/fine_tuning_vlm_for_speech_understanding_trl.ipynb`](../templates/fine_tuning_vlm_for_speech_understanding_trl.ipynb).

**Obiectivul principal e învățarea** (MentorCruise). Notebook-ul funcțional vine al doilea.

## Cum lucrăm

- **Teorie:** [docs/](../docs/PLANNING.md)
- **Exerciții:** [practice/](../practice/README.md)
- **Spec:** [specs/](README.md)
- Abia apoi **implementare**.

Explicațiile sunt scurte, în română, cu link-uri. La exerciții primești indicii, nu soluții.

## Arhitectura (din [diagramă](../docs/VLA.png))

**Data collator:**
1. Conversie în format conversație (system / user cu audio / assistant cu transcrierea).
2. Chat template cu tokenii `<|audio_start|>`, `<|audio_pad|>`, `<|audio_end|>`.
3. `fetch_audio` în `qwen-vl-utils`: bytes → array.
4. Processor-ul: repetă `<|audio_pad|>` după lungimea fiecărui audio, concatenează audio-urile și păstrează lungimile.

**Model:**

5. Encoder audio → audio tokens → proiector → înlocuiește placeholder-ele → LLM → loss doar pe răspunsul asistentului.

**Antrenare:**
- **Stage 1:** doar proiectorul (LLM și encoder înghețate, bf16, fără cuantizare).
- **Stage 2:** end-to-end cu QLoRA.

## Fapte tehnice stabilite

| Fapt | Sursă / verificat |
|---|---|
| `speechbrain/LargeScaleASR` a fost redenumit **`speechbrain/LoquaciousSet`**. URL-ul vechi redirecționează, nu e gated. Configurații: `small`, `medium`, `large`, `clean`. `small/` are 72 de shard-uri parquet, `test/` are 6. | HF API, 2026-11-23 |
| Un eșantion are `ID, duration, sex, spk_id, text, wav{bytes, path}`. Audio-ul e **bytes WAV**, textul e **UPPERCASE fără punctuație**. | screenshot din diagramă (de reconfirmat în 01) |
| **Whisper large-v3-turbo:** 128 mel bins; intrare de 30 s → 3000 cadre → **1500 vectori** (50/s, câte 20 ms), `d_model = 1280`. | model card + curs Unit 3 |
| **Qwen2-VL-7B:** `hidden_size = 3584`, deci proiectorul face 1280 → 3584. | config Qwen2-VL |
| Versiuni curente: transformers 5.24, datasets 5.2, torch 2.15. Mentorul fixează `transformers==4.47.0` și `torch==2.4.1+cu121`. | pip, 2026-11-23 |
| datasets 5.x decodează audio prin **torchcodec + FFmpeg**. | testat 2026-11-23 |
| Mac: M3 Pro, 36 GB, fără CUDA. **FFmpeg-ul din Homebrew e stricat** (`x265` actualizat); se repară cu `brew reinstall ffmpeg` (de făcut de utilizator). | testat 2026-11-23 |

## Decizii

| # | Data | Decizie | De ce | Status |
|---|---|---|---|---|
| D1 | 2026-11-23 | Fluxul e întâi învățare: teorie → exerciții → spec → implementare | cerința utilizatorului | ✅ decis |
| D2 | 2026-11-23 | Etapele 00–05 pe Mac; GPU prin **Colab Pay-As-You-Go din VSCode** (100 de unități la $9.99, plătite dinainte) | cel mai ieftin și mai simplu, cu plafon de buget sigur | ✅ decis |
| D3 | 2026-11-23 | Checkpoint-urile se salvează pe **HF Hub** (repo-uri private `-stage1-ckpt` / `-stage2-ckpt`, `hub_strategy="checkpoint"`) | Colab poate întrerupe sesiunea; reluăm de pe Hub | ✅ decis |
| D4 | 2026-11-23 | Exercițiile stau în `practice/`, cu verificări validate de o soluție de referință ținută în afara repo-ului | utilizatorul vrea să-și exerseze mâna | ✅ decis |
| D5 | 2026-11-23 | Lucrăm într-o copie a notebook-ului (`notebooks/speech_vlm.ipynb`); `templates/` rămâne neatins | păstrăm originalul mentorului | 🟡 propus |
| D6 | 2026-11-23 | Decodăm audio din bytes cu `soundfile` (`Audio(decode=False)`), nu prin decodarea automată din datasets | fără dependența de FFmpeg; e același cod ca în `fetch_audio` | 🟡 propus |
| D7 | 2026-11-23 | La început folosim `attn_implementation="sdpa"`, fără `flash-attn` | nu necesită compilare pe Colab | 🟡 propus |
| D8 | 2026-11-23 | Pe Colab folosim torch-ul preinstalat, nu pin-ul 2.4.1 al mentorului, decât dacă reapare bug-ul | setup mai simplu | 🟡 propus |
| D9 | 2026-11-23 | La Stage 1 salvăm doar proiectorul (câțiva MB), nu tot modelul (~18 GB) | checkpoint-uri mici pe Hub | 🟡 propus |

## Întrebări deschise

| # | Întrebare | Pentru cine | Unde contează |
|---|---|---|---|
| Q1 | Pe ce versiune de `transformers` facem fork-ul: 4.47.0 sau v5? (notebook-ul pomenește un flag din v5) | mentor | 04, 05 |
| Q2 | Mai merge `load_dataset` cu numele vechi `speechbrain/LargeScaleASR`? | de testat | 01 |
| Q3 | Câți `<|audio_pad|>` per audio: 1500 fix, `ceil(durată × 50)` sau cu pooling? | noi + mentor | 04, 05 |
| Q4 | Ce arhitectură are proiectorul: un singur `Linear` sau un MLP? Cu sau fără downsampling? | noi + mentor | 05 |
| Q5 | Ce instrument lipsește în §5 („Use  to convert”)? Probabil `ffmpeg`. | mentor | 09 |

## Unde suntem

- [x] Cursul audio parcurs; [notițe](../docs/audio_course_notes.md)
- [x] `docs/`: PLANNING + READING pentru toate etapele; SETUP pentru Colab și checkpoint-uri
- [x] `practice/`: caietele 00 și 01
- [ ] Utilizatorul face setup-ul (HF, Colab, W&B) și `brew reinstall ffmpeg`
- [ ] Caietul `practice/01`, apoi spec-ul [01_dataset](01_dataset.md), apoi implementarea TASK-urilor din §2
