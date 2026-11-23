# Proiect: context, decizii, întrebări deschise

> Sursa unică de adevăr pentru context. O citim la începutul fiecărei sesiuni și o actualizăm când luăm o decizie sau descoperim un fapt nou.
> Ultima actualizare: 2026-11-23

## Scop

Fine-tuning pentru **Qwen2-VL-7B**, ca să înțeleagă vorbirea (ASR, speech-to-text). Îi adăugăm un **encoder audio Whisper turbo** și un **proiector audio**, după notebook-ul mentorului: [`templates/fine_tuning_vlm_for_speech_understanding_trl.ipynb`](../templates/fine_tuning_vlm_for_speech_understanding_trl.ipynb).

**Obiectivul principal e învățarea** (MentorCruise). Notebook-ul funcțional vine al doilea.

## Cum lucrăm

- **Teorie:** [docs/](../docs/PLANNING.md)
- **Practică:** ghidurile din `docs/NN_.../practice/`, de exemplu [01_audio_data/practice](../docs/01_audio_data/practice/)
- **Spec:** [specs/](README.md)
- Abia apoi **implementare**.

Explicațiile sunt scurte, în română, cu link-uri. Ghidurile au codul rezolvat: comentariile `# TODO` și soluția dedesubt.

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
| datasets 5.x decodează audio prin **torchcodec + FFmpeg**: fără `torchcodec` apare `ImportError`. Coloana audio e un `AudioDecoder`, dar `ex["audio"]["array"]` și `["sampling_rate"]` merg în continuare. MINDS-14 (`PolyAI/minds14`) se încarcă normal. | testat 2026-11-23 |
| Mac: M3 Pro, 36 GB, fără CUDA. FFmpeg-ul din Homebrew era stricat (`x265` actualizat). **Reparat cu `brew reinstall ffmpeg`: acum e FFmpeg 9.0.2**, suportat de torchcodec. | testat 2026-11-23 |
| Colab: utilizatorul are deja **Google AI Pro (5 TB)**, care include Colab. Avea **600 de compute units** la 2026-11-23; butoanele Pay As You Go sunt inactive peste 500 de unități. Unitățile expiră după 90 de zile. **AI Pro = 200 de unități pe lună, fără background execution** (doar Ultra o are). Închiderea laptopului poate opri training-ul, de aceea salvăm checkpoint-uri pe Hub. | screenshot + [Google One](https://one.google.com/intl/en_us/about/google-ai-plans/) + [Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/), 2026-11-23 |
| Colab nu garantează GPU-uri: la primul test din VSCode (2026-11-23) a apărut *„All GPU accelerators are unavailable: L4, A100, T4, G4”*. Soluția e să reîncerci mai târziu. Extensia Colab și login-ul Google funcționează. | test utilizator, 2026-11-23 |
| **Capcane verificate în versiunile actuale:**<br>• `WhisperFeatureExtractor(..., padding=True)` completează doar la cel mai lung exemplu din batch (ex. `(1, 80, 780)`), nu la 3000. Pentru Whisper folosești apelul implicit (`max_length`).<br>• Pe `IterableDataset`, `ds[0]` întoarce un `IterableColumn`, nu primul exemplu; folosești `next(iter(ds))`.<br>• `set_epoch` schimbă ordinea doar la iterarea directă; un `.take(n)` derivat pornește de la epoca 0. | testat 2026-11-23 |
| Cont W&B: **`ceachi-bogdan`** (W&B rulează acum pe `api.forge.coreweave.com`). | `wandb login`, 2026-11-23 |
| Environment conda: **`speech-vlm`** (Python 3.11). Pe Mac: torch 2.15.0, torchaudio 2.11.1 (merg împreună), transformers 5.24.0, datasets 5.2.0, qwen-vl-utils 0.0.14 (cere `torchvision`). `huggingface_hub` 2.4: CLI-ul e **`hf`**; `huggingface-cli` nu mai există. | 2026-11-23 |
| Cont HF: **`Ceachi`**. Token-ul `cncc` e fine-grained și are `repo.write` pe contul personal, suficient pentru creare de repo-uri și push. Repo-ul de model: **`Ceachi/project-finetune-vlm-for-speech-understanding`** (privat, **creat**). Codul stă pe GitHub: `Ceachi/Project-Finetune_vlm_for_speech_understanding`. | HF API, 2026-11-23 |

## Decizii

| # | Data | Decizie | De ce | Status |
|---|---|---|---|---|
| D1 | 2026-11-23 | Fluxul e întâi învățare: teorie → exerciții → spec → implementare | cerința utilizatorului | ✅ decis |
| D2 | 2026-11-23 | Etapele 00–05 pe Mac; GPU prin **Colab din VSCode**, folosind unitățile din abonamentul **Google AI Pro** pe care utilizatorul îl are deja (600 de unități). Nu cumpărăm Pay As You Go și nu facem upgrade. | cost suplimentar zero; setup simplu | ✅ decis |
| D3 | 2026-11-23 | Checkpoint-urile se salvează pe **HF Hub**, în același repo, pe **branch-uri** separate: `stage1-checkpoints` și `stage2-checkpoints`. `main` ține processor-ul și modelul. Trainer-ul primește `hub_revision=<branch>` și `hub_strategy="checkpoint"`. | Colab poate întrerupe sesiunea. Trainer-ul nu poate urca într-un subfolder (doar în rădăcina unui branch), deci folosim branch-uri. | ✅ decis |
| D4 | 2026-11-23 | Practica stă în `docs/NN_.../practice/`: câte un ghid `.ipynb` pe fiecare modul din curs, cu teorie pe scurt, codul din curs, „ce să observi” și exerciții cu `# TODO` + soluție. Ghidul stă în folderul etapei căreia îi aparține modulul și se creează **doar când ajungem la etapa respectivă**, împreună. Fiecare ghid e rulat cap-coadă înainte de livrare. | utilizatorul vrea totul organizat pe etape și cod pe care să-l ruleze local | ✅ decis |
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
| ~~Q6~~ | ~~Include Google AI Pro background execution?~~ **Răspuns: NU**, doar Ultra o include (vezi faptele tehnice). | rezolvat 2026-11-23 | 07, 08 |

## Unde suntem

- [x] Cursul audio parcurs; [notițe](../docs/audio_course_notes.md)
- [x] `docs/`: PLANNING + READING pentru toate etapele; SETUP pentru Colab și checkpoint-uri
- [x] Ghidurile [docs/01_audio_data/practice](../docs/01_audio_data/practice/) 01–05 (Introduction, Load & explore, Preprocessing, Streaming, Datasets·Stream), rulate cap-coadă, doar pe seturi mici. READING-ul etapei are 57 de Q&A.
- [x] Setup pe Mac ([SETUP.md](../docs/00_setup/SETUP.md) pașii 1–7): environment, pachete, FFmpeg, HF, `HF_TOKEN`, repo HF, W&B, `torchvision`. Extensia Colab și login-ul Google merg; la test nu erau GPU-uri libere.
- [ ] Fork-uri (`transformers`, `Qwen3-VL`). Branch-ul de lucru așteaptă răspunsul mentorului la Q1.
- [ ] Utilizatorul parcurge ghidurile 01–05 și pune întrebări (le adăugăm în notebook). Apoi spec-ul [01_dataset](01_dataset.md), apoi TASK-urile din §2.
