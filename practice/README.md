# Practice

Aici îți exersezi mâna. **Teoria** e în [`docs/`](../docs/PLANNING.md), iar aici **scrii cod**. Fiecare etapă are un caiet `exercises.ipynb` cu exerciții mici, separate de proiect.

## Cum lucrăm

1. Citești teoria etapei: `docs/NN_.../READING.md`.
2. Deschizi `practice/NN_.../exercises.ipynb`.
3. Scrii codul în celulele `# TODO`. **Tastezi tu, fără copy-paste din curs.** Ăsta e tot rostul exercițiilor.
4. Rulezi celula **✅ Verificare** de sub fiecare exercițiu. Dacă trece, rezultatul e corect.
5. Dacă te blochezi, deschizi **💡 Indiciu**. Dacă tot nu merge, mă întrebi pe mine: îți dau un indiciu, nu soluția.
6. La ✍️ răspunzi în cuvintele tale, direct în caiet.
7. La final îmi spui „review practice/NN” și îți comentez codul și răspunsurile.

**Reguli:**
- Soluțiile nu sunt în repo.
- Fișierele generate (checkpoint-uri, audio) merg în `outputs/`, care e ignorat de git.

## Caiete

| Etapă | Caiet | Teorie | Rulează pe | Status |
|---|---|---|---|---|
| 00 · Setup | [00_setup/exercises.ipynb](00_setup/exercises.ipynb) | [READING](../docs/00_setup/READING.md) · [SETUP](../docs/00_setup/SETUP.md) | Mac | ⬜ |
| 01 · Audio data | [01_audio_data/exercises.ipynb](01_audio_data/exercises.ipynb) | [READING](../docs/01_audio_data/READING.md) | Mac | ⬜ |
| 02–09 | se adaugă când ajungem la etapa respectivă | | | |

## Environment (Mac)

Pachetele de care ai nevoie: `numpy librosa soundfile matplotlib datasets transformers accelerate torch huggingface_hub`.

> **FFmpeg:** pe Mac-ul tău, FFmpeg-ul din Homebrew e stricat. Dependența `x265` a fost actualizată și `ffmpeg` nu mai pornește. Se repară cu `brew reinstall ffmpeg`.
> Exercițiile din 01 merg și fără FFmpeg: decodăm audio din bytes cu `soundfile`.
> Ai nevoie de FFmpeg funcțional pentru decodarea automată din `datasets` și, la etapa 09, pentru conversia Voice Memos → WAV.
