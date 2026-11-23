# 01 · Dataset: streaming, un eșantion, vizualizare

**Status:** 📝 draft (de discutat împreună) · **Teorie:** [docs/01_audio_data](../docs/01_audio_data/READING.md) · **Practică:** [docs/01_audio_data/practice](../docs/01_audio_data/practice/) · **Notebook:** §2 Understand Dataset

## Context
Primul contact cu datele proiectului. Ce învățăm aici despre format, mai ales despre **bytes WAV**, refolosim direct în `fetch_audio` (pasul 3 din [diagramă](../docs/VLA.png)) și în data collator.

## Requirements
Luate din TASK-urile mentorului.
- **R1:** încărcăm dataset-ul `speechbrain/LargeScaleASR` (acum `LoquaciousSet`) în **streaming**, fără descărcare completă.
- **R2:** luăm **un** eșantion cu `.take(1)` și îi inspectăm câmpurile.
- **R3:** vizualizăm forma de undă cu `librosa`.
- **R4 (propus de noi):** decodăm bytes-ii WAV într-un array `float32` cu un helper reutilizabil, de care vom avea nevoie la etapa 04.

## Non-goals
- Descărcarea shard-urilor pentru training: asta e la etapa 07 (celula „Download Dataset”).
- Resampling-ul sau filtrarea întregului dataset.

## Interfață
```python
stream = load_dataset(DATASET_NAME, ..., split="train", streaming=True)  # -> IterableDataset
sample = next(iter(stream.take(1)))                                       # -> dict
def decode_wav_bytes(b: bytes) -> tuple[np.ndarray, int]: ...           # (float32 1-D, sampling_rate)
```

## Design (propus, de discutat)
- **Numele dataset-ului:** încercăm întâi numele vechi, din notebook. Dacă nu merge, folosim `speechbrain/LoquaciousSet` (Q1).
- **Ce citim:** configurația `small` sau `data_files=["small/train-00000-*"]` (Q3). În streaming se citește oricum doar ce e necesar.
- **Decodarea:** dacă `wav` e de tip `Audio`, facem `cast_column("wav", Audio(decode=False))` și decodăm cu `soundfile` din `io.BytesIO` (decizia D6 din [PROJECT.md](PROJECT.md#decizii)).
- **Vizualizarea:** `waveshow` + mel spectrogram cu 128 de benzi (ca la Whisper turbo), ca extra.

## Acceptance criteria
- **AC1:** `stream` este un `IterableDataset`, deci nu am descărcat tot dataset-ul.
- **AC2:** `sample` conține cel puțin `text`, `duration` și coloana audio (`wav`). Schema exactă se confirmă la Q2.
- **AC3:** `decode_wav_bytes` întoarce un array 1-D `float32` cu valori în [-1, 1], `sr == 16000`, iar `abs(len/sr - sample["duration"]) < 0.05`.
- **AC4:** graficul waveform are axa timpului egală cu durata eșantionului.
- **AC5:** am confirmat că `text` e UPPERCASE fără punctuație. Asta va conta la evaluare (normalizarea WER).

## Tasks
- [ ] Q1–Q3 lămurite (rulăm `print(stream.features)`)
- [ ] celula de streaming + `take(1)`
- [ ] helper-ul `decode_wav_bytes`
- [ ] vizualizarea
- [ ] o celulă cu assert-urile AC1–AC5
- [ ] faptele confirmate trecute în [PROJECT.md](PROJECT.md#fapte-tehnice-stabilite)

## Open questions
- **Q1:** Mai merge `load_dataset` cu numele vechi?
- **Q2:** Ce tip are coloana `wav` în `features`: `Audio` sau o structură `{bytes, path}`? Care e schema exactă?
- **Q3:** Folosim configurația `small` sau `data_files`?
- **Q4:** Toate eșantioanele sunt la 16 kHz?

## Decizii
_(se completează după discuție)_
