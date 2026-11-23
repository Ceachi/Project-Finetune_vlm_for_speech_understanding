# 01 · Audio data

**Scop:** înțelegi ce e audio-ul digital și cum arată datele din dataset; încarci un eșantion în streaming și îl vizualizezi.
**În notebook:** §2 Understand Dataset, cu „TASK: load the dataset in the streaming mode” și „TASK: use take and grab 1 sample… visualize the wave using librosa”.
**Rulează pe:** Mac.

> Dataset-ul `speechbrain/LargeScaleASR` se numește acum [`speechbrain/LoquaciousSet`](https://huggingface.co/datasets/speechbrain/LoquaciousSet). Nu e gated și are configurațiile `small`, `medium`, `large`, `clean`. Audio-ul vine ca **bytes WAV**, iar textul e UPPERCASE, fără punctuație.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [Audio course · Introduction to audio data](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) | sampling rate, amplitudine, waveform, spectru, spectrogramă, mel | Obligatoriu |
| 2 | [Audio course · Load and explore](https://huggingface.co/learn/audio-course/en/chapter1/load_and_explore) | structura unui eșantion audio în 🤗 Datasets | Obligatoriu |
| 3 | [Audio course · Preprocessing](https://huggingface.co/learn/audio-course/en/chapter1/preprocessing) | resampling, filtrare după durată, feature extractor | Obligatoriu |
| 4 | [Audio course · Streaming](https://huggingface.co/learn/audio-course/en/chapter1/streaming) | `streaming=True`, `next(iter(...))`, `.take(n)` | Obligatoriu |
| 5 | [🤗 Datasets · Stream](https://huggingface.co/docs/datasets/stream) | `IterableDataset`, `take`, `shuffle` cu buffer | Obligatoriu |
| 6 | [Dataset card · LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) | coloane, configurații, licență | Obligatoriu |
| 7 | [🤗 Datasets · Load audio](https://huggingface.co/docs/datasets/audio_load) / [Process audio](https://huggingface.co/docs/datasets/audio_process) | feature-ul `Audio`, `cast_column` | Opțional |
| 8 | librosa: [waveshow](https://librosa.org/doc/0.11.0/generated/librosa.display.waveshow.html), [specshow](https://librosa.org/doc/0.11.0/generated/librosa.display.specshow.html), [melspectrogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html), [load](https://librosa.org/doc/0.11.0/generated/librosa.load.html) | parametrii funcțiilor de vizualizare | Referință |
| 9 | [soundfile](https://python-soundfile.readthedocs.io/en/latest/) | citirea unui WAV din bytes | Referință |
| 10 | [3Blue1Brown · Fourier Transform (video)](https://www.youtube.com/watch?v=spUNpyF58BY) | intuiția vizuală pentru DFT | Opțional |
| 11 | [Digital Signals Theory (B. McFee)](https://brianmcfee.net/dstbook-site/content/intro.html) | aprofundare în procesarea de semnal | Opțional |

## De exersat

**Caietul cu exerciții:** [practice/01_audio_data/exercises.ipynb](../../practice/01_audio_data/exercises.ipynb). Conține 9 exerciții cu verificări, inclusiv o demonstrație de aliasing.

1. Reprodu exemplele din Unit 1 cu `librosa.ex("trumpet")`: waveform, spectru, spectrogramă, mel spectrogram.
2. Resamplează sunetul la 16 kHz cu `librosa.resample` și compară `len(array)` înainte și după.
3. Fă [hands-on-ul din Unit 2](https://huggingface.co/learn/audio-course/en/chapter2/hands_on), pașii 2–4: streaming, explorarea unui exemplu, waveform + spectrogramă. Dacă dataset-ul din exercițiu nu se încarcă, folosește `hf-internal-testing/librispeech_asr_dummy`.
4. Citește un fișier WAV oarecare ca bytes (`open(path, "rb").read()`) și decodează-l cu `soundfile.read(io.BytesIO(b))`. E exact situația din dataset-ul nostru.

## Verifică-te

- De ce 16 kHz e suficient pentru vorbire? (cuvânt-cheie: Nyquist)
- Câte valori are un clip de 7,8 s la 16 kHz?
- De ce nu faci downsampling pur și simplu aruncând fiecare al doilea eșantion?
- Ce îți arată în plus spectrograma față de spectru? Dar mel spectrograma față de spectrograma obișnuită?
- Ce pierzi în modul streaming față de descărcarea completă?
