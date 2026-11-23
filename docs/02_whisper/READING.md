# 02 · Whisper

**Scop:** înțelegi cum transformă Whisper audio-ul în log-mel spectrogram și apoi în vectori (audio encodings). Folosim **doar encoderul**.
**În notebook:** „TASK: Learn how to use whisper library to process an audio file from the dataset”.
**Rulează pe:** Mac. Encoderul Whisper turbo încape fără probleme.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [Audio course · Unit 3 intro](https://huggingface.co/learn/audio-course/en/chapter3/introduction) | input ca spectrogramă, de ce e mai scurt decât waveform-ul | Obligatoriu |
| 2 | [Audio course · Seq2Seq](https://huggingface.co/learn/audio-course/en/chapter3/seq2seq) | encoderul și decoderul Whisper, cross-attention | Obligatoriu |
| 3 | [Audio course · CTC](https://huggingface.co/learn/audio-course/en/chapter3/ctc) | alternativa encoder-only, pentru context | Opțional |
| 4 | [Audio course · Pre-trained ASR models](https://huggingface.co/learn/audio-course/en/chapter5/asr_models) | Whisper în practică, limita de 30 s, chunking | Obligatoriu |
| 5 | [openai/whisper (GitHub)](https://github.com/openai/whisper) | `load_model`, `pad_or_trim`, `log_mel_spectrogram` (notebook-ul cere pachetul OpenAI) | Obligatoriu |
| 6 | [Model card · whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | ce diferă turbo față de large-v3 | Obligatoriu |
| 7 | [Transformers · Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) | `WhisperFeatureExtractor`, `WhisperModel`, encoderul | Obligatoriu |
| 8 | [Blog · Fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper) | secțiunea despre feature extractor | Opțional |
| 9 | [Whisper paper](https://arxiv.org/abs/2212.04356) | secțiunea 2 (Approach) | Opțional |

## De exersat

1. Rulează `pipeline("automatic-speech-recognition", model="openai/whisper-tiny")` pe un eșantion ([Unit 2](https://huggingface.co/learn/audio-course/en/chapter2/asr_pipeline)).
2. Calculează log-mel cu `WhisperFeatureExtractor` pentru `openai/whisper-tiny` și pentru `openai/whisper-large-v3-turbo`. Compară shape-urile.
3. Cu pachetul `openai-whisper`: `load_model("tiny")`, `pad_or_trim`, `log_mel_spectrogram`, apoi rulezi doar `model.encoder(mel)`. Notează shape-ul output-ului.
4. Vizualizează log-mel-ul cu `librosa.display.specshow`. Codul e în [Unit 1 · Preprocessing](https://huggingface.co/learn/audio-course/en/chapter1/preprocessing).

## Verifică-te

- De ce aduce Whisper orice audio la exact 30 s?
- Ce shape are output-ul encoderului pentru 30 s? Câți vectori revin unei secunde?
- De ce folosim în proiect doar encoderul, nu și decoderul Whisper?
- Ce diferă între `tiny` și `turbo` la numărul de mel bins și la dimensiunea vectorilor?
