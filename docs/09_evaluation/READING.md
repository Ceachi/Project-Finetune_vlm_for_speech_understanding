# 09 · Evaluare și testare

**Scop:** măsori calitatea transcrierilor (WER) și testezi modelul fine-tuned pe propria ta voce.
**În notebook:** §5 Testing the Fine-Tuned Model: încărcarea modelului și a adapterului, pregătirea `.wav`-ului, inferența.
**Rulează pe:** Colab pentru modelul final. Exercițiile merg pe Mac.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [Audio course · Evaluation](https://huggingface.co/learn/audio-course/en/chapter5/evaluation) | WER, CER, RTFx, normalizarea textului | Obligatoriu |
| 2 | [Metrica WER (evaluate)](https://huggingface.co/spaces/evaluate-metric/wer) | `evaluate.load("wer")` | Obligatoriu |
| 3 | [torchaudio · Resample](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html) | resampling la 16 kHz (cerut în notebook) | Obligatoriu |
| 4 | [PEFT · Models](https://huggingface.co/docs/peft/package_reference/peft_model) | `PeftModel.from_pretrained`: încărcarea adapterului | Obligatoriu |
| 5 | [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) | ce WER au modelele bune, ca reper | Opțional |
| 6 | [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) | biblioteca de metrici | Referință |
| 7 | [librosa · resample](https://librosa.org/doc/0.11.0/generated/librosa.resample.html) | alternativă la torchaudio | Referință |
| 8 | [ffmpeg](https://ffmpeg.org/ffmpeg.html) | conversia `.m4a` (Voice Memos) → `.wav` | Referință |

## De exersat

1. Calculează de mână WER-ul pentru referința „the cat sat on the mat” și predicția „the cat sit on the”, apoi verifică rezultatul cu `evaluate`.
2. Înregistrează un Voice Memo, convertește-l în WAV cu `ffmpeg`, resamplează-l la 16 kHz și transcrie-l cu `openai/whisper-tiny`. Asta e baseline-ul.
3. Aplică `BasicTextNormalizer` pe predicție și pe referință și compară WER-ul înainte și după normalizare.

## Verifică-te

- Poate WER-ul să depășească 100%? În ce situație?
- WER ortografic vs WER normalizat: pe care îl raportezi și de ce?
- De ce trebuie resamplată înregistrarea ta la 16 kHz înainte s-o dai modelului?
