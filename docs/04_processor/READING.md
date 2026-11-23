# 04 · Încărcarea audio-ului și extinderea processor-ului (pașii 3–4 din diagramă)

**Scop:**
- adaugi `fetch_audio`, care încarcă audio-ul din bytes, în `qwen-vl-utils`;
- extinzi processor-ul Qwen2-VL să repete `<|audio_pad|>` după lungimea fiecărui audio;
- tot în processor, concatenezi audio-urile într-un tensor lung și păstrezi lungimile.

**În notebook:** Step #3 Load audio, Step #4 Extend the Processor și „TASK: load your processor and test how it works”.
**Rulează pe:** Mac.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [qwen-vl-utils (GitHub)](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils) | README, cum se folosește `process_vision_info` (repo-ul s-a mutat în Qwen3-VL) | Obligatoriu |
| 2 | [`vision_process.py`](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py) | `fetch_image`, `fetch_video`, `process_vision_info`: aici adaugi `fetch_audio` | Obligatoriu |
| 3 | [`processing_qwen2_vl.py` (v4.47.0)](https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/qwen2_vl/processing_qwen2_vl.py) | cum se repetă `<|image_pad|>` în funcție de `image_grid_thw` | Obligatoriu |
| 4 | [`processing_qwen2_audio.py` (v4.47.0)](https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/qwen2_audio/processing_qwen2_audio.py) | cum face Qwen2-Audio același lucru pentru audio | Obligatoriu |
| 5 | [Transformers · Processors](https://huggingface.co/docs/transformers/processors) | ce e un processor (tokenizer + feature extractor/image processor) | Obligatoriu |
| 6 | [Qwen2-VL blog](https://qwenlm.github.io/blog/qwen2-vl/) | „Naive Dynamic Resolution”: de ce numărul de tokeni depinde de imagine | Opțional |
| 7 | [GitHub · Fork a repo](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo) | fork-ul `transformers` și `qwen-vl-utils` | Referință |

> Versiunea de `qwen-vl-utils` instalată cu pip poate diferi de `main`. Vezi ce fișier rulează de fapt: `python -c "import qwen_vl_utils; print(qwen_vl_utils.__file__)"`.

## De exersat

1. Rulează `process_vision_info` pe o conversație cu o imagine și inspectează ce returnează.
2. Rulează `Qwen2VLProcessor` (`Qwen/Qwen2-VL-2B-Instruct`) pe imagine + text. Numără câte `<|image_pad|>` apar în `input_ids` și compară cu `image_grid_thw`.
3. În `processing_qwen2_audio.py`, găsește locul unde se calculează numărul de tokeni audio și unde se expandează placeholder-ul.
4. Clonează fork-ul tău de `transformers`, instalează-l editabil (`pip install -e .`) într-un env de test și adaugă un `print` în processor, ca să vezi că rulează codul tău.

## Verifică-te

- Cum decide Qwen2-VL câți `<|image_pad|>` pune pentru o imagine?
- Care e analogul pentru audio? (legătura cu numărul de vectori pe secundă scoși de encoderul Whisper, de la etapa 02)
- De ce concatenăm toate audio-urile din batch într-un singur tensor și păstrăm lungimile? (vezi nota din [diagramă](../VLA.png))
- De ce `fetch_audio` trebuie să returneze audio la 16 kHz?
