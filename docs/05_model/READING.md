# 05 · Modelul (pasul 5 din diagramă)

**Scop:** construiești clasa modelului (Qwen2-VL + encoder Whisper + proiector audio), o inițializezi din checkpoint-urile existente și o urci pe HF împreună cu config-ul actualizat.
**În notebook:** Step #5 Create the Model, Initialize & Push Model, „TASK: Update and upload model_config”.
**Rulează pe:** Mac, cu un config mic și greutăți aleatorii. Modelul 7B real îl încărcăm pe Colab.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [`modeling_qwen2_vl.py` (v4.47.0)](https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py) | `Qwen2VLForConditionalGeneration.forward`: unde embedding-urile imaginii înlocuiesc `<|image_pad|>` | Obligatoriu |
| 2 | [`configuration_qwen2_vl.py` (v4.47.0)](https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py) | cum e structurat config-ul (text + vision) | Obligatoriu |
| 3 | [`modeling_qwen2_audio.py` (v4.47.0)](https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py) | audio tower (encoder tip Whisper) + `multi_modal_projector`: cel mai apropiat model existent de ce construim | Obligatoriu |
| 4 | [Transformers · Qwen2-Audio](https://huggingface.co/docs/transformers/model_doc/qwen2_audio) | arhitectura la nivel înalt | Obligatoriu |
| 5 | [Transformers · Customizing models](https://huggingface.co/docs/transformers/custom_models) | `PreTrainedModel` + `PretrainedConfig` | Obligatoriu |
| 6 | [LLaVA paper](https://arxiv.org/abs/2304.08485) | proiector + antrenare în 2 etape: exact ideea din notebook | Obligatoriu (secțiunile despre arhitectură și antrenare) |
| 7 | [Transformers · Models](https://huggingface.co/docs/transformers/main_classes/model) | `from_pretrained`, `save_pretrained`, `push_to_hub`, `resize_token_embeddings` | Referință |
| 8 | [Transformers · Instantiate a big model](https://huggingface.co/docs/transformers/main/en/big_models) | checkpoint-uri sharded (mai multe fișiere `.safetensors`) | Opțional |
| 9 | [Vision Language Models Explained](https://huggingface.co/blog/vlms) | imaginea de ansamblu | Opțional |
| 10 | [Qwen2-VL paper](https://arxiv.org/abs/2409.12191) / [Qwen2-Audio paper](https://arxiv.org/abs/2407.10759) | detalii de arhitectură | Opțional |

## De exersat

1. Creează un `Qwen2VLConfig` mic (2 straturi, hidden size mic) și instanțiază `Qwen2VLForConditionalGeneration(config)` cu greutăți aleatorii. Rulează un forward pe Mac.
2. Încarcă `WhisperModel.from_pretrained("openai/whisper-tiny")`, păstrează doar `.encoder` și rulează-l pe un log-mel.
3. Scrie un proiector `nn.Linear(d_whisper, hidden_llm)` și verifică shape-urile de intrare și de ieșire.
4. În `modeling_qwen2_audio.py`, găsește funcția care îmbină embedding-urile audio cu cele de text.
5. Salvează un model mic cu `save_pretrained`, adaugă un modul nou în clasă și reîncarcă-l. Observă warning-ul „newly initialized”.

## Verifică-te

- Care parametri vin din Qwen2-VL, care din Whisper și care sunt aleatori la început?
- De ce dimensiunea de ieșire a proiectorului trebuie să fie egală cu hidden size-ul LLM-ului?
- Ce e `_name_or_path` în config și de ce trebuie actualizat?
- De ce trebuie config-ul să conțină id-urile tokenilor audio?
