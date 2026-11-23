# 06 · Inferență

**Scop:**
- scrii o funcție de inferență: conversație → chat template → processor → model → decodare;
- verifici că partea vision dă același rezultat ca Qwen2-VL original;
- vezi ce face modelul cu audio înainte de fine-tuning.

**În notebook:** secțiunea Inference: „TASK: write an inference function”, testul VL cu mașina roșie, testul cu audio.
**Rulează pe:** Colab (L4) pentru 7B. Exercițiile merg pe Mac cu modelul 2B.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [Transformers · Text generation](https://huggingface.co/docs/transformers/llm_tutorial) | `generate`, `max_new_tokens`, decodarea | Obligatoriu |
| 2 | [Transformers · Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) | greedy vs sampling vs beam search | Obligatoriu |
| 3 | [Transformers · Qwen2-VL](https://huggingface.co/docs/transformers/model_doc/qwen2_vl) | exemplul de inferență cu imagine | Obligatoriu |
| 4 | [Cookbook · Fine-tuning VLM cu TRL](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) | funcția de inferență din cookbook | Obligatoriu |
| 5 | [Transformers · Attention backends](https://huggingface.co/docs/transformers/attention_interface) | `attn_implementation`: `"sdpa"` (inclus în PyTorch) vs `"flash_attention_2"` | Obligatoriu |
| 6 | [Colab în VSCode · User guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) | server nou cu GPU, `Colab: Remove Server` | Obligatoriu |

## De exersat

1. Pe Mac: rulează `Qwen/Qwen2-VL-2B-Instruct` pe [imaginea din notebook](https://t4.ftcdn.net/jpg/01/57/82/05/360_F_157820583_agejYX5XeczPZuWRSCDF2YYeCGwJqUdG.jpg) cu promptul „Detect the bounding box of the red car.”
2. Compară `do_sample=False` cu `do_sample=True`, rulând fiecare variantă de două ori.
3. Decodează output-ul o dată cu `skip_special_tokens=True` și o dată cu `False`, și compară.

## Verifică-te

- De ce trebuie ca output-ul VL al modelului nostru să fie identic cu cel al Qwen2-VL?
- De ce ai nevoie de greedy decoding ca să compari corect două modele?
- Ce te aștepți să răspundă modelul la audio înainte de fine-tuning, și de ce?
