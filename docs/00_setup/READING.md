# 00 · Setup

**Scop:** pregătești conturile și uneltele de care ai nevoie înainte de studiu:
- Hugging Face;
- Colab în VSCode, pentru GPU;
- W&B, pentru monitorizarea training-ului;
- fork-urile pe GitHub.

Tot aici înveți cum salvăm și cum reluăm checkpoint-uri.
**În notebook:** §1 Install Dependencies (inclusiv „TASK: Reinstall transformers and qwen-vl-utils from your own forks”) și §2 HF Login.
**Pașii concreți:** [SETUP.md](SETUP.md)

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [HF · User access tokens](https://huggingface.co/docs/hub/security-tokens) | tipuri de token (read/write), de ce nu-l pui în cod | Obligatoriu |
| 2 | [huggingface_hub · Quickstart](https://huggingface.co/docs/huggingface_hub/quick-start) | login, variabila `HF_TOKEN` | Obligatoriu |
| 3 | [huggingface_hub · Repositories](https://huggingface.co/docs/huggingface_hub/guides/repository) | `create_repo`, repo-uri private | Obligatoriu |
| 4 | [huggingface_hub · Download files](https://huggingface.co/docs/huggingface_hub/guides/download) | `snapshot_download`: cum aduci un checkpoint înapoi | Obligatoriu |
| 5 | [Colab în VSCode · User guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) | server nou cu GPU, monitorizarea consumului, `Colab: Remove Server` | Obligatoriu |
| 6 | [Colab · FAQ](https://research.google.com/colaboratory/faq.html) | limitele sesiunilor (inactivitate, durată maximă) | Obligatoriu |
| 7 | [Transformers · Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) | `save_steps`, `save_total_limit`, `push_to_hub`, `hub_strategy`, `resume_from_checkpoint` | Obligatoriu |
| 8 | [W&B · Hugging Face integration](https://docs.coreweave.com/products/wandb/integrations/huggingface) | `wandb.login()`, `report_to="wandb"` | Obligatoriu |
| 9 | [pip · VCS support](https://pip.pypa.io/en/stable/topics/vcs-support/) | `pip install "git+https://...@branch#subdirectory=..."` | Obligatoriu (la etapa 04) |
| 10 | [GitHub · Fork a repo](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo) | fork-ul `transformers` și `qwen-vl-utils` | Obligatoriu (la etapa 04) |

## De exersat

**Caietul cu exerciții:** [practice/00_setup/exercises.ipynb](../../practice/00_setup/exercises.ipynb). Conține exercițiul de checkpoint de mai jos, pas cu pas, cu verificări.

1. Fă pașii 1–4 din [SETUP.md](SETUP.md): Hugging Face, Colab, W&B, fork-uri.
2. **Test Colab:**
   - deschide un notebook gol, pornește un server nou cu L4 și rulează `!nvidia-smi`;
   - închide serverul cu `Colab: Remove Server`;
   - verifică în status bar cât ai consumat.
3. **Exercițiu de checkpoint, gratuit, pe Mac:**
   - antrenează `gpt2` pe câteva propoziții, cu `max_steps=10`, `save_steps=5`, `push_to_hub=True`, `hub_strategy="checkpoint"`, într-un repo privat de test;
   - creează un Trainer nou cu `max_steps=20`;
   - descarcă `last-checkpoint` cu `snapshot_download` și pornește `trainer.train(resume_from_checkpoint=...)`;
   - verifică în log că a continuat de la pasul 10.

## Verifică-te

- De ce nu salvăm checkpoint-urile pe disk-ul serverului Colab (`/content`)?
- Ce conține un checkpoint de Trainer în afară de greutăți, și de ce e nevoie de acele fișiere ca să reiei?
- Ce se întâmplă cu creditele Colab dacă lași serverul pornit fără să rulezi nimic?
- De ce token-ul HF nu se scrie în notebook?
