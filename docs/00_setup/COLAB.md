# Colab: sesiuni, checkpoint-uri, costuri

Referință pentru etapele cu GPU (06–09). Setup-ul inițial e în [SETUP.md](SETUP.md).

> Serverul Colab **nu vede fișierele de pe Mac**.
> - Codul vine de pe GitHub, din fork-uri.
> - Modelele, dataset-ul și checkpoint-urile vin de pe HF Hub.
> - Fișierele mici le poți urca cu click dreapta → *Upload to Colab*.

## Prima celulă din fiecare sesiune Colab

```python
# 1) Dependențe: fork-urile + restul
!pip install -q "git+https://github.com/Ceachi/transformers.git@audio" \
                "git+https://github.com/Ceachi/Qwen3-VL.git@audio#subdirectory=qwen-vl-utils" \
                trl peft bitsandbytes datasets librosa wandb accelerate

# 2) Token-uri: nu se scriu niciodată în notebook
import os, getpass
os.environ["HF_TOKEN"] = getpass.getpass("HF token: ")
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

import wandb
wandb.login()

# 3) Verificare GPU
!nvidia-smi
```

- **PyTorch:** folosim versiunea preinstalată în Colab. Pin-ul `torch==2.4.1+cu121` din notebook ocolea un bug din 2024; revenim la el doar dacă dăm de acel bug.
- **Attention:** folosim `attn_implementation="sdpa"`, care e inclus în PyTorch. Fără `flash-attn`.
- **Fluxul pentru cod:** modifici pe Mac → `git push` → reinstalezi pe Colab cu celula de mai sus.

## Checkpoint-uri: cum salvăm și cum reluăm

**Organizare: un singur repo, cu branch-uri separate**

| Branch | Ce conține |
|---|---|
| `main` | processor + model: ce încarcă notebook-ul cu `from_pretrained` |
| `stage1-checkpoints` | checkpoint-urile din Stage 1 |
| `stage2-checkpoints` | checkpoint-urile din Stage 2 (ultimul e mereu în `last-checkpoint/`) |

Folosim branch-uri, nu un folder `checkpoints/`, din cauza modului în care urcă Trainer-ul:
- când face push, urcă în **rădăcina** repo-ului și ar suprascrie modelul din `main`;
- nu are opțiune de subfolder, dar are `hub_revision`, cu care poate urca pe alt branch.

Pe huggingface.co, branch-urile le vezi în *Files and versions*, din meniul cu `main`.

**Reguli:**
- Disk-ul serverului Colab (`/content`) se șterge când serverul e închis, de aceea salvăm pe HF Hub.
- **Salvăm la ~15–30 de minute de training.** Alegem `save_steps` după ce măsurăm câte secunde durează un pas în primii ~20 de pași.
- **`save_total_limit=2`**, ca să nu se umple disk-ul.

**Configurare (exemplu pentru Stage 2, unde checkpoint-urile LoRA sunt mici):**
```python
from huggingface_hub import create_branch
from trl import SFTConfig

REPO = "Ceachi/project-finetune-vlm-for-speech-understanding"
create_branch(REPO, branch="stage2-checkpoints", exist_ok=True)   # o singură dată

args = SFTConfig(
    output_dir="stage2",
    save_strategy="steps",
    save_steps=100,                    # ajustat după viteza măsurată
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id=REPO,
    hub_revision="stage2-checkpoints", # pe branch-ul de checkpoint-uri, nu pe main
    hub_strategy="checkpoint",         # ultimul checkpoint ajunge și în folderul last-checkpoint/
    report_to="wandb",
    # ... restul parametrilor
)
```

**Reluare, într-o sesiune nouă:**
```python
from huggingface_hub import snapshot_download

snapshot_download(REPO, revision="stage2-checkpoints",
                  allow_patterns=["last-checkpoint/*"], local_dir="ckpt")
# construiești trainer-ul exact ca înainte, apoi:
trainer.train(resume_from_checkpoint="ckpt/last-checkpoint")
```

**Stage 1:** un checkpoint complet ar avea ~18 GB. Acolo salvăm doar proiectorul, care are câțiva MB; stabilim cum la etapa 07.

**Plecat de acasă:**
- **Google AI Pro nu include background execution.** Doar Ultra o are ([sursa](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)). Dacă închizi laptopul, training-ul se poate opri.
- Pierzi doar ce s-a calculat după ultimul checkpoint.
- Acasă pornești un server nou, rulezi prima celulă și reiei ca mai sus.

## Reguli de cost

- Folosim unitățile din Google AI Pro (600 la 2026-11-23). Nu cumpărăm altele și nu facem upgrade.
- L4 pentru teste; A100 doar când L4 nu are destulă memorie.
- `Colab: Remove Server` când termini, sau când pleci și nu rulează nimic.
- Înainte de orice rulare lungă: măsori secunde/pas, apoi calculezi câte unități consumă (ore × unități/oră).
