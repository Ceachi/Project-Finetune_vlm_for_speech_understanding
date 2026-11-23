# Setup înainte de studiu

Unde rulăm:
- **Mac:** etapele 00–05, gratuit.
- **Colab Pay-As-You-Go din VSCode:** tot ce are nevoie de GPU (06–09).

Pe Mac folosești environment-ul tău obișnuit (Python 3.11) cu `torch torchaudio transformers datasets librosa soundfile matplotlib huggingface_hub openai-whisper qwen-vl-utils`. Nu instala pachetele care merg doar pe CUDA (`flash-attn`, `bitsandbytes`).

> **FFmpeg pe Mac:** FFmpeg-ul din Homebrew e stricat. Dependența `x265` a fost actualizată și `ffmpeg` nu mai pornește. Rulează `brew reinstall ffmpeg`.
> E necesar pentru decodarea automată a audio-ului în `datasets` (prin `torchcodec`) și pentru conversia Voice Memos → WAV de la etapa 09.

---

## 1. Hugging Face (o singură dată)

1. Cont, apoi un token cu drept de **write**: <https://huggingface.co/settings/tokens>
2. Pe Mac:
   - `hf auth login` (pe versiunile mai vechi: `huggingface-cli login`);
   - `export HF_TOKEN=...` în `~/.zshrc`, pentru că notebook-ul citește `os.environ['HF_TOKEN']`.
3. Repo-ul principal al proiectului (privat). Aici urcă processor-ul (etapa 03) și modelul (etapa 05):
   ```python
   from huggingface_hub import create_repo
   create_repo("<user>/qwen2vl-audio", private=True, exist_ok=True)
   ```
   Pentru checkpoint-urile din training folosim repo-uri separate (`<user>/qwen2vl-audio-stage1-ckpt`, `...-stage2-ckpt`), ca repo-ul principal să rămână curat. Le creăm la etapele 07–08.
   Verifică în setările contului HF limita de stocare privată.

## 2. Colab în VSCode

1. Cumpără **Pay As You Go: 100 compute units la $9.99**, fără abonament: <https://colab.research.google.com/signup>. Unitățile expiră după 90 de zile.
2. Instalează extensia [Google Colab](https://marketplace.visualstudio.com/items?itemName=google.colab) în VSCode.
3. Deschide un `.ipynb` și urmează calea **Select Kernel → Colab → New Colab Server**. Alege GPU-ul: **L4** pentru teste, **A100** doar când L4 nu are destulă memorie. Te loghezi cu contul Google.
4. Verifici cu `!nvidia-smi`.
5. **Monitorizarea costului:** în status bar (dreapta jos) vezi rata de consum, iar la hover vezi soldul. Panoul *Resources* din bara laterală Colab arată RAM, disk și GPU.
6. **Când termini: `Cmd+Shift+P` → `Colab: Remove Server`.** Un server pornit consumă unități chiar dacă nu rulează nimic.

> Serverul Colab **nu vede fișierele de pe Mac**. Tot ce îi trebuie vine de pe GitHub (codul din fork-uri) sau de pe HF Hub (modele, dataset, checkpoint-uri). Fișierele mici le poți urca cu click dreapta → *Upload to Colab*.

## 3. W&B (Weights & Biases)

1. Cont gratuit și API key: <https://wandb.ai/authorize>
2. În notebook: `import wandb; wandb.login()`. Îți cere cheia o singură dată pe sesiune.
3. Așa urmărești loss-ul de oriunde, inclusiv de pe telefon, cât timp ești plecat.

## 4. Fork-uri pe GitHub (necesare la etapa 04; le poți face și acum)

1. Fork la [`huggingface/transformers`](https://github.com/huggingface/transformers). Creezi un branch de lucru pornind de la versiunea pe care o confirmă mentorul (ex. `git checkout -b audio v4.47.0`).
2. Fork la [`QwenLM/Qwen3-VL`](https://github.com/QwenLM/Qwen3-VL). Conține folderul `qwen-vl-utils/`.
3. **Pe Mac (dezvoltare):** clonezi fork-urile și le instalezi editabil cu `pip install -e <folder>`. Modificările se văd imediat.
4. **Pe Colab:** instalezi direct din branch-ul de pe GitHub:
   ```bash
   pip install "git+https://github.com/<user>/transformers.git@audio"
   pip install "git+https://github.com/<user>/Qwen3-VL.git@audio#subdirectory=qwen-vl-utils"
   ```
   Fluxul: modifici pe Mac → `git commit` + `git push` → reinstalezi pe Colab.

## 5. Prima celulă din fiecare sesiune Colab

```python
# 1) Dependențe: fork-urile + restul
!pip install -q "git+https://github.com/<user>/transformers.git@audio" \
                "git+https://github.com/<user>/Qwen3-VL.git@audio#subdirectory=qwen-vl-utils" \
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

- **PyTorch:** folosim versiunea preinstalată în Colab. Pin-ul `torch==2.4.1+cu121` din notebook ocolea un bug din toamna lui 2024; revenim la el doar dacă dăm de acel bug.
- **Attention:** nu instalăm `flash-attn` la început. Folosim `attn_implementation="sdpa"`, care e inclus în PyTorch și nu cere compilare.

## 6. Checkpoint-uri: cum salvăm și cum reluăm

**Reguli:**
- **Salvăm pe HF Hub** (repo privat). Disk-ul serverului Colab (`/content`) se șterge când serverul e închis sau expiră.
- **Salvăm la ~15–30 de minute de training.** Alegem `save_steps` după ce măsurăm viteza (secunde/pas) în primii ~20 de pași.
- **`save_total_limit=2`**, ca să nu se umple disk-ul.

**Configurare (exemplu pentru Stage 2, LoRA, unde checkpoint-urile sunt mici):**
```python
from trl import SFTConfig

args = SFTConfig(
    output_dir="stage2",
    save_strategy="steps",
    save_steps=100,                    # ajustat după viteza măsurată
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id="<user>/qwen2vl-audio-stage2-ckpt",
    hub_strategy="checkpoint",         # ultimul checkpoint ajunge și în folderul last-checkpoint/
    hub_private_repo=True,
    report_to="wandb",
    # ... restul parametrilor de training
)
```

**Reluare, într-o sesiune nouă:**
```python
from huggingface_hub import snapshot_download

snapshot_download(
    "<user>/qwen2vl-audio-stage2-ckpt",
    allow_patterns=["last-checkpoint/*"],
    local_dir="ckpt",
)
# construiești trainer-ul exact ca înainte (același model, aceleași date, aceleași args), apoi:
trainer.train(resume_from_checkpoint="ckpt/last-checkpoint")
```
Training-ul continuă de la pasul salvat, cu tot cu starea optimizer-ului și a learning rate-ului.

**Stage 1:** un checkpoint complet ar avea ~18 GB (tot modelul), deși antrenăm doar proiectorul. Acolo vom salva doar proiectorul, care are câțiva MB. Detaliile le stabilim la etapa 07.

**Plecat de acasă în timpul training-ului:**
- Pe Pay-As-You-Go, training-ul poate continua după ce închizi laptopul, dar nu e garantat. Pornește de la ideea că se oprește.
- Pierzi doar ce s-a calculat după ultimul checkpoint.
- Acasă deschizi un server nou, rulezi celula de setup și reiei ca mai sus.

## 7. Reguli de cost (checklist)

- [ ] Doar unități preplătite, fără abonament.
- [ ] L4 pentru teste; A100 doar când L4 nu are destulă memorie.
- [ ] `Colab: Remove Server` când termini, sau când pleci și nu rulează nimic.
- [ ] Înainte de orice rulare lungă: măsori secunde/pas pe primii ~20 de pași, apoi calculezi costul (ore × unități/oră × ~$0,10).
- [ ] Te uiți la sold (status bar) la începutul și la sfârșitul fiecărei sesiuni.
