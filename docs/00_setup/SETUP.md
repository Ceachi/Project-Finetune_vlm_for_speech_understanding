# Setup: ce rulezi, în ordine

✅ = făcut · ⬜ = de făcut

## Acum, pe Mac

### 1. Environment ✅ (+ de instalat: `torchcodec`)
```bash
conda activate speech-vlm
pip install -U torch torchaudio torchvision transformers accelerate datasets librosa soundfile matplotlib ipykernel openai-whisper qwen-vl-utils wandb torchcodec
```
`torchcodec` e folosit de `datasets` 5 ca să decodeze audio; are nevoie de FFmpeg (pasul 2). Dacă îți lipsește, rulezi doar `pip install torchcodec`.
`qwen-vl-utils` are nevoie de `torchvision`. Dacă ai instalat lista fără el, rulezi doar `pip install torchvision`.

### 2. Repară FFmpeg ✅ (FFmpeg 9.0.2 funcționează)
```bash
brew reinstall ffmpeg
ffmpeg -version | head -1
```
Ultima comandă trebuie să afișeze versiunea, fără erori.

### 3. Login Hugging Face ✅
```bash
hf auth whoami
```
Trebuie să afișeze `Ceachi`.

### 4. Variabila `HF_TOKEN` ✅ (adăugată în `~/.zshrc`)
```bash
echo 'export HF_TOKEN="$(cat ~/.cache/huggingface/token)"' >> ~/.zshrc
source ~/.zshrc
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN e setat"
```
Apoi închide VSCode complet (`Cmd+Q`) și redeschide-l.

### 5. Repo-ul de model pe Hugging Face ✅ ([privat](https://huggingface.co/Ceachi/project-finetune-vlm-for-speech-understanding))
Aici urcă processor-ul (etapa 03) și modelul (etapa 05). Codul rămâne pe GitHub.
```bash
hf repos create Ceachi/project-finetune-vlm-for-speech-understanding --private --exist-ok
```

### 6. W&B (graficele de training) ✅ (logat ca `ceachi-bogdan`)
```bash
wandb login
```
Îți cere API key-ul de pe <https://wandb.ai/authorize>.

### 7. Verificare finală ✅
```bash
python -c "import torch, transformers, datasets, librosa, soundfile, whisper, wandb, qwen_vl_utils, os; from huggingface_hub import whoami; print('HF:', whoami()['name'], '| HF_TOKEN:', 'HF_TOKEN' in os.environ, '| MPS:', torch.backends.mps.is_available())"
```
Trebuie să afișeze: `HF: Ceachi | HF_TOKEN: True | MPS: True`

În VSCode: deschizi un notebook, apoi **Select Kernel → Python Environments → `speech-vlm`**.

**Cu asta ești gata pentru etapele 00–05.**

---

## Înainte de etapa 04: fork-uri pe GitHub ⬜ (`gh` e deja logat ca `Ceachi`)

```bash
gh repo fork huggingface/transformers --clone=false
gh repo fork QwenLM/Qwen3-VL --clone=false
```
Poți folosi și butonul **Fork** de pe GitHub. Dacă `gh` îți cere login, rulezi întâi `gh auth login`.

Branch-ul de lucru îl creăm la etapa 04, după ce mentorul confirmă versiunea de `transformers` ([Q1](../../specs/PROJECT.md#întrebări-deschise)).

## Înainte de etapa 06: Colab pentru GPU

1. ✅ **Nu cumperi nimic.** Ai deja Colab inclus în abonamentul **Google AI Pro**, cu **600 de compute units** la 2026-11-23. Planul dă 200 pe lună, iar cele 600 sunt probabil adunate din ultimele 3 luni. Soldul îl vezi pe <https://colab.research.google.com/signup>. Unitățile expiră după 90 de zile.
2. Instalezi extensia:
   ```bash
   code --install-extension google.colab
   ```
3. Testezi:
   - deschizi un notebook nou;
   - mergi pe **Select Kernel → Colab → New Colab Server → L4**;
   - rulezi `!nvidia-smi`.

   Dacă apare *„Unable to assign server. All GPU accelerators are unavailable”*, nu e o greșeală de setup: Colab nu are GPU-uri libere în acel moment. Reîncerci mai târziu. Dacă rămâne pornit un server fără GPU (vezi rata, de exemplu `0.08/hr`, în status bar), îl închizi cu `Colab: Remove Server`.
4. **Când termini:** `Cmd+Shift+P` → `Colab: Remove Server`. Altfel serverul consumă unități în continuare.

Soldul și consumul le vezi în status bar, dreapta jos.

Prima celulă din fiecare sesiune, checkpoint-urile și regulile de cost sunt în [COLAB.md](COLAB.md).
