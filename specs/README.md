# Specs: spec-driven development

Înainte să implementăm un TASK din notebook-ul mentorului, scriem **împreună** un spec scurt: ce construim, cu ce interfață, cum verificăm că e corect. Abia apoi scriem codul.

Specs-urile sunt și **memoria proiectului**. [PROJECT.md](PROJECT.md) ține contextul, deciziile și întrebările deschise, ca nimic să nu se piardă între sesiuni.

## Fluxul pentru fiecare componentă

1. **Teorie:** `docs/NN_.../READING.md`
2. **Exercițiu:** `practice/NN_.../exercises.ipynb`
3. **Spec draft:** copiem [_TEMPLATE.md](_TEMPLATE.md) în `NN_nume.md` și completăm împreună:
   - requirements, luate din TASK-ul mentorului;
   - interfața;
   - designul;
   - criteriile de acceptare.
4. **Review:** rezolvăm întrebările deschise, apoi tu aprobi spec-ul (status ✅).
5. **Implementare:** în copia notebook-ului sau în fork-uri, exact după spec.
6. **Verificare:** bifăm criteriile de acceptare (status ✔️) și trecem deciziile și faptele noi în [PROJECT.md](PROJECT.md).

**Status:** 📝 draft · ✅ aprobat · 🚧 în lucru · ✔️ gata

## Reguli

- Un spec are **cel mult o pagină**. Dacă iese mai mare, componenta trebuie împărțită.
- **Criteriile de acceptare sunt testabile:** fiecare devine un `assert` în notebook.
- **Orice decizie nouă** ajunge în tabelul de decizii din [PROJECT.md](PROJECT.md), cu motivul ei.
- `templates/` e notebook-ul mentorului și **nu se modifică**.

## Index

| Spec | Etapă | Notebook (secțiune) | Status |
|---|---|---|---|
| [PROJECT.md](PROJECT.md) | toate | — | contextul proiectului, se actualizează mereu |
| [01_dataset.md](01_dataset.md) | 01 · Audio data | §2 Understand Dataset | 📝 draft |
| 02_whisper_encoder | 02 · Whisper | TASK whisper | se scrie când ajungem acolo |
| 03_chat_template | 03 · Chat template | Step #1–2 | — |
| 04_audio_processor | 04 · Processor | Step #3–4 | — |
| 05_model | 05 · Model | Step #5, Initialize & Push | — |
| 06_inference | 06 · Inferență | Inference | — |
| 07_stage1 | 07 · Training Stage 1 | Stage 1 | — |
| 08_stage2 | 08 · Training Stage 2 | Stage 2 | — |
| 09_evaluation | 09 · Evaluare | Testing | — |
