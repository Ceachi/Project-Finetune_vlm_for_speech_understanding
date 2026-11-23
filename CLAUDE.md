# Context pentru Claude

- **Proiect de învățare (MentorCruise):** fine-tuning Qwen2-VL-7B pentru ASR, cu encoder Whisper turbo + proiector audio. Utilizatorul învață: explicații scurte, în română, cu link-uri; în ghiduri, cod rezolvat, cu comentarii `# TODO`.
- **La începutul fiecărei sesiuni citește [specs/PROJECT.md](specs/PROJECT.md).** Conține contextul, faptele tehnice, deciziile, întrebările deschise și unde am rămas. Statusul spec-urilor e în [specs/README.md](specs/README.md).
- **Flux:** `docs/NN/READING.md` (teorie) → `docs/NN/practice/` (ghiduri rulabile) → `specs/NN` (spec aprobat) → implementare.
- **La finalul fiecărui `docs/NN/READING.md`** e secțiunea „Întrebări și răspunsuri”, cu răspunsuri ascunse în `<details>`. Acoperă toată teoria etapei și o completezi pe măsură ce învățăm.
- **Orice decizie nouă sau fapt descoperit** se trece în `specs/PROJECT.md`, la secțiunile „Decizii” și „Fapte tehnice”.
- **`templates/`** e notebook-ul mentorului și nu se modifică.
- **Ghiduri noi în `docs/NN/practice/`:** câte un `.ipynb` pe modul de curs, cu teorie pe scurt, codul din curs, „ce să observi” și exerciții cu `# TODO` + soluție. **Doar seturi mici** în practică (MINDS-14, `librispeech_asr_dummy`, seturi-jucărie); fără download-uri mari, nici măcar la testare. Ghidul merge în folderul **etapei căreia îi aparține modulul** (ex. Unit 1 → `01_audio_data`, Unit 3 → `02_whisper`). Îl creezi **doar când utilizatorul ajunge la etapa respectivă, împreună cu el**, nu în avans. Înainte de livrare, rulează-l cap-coadă, în afara repo-ului, și verifică că răspunsurile scrise se potrivesc cu output-ul real.
