# Context pentru Claude

- **Proiect de învățare (MentorCruise):** fine-tuning Qwen2-VL-7B pentru ASR, cu encoder Whisper turbo + proiector audio. Utilizatorul învață: explicații scurte, în română, cu link-uri; la exerciții, indicii, nu soluții.
- **La începutul fiecărei sesiuni citește [specs/PROJECT.md](specs/PROJECT.md).** Conține contextul, faptele tehnice, deciziile, întrebările deschise și unde am rămas. Statusul spec-urilor e în [specs/README.md](specs/README.md).
- **Flux:** `docs/NN` (teorie) → `practice/NN` (exerciții) → `specs/NN` (spec aprobat) → implementare.
- **Orice decizie nouă sau fapt descoperit** se trece în `specs/PROJECT.md`, la secțiunile „Decizii” și „Fapte tehnice”.
- **`templates/`** e notebook-ul mentorului și nu se modifică.
- **Caiete noi în `practice/`:** înainte să le dai utilizatorului, validează verificările cu o soluție de referință rulată în afara repo-ului.
