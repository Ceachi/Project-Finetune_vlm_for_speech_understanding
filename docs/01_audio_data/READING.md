# 01 · Audio data

**Scop:** înțelegi ce e audio-ul digital și cum arată datele din dataset; încarci un eșantion în streaming și îl vizualizezi.
**În notebook:** §2 Understand Dataset, cu „TASK: load the dataset in the streaming mode” și „TASK: use take and grab 1 sample… visualize the wave using librosa”.
**Rulează pe:** Mac.

> Dataset-ul `speechbrain/LargeScaleASR` se numește acum [`speechbrain/LoquaciousSet`](https://huggingface.co/datasets/speechbrain/LoquaciousSet). Nu e gated și are configurațiile `small`, `medium`, `large`, `clean`. Audio-ul vine ca **bytes WAV**, iar textul e UPPERCASE, fără punctuație.

## De citit

| # | Resursă | Ce urmărești | Prioritate |
|---|---|---|---|
| 1 | [Audio course · Introduction to audio data](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) | sampling rate, amplitudine, waveform, spectru, spectrogramă, mel | Obligatoriu |
| 2 | [Audio course · Load and explore](https://huggingface.co/learn/audio-course/en/chapter1/load_and_explore) | structura unui eșantion audio în 🤗 Datasets | Obligatoriu |
| 3 | [Audio course · Preprocessing](https://huggingface.co/learn/audio-course/en/chapter1/preprocessing) | resampling, filtrare după durată, feature extractor | Obligatoriu |
| 4 | [Audio course · Streaming](https://huggingface.co/learn/audio-course/en/chapter1/streaming) | `streaming=True`, `next(iter(...))`, `.take(n)` | Obligatoriu |
| 5 | [🤗 Datasets · Stream](https://huggingface.co/docs/datasets/stream) | `IterableDataset`, `take`, `shuffle` cu buffer | Obligatoriu |
| 6 | [Dataset card · LoquaciousSet](https://huggingface.co/datasets/speechbrain/LoquaciousSet) | coloane, configurații, licență | Obligatoriu |
| 7 | [🤗 Datasets · Load audio](https://huggingface.co/docs/datasets/audio_load) / [Process audio](https://huggingface.co/docs/datasets/audio_process) | feature-ul `Audio`, `cast_column` | Opțional |
| 8 | librosa: [waveshow](https://librosa.org/doc/0.11.0/generated/librosa.display.waveshow.html), [specshow](https://librosa.org/doc/0.11.0/generated/librosa.display.specshow.html), [melspectrogram](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html), [load](https://librosa.org/doc/0.11.0/generated/librosa.load.html) | parametrii funcțiilor de vizualizare | Referință |
| 9 | [soundfile](https://python-soundfile.readthedocs.io/en/latest/) | citirea unui WAV din bytes | Referință |
| 10 | [3Blue1Brown · Fourier Transform (video)](https://www.youtube.com/watch?v=spUNpyF58BY) | intuiția vizuală pentru DFT | Opțional |
| 11 | [Digital Signals Theory (B. McFee)](https://brianmcfee.net/dstbook-site/content/intro.html) | aprofundare în procesarea de semnal | Opțional |

## Practică

Ghiduri pe module din curs, în [`practice/`](practice/). Fiecare conține teoria pe scurt, codul din curs, ce să observi și exerciții cu `# TODO`, cu soluția dedesubt. Le rulezi cu kernel-ul `speech-vlm`.

| # | Ghid | Pagina din curs | Status |
|---|---|---|---|
| 01 | [01_introduction.ipynb](practice/01_introduction.ipynb) | [Introduction to audio data](https://huggingface.co/learn/audio-course/en/chapter1/audio_data) | ✅ gata |
| 02 | [02_load_and_explore.ipynb](practice/02_load_and_explore.ipynb) | [Load and explore](https://huggingface.co/learn/audio-course/en/chapter1/load_and_explore) | ✅ gata |
| 03 | [03_preprocessing.ipynb](practice/03_preprocessing.ipynb) | [Preprocessing](https://huggingface.co/learn/audio-course/en/chapter1/preprocessing) | ✅ gata |
| 04 | [04_streaming.ipynb](practice/04_streaming.ipynb) | [Streaming](https://huggingface.co/learn/audio-course/en/chapter1/streaming) | ✅ gata |
| 05 | [05_datasets_stream.ipynb](practice/05_datasets_stream.ipynb) | [🤗 Datasets · Stream](https://huggingface.co/docs/datasets/stream) | ✅ gata |

Ghidurile 02–05 au nevoie de `pip install torchcodec`. Folosesc doar seturi mici (MINDS-14, `librispeech_asr_dummy`), deci nu descarci nimic mare.

## Întrebări și răspunsuri

Tot ce e important să reții din etapa asta. Încearcă întâi să răspunzi singur, apoi dă click pe întrebare ca să vezi răspunsul.

### A. Audio digital

<details><summary><b>1. Ce înseamnă să digitizezi un sunet?</b></summary>

Sunetul e o undă **continuă** de presiune. Microfonul o transformă în semnal electric, iar un convertor analog-digital îl **eșantionează**, adică îi măsoară valoarea la intervale fixe. Rezultatul e un șir **finit** de numere.

</details>

<details><summary><b>2. Ce e sampling rate-ul?</b></summary>

Numărul de eșantioane pe secundă, măsurat în Hz. CD: 44.100 Hz. Audio high-res: 192.000 Hz. Modele de vorbire, inclusiv Whisper: **16.000 Hz**.

</details>

<details><summary><b>3. Ce spune limita Nyquist?</b></summary>

Frecvența maximă pe care o poți captura e **jumătate din sampling rate**. La 16 kHz, asta înseamnă până la 8 kHz.

</details>

<details><summary><b>4. De ce e suficient 16 kHz pentru vorbire?</b></summary>

Frecvențele audibile din vorbire sunt sub 8 kHz, adică exact limita Nyquist la 16 kHz. Un sampling rate mai mare nu aduce informație în plus, doar cost de calcul. Unul mai mic (8 kHz) face vocea să sune „înfundat”.

</details>

<details><summary><b>5. Câte valori are un clip de 7,8 s la 16 kHz?</b></summary>

7,8 × 16.000 = **124.800** de eșantioane.

</details>

<details><summary><b>6. De ce trebuie ca toate exemplele să aibă același sampling rate?</b></summary>

Același sunet, la sampling rate-uri diferite, dă secvențe de lungimi diferite: 5 s înseamnă 80.000 de valori la 16 kHz și 40.000 la 8 kHz. Modelul nu generalizează între ele. La fine-tuning, datele trebuie aduse la sampling rate-ul cu care a fost pre-antrenat modelul.

</details>

<details><summary><b>7. Ce e aliasing-ul?</b></summary>

Când o frecvență e peste limita Nyquist, eșantioanele o „văd” ca pe o frecvență mai joasă, care nu există în realitate. Exemplu: un ton de 6 kHz eșantionat la 8 kHz apare la **2 kHz**.

</details>

<details><summary><b>8. Ce diferă între `.wav`, `.flac` și `.mp3`?</b></summary>

Modul în care comprimă datele: `.wav` e de obicei necomprimat, `.flac` e comprimat fără pierderi, `.mp3` e comprimat cu pierderi.

</details>

<details><summary><b>9. Ce e bit depth-ul?</b></summary>

Precizia cu care salvezi fiecare eșantion: 16 biți = 65.536 de niveluri, 24 de biți = 16.777.216. Rotunjirea la cel mai apropiat nivel produce **zgomot de cuantizare**. La 16 biți e deja inaudibil.

</details>

<details><summary><b>10. În ce format lucrează modelele ML cu audio?</b></summary>

**float32**, cu valori în intervalul **[-1.0, 1.0]**. Un float pe 32 de biți are o precizie echivalentă cu 24 de biți.

</details>

<details><summary><b>11. Cum citești decibelii în audio digital?</b></summary>

**0 dB** e sunetul cel mai tare posibil, iar restul valorilor sunt negative. **-6 dB** înseamnă aproximativ jumătate din amplitudine. Sub **-60 dB** sunetul practic nu se mai aude. Formula pentru amplitudine: `20 · log10(A / A_ref)`.

</details>

### B. Reprezentări în timp și în frecvență

<details><summary><b>12. Ce arată waveform-ul și la ce îl folosești?</b></summary>

Arată valoarea eșantioanelor în timp (domeniul timpului). Te ajută să vezi:
- când apar sunetele, cât de tare e semnalul, eventualul zgomot sau distorsiunile;
- dacă preprocesarea a făcut ce trebuia;
- cum arată exemplele pe care modelul greșește.

</details>

<details><summary><b>13. Ce face DFT-ul și ce legătură are cu FFT-ul?</b></summary>

DFT-ul descompune un fragment de semnal în frecvențele care îl compun și spune cât de puternică e fiecare. **FFT** e algoritmul rapid cu care se calculează DFT-ul. În practică, cei doi termeni se folosesc unul în locul celuilalt.

</details>

<details><summary><b>14. Ce conține rezultatul DFT-ului?</b></summary>

Numere complexe. Modulul (`np.abs`) e **amplitudinea**, iar unghiul e **faza**, pe care ML-ul o ignoră de obicei. **Power spectrum** înseamnă amplitudinea la pătrat.

</details>

<details><summary><b>15. Waveform-ul și spectrul conțin informații diferite?</b></summary>

Nu. Conțin **aceeași informație**, văzută diferit: waveform-ul arată amplitudinea în timp, spectrul arată frecvențele la un moment dat.

</details>

<details><summary><b>16. De ce aplici o fereastră (Hanning) înainte de FFT?</b></summary>

Tăietura bruscă de la capetele fragmentului creează frecvențe false (*spectral leakage*). Fereastra atenuează capetele treptat.

</details>

<details><summary><b>17. Ce sunt vârfurile din spectrul unei note muzicale?</b></summary>

**Armonicile**: fundamentala și multiplii ei. La trompeta din curs, fundamentala e ~624 Hz, dar cel mai puternic vârf e armonica a 2-a, la ~1249 Hz.

</details>

<details><summary><b>18. Ce e o spectrogramă și cum se calculează?</b></summary>

Arată frecvențele în timp. Calculezi multe DFT-uri pe ferestre scurte și le pui una lângă alta: asta e **STFT**-ul. Pe grafic, axa x e timpul, axa y e frecvența, iar culoarea e intensitatea, în dB.

</details>

<details><summary><b>19. Ce înseamnă `n_fft` și `hop_length`, și ce valori implicite are librosa?</b></summary>

`n_fft` e lungimea ferestrei, implicit **2048** de eșantioane. `hop_length` e pasul dintre ferestre, implicit `n_fft / 4` = **512**. Rezultatul are `n_fft / 2 + 1` frecvențe (1025) și `1 + len / hop` cadre.

</details>

<details><summary><b>20. Ce compromis impune lungimea ferestrei?</b></summary>

Ferestrele **scurte** dau precizie în **timp**, dar frecvențele sunt vagi. Ferestrele **lungi** dau frecvențe precise, dar timpul e vag. Nu le poți avea pe amândouă.

</details>

<details><summary><b>21. Poți reface sunetul dintr-o spectrogramă?</b></summary>

Da, cu ISTFT, dar **doar dacă ai și faza**. Modelele care generează spectrograme scot doar amplitudinea, așa că faza se estimează cu **Griffin-Lim** sau cu un **vocoder** neural (ex. HiFiGAN).

</details>

### C. Mel spectrogram

<details><summary><b>22. Ce e scara mel și de ce o folosim?</b></summary>

E o scară **perceptuală**. Urechea distinge mai fin diferențele între frecvențele joase decât între cele înalte, iar scara mel imită asta. Pune în evidență trăsăturile relevante pentru vorbire.

</details>

<details><summary><b>23. Cum se calculează un mel spectrogram?</b></summary>

Calculezi STFT-ul, apoi treci fiecare spectru printr-un **mel filterbank**. Filtrele sunt triunghiulare: **înguste** la frecvențe joase, **late** la cele înalte.

</details>

<details><summary><b>24. Ce e `n_mels` și ce valori se folosesc?</b></summary>

Numărul de benzi mel. Valorile uzuale sunt 40 sau 80. Whisper, până la large-v2, folosește 80. **Whisper large-v3 și turbo folosesc 128.**

</details>

<details><summary><b>25. Ce e un log-mel spectrogram?</b></summary>

Mel spectrogram-ul convertit în dB (logaritmic). E intrarea standard pentru modelele de vorbire, inclusiv pentru **Whisper**.

</details>

<details><summary><b>26. De ce nu calculezi mel spectrogram-ul „cum vrei”?</b></summary>

Pentru că există variante: scara htk vs slaney, putere vs amplitudine, log vs dB. Modelul se așteaptă la **exact** varianta de la antrenare, de aceea folosești **feature extractor-ul** lui.

</details>

<details><summary><b>27. De ce pierde informație un mel spectrogram?</b></summary>

Filtrele combină mai multe frecvențe într-o singură bandă, deci o parte din informație se pierde. Ca să refaci sunetul ai nevoie de un model (vocoder) care estimează ce lipsește.

</details>

### D. 🤗 Datasets: încărcare și explorare

<details><summary><b>28. Ce parametri are `load_dataset` și ce înseamnă?</b></summary>

- id-ul de pe Hub;
- `name`: **configurația**, adică subsetul (ex. limba);
- `split`: `train`, `test` etc.

La prima rulare, dataset-ul se descarcă și ajunge în cache, în `~/.cache/huggingface`.

</details>

<details><summary><b>29. Ce conține o coloană `Audio`?</b></summary>

`array` (vector NumPy 1-D, `float32`, cu eșantioanele), `sampling_rate` și, de obicei, `path`.

În `datasets` 5, coloana e un obiect `AudioDecoder`, care decodează doar când îl accesezi. `ex["audio"]["array"]` merge ca înainte. Pentru decodare ai nevoie de `torchcodec` și FFmpeg.

</details>

<details><summary><b>30. Ce e un `ClassLabel`?</b></summary>

O coloană care stochează etichetele ca **numere** și ține lista de nume în `features`. Treci la text cu `int2str`, de exemplu 13 → `"pay_bill"`, sau cu `.names`.

</details>

<details><summary><b>31. De ce scoți coloane cu `remove_columns`?</b></summary>

Ca să păstrezi doar ce îți trebuie pentru task. De exemplu, în MINDS-14 en-AU, `lang_id` are aceeași valoare peste tot, iar `english_transcription` repetă `transcription`.

</details>

<details><summary><b>32. Ce verifici când explorezi un dataset audio?</b></summary>

Câmpurile, sampling rate-ul, distribuția duratelor (cu extremele ei) și distribuția etichetelor. Asculți câteva exemple și te uiți la waveform și la spectrogramă.

La MINDS-14 en-AU iese: 654 de exemple, la **8 kHz**, între 2 și 57 s, dintre care 30 peste 20 s, cu clase relativ echilibrate.

</details>

### E. Preprocessing

<details><summary><b>33. Care sunt pașii de preprocesare pentru audio?</b></summary>

1. **Resampling** la sampling rate-ul modelului.
2. **Filtrare**, de exemplu după durată.
3. **Conversia în input features**, cu feature extractor-ul modelului.

</details>

<details><summary><b>34. Cum faci resampling cu 🤗 Datasets?</b></summary>

`ds.cast_column("audio", Audio(sampling_rate=16_000))`. Nu schimbă datele pe loc: resampling-ul se face **on-the-fly**, când încarci fiecare exemplu.

</details>

<details><summary><b>35. De ce downsampling-ul nu e doar „aruncă fiecare al doilea eșantion”?</b></summary>

Întâi trebuie eliminate frecvențele de peste noul Nyquist, cu un filtru **anti-aliasing**. Altfel apar alias-uri. Upsampling-ul, în schimb, interpolează valori noi între cele existente. Lași ambele operații pe seama librăriilor (librosa, datasets).

</details>

<details><summary><b>36. De ce filtrezi exemplele după durată?</b></summary>

Exemplele foarte lungi pot da **out-of-memory** la antrenare și sunt tăiate de modelele cu lungime fixă (Whisper taie la 30 s). În curs, MINDS-14 en-AU scade de la 654 la **624** de exemple după filtrul de 20 s.

</details>

<details><summary><b>37. Ce face feature extractor-ul Whisper?</b></summary>

Completează cu zerouri sau taie fiecare exemplu la **30 s**, apoi calculează **log-mel spectrogram-ul**. Fiindcă toate exemplele au aceeași lungime, Whisper **nu are nevoie de attention mask**: învață singur să ignore liniștea.

</details>

<details><summary><b>38. Ce e un processor și de ce e util?</b></summary>

Un obiect care împachetează **feature extractor-ul** (audio → features) și **tokenizer-ul** (text ↔ tokeni), pentru modele multimodale ca ASR-ul. Îl încarci cu `AutoProcessor.from_pretrained(...)`.

</details>

<details><summary><b>39. Cum aplici preprocesarea pe tot dataset-ul?</b></summary>

Scrii o funcție care procesează un exemplu și o aplici cu `ds.map(...)`. Rezultatul apare ca o coloană nouă, de exemplu `input_features`.

</details>

<details><summary><b>40. ⚠️ De ce nu e bun `padding=True` pentru Whisper, în versiunile actuale de transformers?</b></summary>

Pentru că `padding=True` completează doar până la **cel mai lung exemplu din batch**. Un singur exemplu de 7,8 s rămâne `(1, 80, 780)`. Whisper vrea mereu **3000 de cadre** (30 s), deci folosești apelul implicit, care e `padding="max_length"`.

</details>

<details><summary><b>41. Ce tokeni speciali adaugă tokenizer-ul Whisper?</b></summary>

La început `<|startoftranscript|>` și `<|notimestamps|>`, iar la final `<|endoftext|>`. Cu `skip_special_tokens=True`, dispar la decodare. În proiect vom adăuga tokeni speciali proprii (`<|audio_start|>` etc.), dar în tokenizer-ul Qwen.

</details>

### F. Streaming

<details><summary><b>42. Ce problemă rezolvă streaming-ul?</b></summary>

Dataset-urile audio sunt uriașe. De exemplu, GigaSpeech `xl` are 10.000 de ore și ocupă peste 1 TB. Cu `streaming=True`, exemplele se încarcă **pe rând**, în timp ce iterezi, fără să descarci tot.

</details>

<details><summary><b>43. Ce avantaje are streaming-ul?</b></summary>

Nu ocupă spațiu pe disk. Poți începe imediat, de la primul exemplu. Experimentezi rapid pe câteva exemple. Poți evalua ușor pe mai multe dataset-uri.

</details>

<details><summary><b>44. Ce pierzi în streaming?</b></summary>

Nu se salvează nimic local: nici datele brute, nici cele procesate. La fiecare refolosire, totul se descarcă și se procesează din nou. Dataset-urile pe care le folosești des e mai bine să le descarci.

</details>

<details><summary><b>45. Cum accesezi exemplele unui dataset în streaming?</b></summary>

Indexarea (`ds[0]`) nu merge. Iterezi: `next(iter(ds))` îți dă primul exemplu, iar `ds.take(n)` primele n.

</details>

<details><summary><b>46. Cum transformi un `Dataset` local într-un `IterableDataset`, și de ce ai face asta?</b></summary>

Cu `ds.to_iterable_dataset(num_shards=N)`. E mai rapid decât `streaming=True`, pentru că citește din fișierele locale, iar **shard-urile** permit amestecarea și citirea în paralel (de exemplu, cu workerii unui DataLoader).

</details>

<details><summary><b>47. ⚠️ Ce întoarce `ds[0]` pe un `IterableDataset` în `datasets` 5?</b></summary>

**Nu** primul exemplu, ci un `IterableColumn`: interpretează `0` ca nume de coloană. Primul exemplu îl iei cu `next(iter(ds))`.

</details>

<details><summary><b>48. Cum amestecă un `IterableDataset` fără să aibă tot dataset-ul în memorie?</b></summary>

Cu un **buffer**: ține `buffer_size` exemple (implicit 1000), scoate unul la întâmplare și pune în locul lui următorul din stream. Amestecă și ordinea shard-urilor. Un buffer mic amestecă doar local.

</details>

<details><summary><b>49. Ce face `set_epoch`, și ce capcană are?</b></summary>

Schimbă ordinea de amestecare de la o epocă la alta: seed-ul efectiv devine `seed + epoch`. ⚠️ Funcționează doar când iterezi **direct** dataset-ul amestecat. Un `.take(n)` creat din el pornește din nou de la epoca 0.

</details>

<details><summary><b>50. De ce faci shuffle înainte de `take` și `skip`?</b></summary>

`take(n)` (primele n) și `skip(n)` (restul) blochează ordinea shard-urilor, deci după ele nu mai poți amesteca. Cu shuffle, apoi `take` și `skip`, obții un split train/validare aleator și fără suprapuneri.

</details>

<details><summary><b>51. Ce face `interleave_datasets`?</b></summary>

Alternează exemple din mai multe dataset-uri cu aceleași coloane, în proporțiile date de `probabilities` (ex. `[0.8, 0.2]`). `stopping_strategy` decide când se oprește: `first_exhausted` sau `all_exhausted`.

</details>

<details><summary><b>52. Când rulează de fapt `map` și `filter` pe un `IterableDataset`?</b></summary>

**Leneș** (*lazy*), adică la iterare, exemplu cu exemplu. Construirea pipeline-ului e instantanee. Tot așa, `cast_column(..., Audio(sampling_rate=16_000))` resamplează din mers.

</details>

<details><summary><b>53. Cum reiei citirea datelor după o întrerupere?</b></summary>

Salvezi `ds.state_dict()` împreună cu checkpoint-ul modelului și apelezi `ds.load_state_dict(...)` la reluare. Shard-urile deja citite sunt sărite. ⚠️ Exemplele aflate în buffer-ul de shuffle se pierd. Pentru DataLoader există `StatefulDataLoader` (din `torchdata`).

</details>

### G. 🎯 Legătura cu proiectul

<details><summary><b>54. Ce intră concret în encoderul Whisper turbo?</b></summary>

1. Audio la **16 kHz**, completat sau tăiat la **30 s**.
2. Log-mel cu **128 de benzi**, calculat pe ferestre de 25 ms (`n_fft=400`) mutate din 10 în 10 ms (`hop_length=160`).
3. Rezultă 100 de cadre pe secundă, deci shape-ul **(1, 128, 3000)**.

</details>

<details><summary><b>55. Câți vectori scoate encoderul și ce reprezintă fiecare?</b></summary>

**1500 de vectori** pentru 30 s, pentru că encoderul înjumătățește numărul de cadre. Fiecare corespunde la **20 ms**, deci 50 pe secundă. Trecuți prin proiector, acești vectori devin tokenii audio pentru LLM.

</details>

<details><summary><b>56. Ce știm despre dataset-ul proiectului?</b></summary>

- Numele: [`speechbrain/LoquaciousSet`](https://huggingface.co/datasets/speechbrain/LoquaciousSet), fost `LargeScaleASR`.
- Mărimea: ~25.000 de ore, în configurațiile `small`, `medium`, `large` și `clean`. Nu e gated.
- Audio-ul vine ca **bytes WAV**, iar textul e **UPPERCASE, fără punctuație**.
- E prea mare pentru download complet, deci lucrăm în **streaming**.

</details>

<details><summary><b>57. Cum decodezi audio-ul când vine ca bytes?</b></summary>

Cu `soundfile.read(io.BytesIO(b))`, care întoarce `(array, sampling_rate)`. Dacă vrei bytes-ii nedecodați din `datasets`, folosești `cast_column(..., Audio(decode=False))`. Exact asta va face `fetch_audio` în proiect.

</details>
