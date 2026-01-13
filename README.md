# Parkinson-Detection
Dieses Repository enthält das Semesterprojekt für den Kurs B52 Künstliche Intelligenz und Big Data Analytics (Wintersemester 2025/26) im Studiengang Informatik in Kultur und Gesundheit an der HTW Berlin.Der Kurs wird von Alexander Hinzer gelehrt.

Das Projekt untersucht den Einsatz von Machine-Learning- und audio-basierten KI-Methoden zur Analyse von Sprach- und Frequenzdaten im Zusammenhang mit Morbus Parkinson. 

# 1. Problemstellung
 
## Was ist Parkinson?
Der Morbus Parkinson ist eine chronisch fortschreitende, neurodegenerative Erkrankung, die unter anderem zu steifen Muskeln, verlangsamten Bewegungen und unkontrollierbarem Zittern führt [1]. 

## Was sind Ursachen für die Erkrankung?
Das primäre Parkinson-Syndrom geht von einer bestimmten Hirnregion aus, der sogenannten schwarzen Substanz (Substantia nigra) im Mittelhirn. Hier befinden sich spezielle Nervenzellen, die den Nervenbotenstoff Dopamin produzieren und mit ihm mit anderen Nervenzellen kommunizieren. Dopamin ist unter anderem wichtig für die Bewegungssteuerung. Durch die Parkinson-Erkrankung sterben immer mehr dieser Nervenzellen ab, es entwickeln sich ein Dopaminmangel und ein Ungleichgewicht der Nervenbotenstoffe im Gehirn [1]. 

Die Ursache für den Zelltod bei der Parkinson-Krankheit ist noch nicht eindeutig nachgewiesen. der primäre Parkinson macht etwa 75 Prozent aller Parkinson-Syndrome aus. Von diesem „klassischen“ Parkinson unterscheidet man die sehr seltenen genetischen Formen von Parkinson, das „Sekundäre Parkinson-Syndrom“, das z.B. durch Medikamente, Vergiftungen oder bestimmte Erkrankungen ausgelöst werden kann, und das „Atypische Parkinson-Syndrom“ als Folge verschiedenartiger anderer neurodegenerativer Erkrankungen [1]. 

## Was sind die Symptome der Parkinson Erkrankung? 
Zu den typischen Symptomen gehören das Zittern, weitere Bewegungsstörungen wie Steifheit der Muskeln, verlangsamte Bewegungen und Gleichgewichtsstörungen. Zusätzliche Symptome können das „Einfrieren“ von Bewegungen, Schwierigkeiten beim Sprechen und Schlucken, Störungen der vegetativen Funktionen (z. B. Blutdruck und Verdauung), Schlafstörungen, Depressionen und geistige Beeinträchtigungen bis hin zur Demenz sein [1].

Sprach- und Stimmveränderungen treten häufig auf und können bereits Jahre vor den klassischen motorischen Symptomen beobachtet werden. Diese frühen Veränderungen bieten ein Potenzial für eine frühere Erkennung und Überwachung der Krankheit, was wichtig wäre, da Diagnosen derzeit meist erst erfolgen, nachdem erheblicher neuronaler Verlust stattgefunden hat. Dies sind für Ärzt:innen schwer nur durchs hören erkennbar [2]. 

Zu den Veränderungen zählen eine verminderte Lautstärke (Hypophonie), eine monotone Sprechweise mit reduzierter Tonhöhen- und Lautstärkenvariation, eine verlangsamte oder unregelmäßige Sprechgeschwindigkeit sowie eine unpräzise Artikulation, bei der insbesondere Konsonanten weniger deutlich gebildet werden. Zusätzlich zeigen sich veränderte Pausenmuster, eine eingeschränkte Prosodie ((Prosodie = Betonung, Rhythmus, Melodie der Sprache)) mit flacher Betonung und reduziertem emotionalem Ausdruck sowie eine verminderte Stimmstabilität mit Schwankungen in Tonhöhe und Stimmqualität. Bei komplexeren Sprachaufgaben können außerdem Wortfindungsprobleme und eine reduzierte sprachliche Vielfalt auftreten. Diese Veränderungen sind oft so subtil, dass sie klinisch allein durch Zuhören schwer erkennbar sind und erst durch objektive, akustische und digitale Analyseverfahren zuverlässig erfasst werden können [2].

## Zielstellung
Die manuelle Analyse von Sprachaufnahmen ist häufig zeitaufwendig und mit einer begrenzten Zuverlässigkeit verbunden. Vor diesem Hintergrund stellt sich die Frage, ob man mit machinellem Lernen in der Lage ist, Sprach- und Frequenzdaten automatisiert auszuwerten, um Hinweise auf eine neurodegenerative Erkrankung wie Parkinson zu erkennen. Ziel ist die Entwicklung eines ML-Modells, das anhand von Sprachaufnahmen und akustischen Merkmalen eine Parkinson-Erkrankung  identifizieren kann.

## 2. Datenbeschaffung

Es gibt zahlreiche öffentlich verfügbare Datensätze zur Analyse von Parkinson anhand von Sprach- und Stimmmerkmalen. Die folgende Tabelle gibt einen Überblick über relevante Datensätze:

| Datensatzname | Datentyp | Inhalt | Größe / Umfang | Quelle |
|---------------|----------|--------|----------------|--------|
| UCI Parkinsons Dataset | Tabellarische Features | Frequenzmerkmale (Jitter, Shimmer, HNR, F0…) aus Sprachaufnahmen | 31 Personen, 195 Aufnahmen | https://archive.ics.uci.edu/dataset/174/parkinsons |
| UCI Parkinson Speech with Multiple Types of Audio | Tabellarische Features | Sustained Vowel, Wörter, Zahlen, Sätze | 40 Personen, mehrere hundert Audios | https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset |
| Figshare Parkinson Voice Samples | Audio (.wav) | Sustained Vowel /a/, Parkinson vs. Healthy | 100+ Aufnahmen | https://figshare.com/articles/dataset/23849127 |
| SJTU Parkinson Speech Dataset | Audio (.wav) | Verschiedene Sprachproben (Original-Speech) | > 150 Dateien | https://github.com/SJTU-YONGFU-RESEARCH-GRP/Parkinson-Patient-Speech-Dataset |
| Italian Parkinson’s Voice & Speech | Audio | Italienische Sprecher, Wörter, Vokale, Sätze | > 800 Aufnahmen, 65 Sprecher | https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech |
| Mendeley Parkinson Acoustic Features | Tabellarische Features | Akustische Merkmale (MFCC, Jitter, Shimmer…) | ca. 200+ Einträge | https://data.mendeley.com/datasets/fjd6fcfkwn |
| NeuroVoz Parkinsonian Speech Corpus | Audio | Monologe, Vokale, Wörter, Wiederholungen, längere Sprache | 108 Sprecher |https://zenodo.org/records/10777657 (nicht öffentlich zugänglich)|
| PC-GITA Corpus | Audio | Spanische Sprecher, mehrere Sprachaufgaben, Parkinson vs. Healthy | 235 Sprecher | https://perception.csl.illinois.edu/PC-GITA.html (nicht öffentlich verfügbar)|
| MDVR-KCL Parkinson Voice Dataset | Audio | Sustained vowel phonation „a“, klinisch diagnostiziert | 40 Sprecher, 400+ Samples | https://www.kaggle.com/datasets/nutansingh/mdvr-kcl-dataset |
| mPower Parkinson Dataset | Audio + Sensorik | Smartphone-Sprachaufnahmen, longitudinal, Real-World-Daten | Tausende Teilnehmende | https://www.synapse.org/#!Synapse:syn4993293 |
| Parkinson Telemonitoring Dataset | Zeitreihen / Features | Sprachbasierte UPDRS-Messungen | 42 Personen, 5.875 Messungen | https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring |
| German Parkinson Speech Corpus (nicht offen) | Audio | Deutsche Sprache, klinische Studien | Variabel | Nur über Forschungskooperation |
| RBD / Prodromal PD Speech Datasets | Audio | Sprachdaten von Hochrisikogruppen (prodromales Parkinson) | Forschungsdaten | Nicht öffentlich |



Für dieses Projekt wurden bewusst zwei unterschiedliche, sich ergänzende Datensätze ausgewählt. Der **UCI Parkinsons Datensatz** [5] dient als Einstieg und Benchmark, da er bereits extrahierte akustische Frequenzmerkmale enthält und eine schnelle, methodisch saubere Evaluation klassischer Machine-Learning-Modelle ermöglicht.

## 3. Exploratory Data Analysis
### Zielvariable

####  **`status`**  
  - `0` → gesunde Kontrollperson  
  - `1` → Parkinson-Erkrankung  

Ziel des Datensatzes ist die binäre Klassifikation von Parkinson-Erkrankten und gesunden Personen anhand von biomedizinischen Voice-Features. 

Der Datensatz weist eine deutliche Klassenungleichverteilung auf. Sowohl auf Personenebene als auch auf Aufnahmeebene sind Parkinson-Patient:innen etwa dreimal so häufig vertreten wie gesunde Kontrollpersonen. Von insgesamt 195 Sprachaufnahmen stammen 147 von an Parkinson erkrankten Personen, während lediglich 48 Aufnahmen von gesunden Personen vorliegen. Diese ausgeprägte Klassenimbalance könnte eine Herausforderung für Klassifikationsmodelle darstellen.

![Zielvariable: statu](/images/status.png)

### Featureanalyse
Der Datensatz umfasst akustische, rauschbasierte sowie nichtlineare Merkmale, die unterschiedliche Aspekte der Stimmproduktion abbilden. Während Jitter- und Shimmer-Parameter kurzfristige Instabilitäten der Frequenz und Amplitude beschreiben, erfassen nichtlineare Maße wie RPDE, DFA oder PPE komplexe zeitliche Strukturen des Sprachsignals. 

#### Grundfrequenz / Pitch (3 Variablen)
- **MDVP:Fo(Hz):** durchschnittliche Grundfrequenz (wahrgenommene Tonhöhe)
- **MDVP:Fhi(Hz):** maximale Grundfrequenz (obere Pitch-Spanne)
- **MDVP:Flo(Hz):** minimale Grundfrequenz (untere Pitch-Spanne) ➡️ Zusammen: Pitch-Level + Pitch-Variabilität (Spannweite der Stimme)

#### Jitter – Frequenzinstabilität (5 Variablen)
- **MDVP:Jitter(%):** relative (normierte) Periodenschwankung → gut vergleichbar
- **MDVP:Jitter(Abs):** absolute Periodenschwankung (sek) → stärker abhängig von F0
- **MDVP:RAP:** Jitter über 3 Perioden gemittelt → robuster, “intrinsische” Stabilität
- **MDVP:PPQ:** Jitter über 5 Perioden gemittelt → noch stärker geglättet
- **Jitter:DDP:** abgeleitetes Maß (≈ 3×RAP) → betont kurzfristige Instabilität
➡️ Hohe Werte = unregelmäßige Stimmlippenschwingung (bei Parkinson oft erhöht)

#### Shimmer – Amplitudeninstabilität (6 Variablen)
- **MDVP:Shimmer(%):** relative Amplitudenschwankung (lokal)
- **MDVP:Shimmer(dB):** gleiches Konzept, aber logarithmisch in dB
- **Shimmer:APQ3 / APQ5:** Amplitudenstörung über 3 bzw. 5 Perioden gemittelt (robuster)
- **MDVP:APQ (APQ11):** über 11 Perioden (stärkste Glättung)
- **Shimmer:DDA:** abgeleitet (≈ 3×APQ3) → betont kurzfristige Unregelmäßigkeiten ➡️ Hohe Werte = instabile Lautstärkeentwicklung / reduzierte Stimmkontrolle

#### Rauschmaße (2 Variablen)
- **NHR:** Noise-to-Harmonics → höher = mehr Rauschen / schlechtere Stimmqualität
- **HNR (dB):** Harmonics-to-Noise → höher = klarer/periodischer, “sauberer” Klang

#### Nichtlineare / Komplexität (8 Variablen)
- **RPDE:** Unregelmäßigkeit der Perioden (höher = unregelmäßiger/pathologischer)
- **DFA:** Langzeitkorrelation / fraktale Stabilität (höher = komplexer/oft weniger stabil)
- **D2:** Korrelationsdimension (höher = komplexere/chaotischere Dynamik)
- **PPE:** Entropie/Unvorhersagbarkeit der Pitch-Perioden (höher = unregelmäßiger)
- **spread1 / spread2:** Form & Streuung der Pitch-Verteilung (spread1 meist negativ/log-transformiert; spread2 positiv)

Die Pitch-Variablen (Fo/Fhi/Flo) zeigen insgesamt eine breite Streuung; Fo ist leicht rechtsschief verteilt, Fhi ist besonders stark rechtsschief und weist extrem hohe Ausreißer auf, und Flo ist ebenfalls rechtsschief. Die Jitter- und Shimmer-Variablen sind fast alle klar rechtsschief verteilt: Es gibt viele kleine „Normalwerte“ und nur wenige sehr hohe Werte, wodurch häufig eine hohe Schiefe und Kurtosis entsteht und die Verteilungen insgesamt ausreißerlastig sind. Das gilt besonders für Jitter(%), RAP/PPQ/DDP sowie MDVP:APQ (und teilweise auch für Shimmer(dB)). NHR ist ebenfalls stark rechtsschief mit wenigen extrem hohen Werten, während HNR dagegen relativ ausgewogen und leicht linksschief verteilt ist, mit einigen sehr niedrigen Werten. Die nichtlinearen Maße RPDE, DFA und D2 sind eher nahezu symmetrisch mit moderater Streuung, PPE ist hingegen leicht rechtsschief. spread1 liegt vollständig im negativen Bereich (log-transformiert) und wirkt insgesamt eher „normaler“ verteilt, während spread2 positiv ist und ebenfalls relativ unauffällig beziehungsweise symmetrisch erscheint.

### Peason Korrealation aller numerischen Features
Es zeigen sich deutlich erkennbare Feature-Gruppen bzw, Cluster:

1. **Jitter-Cluster** (MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP): Alle Variablen beschreiben die Instabilität der Grundfrequenz und unterscheiden sich nur im Berechnungsfenster bzw. Glättungsgrad
2. **Shimmer-Cluster** (MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA: Alle Variablen messen die  Amplitudeninstabilität und zeigen ein ähnliches Verhalten. 

Diese Variablen haben eine sehr stark positive korrelation und scheinen alle dassselbe Grundphänomen zu beschreibung und sind dadurch redundant in diesem Datensatz. Dies macht eine Featrue-Reduktion sinnvoll. 

**Zusammenhang zwischen den Jitter- und Shimmer-Maße:** Dies deutet darauf hin, das instabile Frequenz und instabilie Ampliture zusammen auftreten.

**Spiegelbildlicher Effekt bei NHR und HNR:** HNR korreliert stark negativ mit Jitter, Shimmer und NHR und NHR korreliert stark positiv mit Jitter- und Shimmer-Maße. Dies kann bedeuten: je instabilder und verauschter die Stimme desto niedriger ist der HNR und desto höher ist der NHR. Für die Klassifikation scheint HNR ein sehr wichtiges Feature zu sein.

**Grundfrequenz-Maße (Fo, Fhi, Flo)**: Diese Maße korrelieren nur mäßig untereinander und eher schwach mit Jitter und Shimmer. Dies weißt darauf hin, dass die Pitch-Höhe und die Pitch-Stabilität relativ unabhänig voneinander sind. 

**Nichtlineare Features (RPDE, DFA, spread1, spread2, D2, PPE)**: Diese Features komplexe nicht lineare zusammenhänge, was sie wertvoll für die Klassifikation macht. 

![Beschreibungstext](/images/corr_numerical.png)

### Pearson-Korrelation von `status`und den numerischen Featrues

Es zeigen sich überwiegend moderate Zusammenhänge zwischen den akustischen Merkmalen und dem binären Krankheitsstatus. Insbesondere nichtlineare Merkmale wie spread1, spread2 und PPE weisen die stärksten Korrelationen auf (> 0,5), was darauf hindeutet, dass nichtlineare Stimm­dynamiken besonders aussagekräftig für die Erkennung der Parkinson-Krankheit sind. Klassische Perturbationsmaße (Jitter und Shimmer) zeigen ebenfalls konsistente moderate positive Korrelationen, während Maße der Harmonizität sowie der Grundfrequenz negativ mit dem Krankheitsstatus korrelieren.

![Beschreibungstext](/images/corr_status.png)

# 4. Baseline-Modelle

Zur Definition einer belastbaren Referenzleistung wurden zunächst mehrere Baseline-Modelle unter Verwendung aller verfügbaren Merkmale trainiert. Alle Modelle wurden in ihrem Default-Zustand angewendet, ohne eine explizite Anpassung der Hyperparameter vorzunehmen. Dadurch konnte eine objektive Vergleichsbasis geschaffen werden, anhand derer der Einfluss nachfolgender Schritte wie der Feature Selection bewertet werden konnte.

Folgende Modelle wurden genutzt:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Suport Vector Machine
5. k-NN Classifier

|  Rang | Modell                           | ROC-AUC (%) | PR-AUC (%) | Accuracy (%) | F1-Score (weighted, %) |
| ----: | -------------------------------- | ----------: | ---------: | -----------: | ---------------------: |
| **1** | **Random Forest**                |  **97.586** | **99.251** |   **94.872** |             **94.872** |
| **2** | **Support Vector Machine (SVM)** |      95.517 |     98.297 |       92.308 |                 91.818 |
| **3** | **k-NN Classifier**              |      95.517 |     98.297 |       92.308 |                 91.818 |
| **4** | **Logistic Regression**          |      92.414 |     97.202 |       92.308 |                 92.170 |
| **5** | **Gradient Boosting Classifier** |      96.897 |     99.056 |       89.744 |                 89.744 |

# 5. Cross-Validation

Zur robusten Bewertung der Generalisierungsleistung wurden alle Baseline-Modelle zusätzlich mittels stratifizierter 5-facher Cross-Validation evaluiert. Dabei wurde in jedem Fold die Klassenverteilung beibehalten, um dem unausgeglichenen Datensatz Rechnung zu tragen. Als Evaluationsmetrik wurde die ROC-AUC verwendet, da sie unabhängig von einem festen Entscheidungsschwellenwert ist und sich besonders für Klassifikationsprobleme mit Klassenungleichgewicht eignet.
Der Mittelwert der ROC-AUC gibt die durchschnittliche Modellleistung an, während die Standardabweichung Aufschluss über die Stabilität der Ergebnisse über verschiedene Datenaufteilungen hinweg liefert.


|  Rang | Modell                           | CV ROC-AUC (Mittelwert ± Std.) |
| ----: | -------------------------------- | -----------------------------: |
| **1** | **k-Nearest Neighbors (k-NN)**   |           **97.17 % ± 1.01 %** |
| **2** | **Gradient Boosting Classifier** |           **96.23 % ± 3.08 %** |
| **3** | **Random Forest Classifier**     |           **95.92 % ± 2.85 %** |
| **4** | **Logistic Regression**          |           **90.50 % ± 4.25 %** |
| **5** | **Support Vector Machine (SVM)** |           **89.06 % ± 6.14 %** |

Die Ergebnisse zeigen, dass insbesondere nichtlineare Modelle eine hohe und stabile Trennschärfe erzielen. Der k-Nearest Neighbors Classifier erreicht sowohl die höchste mittlere ROC-AUC als auch die geringste Varianz und weist damit die stabilste Leistung auf. Random Forest und Gradient Boosting erzielen ebenfalls hohe Werte, zeigen jedoch etwas stärkere Schwankungen über die 5 Folds hinweg.

Die Logistic Regression zeigt als lineares Modell geringere Leistungen, während nichtlineare Modelle insgesamt besser abschneiden. Die Support Vector Machine weist trotz nichtlinearer Modellierung eine höhere Varianz auf, was auf eine stärkere Abhängigkeit von Datenaufteilung und Hyperparametern schließen lässt.

# 6. Feature Selection
In einer Studie von Wrobel (2021) konnte gezeigt werden, dass der Einsatz von Feature-Selection-Methoden bei der Klassifikation von Parkinson-Sprachdaten zu einer deutlichen Verbesserung der Modellleistung führt. Insbesondere erzielten nahezu alle untersuchten Klassifikatoren nach Merkmalsreduktion höhere Klassifikationsgenauigkeiten. Vor diesem Hintergrund wurde auch in der vorliegenden Arbeit eine Feature Selection durchgeführt [15]. 

Die Feature Selection erfolgt mittels Random Forest auf dem Trainingsdatensatz. Ziel ist es, redundante Features zu identifizieren und aus dem Datensatz zu entfernen. Dabei hat der Random-Forest-Algorithmus alle Shimmer-Merkmale ausgeschlossen, da diese stark mit Jitter- und Rauschmaßen korrelieren und keinen zusätzlichen Informationsgewinn liefern.

Ausgewählte Features:

- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- MDVP:RAP
- Jitter:DDP
- NHR 
- spread1
- PPE

# 7. Modelltraining mit selektierten Features

Die Modelle zeigten nach Anwendung der Feature Selection eine durchgängig geringere Performance. Dies deutet darauf hin, dass die entfernten Merkmale – insbesondere Shimmer-Features – trotz Redundanz relevante Zusatzinformationen enthielten.

|  Rang | Modell                           | ROC-AUC (%) | PR-AUC (%) | Accuracy (%) | F1-Score (weighted, %) |
| ----: | -------------------------------- | ----------: | ---------: | -----------: | ---------------------: |
| **1** | **Gradient Boosting Classifier** |  **93.793** | **97.991** |   **79.487** |             **79.487** |
| **2** | **k-NN Classifier**              |      92.586 |     96.627 |       87.179 |                 86.951 |
| **3** | **Support Vector Machine (SVM)** |      91.724 |     96.530 |       89.744 |                 89.345 |
| **4** | **Logistic Regression**          |      89.310 |     95.725 |       87.179 |                 87.372 |

Da der Random Forest bereits zur Feature Selection eingesetzt wurde und während des Fit-Prozesses implizit eine Merkmalsauswahl vornimmt, wurde auf ein erneutes Training dieses Modells auf dem reduzierten Feature-Set verzichtet.

# 8. Fazit
Obwohl die gesunde Klasse deutlich unterrepräsentiert ist, erzielten alle untersuchten Modelle hohe Klassifikationsleistungen. Dies lässt darauf schließen, dass insbesondere nichtlineare Sprachmerkmale eine starke Trennschärfe zur Parkinsondiagnostik aufweisen.

Die Evaluation der Baseline-Modelle zeigte, dass nichtlineare Verfahren wie Random Forest, Gradient Boosting und k-Nearest Neighbors den linearen Ansätzen überlegen sind. Die zusätzliche Bewertung mittels stratifizierter Cross-Validation bestätigte diese Beobachtung und verdeutlichte zugleich, dass schwellenwertunabhängige Metriken wie ROC-AUC und PR-AUC bei kleinen Testmengen besser geeignet sind als Accuracy. Der Einsatz einer Random-Forest-basierten Feature Selection führte entgegen der Erwartung zu einer durchgängig schlechteren Modellleistung, was darauf hindeutet, dass auch stark korrelierte Merkmale – insbesondere Shimmer-Features – relevante Zusatzinformationen liefern und zur Stabilisierung der Modelle beitragen.

Insgesamt erwies sich der Random Forest Classifier als das leistungsstärkste und zugleich robusteste Modell im gesamten Analyseprozess. Er erzielte sowohl in der Baseline-Evaluation als auch im Vergleich der Modellvarianten die besten Ergebnisse.

# 9. Fazit
Obwohl die gesunde Klasse deutlich unterrepräsentiert ist, erzielten alle untersuchten Modelle hohe Klassifikationsleistungen. Dies lässt darauf schließen, dass insbesondere nichtlineare Sprachmerkmale eine starke Trennschärfe zur Parkinsondiagnostik aufweisen.

Die Evaluation der Baseline-Modelle zeigte, dass nichtlineare Verfahren wie Random Forest, Gradient Boosting und k-Nearest Neighbors den linearen Ansätzen überlegen sind. Die zusätzliche Bewertung mittels stratifizierter Cross-Validation bestätigte diese Beobachtung und verdeutlichte zugleich, dass schwellenwertunabhängige Metriken wie ROC-AUC und PR-AUC bei kleinen Testmengen besser geeignet sind als Accuracy. Der Einsatz einer Random-Forest-basierten Feature Selection führte entgegen der Erwartung zu einer durchgängig schlechteren Modellleistung, was darauf hindeutet, dass auch stark korrelierte Merkmale – insbesondere Shimmer-Features – relevante Zusatzinformationen liefern und zur Stabilisierung der Modelle beitragen.

Insgesamt erwies sich der Random Forest Classifier als das leistungsstärkste und zugleich robusteste Modell im gesamten Analyseprozess. Er erzielte sowohl in der Baseline-Evaluation als auch im Vergleich der Modellvarianten die besten Ergebnisse.

# 10. Ausblick

In zukünftigen Arbeiten würde ich gerne reale Audiodateien direkt analysieren. Die vorliegende Arbeit diente primär dazu, ein grundlegendes Verständnis relevanter Merkmale für die Parkinson-Diagnostik auf Basis tabellarischer Sprachfeatures zu entwickeln. Aufbauend auf diesen Erkenntnissen bietet es sich an, den Figshare-Datensatz mit Roh-Audioaufnahmen [4] zu verwenden, um realistische Sprachsignale zu untersuchen und moderne, audio-basierte Analyseansätze umzusetzen. Die Auswertung der Audiodaten könnte mithilfe von Parselmouth [3], einer Python-Schnittstelle zu Praat, erfolgen, wodurch eine präzise und reproduzierbare Extraktion phonetischer und akustischer Merkmale ermöglicht wird.

# 11. Quellen
**[1]** Deutsche Gesellschaft für Parkinson und Bewegungsstörungen e. V. (o. J.). *Die Parkinson-Krankheit*. Abgerufen am *29. Dezember 2025*, von https://parkinson-gesellschaft.de/fuer-betroffene/die-parkinson-krankheit/

**[2]** Cao, F., Vogel, A. P., Gharahkhani, P., & Renteria, M. E. (2025). *Speech and language biomarkers for Parkinson’s disease prediction, early diagnosis and progression.* *npj Parkinson’s Disease, 11*(57). https://doi.org/10.1038/s41531-025-00913-4

**[3]** Jadoul, Y., Thompson, B., & de Boer, B. (2018). *Introducing Parselmouth: A Python interface to Praat*. *Journal of Phonetics, 71*, 1–15. [https://doi.org/10.1016/j.wocn.2018.07.001](https://doi.org/10.1016/j.wocn.2018.07.001) 

**[4]** Sakar, C. O., Serbes, G., & Gunduz, A. (2023). *Voice samples for patients with Parkinson’s disease and healthy controls* [Data set]. Figshare. [https://doi.org/10.6084/m9.figshare.23849127](https://doi.org/10.6084/m9.figshare.23849127)

**[5]** A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
'Accurate telemonitoring of Parkinson.s disease progression by non-invasive speech tests',IEEE Transactions on Biomedical Engineering [https://doi.org/10.24432/C56C7T](https://doi.org/10.24432/C56C7T)

**[6]** DataCamp. (n.d.). Measures of central tendency in Python. Abgerufen am 6. Januar 2026, von https://www.datacamp.com/de/tutorial/central-tendency

**[7]** Yulianti, Y., Syapariyah, A. N., & Saifudin, A. (2020). Feature selection techniques to choose the best features for Parkinson’s disease predictions based on decision tree. Journal of Physics: Conference Series, 1477(3), 032008. https://doi.org/10.1088/1742-6596/1477/3/032008

**[8]** Farrús, M., & Morales, M. (2007). *Evaluation of voice perturbation measures for automatic speaker recognition*. In *Proceedings of Interspeech 2007* (pp. 1414–1417). ISCA. [https://www.isca-archive.org/interspeech_2007/farrus07_interspeech.pdf](https://www.isca-archive.org/interspeech_2007/farrus07_interspeech.pdf)

**[9]** CVT Research. (n.d.). *Relative Average Perturbation (RAP)*. Abgerufen am 8. Januar 2026, von [https://cvtresearch.com/relative-average-perturbation/](https://cvtresearch.com/relative-average-perturbation/)

**[10]** Boersma, P., & Weenink, D. (n.d.). *PointProcess: Get jitter (ddp)*. Praat Manual. Abgerufen am 8. Januar 2026, von [https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ddp____.html](https://www.fon.hum.uva.nl/praat/manual/PointProcess__Get_jitter__ddp____.html)

**[11]** Wikipedia. (n.d.). *Shimmer*. Abgerufen am 8. Januar 2026, von [https://de.wikipedia.org/wiki/Shimmer](https://de.wikipedia.org/wiki/Shimmer)

**[12]** Boersma, P., & Weenink, D. (n.d.). *Voice 3: Shimmer*. Praat Manual. Abgerufen am 8. Januar 2026, von [https://www.fon.hum.uva.nl/praat/manual/Voice_3__Shimmer.html](https://www.fon.hum.uva.nl/praat/manual/Voice_3__Shimmer.html)

**[13]** Boersma, P., & Weenink, D. (n.d.). *Harmonicity*. Praat Manual. Abgerufen am 8. Januar 2026, von [https://www.fon.hum.uva.nl/praat/manual/Harmonicity.html](https://www.fon.hum.uva.nl/praat/manual/Harmonicity.html)

**[14]** Wijaya, C. Y. (2024). Is it necessary for feature scaling in tree-based models? - NBD Lite #20. Retrieved from https://www.nb-data.com/p/is-it-necessary-for-feature-scaling

**[15]** Wrobel, K. (2021). Diagnosing Parkinson’s disease by means of ensemble classification of patients’ voice samples. Procedia Computer Science, 192, 3905–3914. https://doi.org/10.1016/j.procs.2021.09.165
 