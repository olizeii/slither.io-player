# slither.io-player

Ein Computer-Vision-Bot für [slither.io](http://slither.io). Er liest nur den Bildschirm und steuert nur die Maus, genau wie ein Mensch. Die Pipeline: **Capture → Vision → Brain → Controller**, ausgelegt auf 60 FPS mit etwa 10–15 ms Latenz pro Frame.

> **Hinweis:** Bots verstoßen sehr wahrscheinlich gegen die Nutzungsbedingungen von slither.io und können zu Bans führen. Das Projekt ist als Lern- und Forschungsprojekt für Computer Vision und Echtzeit-Agenten gedacht. Nutzung auf eigenes Risiko.

---

## 1. Architektur

```
 ┌────────────── Thread "capture" ──────────────┐
 │ dxcam/mss grab (Spielregion)                 │
 │   → cv2.resize(scale=0.5, INTER_AREA)        │
 │   → LatestFrameBuffer (1 Slot, alte Frames   │
 │     werden verworfen statt gepuffert)        │
 └──────────────────────┬───────────────────────┘
                        │ neuester Frame
 ┌────────────── Main-Thread ───────────────────┐
 │ Vision.analyze()  ~6–10 ms → GameState        │
 │   Food, Hindernis-Punkte, Gegnerköpfe,        │
 │   eigener Radius/Heading, Rand (Minimap)      │
 │ Brain.decide()    < 1 ms   → Action           │
 │   Winkel, Boost, Modus                        │
 │ Hotkeys (F8/F9/F12), Debug-Overlay, Stats     │
 └──────────────────────┬───────────────────────┘
                        │ Zielwinkel + Boost (Lock)
 ┌────────────── Thread "mouse" (240 Hz) ───────┐
 │ dreht den Cursor mit begrenzter Winkel-       │
 │ geschwindigkeit zum Ziel (SetCursorPos),      │
 │ Boost = linke Maustaste via SendInput         │
 └──────────────────────────────────────────────┘
```

**Warum Threads statt Multiprocessing?** Die teuren Teile (DXGI-Capture, `cv2.*`, numpy) geben den GIL frei. Threads teilen den Frame ohne Kopie. Bei Multiprocessing müsste jeder Frame (1–6 MB) serialisiert oder über Shared Memory übergeben werden. Das kostet mehr, als es bei ~10 ms Rechenzeit pro Frame bringt. Die drei Threads bilden trotzdem eine echte Pipeline: Während Frame *n* analysiert wird, wird *n+1* schon aufgenommen und verkleinert.

**Warum ein eigener Maus-Thread?** Das Brain entscheidet mit Frame-Rate (30–60 Hz). Der Maus-Thread interpoliert mit 240 Hz. Die Steuerung wird dadurch flüssig, und Sprünge zwischen zwei Entscheidungen werden geglättet (`max_angular_speed`).

| Modul | Datei | Aufgabe |
|---|---|---|
| Config | `slither_bot/config.py` | alle Parameter als Dataclasses, Overrides per `config.json` |
| Capture | `slither_bot/capture.py` | dxcam/bettercam/mss, Downscaling, Latest-Frame-Buffer |
| Vision | `slither_bot/vision.py` | HSV-Thresholding, Komponenten-Klassifikation, Augen-Erkennung, Minimap |
| Tracker | `slither_bot/tracker.py` | Gegnerköpfe über Frames verfolgen → Geschwindigkeit |
| Brain | `slither_bot/brain.py` | Sektor-Potentialfeld + Taktik-Zustandsmaschine |
| Controller | `slither_bot/controller.py` | Cursor/Boost über Win32-API (pynput als Fallback) |
| Win32 | `slither_bot/win32.py` | ctypes: DPI-Awareness, Fenstersuche, SendInput, Hotkeys |
| Debug | `slither_bot/debug.py` | Overlay: Masken, Strahlen, Köpfe, gewählte Richtung |
| HUD | `slither_bot/hud.py` | Windows-OCR: Länge, Rang, endgültige Länge |
| Episode | `slither_bot/episode.py` | erkennt Rundenstart und -ende, sammelt Statistiken |
| Minimap | `slither_bot/minimap.py` | eigene Position + Richtung zum nächsten dichten Gebiet |
| Tuner | `slither_bot/tuner.py` | (1+1)-ES: lernt die Parameter zwischen den Runden |
| Main | `slither_bot/main.py` | verbindet alles, Main-Loop, Respawn, Training, Statistik |
| Kalibrierung | `tools/calibrate.py` | HSV-Schwellwerte live per Trackbar einstellen |

---

## 2. Setup & Dependencies

Python 3.10+ (empfohlen: 3.12), Windows 10/11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

| Bibliothek | Wofür | Warum diese |
|---|---|---|
| `dxcam` (oder `bettercam`) | Screen Capture | DXGI Desktop Duplication: 100–240 FPS, ~2–5 ms pro Grab. `mss` (GDI) schafft bei 1080p oft nur 30–40 FPS |
| `mss` | Fallback-Capture | plattformübergreifend, keine Abhängigkeiten |
| `opencv-python` | Vision | `inRange`, `connectedComponentsWithStats`, `distanceTransform`: alles in C++ |
| `numpy` | Vektor-Mathematik | Sektor-Profile ohne Python-Schleifen |
| ctypes (stdlib) | Maus/Tastatur | `SetCursorPos`/`SendInput` brauchen Mikrosekunden. `pyautogui` hat eingebaute Pausen (~100 ms) |
| `pynput` | Fallback außerhalb Windows | |
| `pytest` | Tests | synthetische Frames, kein Spiel nötig |

Optional für spätere Ausbaustufen: `onnxruntime-directml` oder `ultralytics` für ein YOLO-Modell (siehe Strategie), `pytesseract` für die Score-Erkennung.

### Starten

1. slither.io im Browser öffnen, **F11** (Vollbild), damit die Canvas-Mitte der Bildschirmmitte entspricht. Ohne Vollbild `capture.crop_top_px` auf die Höhe der Browserleiste setzen.
2. Einmal kalibrieren (siehe unten).
3. ```bash
   python -m slither_bot.main --debug
   ```
4. Ins Spielfenster klicken, Spiel starten, **F8** drücken. **F8** pausiert, **F9** schaltet das Debug-Fenster, **F12** beendet.

Sicherheitsnetz: Verliert das Spielfenster den Fokus, lässt der Bot sofort die Maus los (`runtime.foreground_check`). Das Debug-Fenster sollte auf einem zweiten Monitor oder neben der Spielregion liegen, sonst sieht der Bot sich selbst.

Optionen: `--region L,T,W,H`, `--backend mss`, `--fps 60`, `--auto-respawn`, `--start`, `--no-foreground-check`, `--aggressive`, `--train`, `--use-best`.

Wichtig: Der slither.io-Tab muss der **aktive Tab** seines Fensters sein (Fenstertitel beginnt mit „slither.io“). Am besten ziehst du ihn in ein eigenes Fenster. dxcam nimmt nur den Hauptmonitor auf. Liegt das Spiel auf einem anderen Monitor, nimmt der Bot automatisch das langsamere `mss`.

### Tests

```bash
python -m pytest
```

Die Tests erzeugen synthetische Frames (dunkler Hintergrund, eigene Schlange, Gegner mit Augen, Futter) und prüfen Vision, Ausweichen, Futtersuche, Korridor-Logik und Tracker.

---

## 3. Color Thresholding im Detail

slither.io ist für Thresholding gut geeignet: Der Hintergrund ist ein **dunkles, entsättigtes Blau-Hex-Gitter** (V ≈ 25–60). Alles Relevante leuchtet:

| Objekt | Aussehen | HSV-Regel (OpenCV: H 0–179) |
|---|---|---|
| Hintergrund | dunkelblau, Hex-Muster | V < 90 → ignorieren |
| Futter | kleine, glühende, gesättigte Kugeln | `lit` = S ≥ 70 **und** V ≥ 90, Fläche 2–90 px |
| Große Kugeln | Überreste toter Schlangen | `lit`, Fläche bis 450 px, rund (Füllgrad ≥ 0,55, Seitenverhältnis ≤ 1,6) |
| Schlangen | große, zusammenhängende helle Flächen | `lit` oder weiß (S ≤ 60, V ≥ 200), Fläche > 90 px, nicht rund |
| Eigene Schlange | Komponente unter der Bildmitte | Kamera folgt dem eigenen Kopf |
| Gegnerköpfe | zwei weiße Augen mit schwarzer Pupille | S ≤ 40, V ≥ 215, Fläche 3–120 px, Pupille V ≤ 70 im Augen-Rechteck, genau 1–2 Augen im Umkreis von 25 px |
| Rand / Todeszone | dunkelrote Fläche außerhalb der Karte | H 0–12 oder 168–179, S ≥ 110, V 20–90, Fläche ≥ 1500 px |
| Position in der Welt | Punkt auf der Minimap | hellster Pixel im Minimap-Rechteck → Abstand zur Kartenmitte |

Der Kern aus `vision.py`:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 1) alles, was leuchtet: farbig-gesättigt ODER weiß (weiße Skins, Augen)
lit = cv2.inRange(hsv, (0, 70, 90), (179, 255, 255))
lit |= cv2.inRange(hsv, (0, 0, 200), (179, 60, 255))
lit &= play_mask                               # Leaderboard/Minimap/Score ausblenden
lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, ellipse3)   # gestreifte Skins verbinden

# 2) Komponenten + vektorisierte Klassifikation
n, labels, stats, cent = cv2.connectedComponentsWithStats(lit, connectivity=8)
area = stats[:, cv2.CC_STAT_AREA]
fill = area / (w * h)                          # Füllgrad der Bounding-Box
small_food = (area >= 2) & (area <= 90)
big_food = (area > 90) & (area <= 450) & (fill >= .55) & (aspect <= 1.6)
snake = (area > 90) & ~big_food
snake[self_label] = False                      # durch den eigenen Körper darf man fahren

# 3) Hindernismaske per Lookup-Table: eine Indexoperation statt Schleife
obstacle = np.where(snake, 255, 0).astype(np.uint8)[labels]

# 4) für Distanzabfragen reicht der Umriss: 10-20x weniger Punkte
outline = cv2.subtract(obstacle, cv2.erode(obstacle, k3))
```

Weitere Details:

- **Downscaling auf 50 %** mit `INTER_AREA` mittelt Pixel. Kleines Futter bleibt so sichtbar, und die gesamte Vision läuft auf 960×540 statt 1920×1080 (4× schneller).
- **Eigene Größe:** `distanceTransform` auf der eigenen Maske nahe der Mitte liefert den Kopfradius. Daraus ergeben sich Sicherheitsabstände (Korridorbreite, Gefahrenradius).
- **Eigene Richtung:** Der Schwerpunkt des eigenen Körpers in einem Ring von 2–4,5 Radien um den Kopf ist der „Hals“. Die Fahrtrichtung zeigt vom Hals zum Kopf.
- **Kalibrieren:** Skins, Browser-Farbprofile und Helligkeit variieren. `python tools/calibrate.py` zeigt live die Maske. Mit `1`–`5` wählst du den Bereich, mit `w` speicherst du nach `config.json`, mit `s` speicherst du einen Screenshot für `--image`.

---

## 4. Entscheidungslogik

360° werden in 72 Sektoren à 5° geteilt. Pro Sektor gibt es drei Werte, alles vektorisiert mit numpy:

1. **Clearance:** freie Strecke in diese Richtung. Ein Hindernis im Abstand *d* und *k* Sektoren neben dem Strahl blockiert ihn, wenn `d·sin(kΔθ) < Korridorbreite`. Die Schlange ist breit, ein Strahl allein reicht also nicht.
2. **Danger:** `((R − clearance)/R)²`. Dazu kommen **vorhergesagte Kopfpositionen** der Gegner (Tracker-Geschwindigkeit × 0–0,6 s, aufgeblasen) und eine Randstrafe aus der Minimap.
3. **Food:** Gewicht · e^(−d/180), nur Futter, das *vor* dem nächsten Hindernis liegt, geglättet über Nachbarsektoren (Gauß). So entsteht eine Dichtekarte statt „nächstes Kügelchen“.

```
score = 1.0·food − 4.0·danger − 0.35·Drehwinkel − 0.15·Richtungswechsel
Sektoren mit clearance < 35 px sind verboten.
```

| Prio | Modus | Auslöser | Verhalten |
|---|---|---|---|
| 1 | `ESCAPE` | Gefahr voraus ≥ 0,55, Kopf nähert sich (< 170 px, > 40 px/s) oder alles blockiert | sicherste Richtung, Boost nur geradeaus mit viel Platz |
| 2 | `FORAGE` | Standard | höchste Futterdichte, ohne Futter Drift zur Kartenmitte |
| 2 | `FEAST` | ≥ 250 Futtermasse in < 220 px, sicher, fast geradeaus | Boost, um Überreste vor anderen zu holen |
| 3 | `CUTOFF` | Gegnerkopf seitlich, nicht dicker als wir, wir erreichen mit Boost einen Punkt vor ihm schneller als er | Boost zum Abfangpunkt, danach Kurs halten: der eigene Körper wird zur Wand |
| 3 | `ENCIRCLE` | wir sind groß (Fläche ≥ 6000), Beute ≤ 0,6× unser Radius in < 260 px | Kreisbahn um den Kopf mit schrumpfendem Radius |
| 3 | `HUNT` | nur im aggressiven Profil: kleinere Schlange in < 420 px | parallel neben ihrem Weg mitfahren, bis ein `CUTOFF` möglich ist |
| 1½ | `LOOT` | eine Schlange stirbt im Bild (eigener Kill oder fremder) **oder** ein großer Futterhaufen | sofort hin, mit Boost bei > 120 px; Vorrang vor allen Taktiken, nur Überleben ist wichtiger |

### Loot, Pits und Minimap

- **Tod erkennen:** Ein stabil verfolgter Kopf verschwindet mitten im Bild, nicht am Rand. Bestätigt wird der Tod erst, wenn dort innerhalb 1 s Überreste auftauchen (große Kugeln oder augenlose Blobs). Verschwindet ein Kopf ohne Überreste, passiert nichts.
- **Überreste statt Wand:** Runde, augenlose Blobs gelten als Futter. Ein Blob mit zwei Augen ist immer eine Schlange. Mehrere verschmolzene Kugeln sehen aus wie ein Körper ohne Kopf. In einer bestätigten Loot-Zone werden solche augenlosen Blobs für 8 s zu Futter statt zu Hindernissen.
- **Mitbewegung:** Die Zone wird mit der eigenen Bewegung mitgeführt (die Kamera folgt uns) und jeden Frame auf die tatsächlich sichtbaren Überreste nachzentriert.
- **Pits:** Ein Raster über das ganze Bild sucht den dichtesten Futterhaufen. Er zählt nur, wenn er mindestens 900 Masse hat **und** 6× dichter ist als der Bilddurchschnitt. Ein echtes Bild ohne Kills kam auf etwa 4×. Ist eine Gegnerschlange deutlich näher dran, lässt der Bot ihn aus.
- **Minimap = Bevölkerungskarte:** Das habe ich direkt aus den Canvas-Elementen der Seite ausgelesen. slither.io zeichnet dort weiße Pixel mit ~42 % Deckkraft, wo Schlangen sind, und die eigene Position als rein weißen Punkt. `minimap.py` sucht auf der Vollauflösung das nächstgelegene dichte Gebiet. Beim Futtersuchen zieht `w_crowd` den Bot dorthin, denn dort gibt es Kills und Loot. Im Debug-Fenster zeigt der orange Pfeil die Richtung, die magenta Kreise sind Loot-Zonen.

Jede Taktik prüft in jedem Frame, ob ihre Richtung sicher ist, und bricht sonst sofort ab. Überleben hat immer Vorrang.

---

## 4b. Trainingsmodus: aggressiv spielen und sich selbst verbessern

```bash
train.bat
```

`train.bat` startet `python -m slither_bot.main --train --start` und startet den Bot neu, falls er abstürzt. F12 beendet ihn endgültig.

So läuft es ab:

1. **Aggressiver Start.** Das Profil `AGGRESSIVE` in `config.py` senkt die Angst (`w_danger`, Gefahrenradius, Sicherheitsabstand), boostet früher und schaltet `HUNT` ein. Außerdem greift der Bot auch etwas dickere Schlangen an (`cutoff_size_ratio` 1,6).
2. **Automatischer Neustart.** Der Bot erkennt das Menü am grünen „Spielen“/„Play Again“-Button unter dem lila Namensfeld (Farben aus dem CSS der Seite) und klickt ihn.
3. **Echte Punktzahl per OCR.** Die eingebaute Windows-Texterkennung (`winocr`, Sprache `de`) liest während der Runde „Deine Länge“ und „Dein rang: X von Y“. Nach dem Tod liest sie „Deine endgültige Länge war N“. Score einer Runde = log(Länge).
4. **Lernen zwischen den Runden.** Der Tuner (`tuner.py`) ist eine rauschfeste (1+1)-Evolutionsstrategie über 23 Parameter (Angst, Gier, Boost, Cutoff, Einkreisen, Jagen, Loot, Crowd):
   - Der *Incumbent* ist der beste Parametersatz bisher. Ein *Kandidat* ist eine Mutation davon und spielt 3 Runden.
   - Der Kandidat ersetzt den Incumbent nur, wenn sein mittlerer Score besser ist. Eine katastrophale erste Runde (Länge < 35 % des Medians) verwirft ihn sofort.
   - Jede 4. Runde spielt wieder der Incumbent. So kann ein Glückstreffer nicht ewig oben bleiben.
   - Die Schrittweite passt sich an: Sie wächst bei Erfolg und schrumpft bei Misserfolg (1/5-Regel).
5. **Gedächtnis.** `tuning/state.json` (Lernstand), `tuning/episodes.csv` (jede Runde mit Länge, Rang, Dauer, Zeit pro Modus und Parametern) und `tuning/best_params.json`. Beim nächsten Start lernt der Bot dort weiter, wo er aufgehört hat.

Mit den besten gefundenen Parametern ohne weiteres Lernen spielen:

```bash
python -m slither_bot.main --use-best
```

Konsolenausgabe pro Runde, zum Beispiel:

```
[train] round 17 (candidate) length=1843 rank=4/512 time=9.3min score=7.52 | incumbent mean=6.91 sigma=0.162 best length=2410 best rank=2 candidate 2/3
```

Ehrliche Erwartung: Eine Runde dauert Minuten, das sind etwa 10–20 Runden pro Stunde. Spürbare Verbesserungen brauchen einige Dutzend Runden, also einen Tag Laufzeit ist realistisch für ein gut getuntes Profil. Platz 1 hängt stark vom Server ab (wie viele große Spieler gerade online sind). In `episodes.csv` siehst du, ob die Kurve steigt.

---

## 5. Strategie: Heuristik oder Reinforcement Learning?

**Empfehlung: zuerst eine regelbasierte Heuristik (wie hier), später optional hybrid. Kein reines End-to-End-DQN auf dem Live-Server.**

Begründung für das Ziel „Platz 1“:

1. **Platz 1 ist vor allem ein Überlebensproblem.** Die Rangliste sortiert nach Länge. Lang wird man in slither.io weniger durch geschickte Kills als dadurch, dass man 30–90 Minuten nicht stirbt. Ein einziger Fehler kostet alles. Eine Heuristik mit harten Constraints („verbotene Sektoren“) ist vorhersagbar und debugbar. Ein DQN hat immer eine Restwahrscheinlichkeit für absurde Aktionen, und die summiert sich über Zehntausende Entscheidungen pro Runde.
2. **Sample-Effizienz.** DQN/PPO auf Pixeln braucht typischerweise 10⁷–10⁸ Frames. Live sind das bei 30 FPS Monate Spielzeit. Das Spiel lässt sich nicht beschleunigen oder parallelisieren, ohne viele Accounts und Server zu belasten.
3. **Sparse und verzögerte Belohnung.** Der Tod kommt selten und spät, und der Fehler lag oft Sekunden davor. Credit Assignment ist schwer. Man müsste die Belohnung stark formen (Länge, Überlebenszeit, Abstand), und dann lernt das Netz im Wesentlichen die Heuristik nach.
4. **Nicht-Stationarität.** Gegner sind echte Menschen und andere Bots, die Serverlast schwankt. Ohne Simulator gibt es keine reproduzierbaren Episoden.
5. **Latenz und Robustheit.** Die Heuristik braucht < 1 ms und läuft auf jeder CPU. Jeder Parameter hat eine physikalische Bedeutung (Pixel, px/s, Radiant) und lässt sich gezielt nachjustieren.

**Wo ML sinnvoll ist:**

- **Wahrnehmung:** Ein kleines YOLO-Modell (YOLOv8n/v11n, ONNX, ~3–5 ms auf GPU) für „Kopf/Körper/Futter“ ist robuster gegen exotische Skins als HSV-Schwellen. Trainingsdaten liefert der heutige HSV-Detektor als Auto-Labeling plus manuelle Korrektur.
- **Parameter-Tuning:** Die ~40 Gewichte und Schwellen der Heuristik per Bayes-Optimierung oder CMA-ES auf „Länge nach 10 min“ optimieren. Das braucht hunderte statt Millionen Episoden.
- **RL nur mit Simulator:** Wer RL will, baut eine eigene, schnelle 2D-Nachbildung (Schlangenphysik, Futter, einfache Gegner-Bots), trainiert dort PPO (stabiler als DQN bei kontinuierlichem Lenkwinkel) auf dem gleichen **Feature-Vektor** (72 Sektoren × clearance/danger/food statt Pixel) und nutzt die Heuristik als Sicherheits-Fallback („Shielded RL“). Das ist die Ausbaustufe, die realistisch über die Heuristik hinauskommt.

Ehrlich gesagt: Pixelbasierte Bots kommen auf vollen Servern selten dauerhaft auf Platz 1. Die stärksten bekannten slither.io-Bots waren Userscripts, die den Spielzustand direkt aus JavaScript lesen (exakte Positionen aller Segmente) und dann ebenfalls heuristisch entscheiden. Der CV-Ansatz hier ist ungenauer, dafür spielunabhängig und lehrreich. Den größten Hebel haben eine gute Kalibrierung und konservative Sicherheitsabstände.

---

## 6. Tuning-Tipps

- Stirbt der Bot durch Kopf-an-Körper-Kollisionen: `brain.clearance_factor`, `danger_radius` oder `w_danger` erhöhen.
- Er zittert: `w_switch` erhöhen, `control.max_angular_speed` senken.
- Er frisst zu zögerlich: `w_food` erhöhen, `food_dist_scale` vergrößern.
- Futter wird als Schlange erkannt oder umgekehrt: `vision.food_max_area` und `big_food_*` im Debug-Fenster prüfen.
- Die Minimap-Position stimmt nicht: `vision.minimap_rect` an deine Auflösung anpassen (Debug-Fenster zeigt `border=`).
- Auto-Respawn findet den Button selbst. Nur wenn das scheitert, klickt er nach 8 s auf `runtime.respawn_click_rel`.
- OCR liest nichts: Prüfen, ob `hud.stats_rect` zur Position von „Deine Länge“ passt (unten links, gemessen auf 1920×1080). Bei englischer Windows-OCR `hud.lang` auf `en` setzen.

## Roadmap

- [ ] YOLO-Detektor als Alternative zu HSV (`vision_yolo.py`)
- [ ] Hintergrund-Phasenkorrelation (`cv2.phaseCorrelate`) für die eigene Geschwindigkeit statt fester `base_speed_px_s`
- [x] Score per OCR, Logging der Rundenergebnisse für automatisches Parameter-Tuning
- [ ] 2D-Simulator + PPO mit Heuristik-Shield
