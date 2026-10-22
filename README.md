# Piper Plus — 공명 방식으로 만든 한국어 낭독 음성 모델

허깅페이스에도 같은 파일이 있습니다: **https://huggingface.co/watchpen/piper-plus**

> **한국어 낭독 음성 모델입니다.** 파이퍼(VITS) 계열 다화자 모델을 ONNX 로 내보냈고(22,050Hz, CPU 로 돕니다),
> **공명 방식**으로 만들었습니다. 글자·성우·감정을 칸마다 외우지 않고, 녹음에서 잰 주파수로만 받습니다 —
> 자모는 공명·터짐·숨·울림 주파수로, 성우는 목소리 주파수 31가지로, 감정은 음질 값 15가지와 공명 노브 3칸으로.
> 같이 든 작은 감정 모델(KoELECTRA-small, int8 ONNX)이 글 한 줄을 읽고 감정을 정하면, 그 감정이 위 값으로 바뀌어
> 들어갑니다. 아래에 표·입력 꼴·예제가 있습니다.

## 기존 파이퍼보다 나아간 점

| | 기존 Piper | **Piper Plus** |
|---|---|---|
| **출력 디코더(보코더)** | HiFi-GAN 생성기 | **4대역 iSTFT 디코더**(PQMF 부대역 4개 · n_fft 16 · hop 4) — 파형을 대역별 짧은 역푸리에 변환으로 바로 만들어 가볍고 빨라, 폰 CPU 에서 실시간으로 돕니다 |
| **한국어 입력** | eSpeak 음소 변환(한국어 발음 규칙이 부정확) | **한글 자모를 그대로**(NFD) — eSpeak 없음. 자모마다 녹음에서 잰 공명 주파수 80칸으로 넣어, 처음 보는 낱말도 소리 부품으로 짜서 읽습니다 |
| **성우** | 화자 번호마다 외운 임베딩 | **목소리 주파수 31가지**(음높이·공명·성도·밝기 …)로 — 다른 목소리를 재서 넣을 수 있습니다 |
| **감정** | 없음 | **감정 공명 입력 `res`**(음질 값 15 + 공명 노브 3) — 감정과 세기를 읽을 때 정하고, 같이 든 한국어 감정 모델이 줄마다 감정을 골라 줍니다 |
| **학습 판별기** | MPD · MSD | MPD · MSD 에 더해 **MRD**(다해상도) · **WavLM 판별기**(말소리 자연스러움) · **세기판별기**(글자에 안 적힌 된소리·거센소리를 소리에서 가려냄) |

## 공명 방식이란

보통의 음성 모델은 글자 하나, 화자 하나, 감정 하나마다 칸을 두고 그 칸의 값을 **외웁니다**. 이
모델은 그렇게 하지 않고, **소리에서 잰 주파수**만 입력으로 받습니다. 글자·화자·감정은 이름이 아니라
"어떤 주파수로 나는 소리인가"로만 모델에 들어갑니다.

| 갈래 | 무엇을 재서 넣나 | 칸 수 | 표 |
|---|---|---|---|
| **발음 공명** | 자모(초성·중성·종성)마다 공명(F1·F2·F3), 터짐, 숨, 울림, 받침 콧소리 등 | 80 | `resonance/jamo_features.json` |
| **성우 공명** | 성우마다 음높이·공명·성도·저역·배우 포만트·밝기·잡음 결·MFCC 껍질 | 31 | `resonance/speaker_features.json` (모델 안에도 들어 있음) |
| **감정 공명** | 감정마다 음높이·크기·jitter·shimmer·HNR·H1H2·F1~F3·B1~B3·alpha·Hammarberg·콧소리 섞임 | 15 + 노브 3 | `emotion/emotion_resonance.json` |

- **발음**은 훈민정음의 제자 원리를 따릅니다. 모음은 입안 공명(혀 높이 F1 · 혀 앞뒤 F2 · 입술
  둥글기), 자음은 소리 내는 자리(입술·혀끝·센입천장·여린입천장·목청)·방식(터짐·갈림·붙갈림·코울림·
  흐름)·세기(예사·된·거센)로 나뉘고, 그 하나하나를 녹음에서 잰 주파수로 적습니다. 겹모음은 출발
  공명 → 도착 공명으로 한 칸 안에 있고, 겹받침은 두 받침의 합입니다. 그래서 처음 보는 낱말도
  아는 소리 부품으로 짜서 읽습니다. 글 인코더에는 발음 규칙(연음·비음화 등)이 한 줄도 들어 있지
  않습니다 — 규칙 발음은 학습 자료가 가르치고, 글자로 답이 안 나오는 불규칙 발음은 사전 몫입니다.
- **성우**는 번호가 아니라 그 목소리의 주파수 31가지로 들어갑니다(`sid` 는 표에서 그 줄을 꺼내는
  열쇠일 뿐입니다). 표준편차로 나눈 z 값이라 다른 목소리를 재서 넣는 것도 됩니다.
- **감정**은 같은 목소리의 무감정 녹음 대비 얼마나 달라지는지(z)를 15칸으로 적고, 여기에 공명
  노브 3칸(성도 척도 · 저역/흉성 · 배우 포만트)을 더합니다. 말투 태그 번호를 외우게 하지 않았습니다.
- 이 모델은 자모·성우·감정의 기억 칸을 비우고 위 세 가지 주파수만으로 학습한 판입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `piper-plus.onnx` | 낭독 모델 (에폭 790, 85.6MB) |
| `piper-plus.onnx.json` | 모델 설정 — 표본율, 글자·태그 번호표(`phoneme_id_map`), 화자 번호표(`speaker_id_map`), 기본 읽기 값 |
| `resonance/jamo_features.json` | 자모 성질표(104칸 이름, 자모·태그 칸마다 값) |
| `resonance/speaker_features.json` | 성우 주파수 31가지(이름, 평균·표준편차, 성우마다 z) |
| `emotion/emotion_resonance.json` | 감정 이름 → `res` 입력 값(노브 3 + 감정 꼬리 15) |
| `emotion/emotion_int8.onnx` | 감정 모델 (KoELECTRA-small v3 기반, int8, 14.6MB) |
| `emotion/vocab.txt` | 감정 모델 토크나이저 어휘 (WordPiece) |
| `emotion/emotion_labels.json` · `emotion_labels_full.json` | 머리마다 라벨 이름과 차례 |
| `emotion/emotion_config.json` | 감정 모델 설정 (`max_len` 192 등) |
| `emotion/emotion_tokens.json` | 감정 이름·말투 낱말 → 번호 (옛 판과 번호를 맞추는 표) |

## 낭독 모델 쓰는 법

### 입력

| 이름 | 모양 | 뜻 |
|---|---|---|
| `input` | `[1, n]` int64 | 번호열 (아래) |
| `input_lengths` | `[1]` int64 | `n` |
| `scales` | `[3]` float32 | `[noise_scale, length_scale, noise_w]` — 기본 `[0.667, 1.0, 0.8]`. `length_scale` 이 크면 느리게 |
| `sid` | `[1]` int64 | 화자 번호 (`speaker_id_map`) — 모델 안 성우 주파수표의 줄 번호 |
| `res` | `[1, 34]` float32 | 공명 입력. `[0:3]` 노브, `[19:34]` 감정 꼬리, `[3:19]` 는 안 씀. 모두 0 이면 무감정 기본 목소리 |

출력 `output` 은 22,050Hz 소리(−1~1), `durations` 는 번호마다 길이입니다.

### 번호열 만들기

1. 글을 유니코드 **NFD** 로 풀면 한글 음절이 초성·중성·종성 자모로 나뉩니다.
2. `^`, `_` 로 시작하고, 글자마다 그 번호 뒤에 `_` 를 붙이고, `$` 로 끝냅니다(Piper 와 같은 꼴).
   번호표에 없는 글자는 뺍니다.
3. 번호표에는 말투 태그 번호(75~255, 319~383, 416~543)도 있지만, **이 판은 태그 번호를 모두 빈
   칸으로 보내고 길이 0 으로 지웁니다.** 말투는 번호가 아니라 `res` 로 넣습니다(아래).

### 감정·말투 넣기

`emotion/emotion_resonance.json` 에서 감정 이름(기쁨·비탄·분노·공포·냉소·속삭임·죄책감·고통·환희·
놀람·부끄러움·장난·짜증·그리움·불안·애정·안도·감탄·수줍음)으로 번호를 찾고, 그 번호의 노브 3값과
꼬리 15값을 `res` 에 넣습니다. 값에 1보다 작은 수를 곱하면 약하게, 큰 수를 곱하면 세게 표현합니다.
말투 낱말(`낱말` 표, 334개)도 같은 꼴로 씁니다.

### 예제 (파이썬)

```python
import json, unicodedata, wave
import numpy as np, onnxruntime as ort

cfg = json.load(open("piper-plus.onnx.json", encoding="utf-8"))
emo = json.load(open("emotion/emotion_resonance.json", encoding="utf-8"))
ids_of = cfg["phoneme_id_map"]

def to_ids(text):
    ids = ids_of["^"] + ids_of["_"]
    for ch in unicodedata.normalize("NFD", text):
        if ch in ids_of:
            ids += ids_of[ch] + ids_of["_"]
    return ids + ids_of["$"]

def res_for(emotion=None, strength=1.0):
    r = np.zeros((1, emo["res"]["길이"]), dtype=np.float32)
    if emotion:
        key = str(emo["감정"][emotion])          # 무감정(250)은 표에 없음 → 0
        knob, tail = emo["res"]["노브"], emo["res"]["감정꼬리"]
        r[0, knob["자리"]:knob["자리"] + knob["칸"]] = np.array(emo["노브표"].get(key, [0.0] * knob["칸"])) * knob["배수"] * strength
        r[0, tail["자리"]:tail["자리"] + tail["칸"]] = np.array(emo["꼬리표"].get(key, [0.0] * tail["칸"])) * tail["배수"] * strength
    return r

sess = ort.InferenceSession("piper-plus.onnx", providers=["CPUExecutionProvider"])

def speak(text, speaker="nangdok", emotion=None, strength=1.0):
    x = np.array([to_ids(text)], dtype=np.int64)
    return sess.run(None, {
        "input": x,
        "input_lengths": np.array([x.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        "sid": np.array([cfg["speaker_id_map"][speaker]], dtype=np.int64),
        "res": res_for(emotion, strength),
    })[0].squeeze()

audio = speak("안녕하세요. 오늘은 날씨가 참 좋네요.")                 # 무감정
angry = speak("당장 여기서 나가!", speaker="seungwoo", emotion="분노")  # 분노

with wave.open("out.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(cfg["audio"]["sample_rate"])
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
```

숫자·영어·문장부호는 번호표에 있는 것만 읽습니다. 숫자는 한글로 풀어 넣으면 더 자연스럽습니다.

### 화자

`speaker_id_map` 에 26칸이 있습니다. 낭독(해설)은 `nangdok`(1), 성우 목소리는 `seungwoo`(0)·
`joyjjang`(2)·`seungchul`(3)·`소라`(6)·`지우`(23) 등입니다. 나머지 칸(대사체·감탄사·의성어 등)은
학습할 때 갈래를 나눈 칸입니다. 각 칸이 어떤 주파수로 들어가는지는 `resonance/speaker_features.json`
에 있습니다.

## 감정 모델 쓰는 법

- 토크나이저: `emotion/vocab.txt` 의 WordPiece(KoELECTRA 와 같음, 대소문자 그대로).
  `[CLS] 문장 [SEP]`, 최대 192 토큰. 앞뒤 맥락을 줄 때는 `tokenizer(맥락글, 문장, truncation="only_first",
  max_length=192)` 꼴(맥락글 예: `"앞: 비탄. 뒤: 분노."`).
- 입력: `input_ids`, `attention_mask`, `token_type_ids` — 모두 `[b, t]` int64.
- 출력 일곱 머리 (이름과 라벨 차례는 `emotion_labels_full.json`):
  - 문장 머리 `[b, C]`: `감정`(16) · `크기`(3) · `빠르기`(3) · `숨결`(4) · `말투`(46)
  - 토큰 머리 `[b, t, C]`: `끼울태그`(27) · `구간`(11)
- `감정` 머리에서 가장 높은 라벨(예 "분노")을 그대로 위 `res_for("분노")` 에 넣으면 됩니다.
  `말투` 머리의 낱말은 `emotion_resonance.json` 의 `낱말` 표로 번호를 찾아 같은 길로 씁니다.

## 알아 둘 것

- 성인 말투 갈래(`교성`·`섹슈얼` 칸)가 들어 있어 `not-for-all-audiences` 로 표시했습니다.
- 실제 사람 목소리를 흉내 내거나 속이는 데 쓰지 마세요.
- 이용 조건(라이선스)은 정해지는 대로 이 카드에 적습니다. 그 전에 다른 곳에 다시 배포하거나 상업적으로
  쓰려면 저장소 주인에게 먼저 물어 주세요. 감정 모델의 바탕 모델은
  [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator) 입니다.

## English summary

A Korean multi-speaker text-to-speech model in the Piper (VITS) family,
exported to ONNX (22,050 Hz, CPU-friendly), built on a **resonance method**: the model
memorises no letters, voices or emotions as learned lookup entries. Every jamo (consonant /
vowel part of a Hangul syllable) is described by formant, burst, aspiration and voicing
frequencies measured from recordings; every speaker by 31 measured voice frequencies; every
emotion by 15 measured voice-quality values plus 3 resonance knobs. A small Korean text
classifier (KoELECTRA-small, int8 ONNX) reads a line and names its emotion, which maps to those
values. Unlike stock Piper, the output decoder is a 4-band iSTFT (PQMF) instead of HiFi-GAN, light enough for real-time synthesis on a phone CPU; input is Hangul jamo directly (no eSpeak); and training adds MRD, WavLM and a Korean tenseness discriminator. The Korean sections above give the tables, the input format and examples.
