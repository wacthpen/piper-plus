# Piper Plus — 된소리 과잉 일반화를 억제하고 신경공명으로 감정을 표현하는 한글 자모 음파 발성 모델
### *Piper Plus: Korean Neural-Resonance TTS with Fortis-Overgeneralization Suppression, Emotion-Expressive Neural Resonance & Acoustic Jamo Vocalization*

허깅페이스에도 같은 파일이 있습니다: **https://huggingface.co/watchpen/piper-plus**

> 📢 **[2026-10-24 모델 가중치 전면 교체 안내]**
> 어절경계를 제거하고 된소리 과잉 일반화를 억제한 최신 **Epoch 880 EMA (`piper-plus.onnx`)** 모델로 교체되었습니다. 이전에 다운로드하신 분들은 최신 모델로 다시 받아주시기 바랍니다.
>
> 📢 **[2026-10-24 Model Weights Updated]**
> Replaced with the latest **Epoch 880 EMA (`piper-plus.onnx`)** model (word-boundary `eojeol_boundary` CumSum completely removed & excessive fortis/tensification generalization structurally suppressed). If you downloaded an earlier version, please re-download `piper-plus.onnx`.
>
> ⚠️ **[저작자 표시 및 깃허브 주소 첨부 의무 조건 (Mandatory Attribution Requirement)]**
> 본 모델(`piper-plus`), 신경공명(발음·성우·감정 공명 주파수 표), 자모 음파 발성 구조, 제로듀레이션 태그 체계, 또는 포함 코드/가중치를 **응용하거나 수정·파생·재개발할 경우, 반드시 원저작자 이니셜(`watchpen`)과 공식 깃허브 주소(https://github.com/wacthpen/piper-plus)를 함께 첨부(명시)**해야 합니다.
> *If you adapt, modify, fine-tune, or redevelop this model, its neural resonance tables, or its codebase, you MUST explicitly credit and include the original author's handle/initials (`watchpen` / GitHub: `wacthpen`, Hugging Face: `watchpen`) AND attach the official GitHub repository URL (`https://github.com/wacthpen/piper-plus`) in your documentation, model card, and derivative works.*

---

## 🌟 Piper Plus 핵심 정체성 (Core Identities)

1. **된소리(경음화) 일반화 과잉 억제 *(Suppression of Excessive Fortis / Tensification Generalization)***
   - 기존 한국어 TTS와 불완전한 외부 G2P가 만연하게 일으키던 고질병인 **'예사소리의 과도한 된소리(경음화) 일반화'**를 원천 억제한 모델입니다.
   - 신경망의 문맥·임베딩 기억층(Memory Layer)을 그대로 통과시키면 학습 데이터 속 경음화 현상이 과잉 일반화되어 멀쩡한 예사소리(평음 `ㄱ, ㄷ, ㅂ, ㅅ, ㅈ`)까지 된소리(`ㄲ, ㄸ, ㅃ, ㅆ, ㅉ`)로 오염됩니다. **Piper Plus는 출력·텍스트 인코더에서 '된소리화된 기억층(`self.emb`)'을 통과하지 않도록(100% 공명 주파수 대체 모드로 우회) 설계**하고 **예사소리 24배 보호 세기판별기**를 결합해 신경망이 임의로 된소리를 만들어내는 현상을 구조적으로 완전히 피해갔으며, 실제로 소리가 바뀌어야 하는 **불규칙 발음은 발음사전(Dictionary)으로 1:1 정밀 대응**합니다.
2. **감정 표현이 가능한 신경공명 & 문맥 감정 모델 자동 매핑 *(Emotion-Expressive Neural Resonance & Context-Aware Auto Emotion Mapping)***
   - 단순한 평면적 텍스트 낭독을 넘어, 32대 감정·말투 체계와 **5대 발성 물리 축(공명 균형, 성문 긴장도, 호흡기식, 피치·포먼트 이동, 발화 속도)**이 결합된 **'신경공명(Neural Resonance)'** 아키텍처로 살아 숨쉬는 감정 연기를 구현합니다.
   - 특히 자연스러운 감정 표현을 유도하기 위해 **전용 한국어 문맥 감정 분류 모델(`emotion/emotion_int8.onnx`)을 직접 학습**하여 탑재했습니다. 사용자가 일일이 감정 태그를 수동으로 적지 않아도, 감정 모델이 **문장의 앞뒤 문맥(Context)을 읽고 자동으로 감정 태그·말투·감정 세기·발화 속도 및 감정 공명 주파수(`res`)를 매핑**합니다.
   - 또한 대본에 삽입된 감정·발성·소리 태그는 **`_태그칸떼기` (Zero-Duration Scatter Compaction)**를 통해 **프레임 수 0(제로듀레이션, Zero-Duration)**으로 처리되어, **태그 자체가 글자나 군소리로 발음되는 일 없이 오직 순수하게 감정과 발성 효과만** 본문 음성에 입혀집니다.
3. **자모 학습을 통한 음파 글자 발성 *(Acoustic Character Vocalization via Direct Jamo Learning)***
   - **"『훈민정음 해례본』에 따라 음파(발성 주파수)를 만들어서 초·중·종성을 발성하니 한국어 발음이 그대로 나온다."**
   - 영어식 IPA 음소 변환(eSpeak 등)이나 왜곡된 음운 규칙을 거치지 않고, **『훈민정음 해례본』의 제자 원리에 따라 한글 초·중·종성 자모(NFD) 자체를 음파(104차원 고유 공명 주파수)와 1:1로 직접 학습**시켜 글자 그대로의 맑고 순수한 음파 발성을 구현했습니다.
   - *(When we construct the acoustic soundwaves / vocal resonance frequencies according to the Hunminjeongeum Haeryebon and vocalize the initial, medial, and final Jamo, authentic Korean pronunciation emerges just as written.)*
4. **초경량 구동 & 발열이 거의 없음 *(Ultra-Light & Near-Zero Heat on Mobile / CPU)***
   - 무거운 거대 신경망 기억층이나 풀밴드 HiFi-GAN 보코더 대신, **자모·성우·감정 주파수의 단일 선형 투영(`emb_f`, `emb_g_res`, `emb_res`) + 4-부대역 MB-iSTFT-VITS(PQMF) 합성 구조**로 작동합니다. 연산량이 극도로 가벼워 스마트폰(모바일)이나 일반 노트북 CPU에서도 **발열이 거의 없이 가볍고 빠르게 실시간 구동**됩니다.

---

## 기존 파이퍼(Piper)보다 나아간 점

| 구분 | 기존 Piper | **Piper Plus (Epoch 880 EMA)** |
|---|---|---|
| **된소리(경음화) 제어** | 문맥 기억층이 경음화를 과잉 일반화하여 평음이 된소리로 무너짐 | **출력 인코더 기억층(`self.emb`) 100% 미통과(대체) + 세기판별기(예사소리 보호 24배) + 흉성 가중치 억제(`0,2,4`)**로 예사소리 보존, **불규칙 발음은 사전으로 1:1 대응** |
| **한국어 입력 및 발성** | eSpeak IPA 음소 변환 (한국어 발음 규칙 부정확·음소 왜곡) | **한글 NFD 자모 직결 음파 발성** —『훈민정음 해례본』 제자 원리를 따른 104차원 성도 공명 주파수 직접 입력 (글자 기억 칸 0) |
| **외국어·영어 확장** | 미지원 또는 어색한 알파벳 철자 읽기 | **옛모음·옛자음(`ᄫ, ᅀ, ᄽ, ᅗ, ᄼ, ㆍ, ㆎ` 등) 공명 주파수**를 통한 원어급 마찰음/모음 발성 + **미학습 영단어 한국식 발음(콩글리쉬/알파벳 음차)** 자동 변환 |
| **출력 디코더 & 발열** | 무거운 HiFi-GAN 생성기 (모바일 발열·지연 발생) | **단일 선형 공명 투영 + 4대역 MB-iSTFT 디코더**(PQMF 4부대역 · `n_fft` 16 · `hop` 4) — **스마트폰·CPU에서 발열이 거의 없이 초경량 실시간 구동** |
| **성우 표현** | 화자 ID마다 외운 고정 임베딩(`emb_g`) | **성우 공명 주파수 31가지**(`spk_feat 28×31 → Linear 31→512`) — 음높이·성도·포만트·음색 껍질 z 점수로 목소리 생성 |
| **감정 표현 & 태그** | 없음 (단조로운 낭독) | **전용 문맥 감정 모델(`emotion_int8.onnx`) 자동 매핑** + **신경공명 `res`(노브 3 + 꼬리 15)** + **감정 태그 제로듀레이션(길이 0, 소리 없이 감정만 주입)** |
| **호흡 및 끊어 읽기** | 어절마다 기계적 단절 또는 긴 문장 통째 합성 시 억양 붕괴 | **어절 경계(`eojeol_boundary`) 완전 제거** + **표준 정속 `0.85` & 감정 빠르기(`1.08/1.0/0.93`)** + **종결형·연결형 어미 갈라읽기(`0.01초` 이음)** |

---

## 🏛️ 핵심 기술 아키텍처 (Core Technical Architecture)

### 1. 3대 완전공명 아키텍처 (발음공명 · 성우공명 · 감정공명) & 『훈민정음 해례본』 주파수 매핑
> **"『훈민정음 해례본』에 따라 음파(발성 주파수)를 만들어서 초·중·종성을 발성하니 한국어 발음이 그대로 나온다."**
> *"훈민정음 해례본에 본떠 발성 주파수 맞춤 — 훈민정음 해례본에 발하는 발성법을 주파수에 따서 맞춤"*
> *(When we construct the acoustic soundwaves / vocal resonance frequencies according to the Hunminjeongeum Haeryebon and vocalize the initial, medial, and final Jamo, authentic Korean pronunciation emerges just as written.)*

보통의 음성 합성 모델은 글자 하나, 화자 하나, 감정 하나마다 임베딩 칸을 두고 그 값을 단순 암기합니다. 반면 **Piper Plus (`완전공명판` Epoch 880 EMA)**는 글자·성우·감정의 개별 기억 칸을 모두 비우고, 『훈민정음 해례본(訓民正音 解例本)』이 밝힌 발성 기관의 과학적 제자 원리를 현대 음향물리학의 **성도(Vocal Tract) 공명 주파수**로 그대로 옮긴 **3대 완전공명 아키텍처**로 설계되었습니다.

1. **발음공명 (`jamo_feat` `544×104` → `emb_f` `Linear(104 → 192, bias=False)`) — 0/1 기호 칸을 모두 비운 순수 실측 주파수**:
   - **`학습주파수표만들기.py` · `해례본받침.py` 설계 원리**: `표_바탕.json`(80칸)에서 `글자·자리·방식` 같은 인위적인 0/1 기호 표시 칸은 모두 `0`으로 비우고, 오직 **녹음에서 실측한 물리적 주파수 31칸(및 받침-초성 연음 넘어감 주파수)**으로만 초·중·종성 음파를 구성합니다(`표_바탕.json` 80칸 + `표_태그.json` 소리·발성·수식 태그 주파수 24칸 = **총 104칸**, `resonance/jamo_features.json`).
   - **초성 실측 주파수 (16칸) & 해례본 `「ㅇ。喉音。如欲字初發聲」`**:
     해례본 오음(아·설·순·치·후)과 가획 원리에 따른 **초성 16칸(`초성터짐kHz`, `초성숨` ms, `초성세기` dB, `초성F2/F3휨`, `초성울림kHz` F1, `초성뒤음높이`, 터짐·숨소리의 `봉우리kHz·퍼짐kHz·저중비·중고비`)**을 실측해 넣었습니다. 특히 초성 `ㅇ`도 0 줄로 비우지 않고 해례본 ***"ㅇ은 후음(목구멍소리)이니 목청에서 소리를 여는 첫소리(喉音 如欲字初發聲)"***에 따라 실측 주파수를 입력하여, `ㅇ`이 기억 칸(`self.emb`)으로 빠지지 않고 100% 공명 주파수로 발성되게 했습니다.
   - **중성(모음) 실측 주파수 (7칸) — 천·지·인(`ㆍ, ㅡ, ㅣ`) 합용과 포먼트 궤적**:
     입의 개구도(F1), 혀의 전후 위치(F2), 원순성·연구개(F3)를 따라 **모음 시작(10% 지점 `공명F1·F2·F3출발`)과 끝(75% 지점 `공명F1·F2·F3도착`) 주파수 및 `모음길이` 7칸**을 실측해 넣었습니다. 이중모음은 한 칸 안에 출발→도착 주파수 궤적으로 표현됩니다.
   - **종성(받침) 실측 주파수 (8칸 + 연음 넘어감) — 해례본 `「終聲復用初聲」` · `「八終聲可足用」` · 합자해 `「自左而右」`**:
     해례본 종성해의 ***"종성은 초성을 다시 쓴다(終聲復用初聲)"***와 ***"팔종성으로 족히 쓴다(八終聲可足用)"***에 따라 초성 짝 자리 로커스(`받침F2휨 z2짝`, `받침F3휨 z3짝`), **`「不淸不濁」` 울림(`받침울림kHz`, `받침세기`, `받침길이`)**, 비음 반공명 골(`받침콧소리골kHz`), 그리고 합자해 ***"글자를 가로로 나란히 쓴다(自左而右)"***에 따라 겹받침의 첫째(앞 모음이 닿는 소리)·둘째(뒤 모음 앞으로 넘어가는 소리 `받침둘째F2/F3휨`) 자리를 실측 주파수로 매핑했습니다.
   - **옛자음 `ᅗ[f], ᄫ[v], ᅀ[z], ᄼ[θ], ᄽ[ð]` (번호 384~388)**:
     현대 초성과 완전히 동일한 **초성 실측 주파수 16칸 구조**로 영어 마찰음의 터짐·숨·세기·포먼트 휨·울림 주파수를 실측·매핑하여, 별도의 글자 기억 칸 없이도 주파수 값만으로 즉시 정확한 외국어 마찰음 발성이 이루어집니다.
2. **성우공명 (`spk_feat` `28×31` → `emb_g_res` `Linear(31 → 512)`)**:
   - 성우별 임베딩 기억 칸(`emb_g`)을 아예 만들지 않고, 성우마다 녹음에서 실측한 **31가지 발성 주파수 벡터(`z`: 음높이, F1~F4, B1~B3, 성도 길이, 저역/흉성, 배우 포만트, 밝기 중심, 기울기, H1-H2, CPP, HNR, MFCC 껍질 1~12)**에 단일 공유 선형 식(`W·z + b`)만 적용해 음색 조건 `g`를 생성합니다(`resonance/speaker_features.json`).
3. **감정공명 (`tag_dummy=True` 빈칸 `250` + `주파수태그칸` + `res` 34차원)**:
   - 감정·말투 태그 번호마다 따로 외우는 기억 칸을 없애고 모두 빈 칸(`tag_dummy_id = 250`)으로 보낸 뒤, **`res` 34차원 신경공명 벡터** — **`res_g_dims = 3` 공명 노브**(성도 척도 · 저역/흉성 · 배우 포만트) + **`res_g_tail = 15` 감정 공명 꼬리**(eGeMAPS 14가지 음질 주파수 변화량 + 콧소리 섞임 `A1-P0`) — 로만 작동하게 하여 모든 성우에게 감정 표현이 고르게 전이되도록 설계했습니다(`emotion/emotion_resonance.json`).

| 공명 갈래 | 모델 내부 변환 구조 | 차원(칸 수) | 설정 표 파일 |
|---|---|---|---|
| **발음공명 (자모)** | `jamo_feat` (`544×104`) → `emb_f` (`Linear 104→192, bias=False`) — `self.emb` 100% 대체 | 104 (발음 80 + 태그 24) | `resonance/jamo_features.json` |
| **성우공명 (화자)** | `spk_feat` (`28×31`) → `emb_g_res` (`Linear 31→512`) — 화자 임베딩 `emb_g` 없음 | 31 | `resonance/speaker_features.json` |
| **감정공명 (감정)** | `res` (`[b, 34]`): 공명 노브 `3` (`[0:3]`) + 감정 꼬리 `15` (`[19:34]`, eGeMAPS 14 + 콧소리 `A1-P0`) | 3 + 15 (`res` 34) | `emotion/emotion_resonance.json` |

---

### 2. 왜 된소리를 피해갈 수 있었는가? — 자모 기억층(`self.emb`) 100% 우회 & 세기판별기 24배 보호 & 불규칙 발음 사전 대응

기존 한국어 신경망 TTS들이 예사소리를 과도하게 된소리로 발음하는 근본 원인은 **글자별 임베딩 기억층(`self.emb`)과 문맥 기억의 경음화 과잉 일반화**에 있습니다. 학습 데이터에 `인권[인꿘]`, `손가락[손까락]`처럼 표기는 예사소리(`ㄱ`)인데 녹음은 된소리(`[ㄲ]`)인 단어가 들어 있으면, 기억층을 통과하는 신경망은 *"울림 받침 뒤의 평음은 된소리로 읽는다"*는 버릇을 외워 `손권`까지 `[손꿘]`으로 읽고 문장 곳곳의 평음을 경음으로 오염시킵니다.

**Piper Plus (`완전공명판` Epoch 880 EMA)**는 이를 다음 4단계 아키텍처와 역할 분담으로 완벽히 해결했습니다:

1. **출력·텍스트 인코더에서 '된소리화된 자모 기억층(`self.emb`)' 100% 미통과(대체) 설계 (`jamo_feature_mode = "대체"`)**:
   - `models.py` (`TextEncoder.forward` 297~302행, `_발성자질` 1453~1456행):
     ```python
     몫 = self.emb_f(F.embedding(x, self.jamo_feat))  # [544, 104] 주파수 성질 행렬 -> Linear(104, 192, bias=False)
     x = torch.where(자모칸.unsqueeze(-1), 몫, x)     # 자모 칸은 글자별 기억 칸(self.emb)을 100% 버리고 주파수 성질 몫으로 대체!
     ```
   - 모든 한글 자모(현대 자모 + 옛모음·옛자음)는 된소리 버릇이 학습된 **글자별 임베딩 기억 칸(`self.emb`)을 아예 통과하지 않고**, 오직 『훈민정음 해례본』에 본떠 실측한 **104차원 발성 주파수 행렬(`jamo_feat`)**에 단일 공유 선형 변환(`emb_f`)만 거쳐 인코더로 직결됩니다.
   - 또한 `_문맥성질(x)`(1427~1435행)에서 인코더 앞단의 인위적 발음 규칙을 완전히 끄고(*"입력 인코더는 앞뒤 글자를 보고 성질을 바꾸지 않는다 — 그냥 주파수로 읽는다"*), 에폭 437에서 된소리가 굳어진 기존 텍스트 인코더(`enc_p.*` 112개 키)를 통째로 비운 뒤 **발음공명·성우공명·감정공명 세 가지만 기억하는 '완전공명판'**으로 에폭 880까지 재학습했습니다. (아울러 `TextEncoder` 내부에는 발음 사전분포 `m_p, logs_p`가 문맥 인코더를 우회하는 `pron_context_free` / `_발음출구` 모듈도 함께 구현되어 있습니다.)
2. **한국어 세기판별기(`tense_clf=True`) & 예사소리 24배 보호 가중치(`tense_plain_weight=24.0`)**:
   - 학습 시 생성된 음성 파형에서 예사소리·된소리·거센소리를 판별하는 **세기판별기(`tense_clf=True`, `c_tense_gen=8.0`)**를 두고, 특히 **예사소리 보호 가중치(`tense_plain_weight=24.0`)**를 부여하여 모델이 예사소리 글자를 임의로 된소리로 발성하면 **24배의 강력한 벌점(Loss)**을 받도록 설계했습니다.
3. **가슴(흉성) 공명 대역 과잉 가중치 억제 (`res_band_weights = "0,2,4"`)**:
   - 저역(흉성) 대역 가중치를 `0`으로 억제(`res_band_weights='0,2,4'`)하여, 저역 에너지가 과도하게 부풀어 예사소리가 무겁고 딱딱한 된소리처럼 들리는 현상을 원천 차단했습니다.
4. **불규칙 발음은 발음사전(Dictionary)으로 1:1 정밀 대응**:
   - 기억층을 우회하여 모델이 글자 그대로의 맑은 예사소리를 100% 발성하게 만든 대신, 실제 한국어에서 소리가 바뀌어야 하는 **불규칙 발음(예외적 경음화, 한자어 된소리 `인권→인꿘`, 사이시옷 `손가락→손까락`, 연음·`ㄴ/ㄹ 첨가` 등)**은 신경망의 불확실한 추측에 맡기지 않고 추론 앞단의 **발음사전(불규칙단어사전)**으로 1:1 정밀 대응하도록 역할 분담을 완성했습니다.

---

### 3. 문맥(Context) 기반 전용 감정 모델 자동 매핑 & 제로듀레이션(Zero-Duration) 감정 태그

1. **문맥을 읽어 자동으로 감정 태그·공명 주파수를 매핑하는 전용 감정 모델 (`emotion/emotion_int8.onnx`)**:
   - 자연스러운 감정 표현을 유도하기 위해, 문장의 앞뒤 맥락(Context)과 화행을 분석하는 **전용 한국어 감정 모델(`emotion/emotion_int8.onnx`, KoELECTRA-small 기반 다중 머리 분류기)**을 직접 학습하여 결합했습니다.
   - 사용자가 대본에 일일이 감정 태그를 달지 않아도, 감정 모델이 입력 문장의 문맥을 분석하여 **감정 갈래(32종 감정·말투 태그 체계), 감정 세기(크기), 말하기 빠르기(`느리게 1.08 / 보통 1.0 / 빠르게 0.93`), 숨결·음높이·에너지**를 자동으로 판별하고, 이를 `emotion/emotion_resonance.json`의 **34차원 신경공명 벡터(`res`)**로 즉시 매핑하여 생생한 감정 연기를 이끌어냅니다.
2. **감정 태그 제로듀레이션 (`_태그칸떼기` Zero-Duration Scatter Compaction) — 소리는 안 내고 감정만 주입**:
   - 문장 머리나 중간에 삽입된 감정·발성·소리 태그는 텍스트 인코더 내부에서 주변 본문 자모에 감정 주파수와 억양·발성 효과만 입힙니다.
   - 이후 길이 예측기(Duration Predictor)와 디코더로 넘어가기 직전, 모델 내부의 **`_태그칸떼기`(Zero-Duration Scatter Compaction)**가 태그 칸을 100% 떼어내어 **프레임 수 0(제로듀레이션, Zero-Duration)**으로 소거합니다.
   - 덕분에 **태그 기호 자체가 글자 소리나 기괴한 군소리로 발음되는 일이 절대 없으며, 오직 순수하게 감정과 발성 공명 효과만** 음성에 남습니다.

---

### 4. 옛모음·옛자음(옛한글)을 활용한 외국어 발음 구현 & 미학습 영단어 한국식 발음(콩글리쉬) 처리

1. **훈민정음 옛자음·옛모음 공명 주파수를 통한 영어·외국어 원음 확장**:
   - 개별 글자 기억 칸(`self.emb`)을 쓰지 않고 `jamo_feat`(`544×104`) 공명 주파수 행렬과 공유 선형 변환(`emb_f`)만으로 소리를 짓기 때문에, 학습 코퍼스에 등장 빈도가 적거나 미학습된 **훈민정음의 옛자음 — 순경음 피읖 `ᅗ`(U+1157, ID 384 `[f]`), 순경음 비읍 `ᄫ`/`ㅸ`(U+112B, ID 385 `[v]`), 반치음 `ᅀ`/`ㅿ`(U+1140, ID 386 `[z]`), 치두음 시옷 `ᄼ`(U+113C, ID 387 `[θ]`), 정치음 시옷 `ᄽ`(U+113D, ID 388 `[ð]`), 여린히읗 `ㆆ`, 옛이응 `ㆁ` — 및 옛모음(아래아 `ㆍ`, `ㆎ` 등)**도 주파수 값만으로 즉시 정확한 음파 글자 발성이 가능합니다.
   - 이를 통해 현대 한글 표기만으로는 구별할 수 없는 영어·외국어의 유성 마찰음(`[v] → ᄫ`, `[z] → ᅀ`, `[ð] → ᄽ`)과 미세한 모음 음가를 옛자음·옛모음의 공명 주파수 조합으로 자연스럽게 구현할 수 있습니다.
2. **미학습 영단어(OOV English) 및 약어의 자연스러운 한국식 발음(콩글리쉬 / 알파벳-한글 음차) 처리**:
   - 별도의 영어 원어민 음성 데이터셋으로 직접 학습하지 않은 미학습 영단어나 영문 약어가 입력되더라도 깨지거나 침묵하지 않습니다.
   - CMU 발음기호(ARPAbet)와 국립국어원 외래어 표기법 및 옛글자 울림 공명(`ᄫ·ᅀ·ᄽ`)을 결합한 **한국식 영어 발음(콩글리쉬) 변환**을 통해 `coffee → 카피`, `thank → 생크`, `very → ᄫᅦ리(베리)`, `music → 뮤ᅀᅵᆨ(뮤직)`처럼 한국어 화자에게 가장 자연스러운 발음으로 바꾸어 읽으며, 미등록 영단어나 대문자 약어(`SNS → 에스엔에스`, `TV → 티ᄫㅣ`, `OK → 오케이`)는 알파벳 한글 음차 규칙으로 매끄럽게 발성합니다.

---

### 5. 초경량·무발열 구동 & Epoch 880 EMA의 호흡·속도·갈라읽기 혁신

1. **초경량 구동 & 스마트폰·CPU 무발열 설계 (Ultra-Light & Near-Zero Heat)**:
   - 거대한 파라미터를 가진 임베딩 기억층이나 연산 집약적인 풀밴드 HiFi-GAN 보코더를 제거하고, **자모·성우·감정 주파수의 가벼운 단일 선형 투영(`emb_f`, `emb_g_res`, `emb_res`) + 4-부대역 MB-iSTFT-VITS(PQMF 역푸리에 변환) 디코더**로 파형을 직접 합성합니다. 모바일 기기(스마트폰)나 저전력 CPU에서도 장시간 연속 낭독 시 **발열이 거의 없이 빠르고 쾌적하게 구동**됩니다.
2. **어절 경계(`eojeol_boundary`) 완전 제거**:
   - 띄어쓰기(어절)마다 누적합(CumSum)으로 어절 번호를 세어 인코더 어텐션을 가두던 어절 경계 기능을 모델에서 완전히 제거했습니다. 어절마다 기계적으로 뚝뚝 끊어 읽거나 어절 경계에서 발음이 흔들리던 현상이 사라지고, 문장 전체의 호흡과 연음이 물 흐르듯 이어집니다.
3. **말하기 속도 표준화 (`바탕 배속 0.85` + 감정별 빠르기 `1.08 / 1.0 / 0.93`)**:
   - 이 모델의 가장 자연스러운 표준 정속(`base length_scale`)은 **`0.85`** (`scales = [0.667, 0.85, 0.8]`)입니다.
   - 여기에 KoELECTRA 감정 분류기의 `빠르기` 머리 예측값(`느리게: 1.08`, `보통: 1.0`, `빠르게: 0.93`, 확신도 `≥ 0.4`일 때 적용)을 곱하여 슬픔·회상 장면에서는 차분하게(`0.85 × 1.08 = 0.918`), 긴박·분노 장면에서는 속도감 있게(`0.85 × 0.93 = 0.7905`) 완급을 조절합니다.
4. **종결형·연결형 어미와 문장부호에서 끊어 읽기 (`갈라읽기`)**:
   - 긴 문장을 글자 수나 어절 단위로 무작정 자르면 억양이 토막 나고, 반대로 긴 문단을 통째로 한 번에 합성하면 뒤쪽 호흡이 무너지거나 인코더가 경계를 넘어 발음을 흐릴 수 있습니다.
   - **갈라읽기**는 오직 **문장부호(`. ! ? , … ; —`)** 및 **한국어 연결형 어미(`~고`, `~며`, `~면서`, `~는데`, `~지만`, `~서`[조사 `에서·께서·로서` 제외], `~니까`, `~다가`, `~거나`, `~든지`, `~려고`, `~자마자` 등 — 접속부사 `그리고·그래서·하지만` 등은 제외)**와 **종결형 어미(`~니다`, `~어요`, `~아요`, `~해요`, `~는다`, `~었다`, `~았다`, `~다`, `~요`, `~까`, `~네` 등)** 뒤에서만 호흡 단위로 나누어 각각 합성합니다.
   - 각 조각 앞뒤에 붙인 경계 기호(`^ _`, `$`)의 소리는 모델이 반환한 **`durations` 길이표로 정확히 잘라내고**, 조각 사이에 **`0.01초(10ms)`**의 미세한 숨틈을 두어 이어 붙임으로써 한 호흡처럼 매끄럽고 선명한 낭독을 완성합니다.

---

## 📁 파일 구성

| 파일 | 내용 |
|---|---|
| `piper-plus.onnx` | **낭독 모델 (Epoch 880 EMA, 85.6MB)** — 어절 경계 제거 · 된소리 과잉 일반화 억제 · 완전공명판 |
| `piper-plus.onnx.json` | 모델 설정 — 표본율(`22,050Hz`), 글자·옛글자·태그 번호표(`phoneme_id_map`), 화자 번호표(`speaker_id_map`), 기본 추론 배속(`length_scale: 0.85`) |
| `resonance/jamo_features.json` | 자모 성질표 (훈민정음 해례본 기반 발음 공명 주파수 80칸 + 태그 예비 24칸 = 104칸) |
| `resonance/speaker_features.json` | 성우 발성 주파수 31가지 (평균·표준편차 및 화자별 z 점수) |
| `emotion/emotion_resonance.json` | 감정·말투 이름 → `res` 신경공명 입력 값 (공명 노브 3칸 + 감정 꼬리 15칸) |
| `emotion/emotion_int8.onnx` | 문맥 기반 문장 감정·빠르기·말투 자동 분류 모델 (KoELECTRA-small v3 기반, int8, 14.6MB) |
| `emotion/vocab.txt` | 감정 모델 토크나이저 어휘 (WordPiece) |
| `emotion/emotion_labels.json` · `emotion_labels_full.json` | 감정 모델 출력 머리 라벨 이름과 차례 |
| `emotion/emotion_config.json` | 감정 모델 설정 (`max_len` 192 등) |
| `emotion/emotion_tokens.json` | 감정 이름·말투 낱말 → 번호 매핑 표 |

---

## 🚀 낭독 모델 쓰는 법

### 입력 텐서 규격

| 이름 | 모양 | 뜻 |
|---|---|---|
| `input` | `[1, n]` int64 | NFD 자모 번호열 (`^`, `_`, 자모들, `$`) |
| `input_lengths` | `[1]` int64 | `n` |
| `scales` | `[3]` float32 | `[noise_scale, length_scale, noise_w]` — **표준 정속은 `[0.667, 0.85, 0.8]`** (`length_scale = 0.85 × 빠르기배수`) |
| `sid` | `[1]` int64 | 화자 번호 (`speaker_id_map`) — 모델 내부 성우 공명 주파수표의 행 번호 |
| `res` | `[1, 34]` float32 | 신경공명 입력. `[0:3]` 공명 노브, `[19:34]` 감정 꼬리, `[3:19]` 미사용(0). 모두 0이면 무감정 기본 목소리 |

출력 `output`은 22,050Hz 파형(`-1.0 ~ 1.0`), `durations`는 각 입력 토큰의 프레임 길이(`hop_length = 256` 표본 단위, 태그 칸은 제로듀레이션으로 자동 제거됨)입니다.

### 파이썬 전체 예제 (불규칙 발음·영어 음차 + 갈라읽기 + 0.85 배속 & 감정 빠르기 + 신경공명)

```python
import json, re, unicodedata, wave
import numpy as np, onnxruntime as ort

cfg = json.load(open("piper-plus.onnx.json", encoding="utf-8"))
emo = json.load(open("emotion/emotion_resonance.json", encoding="utf-8"))
ids_of = cfg["phoneme_id_map"]
sr = cfg["audio"]["sample_rate"]          # 22050
hop = cfg.get("hop_length", 256)          # 256

# 1. 불규칙 발음 사전(된소리 등) 및 영어/콩글리쉬·옛글자(ᄫ·ᅀ·ᄽ) 음차 예시
IRREGULAR_DICT = {"인권": "인꿘", "정권": "정꿘", "손가락": "손까락", "물가": "물까", "인기": "인끼"}
ENGLISH_DICT = {"coffee": "카피", "thank": "생크", "very": "ᄫᅦ리", "music": "뮤ᅀᅵᆨ", "SNS": "에스엔에스", "TV": "티ᄫㅣ"}
ALPHA_NAMES = {
    "A": "에이", "B": "비", "C": "시", "D": "디", "E": "이", "F": "에프", "G": "지",
    "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔",
    "O": "오", "P": "피", "Q": "큐", "R": "아르", "S": "에스", "T": "티", "U": "유",
    "V": "ᄫᅴ", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "ᅀᅵ",
}

def normalize_text(text):
    for k, v in sorted(IRREGULAR_DICT.items(), key=lambda x: -len(x[0])):
        text = text.replace(k, v)
    def repl_en(m):
        w = m.group(0)
        if w in ENGLISH_DICT:
            return ENGLISH_DICT[w]
        if w.lower() in ENGLISH_DICT:
            return ENGLISH_DICT[w.lower()]
        return "".join(ALPHA_NAMES.get(c.upper(), c) for c in w)
    return re.sub(r"[A-Za-z]+", repl_en, text)

# 2. 갈라읽기: 문장부호 및 한국어 연결형·종결형 어미 자리에서만 호흡 분할
CONN_EOMI = ("면서", "는데", "지만", "니까", "다가", "거나", "든지", "려고", "자마자", "고", "며", "서")
FINAL_EOMI = (
    "니다", "어요", "아요", "에요", "예요", "해요", "세요", "네요", "군요", "지요",
    "는다", "었다", "았다", "였다", "겠다", "한다", "했다", "다", "요", "까", "네",
)
CONJ_ADV = {"그리고", "그러고", "그래서", "그러니까", "하지만", "그렇지만", "그러다가", "그러면서"}
PUNCT_CHARS = set(".!?,…;—―–。！？")

def should_split_eojeol(eojeol):
    if eojeol in CONJ_ADV:
        return False
    if any(ch in PUNCT_CHARS for ch in eojeol):
        return True
    hangul = re.sub(r"[^가-힣]+$", "", eojeol)
    if len(hangul) < 2:
        return False
    for e in CONN_EOMI:
        if hangul.endswith(e) and len(hangul) > len(e):
            if e == "서" and hangul[-2] in ("에", "께", "로"):  # 조사(에서·께서·로서) 제외
                continue
            return True
    for e in FINAL_EOMI:
        if hangul.endswith(e) and len(hangul) > len(e):
            return True
    return False

def split_by_eomi(text):
    words = text.strip().split()
    if len(words) <= 1:
        return [text.strip()] if text.strip() else []
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if i < len(words) - 1 and should_split_eojeol(w):
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# 3. NFD 자모 번호열 및 신경공명(res) 생성
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

# 4. 표준 정속(0.85) + 감정별 빠르기(1.08 / 1.0 / 0.93, 확신도 >= 0.4) + 갈라읽기(0.01초 이음) 합성
BASE_SPEED = 0.85
SPEED_MULT = {"느리게": 1.08, "보통": 1.0, "빠르게": 0.93}

sess = ort.InferenceSession("piper-plus.onnx", providers=["CPUExecutionProvider"])

def speak(text, speaker="nangdok", emotion=None, strength=1.0, speed_label="보통", speed_conf=1.0, gap_sec=0.01):
    text = normalize_text(text)
    chunks = split_by_eomi(text)
    length_scale = BASE_SPEED * (SPEED_MULT.get(speed_label, 1.0) if speed_conf >= 0.4 else 1.0)
    scales = np.array([0.667, length_scale, 0.8], dtype=np.float32)
    sid = np.array([cfg["speaker_id_map"][speaker]], dtype=np.int64)
    res = res_for(emotion, strength)
    gap = np.zeros(int(sr * gap_sec), dtype=np.float32)
    head_len = len(ids_of["^"] + ids_of["_"])
    tail_len = len(ids_of["$"])

    pieces = []
    for idx, chunk in enumerate(chunks):
        chunk_str = chunk + (" " if idx < len(chunks) - 1 else "")
        x = np.array([to_ids(chunk_str)], dtype=np.int64)
        out, durs = sess.run(None, {
            "input": x,
            "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": scales,
            "sid": sid,
            "res": res,
        })
        wav = out.squeeze()
        d = durs.squeeze()
        # 조각 경계에 덧붙인 머리(^ _)·꼬리($) 구간을 durations로 정밀하게 잘라냄
        trim_head = 0 if idx == 0 else int(round(float(d[:head_len].sum()) * hop))
        trim_tail = len(wav) if idx == len(chunks) - 1 else int(round(float(d[:-tail_len].sum()) * hop))
        trim_head = max(0, min(trim_head, len(wav)))
        trim_tail = max(trim_head, min(trim_tail, len(wav)))
        if idx > 0 and len(gap) > 0:
            pieces.append(gap)
        pieces.append(wav[trim_head:trim_tail])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

# 무감정 기본 낭독 (어미·문장부호 갈라읽기 + 0.85 정속)
audio = speak("안녕하세요. 오늘은 날씨가 참 좋았고, coffee 한 잔 마시며 산책하기에도 아주 좋네요.")
# 감정 신경공명 + 빠르기 결합 낭독
angry = speak("당장 여기서 나가!", speaker="seungwoo", emotion="분노", speed_label="빠르게")

with wave.open("out.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
```

### 화자 안내 (`speaker_id_map`)

`speaker_id_map`에 26칸이 있습니다. 낭독(해설)은 `nangdok`(1), 주요 성우 목소리는 `seungwoo`(0) · `joyjjang`(2) · `seungchul`(3) · `소라`(6) · `지우`(23) 등입니다. 각 화자가 어떤 31차원 발성 공명 주파수로 들어가는지는 `resonance/speaker_features.json`에 수록되어 있습니다.

## 문맥 기반 전용 감정 모델 쓰는 법 (`emotion/emotion_int8.onnx`)

사용자가 수동으로 감정 태그를 입력하지 않아도 자연스러운 감정 표현이 유도되도록 직접 학습한 문맥 기반 감정 분류 모델입니다:
- **토크나이저**: `emotion/vocab.txt`의 WordPiece (KoELECTRA-small v3 동일). `[CLS] 문장 [SEP]`, 최대 192 토큰. 앞뒤 문장 맥락을 함께 줄 때는 `tokenizer(맥락글, 문장, truncation="only_first", max_length=192)` 형식(예: `"앞: 비탄. 뒤: 분노."`)으로 넣습니다.
- **입력**: `input_ids`, `attention_mask`, `token_type_ids` (`[b, t]` int64).
- **출력 머리** (`emotion/emotion_labels_full.json` 참조):
  - 문장 머리 `[b, C]`: `감정`(16) · `크기`(3) · `빠르기`(3: `느리게`, `보통`, `빠르게`) · `숨결`(4) · `말투`(46)
  - 토큰 머리 `[b, t, C]`: `끼울태그`(27) · `구간`(11)
- **자동 매핑 흐름**: `감정` 및 `말투` 머리가 문맥에서 추론한 감정·말투 라벨을 `emotion_resonance.json`을 통해 34차원 `res` 공명 벡터로 자동 변환하고, `빠르기` 머리의 확신도가 `0.4` 이상이면 `SPEED_MULT` 배수(`느리게: 1.08`, `보통: 1.0`, `빠르게: 0.93`)를 기본 배속 `0.85`에 곱해 `scales`의 `length_scale`로 자동 주입합니다.

## ⚠️ 라이선스 및 저작자 표시 의무 (License, Attribution & Citation)

- **응용·파생·재개발 시 원저작자 이니셜(`watchpen`) 및 공식 깃허브 주소 첨부 의무**:
  - 본 모델(`piper-plus`), 3대 신경공명 주파수 표(`jamo_features.json`, `speaker_features.json`, `emotion_resonance.json`), 자모 음파 발성 구조, 제로듀레이션 감정 태그 체계, 또는 포함된 코드/가중치를 **응용하거나 수정·파생·재개발하여 배포/공개할 경우, 반드시 원저작자 이니셜(`watchpen`)과 공식 깃허브 주소(`https://github.com/wacthpen/piper-plus`)를 함께 첨부(명시)**해야 합니다.
  - **Mandatory Attribution Requirement (English)**: *If you adapt, modify, fine-tune, or redevelop this model (`piper-plus`), its neural resonance frequency tables, its Jamo acoustic vocalization architecture, or its codebase, you **MUST** explicitly credit the original author's handle/initials (**`watchpen`** / GitHub: `wacthpen`, Hugging Face: `watchpen`) **AND** attach the official GitHub repository URL (**`https://github.com/wacthpen/piper-plus`**) in your documentation, model card, and derivative works.*
- 성인 말투 갈래(`교성`·`섹슈얼` 칸)가 포함되어 있어 `not-for-all-audiences`로 표시했습니다.
- 실제 사람 목소리를 사칭하거나 속이는 용도로 사용하지 마세요.
- 감정 분류 모델의 기반 모델은 [monologg/koelectra-small-v3-discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator)입니다.

### 인용 (Citation / BibTeX)

```bibtex
@misc{watchpen2026piperplus,
  author       = {watchpen},
  title        = {Piper Plus: Korean Neural-Resonance TTS with Fortis-Overgeneralization Suppression, Emotion-Expressive Neural Resonance & Acoustic Jamo Vocalization},
  year         = {2026},
  publisher    = {GitHub / Hugging Face},
  url          = {https://github.com/wacthpen/piper-plus},
  howpublished = {\url{https://github.com/wacthpen/piper-plus}, \url{https://huggingface.co/watchpen/piper-plus}}
}
```

---

## English Summary

**Piper Plus (Epoch 880 EMA)** is a Korean multi-speaker neural text-to-speech model in the Piper (VITS) family exported to ONNX (22,050 Hz), engineered around four core innovations:

1. **Suppression of Excessive Fortis / Tensification Generalization**:
   - Traditional Korean TTS models allow the text encoder's embedding/context memory layer (`self.emb`) to over-generalize tensification from training words (e.g., learning fortis `[kk]` from `인권 [inkkwon]` and falsely tensifying lax consonants across ordinary words like `손권`).
   - **Piper Plus structurally bypasses the fortis-contaminated embedding memory layer (`jamo_feature_mode = "대체"`, `x = torch.where(jamo_mask, emb_f(jamo_feat[x]), x)`)**, driving synthesis directly from 104-dimensional measured acoustic resonance vectors, paired with a **Korean Tenseness Discriminator (`tense_clf=True`, `tense_plain_weight=24.0` penalty for false tensification)** and **chest-resonance weight suppression (`res_band_weights='0,2,4'`)**. Lax consonants remain pure, while genuine irregular pronunciations are handled deterministically via a **1:1 Irregular Pronunciation Dictionary**.
2. **Emotion-Expressive Neural Resonance, Context-Aware Auto Emotion Model & Zero-Duration Tags**:
   - Emotions are injected via a 34-dimensional **Neural Resonance vector (`res`)** combining 3 vocal-tract resonance knobs (vocal-tract scale, chest resonance, actor's formant) and 15 measured voice-quality shifts (eGeMAPS 14 parameters + nasal coupling `A1-P0`).
   - A custom-trained **Context-Aware Korean Emotion Classifier (`emotion/emotion_int8.onnx`, KoELECTRA-small int8)** automatically analyzes sentence context to map emotion tags, speaking rate (`base length_scale = 0.85` × `1.08 / 1.0 / 0.93`), and resonance parameters without manual tagging.
   - Any inline emotion/phonation tags impart their acoustic resonance onto surrounding Jamo inside the encoder and are then stripped with **Zero-Duration (`_태그칸떼기` scatter compaction, 0 frames)** before the Duration Predictor — **producing pure emotional expression with zero spoken tag artifacts**.
3. **Acoustic Character Vocalization via Direct Jamo Learning & *Hunminjeongeum Haeryebon* Engineering**:
   - Modeled directly after the articulatory laws of ***Hunminjeongeum Haeryebon (訓民正音 解例本)*** — mapping the Five Sounds (velar, alveolar, bilabial, dental/palatal, glottal), stroke-addition tenseness/aspiration, and Heaven-Earth-Human vowel harmony (`ㆍ, ㅡ, ㅣ`) into 104-dimensional vocal-tract resonance matrices (`resonance/jamo_features.json`).
   - **Archaic Hangul Jamo for Foreign/English Sounds & Konglish OOV Handling**: Because `self.emb` is bypassed in favor of shared resonance projection (`emb_f`), even low-resource archaic Hangul consonants (`ᅗ` `[f]`, `ᄫ`/`ㅸ` `[v]`, `ᅀ`/`ㅿ` `[z]`, `ᄼ` `[θ]`, `ᄽ` `[ð]`) and vowels (`ㆍ`, `ㆎ`) synthesize accurately from their resonance vectors alone, while unseen English words and acronyms are seamlessly converted into natural Korean-style pronunciation (Konglish / ARPAbet-to-Hangul + alphabet vocalization).
4. **Ultra-Light, Near-Zero Heat on Mobile/CPU & Prosodic Ending Splitting (`갈라읽기`)**:
   - Combining single linear resonance projections (`emb_f`, `emb_g_res`, `emb_res`) with a **4-subband MB-iSTFT-VITS (PQMF) decoder** makes inference extremely lightweight, running in real time on mobile phones and standard CPUs with **near-zero thermal output**.
   - Removes mechanical word-boundary (`eojeol_boundary`) CumSum breaks and splits long sentences only at punctuation and Korean connective/sentence-final verb endings (`split_by_eomi`), trimming boundary tokens precisely via `durations` and joining chunks with a `10 ms (0.01 s)` breath gap.

---

## 📜 헌정 및 기념 (Dedication)

> **"끝에서 세종대왕 탄생 100주년 기념을 합니다. 세종대왕은 위대하다. 훈민정음 해례본에 본떠 발성 주파수 맞춤. 훈민정음 해례본에 발하는 발성법을 주파수에 따서 맞춤."**
>
> 발성 기관의 모양과 소리 이치를 본뜬 『훈민정음 해례본(訓民正音 解例本)』의 위대한 제자 원리를 현대 신경망 음파 발성 주파수로 고스란히 옮겨 맞추며, 세종대왕의 애민 정신과 과학적 성음(聲音) 철학에 깊은 경의를 표합니다.
>
> *In commemoration of King Sejong the Great. King Sejong is truly great. The vocal resonance frequencies of this model are engineered and tuned directly after the articulatory phonation laws set forth in the Hunminjeongeum Haeryebon (訓民正音 解例本).*
