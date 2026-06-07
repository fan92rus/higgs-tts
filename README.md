# 🎙 Higgs Audio v3 TTS — Portable by Neurogen

Портативная сборка экспрессивного синтеза речи **Higgs Audio v3** (`bosonai/higgs-audio-v3-tts-4b`)
с красивым десктоп-интерфейсом. Клонирование голоса, 100+ языков, управление эмоциями,
стилем, просодией и звуковыми эффектами — всё в одном окне.

> **Сборка, движок и интерфейс — Portable by Neurogen.**
> Модель Higgs Audio v3 © Boson AI (лицензия Research and Non-Commercial).

---

## ✨ Возможности

- 🗣 **Зеро-шот клонирование голоса** — загрузите 5–20 секунд референса, получите его голос.
- 🌍 **100+ языков**, включая русский.
- 🎭 **Инлайн-управление** прямо в тексте:
  - **Эмоции**: `<|emotion:amusement|>`, `<|emotion:sadness|>`, … (21 эмоция)
  - **Стиль**: `<|style:whispering|>`, `<|style:singing|>`, `<|style:shouting|>`
  - **Просодия**: скорость, паузы, высота тона, выразительность
  - **Звуки**: `<|sfx:laughter|>Haha`, `<|sfx:sigh|>Uh`, `<|sfx:sneeze|>Achoo`, …
- ⚡ **Авто-подбор режима** под ваше железо: **bf16 / fp16 / 8-bit / 4-bit / CPU**.
- 🖥 **Нативное окно** (без браузерных вкладок), тёмная «glass»-тема.
- 💾 Полностью **офлайн** после первого запуска. Все данные — внутри папки программы.

---

## 🚀 Установка и запуск

1. Распакуйте архив в папку **без кириллицы и пробелов** в пути (желательно), например `D:\HiggsTTS\`.
2. Запустите **`Установка (первый запуск).bat`** и выберите:
   - **[1] GPU (NVIDIA, CUDA 12.1)** — рекомендуется;
   - **[2] CPU** — если нет видеокарты NVIDIA.
3. Запустите **`Запустить Higgs TTS (Portable by Neurogen).bat`**.
   - При **первом** запуске автоматически скачаются веса модели (**~9.3 ГБ**) и выполнится
     однократная конвертация. Дальше запуск — мгновенный.

> **Python 3.11 уже включён в сборку** (папка `python\`) — отдельно ставить ничего не нужно.
> Нужен интернет **один раз** — для загрузки весов.

---

## 🐳 Развёртывание в Docker

### Быстрый старт (локально)

Убедитесь, что установлен [Docker](https://docs.docker.com/engine/install/) с драйвером NVIDIA
([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

```bash
docker compose up -d
```

Контейнер будет слушать на `http://localhost:7861`. При первом запуске автоматически
скачаются веса модели (~9.3 ГБ). Данные хранятся в Docker-томах:
- `higgs-models` — веса модели
- `higgs-outputs` — сгенерированные аудио
- `higgs-cache` — кэш HuggingFace / torch

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `HIGGS_HOST` | `0.0.0.0` | Интерфейс сервера |
| `HIGGS_PORT` | `7861` | Порт |
| `HIGGS_MODE` | `auto` | Режим точности: `bf16`, `fp16`, `8bit`, `4bit`, `cpu` |
| `HIGGS_NO_AUTOSTART` | `0` | `1` — не запускать сервер при старте контейнера |

### Production-развёртывание (с Traefik и Watchtower)

Для домашнего сервера используйте композ-файл с Traefik:

```yaml
# docker-compose.deploy.yml
services:
  higgs-tts:
    image: ghcr.io/fan92rus/higgs-tts:latest
    container_name: higgs-tts
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - higgs-models:/app/models
      - higgs-outputs:/app/outputs
      - higgs-cache:/app/.cache
    networks:
      - proxy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.higgs.rule=Host(`higgs.local`)"
      - "traefik.http.routers.higgs.entrypoints=websecure"
      - "traefik.http.routers.higgs.tls.certresolver=stepca"
      - "traefik.http.services.higgs.loadbalancer.server.port=7861"

volumes:
  higgs-models:
    external: true
  higgs-outputs:
    external: true
  higgs-cache:
    external: true

networks:
  proxy:
    external: true
```

[Watchtower](https://containrrr.dev/watchtower/) автоматически обновляет контейнер
при публикации нового образа:

```bash
docker run -d \
  --name watchtower \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --interval 60 \
  --cleanup \
  higgs-tts
```

### CI/CD (GitHub Actions)

При пуше в ветку `main` GitHub Actions автоматически:
1. Запускает тесты (pytest + ruff)
2. Собирает Docker-образ
3. Пушит его в `ghcr.io/fan92rus/higgs-tts:latest`
4. Watchtower на сервере подхватывает обновление и перезапускает контейнер

### Сборка вручную

```bash
docker build -t higgs-tts .
docker run --gpus all -p 7861:7861 -v higgs-models:/app/models -v higgs-outputs:/app/outputs -v higgs-cache:/app/.cache higgs-tts
```

---

## 🧠 Требования к железу

| Видеопамять (VRAM) | Режим | Комментарий |
|---|---|---|
| ≥ 11 ГБ | **bf16 / fp16** | Лучшее качество и скорость |
| 7–11 ГБ | **8-bit** | Квантизация bitsandbytes |
| 5–7 ГБ | **4-bit** | Для бюджетных карт |
| < 5 ГБ или без GPU | **CPU** | Работает везде, но медленно |

Режим определяется автоматически. Принудительно можно задать переменной окружения
`HIGGS_MODE` = `bf16 | fp16 | 8bit | 4bit | cpu`.

> ⚠️ Это большая модель (4B параметров). На слабых видеокартах (например, GTX 1650 4 ГБ)
> используйте режим CPU — будет медленно, но работать будет.

---

## 🎛 Как пользоваться

1. Вкладка **«Генерация»** — введите текст. Кнопками-пилюлями добавляйте эмоции/стиль/просодию/SFX.
2. Вкладка **«Клон голоса»** — перетащите аудио-референс (`.wav`/`.mp3`), при желании впишите его
   текст («Текст референса улучшает клон»), затем сгенерируйте.
3. Готовые файлы сохраняются в папку **`outputs\`** и доступны во вкладке **«История»**.

**Правила инлайн-токенов:**
- Токены доставки (эмоция, стиль, скорость, высота, выразительность) ставьте **в начало** текста.
- Позиционные токены (`pause`, `long_pause`, `sfx`) — **в нужном месте**; каждый `<|sfx:...|>`
  сопровождайте звукоподражанием (`<|sfx:laughter|>Haha`).

### Пресет-голоса
Положите в папку `voices\` файлы `имя.wav` (24 кГц моно желательно) и, по желанию,
`имя.txt` с транскриптом — они появятся в списке голосов.

---

## 📁 Структура

```
Higgs-Audio-v3-TTS-Portable-by-Neurogen\
├─ Установка (первый запуск).bat
├─ Запустить Higgs TTS (Portable by Neurogen).bat
├─ requirements.txt
├─ src\           ← движок, сервер, интерфейс
├─ models\        ← веса (создаётся при первом запуске)
├─ outputs\       ← сгенерированные аудио
├─ voices\        ← ваши пресет-голоса
└─ runtime\       ← Python-окружение (создаётся установщиком)
```

---

## ⚖️ Лицензия

Модель **Higgs Audio v3** распространяется по лицензии **Boson Higgs Audio v3 Research and
Non-Commercial License** — только для исследовательского и **некоммерческого** использования.
Запрещены: клонирование голоса без согласия, выдача себя за другого, мошенничество и любое
незаконное применение. Подробности — в файле `ЛИЦЕНЗИЯ-NOTICE.txt` и на странице модели
на HuggingFace.

Код сборки и интерфейс — **Portable by Neurogen**.

---

<sub>Higgs Audio v3 TTS — **Portable by Neurogen**. Made with ❤ for the community.</sub>

---

## English (short)

Portable Windows build of **Higgs Audio v3 TTS** with a polished native desktop UI: zero-shot
voice cloning, 100+ languages, inline control of emotion / style / prosody / sound-effects.
Run `Установка (первый запуск).bat` once (choose GPU CUDA 12.1 or CPU), then launch
`Запустить Higgs TTS (Portable by Neurogen).bat`. On first run it auto-downloads the ~9.3 GB
weights and converts them once. Precision auto-selects (bf16/fp16/8-bit/4-bit/CPU) to fit your VRAM.

**Docker deployment** (Linux + NVIDIA GPU):

```bash
git clone https://github.com/fan92rus/higgs-tts.git
cd higgs-tts
docker compose up -d
```

The image is auto-built to `ghcr.io/fan92rus/higgs-tts:latest`. Pair with
[Watchtower](https://containrrr.dev/watchtower/) for automatic updates. See the full
Docker guide in the Russian section above.

The model is under the **Boson Research & Non-Commercial License**. Build & UI: **Portable by Neurogen**.
