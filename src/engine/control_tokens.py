"""Catalog of Higgs Audio v3 inline control tokens (emotion / style / prosody / sfx).

Source: the model card for bosonai/higgs-audio-v3-tts-4b. Used to populate the GUI
pickers and the ``/api/info`` payload. Russian labels are provided for the UI.

Placement rules (from the model card):
  1. Delivery tokens (emotion, style, prosody speed/pitch/expressive) shape the whole
     turn -> put them at the START of the text.
  2. Positional tokens (prosody pause/long_pause, sfx) fire inline -> insert where they
     should occur. Every ``<|sfx:...|>`` should be immediately followed by its
     onomatopoeia (e.g. ``<|sfx:laughter|>Haha``).
"""

from __future__ import annotations

EMOTIONS = [
    ("elation", "Ликование / радость"),
    ("amusement", "Веселье / смех"),
    ("enthusiasm", "Энтузиазм / воодушевление"),
    ("determination", "Решимость / твёрдость"),
    ("pride", "Гордость / уверенность"),
    ("contentment", "Спокойное удовлетворение"),
    ("affection", "Теплота / нежность"),
    ("relief", "Облегчение"),
    ("contemplation", "Задумчивость"),
    ("confusion", "Замешательство"),
    ("surprise", "Удивление"),
    ("awe", "Трепет / благоговение"),
    ("longing", "Тоска / томление"),
    ("arousal", "Возбуждение / страсть"),
    ("anger", "Гнев"),
    ("fear", "Страх"),
    ("disgust", "Отвращение"),
    ("bitterness", "Горечь"),
    ("sadness", "Грусть"),
    ("shame", "Стыд"),
    ("helplessness", "Беспомощность"),
]

STYLES = [
    ("singing", "Пение"),
    ("shouting", "Крик / громкий голос"),
    ("whispering", "Шёпот"),
]

# (name, ru_label, onomatopoeia to insert right after the token)
SFX = [
    ("cough", "Кашель", "Ahem"),
    ("laughter", "Смех", "Haha"),
    ("crying", "Плач", "Sob"),
    ("screaming", "Крик", "Ahh"),
    ("burping", "Отрыжка", "Burp"),
    ("humming", "Хмыканье", "Hmm"),
    ("sigh", "Вздох", "Uh"),
    ("sniff", "Шмыганье носом", "Sff"),
    ("sneeze", "Чихание", "Achoo"),
]

PROSODY_SPEED = [
    ("speed_very_slow", "Очень медленно (~0.65×)"),
    ("speed_slow", "Медленно (~0.85×)"),
    ("speed_fast", "Быстро (~1.2×)"),
    ("speed_very_fast", "Очень быстро (~1.4×)"),
]

PROSODY_PAUSE = [
    ("pause", "Пауза (~400–700 мс)"),
    ("long_pause", "Длинная пауза (~700–1500 мс)"),
]

PROSODY_PITCH = [
    ("pitch_low", "Ниже тон (~−3 полутона)"),
    ("pitch_high", "Выше тон (~+2.5 полутона)"),
]

PROSODY_DELIVERY = [
    ("expressive_high", "Выразительнее"),
    ("expressive_low", "Ровнее / спокойнее"),
]


def _emotion_token(name: str) -> str:
    return f"<|emotion:{name}|>"


def _style_token(name: str) -> str:
    return f"<|style:{name}|>"


def _prosody_token(name: str) -> str:
    return f"<|prosody:{name}|>"


def _sfx_token(name: str) -> str:
    return f"<|sfx:{name}|>"


def catalog() -> dict:
    """Structured catalog consumed by the GUI / ``/api/info``."""
    return {
        "emotions": [
            {"token": _emotion_token(n), "name": n, "label_ru": ru} for n, ru in EMOTIONS
        ],
        "styles": [
            {"token": _style_token(n), "name": n, "label_ru": ru} for n, ru in STYLES
        ],
        "sfx": [
            {
                "token": _sfx_token(n),
                "name": n,
                "label_ru": ru,
                "onomatopoeia": ono,
            }
            for n, ru, ono in SFX
        ],
        "prosody_speed": [
            {"token": _prosody_token(n), "name": n, "label_ru": ru}
            for n, ru in PROSODY_SPEED
        ],
        "prosody_pause": [
            {"token": _prosody_token(n), "name": n, "label_ru": ru}
            for n, ru in PROSODY_PAUSE
        ],
        "prosody_pitch": [
            {"token": _prosody_token(n), "name": n, "label_ru": ru}
            for n, ru in PROSODY_PITCH
        ],
        "prosody_delivery": [
            {"token": _prosody_token(n), "name": n, "label_ru": ru}
            for n, ru in PROSODY_DELIVERY
        ],
    }


__all__ = ["catalog"]
