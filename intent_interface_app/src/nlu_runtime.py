"""Runtime слой для отдельного интерфейса детектора намерений.

Файл повторяет ключевую логику из DetectionIntents.ipynb, но не зависит от ноутбука:
- загрузка train/dev данных;
- построение словаря;
- загрузка best_model.pt;
- построение FAISS/LBD индекса;
- ручной инференс с top-кандидатами и ближайшим примером.
"""

from __future__ import annotations

import json
import re
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:
    FAISS_IMPORT_ERROR = None


@dataclass
class Config:
    SEED: int = 42
    MAX_LEN: int = 128

    EMB_DIM: int = 96
    CNN_CHANNELS: int = 128
    LSTM_HIDDEN: int = 128
    PROJ_DIM: int = 128

    EPOCHS: int = 15
    LR: float = 3e-4
    LAMBDA: float = 0.2
    PATIENCE: int = 3
    BATCH_SIZE: int = 64

    SPECIALS: List[str] = field(default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"])
    PAD_IDX: int = 0
    UNK_IDX: int = 1
    BOS_IDX: int = 2
    EOS_IDX: int = 3

    RANK_ALPHA: float = 0.4
    RANK_BETA: float = 0.4
    RANK_GAMMA: float = 0.1
    THRESHOLD: float = 0.4

    NEGATIVE_WEIGHT: float = 1.5
    NEGATIVE_MARGIN: float = 0.1
    FAISS_K: int = 5


@dataclass
class SkillConfig:
    name: str
    base_weight: float = 0.5
    positive_keywords: List[str] = field(default_factory=list)
    exclusion_keywords: List[str] = field(default_factory=list)
    requires_slot: List[str] = field(default_factory=list)


SKILL_CONFIGS: Dict[str, SkillConfig] = {
    "music.play": SkillConfig(
        name="music.play",
        base_weight=0.5,
        positive_keywords=["включи", "поставь", "запусти", "воспроизведи", "сыграй", "музыку", "песню", "трек"],
        exclusion_keywords=["будильник", "таймер", "напоминание", "alarm"],
    ),
    "alarm.set": SkillConfig(
        name="alarm.set",
        base_weight=0.5,
        positive_keywords=["будильник", "разбуди", "побудку", "аларм", "будик"],
        exclusion_keywords=[],
        requires_slot=["time"],
    ),
    "timer.start": SkillConfig(
        name="timer.start",
        base_weight=0.5,
        positive_keywords=["таймер", "засеки", "отсчет", "засечь"],
        exclusion_keywords=["будильник"],
        requires_slot=["duration"],
    ),
    "reminder.add": SkillConfig(
        name="reminder.add",
        base_weight=0.5,
        positive_keywords=["напомни", "напоминание", "не забыть", "запомни"],
        exclusion_keywords=["будильник", "таймер"],
    ),
    "weather.get": SkillConfig(
        name="weather.get",
        base_weight=0.5,
        positive_keywords=["погода", "прогноз", "градусов", "осадки", "дождь", "снег", "ветер"],
        exclusion_keywords=[],
    ),
    "music.stop": SkillConfig(
        name="music.stop",
        base_weight=0.5,
        positive_keywords=["выключи", "останови", "стоп", "пауза", "хватит"],
        exclusion_keywords=["будильник", "таймер"],
    ),
    "time.now": SkillConfig(
        name="time.now",
        base_weight=0.5,
        positive_keywords=["время", "час", "который час"],
        exclusion_keywords=["погода", "будильник", "напомни"],
    ),
    "news.get": SkillConfig(
        name="news.get",
        base_weight=0.5,
        positive_keywords=["новости", "сводки", "новенького"],
        exclusion_keywords=[],
    ),
    "jokes.tell": SkillConfig(
        name="jokes.tell",
        base_weight=0.5,
        positive_keywords=["анекдот", "шутка", "смешное", "прикол", "угарнем"],
        exclusion_keywords=[],
    ),
    "math.calculate": SkillConfig(
        name="math.calculate",
        base_weight=0.5,
        positive_keywords=["посчитай", "сколько будет", "плюс", "минус", "умножь", "раздели", "корень", "степень"],
        exclusion_keywords=[],
    ),
    "system.help": SkillConfig(
        name="system.help",
        base_weight=0.3,
        positive_keywords=["помощь", "что умеешь", "команды", "функции", "возможности", "как"],
        exclusion_keywords=[],
    ),
}


def get_skill_config(name: str) -> SkillConfig:
    return SKILL_CONFIGS.get(name, SkillConfig(name=name))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


@dataclass
class MaskResult:
    original: str
    masked: str
    slots: Dict[str, List[str]]


TIME_WORDS = r"(сегодня|завтра|послезавтра|утром|днём|днем|вечером|ночью)"
MONTHS = r"(январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[ея]|июн[ьяе]|июл[ьяе]|август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])"
UNITS = r"(секунд(?:а|ы)?|минут(?:а|ы)?|час(?:а|ов)?|дн(?:я|ей|ь))"
CITY_LIST = ["москва", "санкт-петербург", "казань", "берлин", "лондон", "париж", "екатеринбург", "новосибирск"]


def _store(slots, key, value, placeholder, canon_key=None, canon_val=None):
    slots.setdefault(key, []).append(value.strip())
    if canon_key:
        slots.setdefault(canon_key, []).append(canon_val)
    return placeholder


def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _merge_spans(spans):
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = []
    for s in spans:
        if not merged or s[0] >= merged[-1][1]:
            merged.append(s)
        else:
            if (s[1] - s[0]) > (merged[-1][1] - merged[-1][0]):
                merged[-1] = s
    return merged


def _city_regex(city: str):
    return re.compile(re.escape(city)[:-1] + r"[а-я]*", flags=re.I)


CITY_PATTERNS = [(c, _city_regex(c)) for c in CITY_LIST]


def mask_entities(text: str) -> MaskResult:
    slots: Dict[str, List[str]] = {}
    raw = text
    spans = []

    protected = []
    for m in re.finditer(r"\{[a-z_]+\}", raw):
        protected.append((m.start(), m.end()))

    def _is_protected(a, b):
        return any(not (b <= x or a >= y) for x, y in protected)

    for m in re.finditer(r"\b([01]?\d|2[0-3]):[0-5]\d\b", raw):
        if not _is_protected(m.start(), m.end()):
            spans.append((m.start(), m.end(), "{time}", "time"))

    for m in re.finditer(r"\b([0-3]?\d\.[01]?\d\.\d{4})\b", raw):
        if not _is_protected(m.start(), m.end()):
            spans.append((m.start(), m.end(), "{date}", "date"))

    for m in re.finditer(rf"\b\d+\s+{UNITS}\b", raw, flags=re.I):
        if not _is_protected(m.start(), m.end()):
            spans.append((m.start(), m.end(), "{duration}", "duration"))

    for m in re.finditer(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", raw):
        a, b = m.start(), m.end()
        if not any(a < y and b > x for x, y, _, _ in spans) and not _is_protected(a, b):
            spans.append((a, b, "{number}", "number"))

    music_m = re.search(r"(включи|воспроизведи|сыграй|поставь)\s+(.+)", raw, flags=re.I)
    if music_m and not _is_protected(music_m.start(2), music_m.end(2)):
        tail = music_m.group(2).lower()
        if not any(kw in tail for kw in ["будильник", "таймер"]):
            spans.append((music_m.start(2), music_m.end(2), "{song}", "song"))

    for _, pat in CITY_PATTERNS:
        for m in pat.finditer(raw):
            if not _is_protected(m.start(), m.end()):
                spans.append((m.start(), m.end(), "{city}", "city"))

    spans = _merge_spans(spans)
    out = list(raw)
    extracted: Dict[str, List[str]] = {}
    for a, b, ph, slot in sorted(spans, key=lambda t: -t[0]):
        val = raw[a:b]
        _store(extracted, slot, val, ph, f"{slot}_canon", _canon(val))
        out[a:b] = list(ph)

    masked = "".join(out)
    masked = re.sub(r"\s+", " ", masked).strip()
    return MaskResult(original=text, masked=masked, slots=extracted)


class NLUEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__()
        cfg = Config()
        self.emb = nn.Embedding(vocab_size, cfg.EMB_DIM, padding_idx=cfg.PAD_IDX)
        self.conv = nn.Conv1d(cfg.EMB_DIM, cfg.CNN_CHANNELS, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(64)
        self.bi_lstm = nn.LSTM(cfg.CNN_CHANNELS, cfg.LSTM_HIDDEN, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * cfg.LSTM_HIDDEN, cfg.PROJ_DIM)
        self.cls = nn.Linear(2 * cfg.LSTM_HIDDEN, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = F.relu(self.conv(e))
        c = self.pool(c).transpose(1, 2)
        out, _ = self.bi_lstm(c)
        feat = self.dropout(out.mean(dim=1))
        logits = self.cls(feat)
        z = F.normalize(self.proj(feat), dim=-1)
        return logits, z


@dataclass
class LBDEntry:
    text: str
    skill: Optional[str] = None
    is_negative: bool = False
    excludes_skill: Optional[str] = None
    id: Optional[int] = None


class LBDVectorStore:
    def __init__(self):
        if faiss is None:
            raise ImportError(f"faiss-cpu не установлен: {FAISS_IMPORT_ERROR}")
        self.positive_entries: List[LBDEntry] = []
        self.positive_masked: List[str] = []
        self.negative_entries: Dict[str, List[LBDEntry]] = {}
        self.negative_masked: Dict[str, List[str]] = {}
        self.pos_index = None
        self.neg_indices = {}

    def add_data(self, entries: List[LBDEntry]):
        for e in entries:
            if e.is_negative:
                self.negative_entries.setdefault(e.excludes_skill or "", []).append(e)
                self.negative_masked.setdefault(e.excludes_skill or "", []).append(mask_entities(e.text).masked)
            else:
                self.positive_entries.append(e)
                self.positive_masked.append(mask_entities(e.text).masked)

    def build_indices(self, embed_fn, dim: int = 128):
        if self.positive_masked:
            vecs = embed_fn(self.positive_masked)
            faiss.normalize_L2(vecs)
            self.pos_index = faiss.IndexFlatIP(dim)
            self.pos_index.add(vecs)

        for skill, masks in self.negative_masked.items():
            vecs = embed_fn(masks)
            faiss.normalize_L2(vecs)
            idx = faiss.IndexFlatIP(dim)
            idx.add(vecs)
            self.neg_indices[skill] = idx

    def search(self, query_vec, k: int = 5, neg_weight: float = 1.5, neg_margin: float = 0.1):
        if self.pos_index is None:
            return []
        D, I = self.pos_index.search(query_vec, k)
        results = []
        for sim, idx in zip(D[0], I[0]):
            entry = self.positive_entries[int(idx)]
            cand = {"entry": entry, "sim": float(sim), "rejected": False, "reason": None}
            skill = entry.skill
            if skill in self.neg_indices:
                Dn, In = self.neg_indices[skill].search(query_vec, 1)
                neg_sim = float(Dn[0][0])
                if neg_sim * neg_weight > float(sim) + neg_margin:
                    cand["rejected"] = True
                    neg_text = self.negative_entries[skill][int(In[0][0])].text
                    cand["reason"] = f"Too close to negative: '{neg_text}' ({neg_sim:.2f})"
            results.append(cand)
        return results


@dataclass
class IntentResult:
    query: str
    skill: str
    confidence: float
    slots: Dict[str, List[str]]
    matched_example: Optional[str]
    candidates: List[Dict[str, Any]]
    rejected: List[str]


class IntentDetector:
    def __init__(self, model, vector_store: LBDVectorStore, config: Config, stoi, id2label, device):
        self.model = model
        self.vector_store = vector_store
        self.cfg = config
        self.stoi = stoi
        self.id2label = id2label
        self.device = device
        self.skill_weights = {n: c.base_weight for n, c in SKILL_CONFIGS.items()}

    def detect(self, query: str) -> IntentResult:
        m = mask_entities(query)
        masked = m.masked
        self.model.eval()
        with torch.no_grad():
            ids = [self.stoi.get(ch, self.cfg.UNK_IDX) for ch in masked]
            ids = [self.cfg.BOS_IDX] + ids[: self.cfg.MAX_LEN - 2] + [self.cfg.EOS_IDX]
            if len(ids) < self.cfg.MAX_LEN:
                ids += [self.cfg.PAD_IDX] * (self.cfg.MAX_LEN - len(ids))
            x = torch.tensor([ids], dtype=torch.long).to(self.device)
            logits, z = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            q_vec = z.cpu().numpy().astype("float32")
            clf_id = int(probs.argmax())
            clf_skill = self.id2label[clf_id]
            clf_prob = float(probs[clf_id])

        candidates = self.vector_store.search(
            q_vec,
            k=self.cfg.FAISS_K,
            neg_weight=self.cfg.NEGATIVE_WEIGHT,
            neg_margin=self.cfg.NEGATIVE_MARGIN,
        )

        final_candidates = []
        rejected_list = []
        for c in candidates:
            skill = c["entry"].skill
            if c.get("rejected"):
                rejected_list.append(f"{skill} ({c.get('reason')})")
                continue
            s_cfg = get_skill_config(skill or "")
            if any(exc in query.lower() for exc in s_cfg.exclusion_keywords):
                rejected_list.append(f"{skill} (keyword exclusion)")
                continue
            is_clf_match = clf_skill == skill
            prob_feature = clf_prob if is_clf_match else (1.0 - clf_prob)
            score = (
                self.cfg.RANK_ALPHA * float(c["sim"])
                + self.cfg.RANK_BETA * float(prob_feature)
                + self.cfg.RANK_GAMMA * self.skill_weights.get(skill, 0.5)
            )
            c = dict(c)
            c["score"] = float(score)
            final_candidates.append(c)

        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        if not final_candidates or final_candidates[0]["score"] < self.cfg.THRESHOLD:
            return IntentResult(
                query=query,
                skill="system.help",
                confidence=0.0,
                slots=m.slots,
                matched_example=None,
                candidates=final_candidates,
                rejected=rejected_list,
            )

        best = final_candidates[0]
        return IntentResult(
            query=query,
            skill=best["entry"].skill,
            confidence=float(best["score"]),
            slots=m.slots,
            matched_example=best["entry"].text,
            candidates=final_candidates,
            rejected=rejected_list,
        )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_jsonl_from_url(url: str) -> List[Dict[str, Any]]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def load_dataset(path: str | Path, fallback_url: Optional[str] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.exists():
        return load_jsonl(path)
    if fallback_url:
        return load_jsonl_from_url(fallback_url)
    raise FileNotFoundError(f"Не найден датасет: {path}")


def build_vocab(samples_list: List[Dict[str, Any]]):
    charset = set()
    cfg = Config()
    for sample in samples_list:
        m = mask_entities(sample["text"])
        charset.update(list(m.masked))
    itos = cfg.SPECIALS + sorted(charset)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    entry = candidate.get("entry")
    return {
        "intent": getattr(entry, "skill", None),
        "example": getattr(entry, "text", None),
        "score": float(candidate.get("score", 0.0)),
        "similarity": float(candidate.get("sim", 0.0)),
        "rejected": bool(candidate.get("rejected", False)),
        "reason": candidate.get("reason"),
    }


class NLURuntime:
    def __init__(
        self,
        model_path: str | Path = "best_model.pt",
        train_path: str | Path = "lbd_train_augmented.jsonl",
        dev_path: str | Path = "lbd_dev_augmented.jsonl",
        threshold: Optional[float] = None,
    ):
        seed_everything(Config.SEED)
        self.cfg = Config()
        if threshold is not None:
            self.cfg.THRESHOLD = float(threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_url = "https://raw.githubusercontent.com/Fla1lx/neural-network-module-for-determining-user-intent/main"
        self.rows = load_dataset(train_path, f"{base_url}/lbd_train_augmented.jsonl")
        self.dev_rows = load_dataset(dev_path, f"{base_url}/lbd_dev_augmented.jsonl")

        self.itos, self.stoi = build_vocab(self.rows + self.dev_rows)
        self.skills = sorted(list({r["skill"] for r in self.rows}))
        self.label2id = {s: i for i, s in enumerate(self.skills)}
        self.id2label = {i: s for s, i in self.label2id.items()}

        self.model = NLUEncoder(len(self.itos), len(self.skills)).to(self.device)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели {model_path}. Скопируй best_model.pt в корень проекта интерфейса."
            )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.lbd_store = LBDVectorStore()
        initial_entries = [LBDEntry(text=r["text"], skill=r["skill"], is_negative=False) for r in self.rows]
        initial_entries.append(LBDEntry(text="поставь будильник как песню", is_negative=True, excludes_skill="music.play"))
        self.lbd_store.add_data(initial_entries)
        self.lbd_store.build_indices(embed_fn=self.get_embeddings, dim=self.cfg.PROJ_DIM)
        self.detector = IntentDetector(
            self.model,
            self.lbd_store,
            self.cfg,
            self.stoi,
            self.id2label,
            self.device,
        )

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch_texts = texts[i : i + 64]
                ids_list = []
                for t in batch_texts:
                    ids = [self.stoi.get(ch, self.cfg.UNK_IDX) for ch in t]
                    ids = [self.cfg.BOS_IDX] + ids[: self.cfg.MAX_LEN - 2] + [self.cfg.EOS_IDX]
                    if len(ids) < self.cfg.MAX_LEN:
                        ids += [self.cfg.PAD_IDX] * (self.cfg.MAX_LEN - len(ids))
                    ids_list.append(ids)
                x = torch.tensor(ids_list, dtype=torch.long).to(self.device)
                _, z = self.model(x)
                vecs.append(z.cpu().numpy())
        return np.vstack(vecs).astype("float32")

    def rank_skill(self, query: str) -> Dict[str, Any]:
        res = self.detector.detect(query)
        masked = mask_entities(query).masked
        return {
            "intent": res.skill,
            "confidence": float(res.confidence),
            "candidates": res.candidates,
            "slots": res.slots,
            "abstain": res.skill == "system.help",
            "query": res.query,
            "masked": masked,
            "matched_example": res.matched_example,
            "rejected": res.rejected,
        }

    def predict_for_ui(self, text: str, expected_skill: Optional[str] = None) -> Dict[str, Any]:
        result = self.rank_skill(text)
        confidence = float(result.get("confidence", 0.0))
        if confidence >= 0.80:
            status = "OK"
        elif confidence >= 0.60:
            status = "Погранично"
        else:
            status = "Низкая уверенность"

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "text": text,
            "expected_skill": expected_skill,
            "predicted_skill": result["intent"],
            "is_correct": None if expected_skill in (None, "") else result["intent"] == expected_skill,
            "confidence": confidence,
            "masked_input": result.get("masked"),
            "slots": result.get("slots", {}),
            "top_candidates": [normalize_candidate(c) for c in (result.get("candidates") or [])[:5]],
            "matched_example": result.get("matched_example"),
            "rejected": result.get("rejected", []),
            "status": status,
        }
        return record
