from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from src.diagnostics_io import read_json, read_jsonl, append_jsonl
from src.nlu_runtime import NLURuntime


ROOT = Path(__file__).parent
DIAGNOSTICS_DIR = ROOT / "diagnostics"
MODEL_PATH = ROOT / "best_model.pt"
TRAIN_PATH = ROOT / "lbd_train_augmented.jsonl"
DEV_PATH = ROOT / "lbd_dev_augmented.jsonl"
MANUAL_LOGS_PATH = DIAGNOSTICS_DIR / "manual_test_logs.jsonl"

st.set_page_config(
    page_title="Диагностика детектора намерений",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource(show_spinner="Загрузка модели и FAISS-индекса...")
def load_runtime(threshold: float | None):
    return NLURuntime(
        model_path=MODEL_PATH,
        train_path=TRAIN_PATH,
        dev_path=DEV_PATH,
        threshold=threshold,
    )


def fmt_percent(x, digits=1):
    if x is None:
        return "—"
    return f"{float(x) * 100:.{digits}f}%"


def top_candidates_to_text(candidates):
    if not isinstance(candidates, list):
        return ""
    parts = []
    for c in candidates[:3]:
        intent = c.get("intent")
        score = c.get("score", 0)
        parts.append(f"{intent} ({float(score):.3f})")
    return ", ".join(parts)


def status_badge(status: str):
    if status == "OK":
        return "🟢 OK"
    if status == "Погранично":
        return "🟡 Погранично"
    return "🔴 Низкая уверенность"


st.sidebar.title("IntentDiag")
st.sidebar.caption("Диагностический интерфейс NLU-модуля")
page = st.sidebar.radio(
    "Раздел",
    ["Обзор", "Метрики", "Матрица ошибок", "Dev-предсказания", "Ручные тесты", "Логи ручных тестов"],
)

st.sidebar.divider()
st.sidebar.subheader("Параметры модели")
custom_threshold_enabled = st.sidebar.checkbox("Переопределить threshold", value=False)
threshold = None
if custom_threshold_enabled:
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.4, 0.01)

st.sidebar.caption("Модель загружается из `best_model.pt`. Метрики читаются из `diagnostics/`.")

metrics = read_json(DIAGNOSTICS_DIR / "metrics.json", default={}) or {}
f1_rows = read_jsonl(DIAGNOSTICS_DIR / "f1_by_intent.jsonl")
cm_data = read_json(DIAGNOSTICS_DIR / "confusion_matrix.json", default={}) or {}
dev_predictions = read_jsonl(DIAGNOSTICS_DIR / "dev_predictions.jsonl")
manual_logs = read_jsonl(MANUAL_LOGS_PATH)

st.title("Диагностика детектора намерений")
st.caption("Апробация модели, метрики качества, матрица ошибок и ручные тесты")

if not MODEL_PATH.exists():
    st.warning("Файл `best_model.pt` не найден рядом с `app.py`. Ручные тесты не смогут работать, пока модель не будет скопирована в папку интерфейса.")

if not DIAGNOSTICS_DIR.exists():
    st.info("Папка `diagnostics/` не найдена. Сначала запусти диагностический блок в ноутбуке и скопируй папку diagnostics рядом с app.py.")


if page == "Обзор":
    st.subheader("Нынешние результаты апробации")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", fmt_percent(metrics.get("accuracy")))
    c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}" if metrics.get("macro_f1") is not None else "—")
    c3.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.3f}" if metrics.get("weighted_f1") is not None else "—")
    c4.metric("Проверено", metrics.get("total_requests", "—"))

    col_left, col_right = st.columns([1.05, 1])

    with col_left:
        st.subheader("F1 score по интентам")
        if f1_rows:
            f1_df = pd.DataFrame(f1_rows).sort_values("f1", ascending=True)
            fig = px.bar(
                f1_df,
                x="f1",
                y="intent",
                orientation="h",
                hover_data=["precision", "recall", "support"],
                range_x=[0, 1],
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Файл `f1_by_intent.jsonl` пока не найден.")

    with col_right:
        st.subheader("Матрица ошибок")
        if cm_data.get("labels") and cm_data.get("matrix"):
            labels = cm_data["labels"]
            fig = px.imshow(
                cm_data["matrix"],
                x=labels,
                y=labels,
                text_auto=True,
                labels=dict(x="Предсказанный класс", y="Истинный класс", color="Кол-во"),
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Файл `confusion_matrix.json` пока не найден.")

    st.subheader("Последние dev-предсказания")
    if dev_predictions:
        preview = pd.DataFrame(dev_predictions)
        cols = ["text", "expected_skill", "predicted_skill", "confidence", "status", "is_correct", "matched_example"]
        cols = [c for c in cols if c in preview.columns]
        st.dataframe(preview[cols].tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("Файл `dev_predictions.jsonl` пока не найден.")


elif page == "Метрики":
    st.subheader("Общие метрики")
    if metrics:
        st.json(metrics)
    else:
        st.info("Файл `metrics.json` пока не найден.")

    st.subheader("Precision / Recall / F1 по интентам")
    if f1_rows:
        f1_df = pd.DataFrame(f1_rows).sort_values("f1", ascending=False)
        st.dataframe(f1_df, use_container_width=True, hide_index=True)
        fig = px.bar(
            f1_df,
            x="intent",
            y=["precision", "recall", "f1"],
            barmode="group",
            range_y=[0, 1],
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Файл `f1_by_intent.jsonl` пока не найден.")


elif page == "Матрица ошибок":
    st.subheader("Матрица ошибок")
    if cm_data.get("labels") and cm_data.get("matrix"):
        labels = cm_data["labels"]
        cm_df = pd.DataFrame(cm_data["matrix"], index=labels, columns=labels)
        fig = px.imshow(
            cm_df,
            text_auto=True,
            labels=dict(x="Предсказанный класс", y="Истинный класс", color="Кол-во"),
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Файл `confusion_matrix.json` пока не найден.")


elif page == "Dev-предсказания":
    st.subheader("Подробные предсказания на dev-выборке")
    if dev_predictions:
        df = pd.DataFrame(dev_predictions)
        col_a, col_b, col_c = st.columns(3)
        only_errors = col_a.checkbox("Показать только ошибки")
        low_conf = col_b.checkbox("Показать низкую уверенность")
        min_conf = col_c.slider("Минимальная уверенность", 0.0, 1.0, 0.0, 0.01)

        filtered = df.copy()
        if only_errors and "is_correct" in filtered.columns:
            filtered = filtered[filtered["is_correct"] == False]
        if low_conf and "status" in filtered.columns:
            filtered = filtered[filtered["status"] != "OK"]
        if "confidence" in filtered.columns:
            filtered = filtered[filtered["confidence"].astype(float) >= min_conf]

        if "top_candidates" in filtered.columns:
            filtered["top_candidates_text"] = filtered["top_candidates"].apply(top_candidates_to_text)

        cols = [
            "text", "expected_skill", "predicted_skill", "confidence", "status", "is_correct",
            "masked_input", "matched_example", "top_candidates_text",
        ]
        cols = [c for c in cols if c in filtered.columns]
        st.dataframe(filtered[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Файл `dev_predictions.jsonl` пока не найден.")


elif page == "Ручные тесты":
    st.subheader("Ручной тест детектора намерений")
    st.write("Здесь интерфейс загружает `best_model.pt`, строит FAISS-индекс по обучающим данным и выполняет инференс без участия ноутбука.")

    with st.form("manual_test_form"):
        text = st.text_area("Текст запроса", value="поставь будильник на 7:30", height=90)
        expected_skill = st.selectbox(
            "Ожидаемый интент, опционально",
            [""] + sorted([r.get("intent") for r in f1_rows if r.get("intent")]) if f1_rows else ["", "music.play", "music.stop", "alarm.set", "timer.start", "reminder.add", "weather.get", "time.now", "news.get", "jokes.tell", "math.calculate", "system.help"],
        )
        save_log = st.checkbox("Сохранить результат в manual_test_logs.jsonl", value=True)
        submitted = st.form_submit_button("Запустить тест")

    if submitted:
        if not text.strip():
            st.error("Введите текст запроса.")
        else:
            try:
                runtime = load_runtime(threshold)
                result = runtime.predict_for_ui(text.strip(), expected_skill or None)
                if save_log:
                    append_jsonl(MANUAL_LOGS_PATH, result)
                    st.success(f"Результат сохранён: {MANUAL_LOGS_PATH}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Предсказанный скилл", result["predicted_skill"])
                c2.metric("Уверенность", f"{result['confidence']:.3f}")
                c3.metric("Статус", result["status"])

                st.write("**Masked input**")
                st.code(result.get("masked_input") or "")

                st.write("**Извлечённые сущности**")
                st.json(result.get("slots", {}))

                st.write("**Пример из выученных данных**")
                st.info(result.get("matched_example") or "Ближайший пример не найден")

                st.write("**Top-кандидаты**")
                candidates = pd.DataFrame(result.get("top_candidates", []))
                if not candidates.empty:
                    st.dataframe(candidates, use_container_width=True, hide_index=True)
                    fig = px.bar(candidates, x="intent", y="score", hover_data=["example", "similarity"], range_y=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Кандидаты не найдены.")

                if result.get("rejected"):
                    st.write("**Отклонённые кандидаты**")
                    st.write(result["rejected"])

            except Exception as exc:
                st.exception(exc)


elif page == "Логи ручных тестов":
    st.subheader("Логи ручных тестов")
    if manual_logs:
        df = pd.DataFrame(manual_logs)
        if "top_candidates" in df.columns:
            df["top_candidates_text"] = df["top_candidates"].apply(top_candidates_to_text)
        cols = [
            "created_at", "text", "expected_skill", "predicted_skill", "confidence", "status", "is_correct",
            "masked_input", "matched_example", "top_candidates_text",
        ]
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols].sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Логи ручных тестов пока пустые. Запусти тест во вкладке `Ручные тесты` и включи сохранение результата.")
