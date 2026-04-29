# Отдельный интерфейс диагностики детектора намерений

Этот интерфейс не зависит от Jupyter Notebook. Он:

1. читает результаты апробации из папки `diagnostics/`;
2. загружает сохранённую модель `best_model.pt`;
3. строит FAISS-индекс по обучающим данным;
4. выполняет ручные тесты во вкладке интерфейса;
5. сохраняет ручные тесты в `diagnostics/manual_test_logs.jsonl`.

## Что положить рядом с `app.py`

```text
intent_interface_app/
├── app.py
├── best_model.pt
├── lbd_train_augmented.jsonl          # желательно, но если нет — приложение попробует скачать из GitHub
├── lbd_dev_augmented.jsonl            # желательно, но если нет — приложение попробует скачать из GitHub
├── diagnostics/
│   ├── metrics.json
│   ├── f1_by_intent.jsonl
│   ├── confusion_matrix.json
│   └── dev_predictions.jsonl
├── src/
│   ├── diagnostics_io.py
│   └── nlu_runtime.py
└── requirements.txt
```

## Запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Разделы интерфейса

- `Обзор` — KPI, F1 по интентам, матрица ошибок, последние dev-предсказания.
- `Метрики` — полный JSON метрик и таблица precision / recall / f1.
- `Матрица ошибок` — heatmap и таблица confusion matrix.
- `Dev-предсказания` — подробные диагностические записи на dev-выборке.
- `Ручные тесты` — live-инференс через `best_model.pt` и FAISS.
- `Логи ручных тестов` — записи, созданные уже самим интерфейсом.

## Важно

`manual_test_logs.jsonl` не создаётся в ноутбуке. Его создаёт интерфейс, когда пользователь запускает ручные тесты.
