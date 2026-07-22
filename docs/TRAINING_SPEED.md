# Разбор скорости обучения

Документ связывает исходное наблюдение issue #7 с измерениями RX 9070 XT из
issue #13. Здесь optimizer step, token budget и wall-clock считаются отдельно:
уменьшить число micro-batches в шаге — не ускорение, если вместе с ним незаметно
уменьшилось число обучающих токенов.

## Что означало `40 / 60000`

Burn TUI считал optimizer steps. При `micro_batch=1`, `accum=96` и
`seq_len=2048` один шаг содержит 196 608 токенов. Измеренные 1 700 tok/s дают
`1700 × 3600 / 196608 = 31.1` шага в час, поэтому около 40 первых шагов — не
зависание и не ошибка progress bar.

Старые 60 000 шагов означали 11.80B токенов с этой формой batch и около 80.3
суток при постоянной скорости. Default сокращён до 12 500 шагов: для `tiny`
это 3.2768B токенов (`8 × 16 × 2048`), близко к 20 токенам на параметр. Startup
теперь печатает steps, tokens/step и полный token budget до первой аллокации.

## Найденные причины

1. Старый schedule планировал почти в пять раз больше compute-efficient token
   budget.
2. Logging читал scalar loss после каждого micro-batch и каждый раз
   синхронизировал host/device. Теперь loss суммируется на device и читается
   один раз за окно.
3. Burn/burn-mamba в закреплённых ревизиях не имеют AMP. Обучение идёт в fp32;
   pure bf16/f16 не является безопасной заменой autocast + GradScaler.
4. Внешний Burn checkpointing повторял целый block, а внутренний
   burn-mamba `SerialRecalculated` повторял SSD ещё раз.
5. CubeCL удерживал до 128 Fusion streams и вместе с ними live buffers. Один
   stream оказался и быстрее, и стабильнее по VRAM.
6. 512-wide модель запускала много узких GEMM и 20 последовательных слоёв.
   Форма 640 × 12 сохранила параметр/FLOP budget, но лучше загрузила GPU.

Muon не объясняет разницу порядков величины: ортогонализация вызывается один раз
после всех accumulated forward/backward. `--muon false` добавляет второй moment
AdamW и расходует больше памяти; без отдельного A/B это не shortcut.

## Измерения RX 9070 XT

Все варианты — настоящий synchronized optimizer step с Muon, fp32 и полным
language-model loss. Короткие эксперименты выполнялись в GitHub Actions, а
полная методика и ссылки на runs находятся в [`KERNELS.md`](KERNELS.md).

| изменение | matched baseline | candidate | эффект |
| --- | ---: | ---: | ---: |
| SSD `recalculated → serial`, ROCm | 3 616 | 4 485 tok/s | +24.0% |
| SSD `recalculated → serial`, Vulkan | 4 834 | 6 274 tok/s | +29.8% |
| checkpoint on, 6×8 → off, 4×12 | 7 359 | 8 406 tok/s | +14.2% |
| CubeCL streams 128 → 1 | 8 406 | 8 608 tok/s | +2.4% |
| shape 512×20 → 640×12 | 8 709 | 9 457 tok/s | +8.6% |

Итоговая production-форма в повторе дала 9 433 tok/s. `tiny-turbo` поэтому
выбирает 640 × 12, micro-batch 4, `serial`, checkpointing off и один CubeCL
stream. Accumulation по умолчанию поднят до 32: effective batch остаётся
`4 × 32 × 1024 = 131072`, то есть скорость не куплена сокращением обучения.

Увеличивать micro-batch выше измеренного нельзя по принципу «раз помещается»:
6×8 без checkpointing не завершился OOM, но упал до 2 715 tok/s из-за memory
pressure. Пик VRAM и steady throughput нужно проверять вместе.

## Precision и profiler

На Vulkan bf16 отвергнут backend'ом, а f16 попал в panic Burn fusion и получил
non-finite loss. На ROCm оба reduced dtype ранее падали при выборе RDNA4 WMMA.
Поэтому production остаётся fp32. Это согласуется с отсутствием AMP в Burn, но
решение основано именно на конечном full-step probe, а не на предположении.

CubeCL profiler насчитал 9 948 kernel launches даже для одного micro-batch.
Matmul занимает около 79% записанного GPU time, а elementwise/reduce/slice/copy
дают тысячи коротких dispatch. Поэтому оставшийся большой резерв — fused
Mamba-3 forward вместе с custom backward в burn-mamba/CubeCL, не локальная
замена RMSNorm. Профилирование сериализует launches, и его 266 tok/s не является
throughput-замером.

## Wall-clock

Для формы из issue (`4 × 12 × 1024 = 49152` токена/step) 12 500 шагов — 614.4M
токенов. Скриншотные 5 966 tok/s означают 28.6 часа; 9 457 tok/s — 18.0 часа.
Цель 9 часов требует 18 963 tok/s, 8 часов — 21 333 tok/s. Текущий результат
сокращает запуск примерно на 10.6 часа, но до заявленного overnight target ещё
нужен примерно двукратный выигрыш.

Production default обрабатывает 1.6384B токенов за те же 12 500 шагов, поэтому
при 9 457 tok/s занимает около 48.1 часа. Для короткого issue-budget можно явно
задать `--accum 12`; startup сразу покажет, что полный token budget изменился.

Сравнивать модели и реализации нужно по tok/s при одинаковом числе токенов,
effective TFLOP/s и bits-per-byte. Число optimizer steps без batch shape этих
трёх вопросов не отвечает.
