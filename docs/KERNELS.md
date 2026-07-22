# Оптимизация Mamba-3 на RX 9070 XT

Этот документ фиксирует расследование issue #13: измеренный горячий путь,
проверенные альтернативы burn-mamba и границы результата. Все GPU-эксперименты
запускались в GitHub Actions на RX 9070 XT 16 GB. Локально GPU-тесты не
выполнялись.

## Воспроизводимый бенчмарк

`examples/train_bench.rs` выполняет настоящий optimizer step `tiny-turbo`:

- полный словарь 32 768 и настоящий language-model head/loss;
- `seq_len=1024`, `micro_batch=6`, `accum=8`, Muon и fp32, как в ночном
  обучении из issue;
- синтетические детерминированные токены вместо чтения корпуса;
- один полный warm-up step для fusion/autotune, затем один синхронизированный
  измеряемый step.

В измеряемую область входят восемь forward/backward micro-batches и шаг
оптимизатора. Один такой step содержит 49 152 токена и длится меньше минуты;
долгий первый запуск — это компиляция и autotune, а не само измерение.

```sh
cargo run --release --no-default-features --features rocm \
  --example train_bench -- --micro-batch 6 --accum 8 --warmup 1 --steps 1 \
  --dtype f32 --ssd recalculated --checkpointing true --muon true
```

Абсолютная скорость короткого синтетического запуска ниже установившихся
~6000 tok/s ночного обучения: единственный warm-up не прогревает долгоживущий
процесс так же, как тысячи шагов. Поэтому вывод об ускорении основан на парном
A/B в одном job и на одинаковой loss, а не на сравнении этих двух разных
запусков.

## Найденная причина: двойной recompute

В исходной конфигурации одновременно работали два независимых механизма:

1. `device.gradient_checkpointing()` сохраняет вход блока и повторяет весь блок
   в backward;
2. `Mamba3SsdPath::SerialRecalculated` имеет собственный custom backward,
   который ещё раз восстанавливает промежуточные результаты пяти стадий SSD.

На `tiny-turbo` внешний checkpointing нужен, чтобы `micro_batch=6` надёжно
помещался. Но внутренний recompute burn-mamba уже не нужен: обычный
`Mamba3SsdPath::Serial` оставляет autodiff управлять chunkwise-графом. Параметры,
forward и математические градиенты у режимов одинаковы; меняется только
компромисс память/повторные вычисления.

## Результат ROCm

[GitHub Actions run 29901656398](https://github.com/uselessgoddess/quasar/actions/runs/29901656398)
на `gfx1201`, один и тот же commit и входы:

| SSD backward | checkpointing | loss | step | throughput | effective |
| --- | ---: | ---: | ---: | ---: | ---: |
| `recalculated` | да | 10.2408 | 13.591 s | 3 616 tok/s | 1.75 TFLOP/s |
| `serial` | да | 10.2408 | 10.959 s | 4 485 tok/s | 2.17 TFLOP/s |

`serial` даёт **+24.0% tok/s** и сокращает step на **19.4%**. Кандидат прошёл
полный forward, backward и Muon/AdamW step при `micro_batch=6` на 16-GB карте,
то есть это не оценка отдельного kernel и не inference-only результат.

## Подтверждение на Vulkan и выбор варианта

[GitHub Actions run 29903226239](https://github.com/uselessgoddess/quasar/actions/runs/29903226239)
повторил полный step на том же `gfx1201` через Vulkan:

| SSD backward | loss | step | throughput | effective |
| --- | ---: | ---: | ---: | ---: |
| `recalculated` | 10.2460 | 10.169 s | 4 834 tok/s | 2.34 TFLOP/s |
| `serial` | 10.2460 | 7.834 s | 6 274 tok/s | 3.04 TFLOP/s |
| `minimal` | 10.2460 | 7.786 s | 6 313 tok/s | 3.06 TFLOP/s |

`serial` повторяет эффект на втором backend: **+29.8% tok/s**, step короче на
**23.0%**. `minimal` быстрее `serial` лишь на 0.62% в одном измерении — это
меньше надёжного сигнала для смены алгоритма, при этом он хранит более крупный
autodiff-граф. Поэтому measured-default `tiny-turbo` — `serial`, а `minimal`
остаётся доступным только для явных экспериментов.

В quasar добавлен `--ssd`; `tiny-turbo` выбирает `serial` автоматически.
`--ssd recalculated` остаётся безопасным fallback для более крупных форм или
micro-batch, которым не хватает памяти. Другие пресеты не меняют прежний
memory-saving default.

## Отрицательные результаты

Они важны, потому что не дают превратить гипотезу в опасный default.

- `bf16` и `f16` на закреплённых Burn/CubeCL и ROCm завершились ещё при
  компиляции ядра: LLVM не смог выбрать RDNA4-инструкции
  `llvm.amdgcn.wmma.f32.16x16x16.{bf16,f16}`. Поэтому простое переключение dtype
  не является рабочим AMP и в production не включено.
- fp32 `recalculated` без внешнего checkpointing исчерпал VRAM и завершился в
  HIP `vmheap.cpp:175 MapPhysMemory`. Значит, отключить checkpointing для
  `micro_batch=6` нельзя.
- Ручной CubeCL RMSNorm-прототип удалён. Он не был подключён к модели, проверял
  только forward на CPU и занимал 5–9 минут LLVM-JIT сборки. GPU backend Burn
  уже fusion-компилирует поэлементную цепочку RMSNorm, поэтому прототип не
  подтверждал и не давал ускорения обучения.

## Почему остаётся далеко до 48 TFLOP/s

48 TFLOP/s — пик больших плотных матричных умножений. `tiny-turbo` узкая
(`d_model=512`, `d_inner=1024`, `head_dim=64`), а SSD состоит из многих
транспозиций, broadcast/exp/cumsum и небольших batched matmul. Этот путь
ограничен трафиком памяти и launch overhead задолго до ALU-пика. Устранение
лишнего backward поднимает полезную загрузку с 1.75 до 2.17 TFLOP/s на ROCm и
с 2.34 до 3.04 TFLOP/s на Vulkan, но не меняет форму этих операций.

Следующий крупный шаг потребовал бы fused SSD forward **и custom backward** в
burn-mamba/CubeCL. Это не локальная замена одного expression: ядро должно
реализовать chunk recurrence, causal mask, Mamba-3 diagonal correction, MIMO и
градиенты всех входов. Без отдельной проверки параметров после optimizer step
ошибка может молча портить многочасовое обучение. Поэтому этот PR сначала
использует уже существующую и проверенную burn-mamba формулировку `Serial`, а
такой rewrite оставляет отдельной upstream-работой.

## Как применять

Для измеренной конфигурации RX 9070 XT:

```sh
cargo run --release --no-default-features --features rocm -- \
  train tiny-turbo --data data/shards --out runs/turbo-24h \
  --micro-batch 6 --accum 8 --muon true
```

У этого пресета `serial` уже включён; явное `--ssd serial` эквивалентно.

Если форма стала больше или backend сообщает OOM, верните память ценой
скорости:

```sh
--ssd recalculated
```

Обычный `release` намеренно не делает thin LTO: время host-link не влияет на
GPU step, но замедляет каждый эксперимент. Для редкой финальной сборки есть
отдельный `--profile release-lto`.
