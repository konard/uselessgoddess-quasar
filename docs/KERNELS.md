# Оптимизация Mamba-3 на RX 9070 XT

Этот документ фиксирует расследование issue #13: воспроизводимый training-step,
парные измерения на RX 9070 XT 16 GB, профиль CubeCL и пошаговый порт
burn-mamba kernels с correctness- и performance-контролем. Все GPU-замеры
выполнены в GitHub Actions на `gfx1201`; локально GPU не использовался.

## Бенчмарк и правило сравнения

`examples/train_bench.rs` выполняет настоящий optimizer step: полный словарь
32 768, language-model loss, backward, накопление градиента и Muon/AdamW.
Только чтение корпуса и запись checkpoint исключены. Токены синтетические и
детерминированные, `device.sync()` стоит после каждого полного шага.

Каждый вариант получает один warm-up и три измеряемых шага; результат — медиана.
Первый measured step иногда всё ещё оплачивает autotune, поэтому одиночный шаг
не считается устойчивым результатом. В парных A/B сохраняется одинаковое число
токенов на optimizer step — 49 152 (`seq_len=1024`), даже когда меняются
`micro_batch` и `accum`. Каждое измеряемое окно короче минуты.

Production-рецепт можно проверить так:

```sh
cargo run --release --no-default-features --features vulkan \
  --example train_bench -- --model tiny-turbo --micro-batch 4 --accum 32 \
  --warmup 1 --steps 3 --dtype f32 --ssd serial \
  --checkpointing false --muon true
```

Для точного повтора issue-batch замените `--accum 32` на `--accum 12`.

## 1. Двойной recompute

В исходной форме одновременно работали два механизма:

1. `device.gradient_checkpointing()` повторял весь `Block` в backward;
2. burn-mamba `SerialRecalculated` повторно строил пять стадий SSD в своём
   custom backward.

Первый парный A/B оставил внешний checkpointing и заменил только SSD backward.
Loss совпал внутри каждого backend:

| backend | SSD | tok/s | изменение |
| --- | --- | ---: | ---: |
| ROCm, [run 29901656398](https://github.com/uselessgoddess/quasar/actions/runs/29901656398) | `recalculated` | 3 616 | — |
| ROCm | `serial` | 4 485 | +24.0% |
| Vulkan, [run 29903226239](https://github.com/uselessgoddess/quasar/actions/runs/29903226239) | `recalculated` | 4 834 | — |
| Vulkan | `serial` | 6 274 | +29.8% |
| Vulkan | `minimal` | 6 313 | +0.6% к `serial` |

`minimal` не дал сигнала больше шума и хранит более крупный autodiff-граф,
поэтому выбран проверенный `serial`. `recalculated` остаётся memory-saving
fallback для больших форм.

Следующий matched-token A/B ([run 29907443070](https://github.com/uselessgoddess/quasar/actions/runs/29907443070))
проверил внешний checkpointing и число Fusion streams:

| micro-batch × accum | checkpointing | streams | median tok/s |
| --- | ---: | ---: | ---: |
| 6 × 8 | да | 128 | 7 359 |
| 4 × 12 | нет | 128 | 8 406 |
| 4 × 12 | нет | 1 | **8 608** |
| 6 × 8 | нет | 1 | 2 715 |

Micro-batch 4 без checkpointing даёт +14.2% к исходной строке; один stream —
ещё +2.4%, или +17.0% вместе. Вариант batch 6 формально помещается, но VRAM
pressure делает его втрое медленнее; «не OOM» не означает полезную
конфигурацию. Корневой
`cubecl.toml` теперь закрепляет `max_streams=1`, как и примеры burn-mamba:
несколько Fusion streams удерживали одновременно больше live-буферов. Старый
лимит 128 для воспроизведения лежит в `experiments/default-streams`.

## 2. Форма GPU важнее одинакового числа FLOPs

После освобождения VRAM paired shape A/B сохранил практически тот же размер и
арифметику, но дал GPU более широкие GEMM и на 40% меньше последовательных
слоёв ([run 29908071337](https://github.com/uselessgoddess/quasar/actions/runs/29908071337)):

| форма | params | fwd FLOPs/token | активации / micro-batch | median tok/s |
| --- | ---: | ---: | ---: | ---: |
| legacy 512 × 20 | 77.7M | 161.5M | 1 649 MiB | 8 709 |
| production 640 × 12 | 78.4M | 161.2M | 1 304 MiB | **9 457** |

Production-форма быстрее на 8.6%, активации ниже на 20.9%, а attention-head
остаётся шириной 64 (`10 × 64`). Это performance-эквивалент, а не доказательство
равного качества: меньшая глубина может проиграть на длинном обучении, поэтому
quality нужно сравнивать отдельным loss/bpb run на одинаковом числе токенов.

Итого устойчивое paired-ускорение выбранной формы относительно текущего
baseline 7 359 tok/s — **+28.5%**. Относительно 5 966 tok/s на приложенном
ночном скриншоте это +58.5%, но это справочная cross-run цифра, не paired A/B.

## 3. Precision: не AMP

[Run 29908650819](https://github.com/uselessgoddess/quasar/actions/runs/29908650819)
повторил fp32 (9 433 tok/s) и проверил низкие dtype на той же карте:

- Vulkan сразу отвергает bf16: device не поддерживает запрошенный `BF16`;
- f16 падает в `burn-cubecl-fusion` при codegen/autotune и заканчивает шаг с
  non-finite loss;
- прежние ROCm-пробы bf16/f16 не смогли выбрать RDNA4 WMMA-инструкции LLVM.

Даже успешный pure-f16 запуск не был бы AMP: в этой ревизии Burn нет autocast и
`GradScaler`, dtype применяется к параметрам, активациям и optimizer state
сразу. Подтверждение ограничения есть в
[Burn issue #4332](https://github.com/tracel-ai/burn/issues/4332). Поэтому fp32
остаётся единственным проверенным training default.

## 4. Что показал CubeCL profiler

Profiler ставит timestamp вокруг каждого launch и сериализует выполнение,
поэтому его 266 tok/s нельзя сравнивать с throughput. Полезен состав одного
`micro_batch=1`, `accum=1` шага после warm-up:

| группа | GPU duration | launches | доля recorded GPU time |
| --- | ---: | ---: | ---: |
| fused matmul | 99.14 ms | 644 | 27% |
| simple matmul | 96.53 ms | 373 | 26% |
| double-buffered matmul | 86.66 ms | 324 | 24% |
| elementwise fusion | 18.37 ms | 1 720 | 5% |
| add kernels | 21.01 ms | 544 | 5% |
| cumulative sum | 10.23 ms | 92 | 2% |
| ordinary reductions | 4.76 ms | 898 | 1% |
| все группы | 360.72 ms | **9 948** | 100% |

Matmul занимает около 79% recorded GPU time, но один микробатч всё равно
диспетчеризует почти десять тысяч kernels: кроме таблицы это 834 slice-assign,
779 fill, 763 scalar-multiply, 550 copy и сотни других запусков. Wall time шага
с profiler — 3.844 s против 0.361 s суммарных timestamps. Значит узкое место —
и размеры GEMM, и launch/orchestration overhead; оптимизация одного RMSNorm не
может дать требуемый порядок величины.

## 5. Пошаговый CubeCL-порт: K4 state passing и K1 chunk cumsum

После профиля fork burn-mamba добавил первую измеримую CubeCL-границу:
[revision `efea1fdb`](https://github.com/konard/burn-mamba/commit/efea1fdb0289608c26d6d2d31da74a4a03412d4d)
заменяет K4 — перенос state между 16 chunks. Это не весь Mamba-3, а
небольшой шаг с однозначной recurrence:

```text
state[n + 1] = exp(decay[n]) * state[n] + intra[n]
```

Forward запускает один work item на элемент `[batch, head, p, r]` и
проходит chunks в регистре. Backward идёт в обратную сторону, сразу
пишет `d_intra` и contributions для `d_decay`; второй kernel редуцирует
их по `p × r`. При production-shape единственный дополнительный scratch
равен одному `[B,N,H,P,R]` tensor, около 20 MiB fp32, а не новой
копии графа модели.

Прежний high-level K4 остался в том же binary под
`BURN_MAMBA_FUSED_STATE_PASSING=0`. Поэтому
[`examples/state_passing_ab.sh`](../examples/state_passing_ab.sh) сравнивает не
две сборки, а один и тот же artifact, seed, model, warm-up и 49 152 токена.
CI сохраняет логи и VRAM каждого варианта, проверяет loss и после A/B
повторяет полный production-batch 131 072 токена для OOM-контроля.

Первый RX 9070 XT A/B на точном head дал положительный, но ожидаемо
локальный эффект ([CI run 29914868530](https://github.com/uselessgoddess/quasar/actions/runs/29914868530)):

| режим | median throughput | peak VRAM | measured loss |
| --- | ---: | ---: | --- |
| reference K4, `4×12` | 9 454 tok/s | 14.970 GiB | 9.9548 → 8.1049 → 6.1827 |
| CubeCL K4, `4×12` | 9 698 tok/s | 14.922 GiB | 9.9548 → 8.1049 → 6.1827 |
| CubeCL K4, production `4×32` | 9 897 tok/s | 14.096 GiB | 9.9548 → 8.1049 → 6.1827 |

То есть изолированный K4 даёт **+2.6%** на matched issue recipe и не
увеличивает измеренный peak VRAM. Три measured шага заняли 19.970 s у
reference, 17.251 s у CubeCL issue-run и 41.747 s у production-run; каждое
измерительное окно короче минуты. Медленный первый measured step сохранён в
расчёте медианы, а длительная JIT-компиляция warm-up в throughput не входит.

До GPU-замера пройдены два независимых correctness-уровня:

1. CubeCL CPU runtime сравнил values и градиенты `intra`, `decay`,
   `initial` с tensor reference на random input и non-zero initial state;
2. шесть double-SSD tests сравнили Serial с Minimal для SISO, MIMO,
   single-chunk и zero/non-zero initial; Vulkan + Fusion дополнительно
   проходит compile и `clippy -D warnings`.

Официальная реализация Mamba-3 показывает реальный масштаб порта:
[forward](https://github.com/state-spaces/mamba/blob/f577286d/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py),
[combined wrapper](https://github.com/state-spaces/mamba/blob/f577286d/mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py)
и [backward](https://github.com/state-spaces/mamba/blob/f577286d/mamba_ssm/ops/triton/mamba3/mamba3_siso_bwd.py).
Forward фьюзит bias, rotary/QK, decay, intra/inter-chunk state и output;
backward разбит на отдельные `dz/do`, `dqkv`, rotary/bias/angle,
`ddt/dtrap/input-state` и angle-cumsum kernels. Оптимизированные Triton shapes
тоже не совпадают буквально: там основной случай `qk=128, v=64`, здесь
`qk=64, v=64`.

Второй seam — [revision `ea297b38`](https://github.com/konard/burn-mamba/commit/ea297b38b4102c7bc8aa41175c7cf3ec332bab8f) —
заменяет K1, prefix sum decay внутри каждого chunk. Один work item обрабатывает
один `[batch, head, chunk]` scan длины 64; forward одновременно меняет layout
`BNLH → BHNL`, а точный VJP выполняет обратный suffix sum сразу в исходном
layout. K1 не создаёт дополнительный scratch tensor. Reference доступен через
`BURN_MAMBA_FUSED_CHUNK_CUMSUM=0`, поэтому постоянный GPU job теперь измеряет
три последовательных режима одного binary: reference, только K4 и K4+K1.
Результат K1 будет зафиксирован только после fresh RX 9070 XT run.

Каждый следующий seam добавляется только после отдельного A/B. Более широкий
порт считается готовым только если есть:

1. forward parity fp32 на нескольких chunk/shape/MIMO вариантах;
2. gradient parity для каждого входа и параметра, включая finite differences;
3. совпадение параметров после полного Muon/AdamW optimizer step;
4. stress test на sequence, не кратной предпочитаемому tile;
5. GPU A/B с одинаковым token budget, finite loss и проверкой VRAM;
6. custom backward — forward-only kernel обучение не ускорит достаточно.

## 6. Wall-clock без обещаний

В issue один step содержал 49 152 токена. Поэтому 12 500 шагов — 614.4M
токенов: при 5 966 tok/s это 28.6 часа, при измеренных 9 457 tok/s — **18.0
часа**, а после K4 при 9 698 tok/s — **17.6 часа**. Для 9 часов нужно
18 963 tok/s, для 8 часов — 21 333 tok/s. Этот PR
заметно сокращает ночь, но не выдаёт 18 часов за 8–9.

Production default сохраняет compute-efficient batch `4 × 32 × 1024 = 131072`
токена на step. Его 12 500 шагов — 1.6384B токенов и около 46.0 часов при
измеренных 9 897 tok/s. Если нужен именно issue-budget, задайте `--accum 12`;
это меняет token budget и должно быть осознанным решением.

`tiny-turbo` уже выбирает `serial`, micro-batch 4, accum 32 и checkpointing
off. При OOM после shape override верните память явно:

```sh
--checkpointing true --ssd recalculated
```
