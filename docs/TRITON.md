# Mamba-3 SSD на Triton

Это путь максимальной производительности для одной AMD GPU. Он использует
`Mamba3` из upstream [`state-spaces/mamba`](https://github.com/state-spaces/mamba)
и его autograd-операцию `mamba3_siso_combined`: forward и backward SSD написаны
на Triton. Подготовку датасета менять не нужно — Python читает те же little-endian
`u16` shards из `data/shards/{train,valid}`, которые создаёт Rust CLI.

Тяжёлый тест на RX 9070 XT в репозитории намеренно не запускался. При старте
harness печатает GPU, ROCm, версию Triton и точное имя ядра; владелец карты может
сразу отличить этот путь от Burn.

## Почему SISO, а не MIMO

В текущем upstream Mamba-3 это два разных backend:

- `is_mimo=False` вызывает Triton `mamba3_siso_combined`;
- `is_mimo=True` вызывает `mamba3_mimo_combined` из TileLang.

Поэтому Triton preset фиксирует `mimo_rank=1`. Оставить ранги 2/4 из Burn-конфига
и назвать результат Triton было бы неверно. Глубина, ширина, state rank, гибридные
attention layers, SwiGLU и token budget сохранены. Получаются 155.7M параметров
для `tiny` и 1.003B для `base`; точное число сверяется с построенным upstream
модулем до первого batch.

## Установка ROCm

Поддерживаемый upstream путь — Linux и Python 3.12. Нативные Windows ROCm wheels
PyTorch из описания issue не включают официальный Linux Triton/Mamba runtime;
для них остаётся Rust/Vulkan путь. Конкретно RDNA4 проверяется владельцем карты:
официальный Mamba CI не содержит RX 9070 XT.

Автоматическая установка из корня репозитория:

```sh
examples/install-triton-rocm.sh
source .venv-triton/bin/activate
```

Скрипт сначала ставит ROCm-сборку PyTorch (она предоставляет `triton-rocm`),
затем зависимости Quasar и закреплённую ревизию upstream Mamba без её лишних
MIMO/TileLang dependencies. Обычный `pip install mamba-ssm` здесь не подходит:
его generic dependency `triton` может заменить ROCm-вариант.

Проверить маленькую форму и компиляцию ядра:

```sh
python -m quasar_triton toy --data data/shards \
  --out runs/toy-triton --steps 2 --micro-batch 1 --accum 1 \
  --eval-every 0 --save-every 0
```

## Обучение

Сначала Rust CLI строит tokenizer и shards как обычно:

```sh
cargo run --release -- tokenizer data/fineweb-edu --vocab-size 32768
cargo run --release -- prepare data/fineweb-edu --out data/shards
```

Затем запускается fused путь:

```sh
python -m quasar_triton tiny --data data/shards --out runs/tiny-triton
```

Default — bf16, activation checkpointing и эффективный batch
`8 × 16 × 2048 = 262144` токена. Начать безопаснее с `--micro-batch 1`, увеличив
`--accum` обратно пропорционально. `--no-checkpointing` быстрее, только если
активации помещаются в 16 GB.

Флаг `--compile` экспериментален и по умолчанию выключен: он компилирует
окружающие PyTorch-операции со статическими shapes. Сам SSD использует Triton и
без этого флага. Такое разделение важно на RDNA4, где ошибка `torch.compile` не
должна маскировать работоспособность upstream SSD kernel.

Run автоматически продолжается из самого нового `checkpoint_*/checkpoint.pt`.
`model.json` и `run.json` не дают случайно продолжить checkpoint с изменённой
архитектурой или schedule. Validation печатает NLL, perplexity и bits-per-byte;
train telemetry — tok/s, effective TFLOP/s и ETA.

## Протокол измерения

Первый step включает JIT-компиляцию Triton и не годится для сравнения. Для
проверки ускорения:

1. Запустить не менее 20 logging windows с одинаковыми `seq_len`, micro-batch и
   accumulation.
2. Сравнивать median steady-state `tok/s`, исключив первое окно, validation и
   checkpoint I/O.
3. Записать `backend=ROCm …; Triton …`, имя GPU и `kernel=mamba3_siso_combined`
   из стартовой строки.
4. Отдельно A/B измерить checkpointing; не смешивать его цену с ценой backend.

Dependency-free tests (`python -m unittest discover`) проверяют preset, shards,
schedule, CLI и сам аргумент `is_mimo=False`. Полная GPU-проверка остаётся
обязательной перед merge, потому что hosted CI не имеет ROCm/RDNA4.
