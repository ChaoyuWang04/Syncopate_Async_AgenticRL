"""固定管线顺序与 Modal 资源分配；不复制各阶段训练参数。"""
from __future__ import annotations

from dataclasses import dataclass

DATA_STAGES = ("cases", "menus", "split", "gates", "supply", "rl-data")
TEACHER_STAGES = ("teacher", "sft-data", "teacher-stop")
TRAIN_STAGES = ("sft-train", "sft-eval", "sft-select", "merge", "exam",
                "rl-train", "rl-adapter", "rl-eval", "opd-train", "opd-eval")
ALL_STAGES = DATA_STAGES + TEACHER_STAGES + TRAIN_STAGES
GROUPS = {"all": ALL_STAGES, "train-all": TRAIN_STAGES,
          "data-prepare": DATA_STAGES, "sft-data-group": TEACHER_STAGES}


@dataclass(frozen=True)
class Resources:
    gpus: int
    cpu: int
    memory_mib: int

    def modal_options(self) -> dict:
        options = {"cpu": self.cpu, "memory": self.memory_mib}
        if self.gpus:
            options["gpu"] = "B200" if self.gpus == 1 else f"B200:{self.gpus}"
        return options


CPU = Resources(0, 8, 16384)
ONE = Resources(1, 16, 131072)
TWO = Resources(2, 32, 262144)


def resources_for(stage: str) -> Resources:
    if stage in DATA_STAGES or stage in {"data-prepare", "sft-select", "sft-data-offline"}:
        return CPU
    if stage == "merge":
        return Resources(0, 16, 196608)
    if stage == "rl-adapter":
        return Resources(0, 8, 65536)
    if stage in {"sft-train", "sft-eval", "exam", "rl-eval", "opd-eval", "sft-data-group"}:
        return ONE
    if stage in {"rl-train", "opd-train"}:
        return TWO
    raise ValueError(f"没有登记资源的阶段：{stage}；教师与建库必须使用 sft-data-group 同容器执行")


def modal_plan(stage: str) -> tuple[tuple[str, Resources], ...]:
    # 教师是常驻进程，这三个阶段不能拆到不同容器。
    names = (("data-prepare", "sft-data-group", *TRAIN_STAGES) if stage == "all"
             else TRAIN_STAGES if stage == "train-all" else (stage,))
    return tuple((name, resources_for(name)) for name in names)


def main() -> None:
    # 给 Bash 的内容全部来自上述固定常量，没有用户输入。
    print("ALL=(" + " ".join(ALL_STAGES) + ")")
    print("TRAIN_ALL=(" + " ".join(TRAIN_STAGES) + ")")
    print("DATA_PREPARE=(" + " ".join(DATA_STAGES) + ")")
    print("TEACHER_GROUP=(" + " ".join(TEACHER_STAGES) + ")")


if __name__ == "__main__":
    main()
