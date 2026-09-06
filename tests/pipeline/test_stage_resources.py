import pytest

from syncopate.pipeline.stages import ALL_STAGES, GROUPS, TRAIN_STAGES, modal_plan, resources_for


def test_full_cloud_plan_has_same_order_as_fixed_runbook():
    expanded = [leaf for stage, _ in modal_plan("all") for leaf in GROUPS.get(stage, (stage,))]
    assert tuple(expanded) == ALL_STAGES
    assert tuple(name for name, _ in modal_plan("train-all")) == TRAIN_STAGES


@pytest.mark.parametrize("stage", ["merge", "rl-adapter", "sft-select", "data-prepare"])
def test_cpu_stages_never_allocate_gpu(stage):
    assert resources_for(stage).gpus == 0
    assert "gpu" not in resources_for(stage).modal_options()


@pytest.mark.parametrize("stage", ["sft-train", "sft-eval", "exam", "rl-eval", "opd-eval"])
def test_single_gpu_work_gets_one_b200(stage):
    assert resources_for(stage).modal_options()["gpu"] == "B200"


@pytest.mark.parametrize("stage", ["rl-train", "opd-train"])
def test_two_gpu_work_gets_two_b200(stage):
    assert resources_for(stage).modal_options()["gpu"] == "B200:2"


def test_no_unknown_stage_or_ephemeral_teacher():
    for stage in ["anything", "teacher", "sft-data", "teacher-stop"]:
        with pytest.raises(ValueError):
            modal_plan(stage)
