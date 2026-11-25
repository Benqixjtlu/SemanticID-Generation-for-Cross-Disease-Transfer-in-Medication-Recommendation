"""Run the examples with Python alone: python -m thesis_starter."""

import argparse
from collections.abc import Sequence

from .baselines import FrequencyBaseline, mention_copy
from .data import CohortSplit, make_split, synthetic_visits
from .decoding import CatalogDecoder, DecodeState
from .evaluation import evaluate, paired_bootstrap, score_visit, transfer_effects
from .ids import TEACHING_CODEBOOKS, TEACHING_VALUES, build_ids, permute_ids, quantize
from .model import PrefixCountModel


def show_ids(seed: int) -> None:
    print("\n[导读第 7–8 章] 药物编码与合法生成")
    ids = build_ids()
    permuted = permute_ids(ids, seed=seed)
    for drug, path in ids.items():
        trace = quantize(TEACHING_VALUES[drug], TEACHING_CODEBOOKS)
        residuals = " → ".join(f"{step.residual_after:g}" for step in trace)
        print(f"  药 {drug}  残差 {residuals}  完整 ID {path}  置换 ID {permuted[drug]}")

    decoder = CatalogDecoder(ids)
    answer = decoder.serialize({"A", "C"})
    state = DecodeState()
    print("\n  训练标签 {A,C} 的合法动作与正确 token")
    for token in answer:
        print(f"    已完成 {state.drugs}  当前前缀 {state.prefix}"
              f"  可选 {decoder.allowed(state)}  选 {token}")
        state = decoder.advance(state, token)
    print(f"  共 {len(answer)} 个 token，反查结果 {decoder.decode(answer)}")


def show_split(split: CohortSplit, budget: int, strict: bool) -> None:
    print(f"\n[导读第 8.4 章] 患者划分（严格排除共病 {'开启' if strict else '关闭'}）")
    partitions = (
        ("源域训练", split.source_train),
        ("目标适配候选池", split.target_pool),
        (f"实际目标适配 k={budget}", split.support(budget)),
        ("目标外层测试", split.target_test),
        ("其他队列外层测试", split.other_test),
    )
    for name, visits in partitions:
        patients = {visit.patient_id for visit in visits}
        print(f"  {name}  {len(patients)} 位患者，{len(visits)} 次有标签就诊")
    source = {visit.patient_id for visit in split.source_train}
    print(f"  P07 是否出现在源训练  {'是' if 'P07' in source else '否'}")
    print(f"  P08 是否出现在源训练  {'是' if 'P08' in source else '否'}")


def show_baselines(split: CohortSplit) -> None:
    print("\n[导读第 11.1 章] 用源队列拟合简单基线")
    baseline = FrequencyBaseline.fit(split.source_train)
    for name, visits in (("目标测试", split.target_test), ("其他队列测试", split.other_test)):
        assignments = {
            "全局频率": {visit.visit_id: baseline.predict(visit, top_n=2) for visit in visits},
            "按诊断频率": {
                visit.visit_id: baseline.predict(visit, top_n=2, by_diagnosis=True)
                for visit in visits
            },
            "历史药名复制": {
                visit.visit_id: mention_copy(visit, TEACHING_VALUES)
                for visit in visits
            },
        }
        for label, predictions in assignments.items():
            scores = evaluate(visits, predictions).averages
            print(f"  {name:8s}  {label:8s}  Jaccard {scores.jaccard:.3f}  F1 {scores.f1:.3f}")


def show_transfer(split: CohortSplit, *, budget: int, seed: int, replicates: int) -> None:
    print("\n[导读第 11.2–11.4 章] 药名集合评分和迁移比较")
    manual = score_visit({"A", "C"}, {"B", "C", "D"})
    print(f"  手算例 {{A,C}} 对 {{B,C,D}}  Jaccard {manual.jaccard:.2f}"
          f"  F1 {manual.f1:.2f}  总分 {manual.overall:.3f}")
    guide = transfer_effects({
        "rq_target": 0.31, "rq_transfer": 0.43,
        "perm_target": 0.37, "perm_transfer": 0.45,
    })
    print(f"  导读假设分数  增益差 {guide.transfer:+.2f}  最终分差 {guide.direct:+.2f}")

    print(f"\n  下列分数由合成病例与计数模型现场计算（k={budget}）")
    support = split.support(budget)
    ids = build_ids()
    predictions_by_arm = {}
    scores_by_arm = {}
    for prefix, mapping in (("rq", ids), ("perm", permute_ids(ids, seed=seed))):
        routes = {}
        if budget > 0:
            routes["target"] = (support,)
        routes["transfer"] = (split.source_train, support)
        for route, training_stages in routes.items():
            arm = f"{prefix}_{route}"
            model = PrefixCountModel(CatalogDecoder(mapping))
            for training_visits in training_stages:
                model.learn(training_visits)
            predictions = {visit.visit_id: model.predict(visit.text) for visit in split.target_test}
            scores = evaluate(split.target_test, predictions).averages
            scores_by_arm[arm] = scores.jaccard
            predictions_by_arm[arm] = predictions
            print(f"  {arm:14s}  Jaccard {scores.jaccard:.3f}  F1 {scores.f1:.3f}")

    if budget == 0:
        direct = scores_by_arm["rq_transfer"] - scores_by_arm["perm_transfer"]
        print(f"  零目标样本的直接分差 {direct:+.3f}")
        return

    result = paired_bootstrap(split.target_test, predictions_by_arm, replicates=replicates, seed=seed)
    print(f"  增益差 {result.point.transfer:+.3f}  95% 区间 {format_ci(result.transfer_ci)}")
    print(f"  最终分差 {result.point.direct:+.3f}  95% 区间 {format_ci(result.direct_ci)}")
    print(f"  对 {result.n_patients} 位测试患者做了 {result.replicates} 次配对重抽样")


def format_ci(interval: tuple[float, float]) -> str:
    lower, upper = interval
    return f"[{lower:+.3f}, {upper:+.3f}]"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="出院用药预测研究的合成数据练习")
    parser.add_argument(
        "section", nargs="?", default="all",
        choices=("all", "ids", "split", "baselines", "transfer"), help="选择要运行的部分",
    )
    parser.add_argument("--seed", type=int, default=42, help="划分、置换和重抽样使用的随机种子")
    parser.add_argument("--k", type=int, default=32, help="目标适配人数，0 表示直接使用源模型")
    parser.add_argument("--replicates", type=int, default=400, help="配对 bootstrap 的抽样次数")
    parser.add_argument("--strict", action="store_true", help="排除源患者中的目标次诊断与已确认病史")
    options = parser.parse_args(argv)
    if options.replicates < 1:
        parser.error("--replicates 必须至少为 1")

    print("出院用药预测研究代码练习｜所有病例与药物均为合成示例")
    if options.section in ("all", "ids"):
        show_ids(options.seed)
    if options.section == "ids":
        return

    split = make_split(
        synthetic_visits(), target_diagnosis="目标组", source_diagnoses=("源组一", "源组二"),
        seed=options.seed, strict=options.strict,
    )
    pool_size = len({visit.patient_id for visit in split.target_pool})
    if not 0 <= options.k <= pool_size:
        parser.error(f"--k 必须介于 0 和目标候选池人数 {pool_size} 之间")
    if options.section in ("all", "split"):
        show_split(split, options.k, options.strict)
    if options.section in ("all", "baselines"):
        show_baselines(split)
    if options.section in ("all", "transfer"):
        show_transfer(split, budget=options.k, seed=options.seed, replicates=options.replicates)


if __name__ == "__main__":
    main()
