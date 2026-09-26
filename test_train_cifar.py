import contextlib
import copy
import io
import json
import unittest
import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train_cifar as train

# Tests must not depend on the script's debug switch.
train.debug = False


def search_config(
    algorithm,
    params,
    metric,
    max_side_steps,
    interval_steps,
    cooldown_steps,
    initial_dropoff_margin,
):
    config = dict(algorithm=algorithm, params=params, metric=metric)
    if algorithm in ("interval", "global_neighbour"):
        config["max_side_steps"] = max_side_steps
    if algorithm == "interval":
        config.update(
            interval_steps=interval_steps,
            cooldown_steps=cooldown_steps,
            initial_dropoff_margin=initial_dropoff_margin,
        )
    return config


class TinyLoader(train.CifarLoader):
    def __init__(self, path, train, batch_size, aug):
        self.images = torch.rand(6, 3, 8, 8)
        self.labels = torch.arange(6) % 2
        self.normalize = torch.nn.Identity()
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug
        self.batch_size = batch_size
        self.drop_last = self.shuffle = train


class TinyModel(torch.nn.Module):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.whiten = torch.nn.Conv2d(3, 4, 1)
        self.norm = train.BatchNorm(4, momentum=0.6, eps=1e-12)
        self.conv = torch.nn.Conv2d(4, 4, 1, bias=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(4, 2, bias=False)
        self.register_buffer("training_steps", torch.tensor(0))

    def reset(self):
        for module in (self.whiten, self.norm, self.conv, self.head):
            module.reset_parameters()
        self.training_steps.zero_()

    def init_whiten(self, images, eps):
        self.clock["time"] += 20

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            self.clock["time"] += 1
            self.clock["training"] += 1
            self.training_steps.add_(1)
        x = self.dropout(self.conv(self.norm(self.whiten(x))))
        return self.head(x.mean((2, 3)))


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_experiment_configs_and_fixed_nesterov(self):
        decays = ("linear_decay", "constant")
        exp2_names = [
            f"exp2_{decay}_momentum_{order}_conditioning"
            for decay in decays
            for order in ("before", "after")
        ]
        self.assertEqual(
            [name for name, _ in train.RUNS],
            [f"exp1_{decay}" for decay in decays] + exp2_names,
        )
        # LR decay is fixed per run instead of searched.
        expected_paths = {
            "exp1": [
                "conv.initial_lr",
                "conv.momentum",
                "head.initial_lr",
                "head.momentum",
            ],
            "exp2": [
                "head.initial_lr",
                "head.momentum",
                "head.svd_mean_percentage_damping",
                "head.input_conditioner_momentum",
            ],
        }
        baseline = train.BASELINE_RUN_CONFIGS[2]
        for name, original in train.RUNS:
            experiment = name.split("_")[0]
            decay = "constant" if "_constant" in name else "linear_decay"
            changed = ("conv", "head") if experiment == "exp1" else ("head",)
            config = train.normalize_config(original, key=None)
            self.assertEqual(config["batch_size"], 2000)
            self.assertEqual(config["num_epochs"], 8)
            self.assertFalse(config["overfit"])
            train.validate_tuning_config(config["hparam_tuning"])
            self.assertEqual(config["hparam_tuning"]["algorithm"], "global_neighbour")
            self.assertEqual(config["hparam_tuning"]["metric"], "tta_val_acc")
            self.assertEqual(
                list(config["hparam_tuning"]["params"]), expected_paths[experiment]
            )
            for group_name in changed:
                self.assertEqual(
                    train.get_hparam(config, f"{group_name}.decay"), decay
                )
            choices, initial = train.interval_search_space(config)
            for group_name, group in config["param_groups"].items():
                train.GroupOptimizer.validate_group(group, runtime=False)
                self.assertEqual(
                    group["nesterov"], baseline["param_groups"][group_name]["nesterov"]
                )
                if group_name not in changed:
                    self.assertEqual(group, baseline["param_groups"][group_name])
            self.assertEqual(choices["head.momentum"], train.MOMENTUM_CHOICES[1:])
            self.assertEqual(initial["head.momentum"], 0.85)
            model = TinyModel(dict(time=0, training=0))
            train.make_optimizer(model, config["param_groups"])
        runs = dict(train.RUNS)
        for decay in decays:
            self.assertEqual(
                runs[f"exp1_{decay}"]["param_groups"]["conv"]["algorithm"], "sgdh"
            )
        for name in exp2_names:
            head = runs[name]["param_groups"]["head"]
            self.assertEqual(head["algorithm"], "input_conditioned")
            self.assertEqual(
                head["gradient_momentum_before_conditioning"], "_before_" in name
            )
            self.assertEqual(
                runs[name]["hparam_tuning"]["params"][
                    "head.input_conditioner_momentum"
                ]["choices"],
                train.MOMENTUM_CHOICES,
            )
        self.assertEqual(
            train.INPUT_CONDITIONER_FIELDS,
            {
                "svd_mean_percentage_damping",
                "input_conditioner_momentum",
                "gradient_momentum_before_conditioning",
            },
        )

    def test_global_decay_candidates_replay_and_preserve_unsearched_schedules(self):
        for preferred_decay in ("constant", "linear_decay"):
            with self.subTest(preferred_decay=preferred_decay):
                config = copy.deepcopy(train.EXPERIMENT_RUN_CONFIGS["exp1_linear_decay"])
                config.update(batch_size=2, num_epochs=3, overfit=True)
                for name, group in config["param_groups"].items():
                    group["lr_scheduler"] = [
                        (3, 0 if name.endswith("_weight") else 0.001)
                    ]
                config["param_groups"]["norm_bias"]["lr_scheduler"] = [
                    (1, 0.002),
                    (2, 0.003, 0.001),
                ]
                config["hparam_tuning"]["params"] = {
                    "conv.initial_lr": dict(initial=0.001, choices=[0.001, 0.002]),
                    "conv.decay": dict(
                        initial="linear_decay", choices=["constant", "linear_decay"]
                    ),
                }
                base = train.configured_hparam_schedules(config["param_groups"], 3)
                original = copy.deepcopy(config)
                model = TinyModel(dict(time=0, training=0))
                updates, starts, scores = [], [], []
                original_step = train.GroupOptimizer.step

                def step(optimizer):
                    updates.append(
                        {group["name"]: group["lr"] for group in optimizer.param_groups}
                    )
                    if model.training_steps.item() == 1:
                        starts.append(copy.deepcopy(model.state_dict()))
                    original_step(optimizer)

                def evaluate(model, loader, tta_level):
                    self.assertEqual(model.training_steps.item(), 3)
                    candidate = updates[-3:]
                    initial = candidate[0]["conv"]
                    decay = (
                        "constant"
                        if candidate[-1]["conv"] == initial
                        else "linear_decay"
                    )
                    scores.append((initial, decay))
                    return (
                        0.8
                        + (0.02 if decay == preferred_decay else 0)
                        + (0.01 if initial == 0.002 else 0)
                    )

                output = io.StringIO()
                with (
                    patch.object(train, "CifarLoader", TinyLoader),
                    patch.object(train.GroupOptimizer, "step", step),
                    patch.object(train, "evaluate", side_effect=evaluate),
                    contextlib.redirect_stdout(output),
                ):
                    result = train.run_experiment("exp1", model, config)["best_result"]
                self.assertEqual(result["intervals"][0]["cooldown_steps"], 0)
                self.assertEqual(
                    result["intervals"][0]["main_hparams"],
                    {"conv.initial_lr": 0.002, "conv.decay": preferred_decay},
                )
                self.assertEqual(
                    result["hparam_schedules"]["conv.initial_lr"],
                    [(3, 0.002)]
                    if preferred_decay == "constant"
                    else [(3, 0.002, 0.0)],
                )
                self.assertEqual(
                    {decay for _, decay in scores}, {"constant", "linear_decay"}
                )
                for path, lines in base.items():
                    if path != "conv.initial_lr":
                        self.assertEqual(result["hparam_schedules"][path], lines)
                for offset in range(0, len(updates), 3):
                    self.assertEqual(
                        [row["norm_bias"] for row in updates[offset : offset + 3]],
                        [0.002, 0.003, 0.001],
                    )
                for state in starts:
                    for key, value in state.items():
                        torch.testing.assert_close(
                            value, starts[0][key], rtol=0, atol=0
                        )
                self.assertEqual(config, original)
                self.assertIn("conv.decay=constant", output.getvalue())
                self.assertIn("conv.decay=linear_decay", output.getvalue())

    def test_global_lr_fields_are_independent(self):
        base = {
            "conv.initial_lr": [(2, 0.1), (3, 0.1, 0.02)],
            "head.initial_lr": [(5, 1300, 0.0)],
            "conv.momentum": [(5, 0.7)],
        }
        initial_only = train.segment_hparam_schedules(
            base, {"conv.initial_lr": 0.2}, 0, 5, global_search=True
        )
        self.assertEqual(initial_only["conv.initial_lr"], [(2, 0.2), (3, 0.2, 0.04)])
        self.assertEqual(initial_only["head.initial_lr"], base["head.initial_lr"])
        decay_only = train.segment_hparam_schedules(
            base, {"head.decay": "constant"}, 0, 5, global_search=True
        )
        self.assertEqual(decay_only["head.initial_lr"], [(5, 1300)])
        self.assertEqual(decay_only["conv.initial_lr"], base["conv.initial_lr"])
        self.assertEqual(base["head.initial_lr"], [(5, 1300, 0.0)])

    def test_optimizer_requires_exact_algorithm_fields(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        extras = {
            "sgd": {},
            "muon": dict(ns_steps=3, ns_eps=0.0),
            "sgdh": dict(normalization_eps=1e-6),
            "input_conditioned": dict(
                svd_mean_percentage_damping=0.01,
                input_conditioner_momentum=0.9,
                gradient_momentum_before_conditioning=True,
            ),
        }
        for algorithm, fields in extras.items():
            group = dict(
                name="test",
                params=[parameter],
                algorithm=algorithm,
                lr=0.1,
                momentum=0.6,
                nesterov=True,
                **fields,
            )
            train.GroupOptimizer([dict(group)])
            for field in group:
                incomplete = dict(group)
                del incomplete[field]
                with (
                    self.subTest(algorithm=algorithm, missing=field),
                    self.assertRaises(ValueError),
                ):
                    train.GroupOptimizer([incomplete])
            for field in {
                "weight_decay",
                "momentun",
                *set().union(*(set(v) for v in extras.values())),
            } - fields.keys():
                with (
                    self.subTest(algorithm=algorithm, extra=field),
                    self.assertRaisesRegex(ValueError, "unexpected fields"),
                ):
                    train.GroupOptimizer([dict(group, **{field: 0.1})])
            config = {
                key: value
                for key, value in group.items()
                if key not in ("params", "name", "lr")
            }
            config["lr_scheduler"] = [(5, 0.1)]
            train.GroupOptimizer.validate_group(config, runtime=False)
            for field in config:
                incomplete = dict(config)
                del incomplete[field]
                with (
                    self.subTest(algorithm=algorithm, config_missing=field),
                    self.assertRaises(ValueError),
                ):
                    train.GroupOptimizer.validate_group(incomplete, runtime=False)

    def test_training_code_has_no_argument_defaults(self):
        for node in ast.walk(ast.parse(inspect.getsource(train))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                self.assertEqual(
                    node.args.defaults, [], getattr(node, "name", "lambda")
                )
                self.assertTrue(all(value is None for value in node.args.kw_defaults))

    def test_search_requires_explicit_options_and_rejects_legacy_shorthand(self):
        for original in (
            train.INTERVAL_RUN_CONFIGS[0],
            train.GLOBAL_NEIGHBOUR_RUN_CONFIGS[0],
        ):
            config = copy.deepcopy(original)
            for field in config["hparam_tuning"]:
                incomplete = copy.deepcopy(config["hparam_tuning"])
                del incomplete[field]
                with self.subTest(missing=field), self.assertRaises(ValueError):
                    train.validate_tuning_config(incomplete)
            for field in ("initial", "choices", "mult"):
                incomplete = copy.deepcopy(config)
                del incomplete["hparam_tuning"]["params"]["conv.initial_lr"][field]
                with self.subTest(missing=field), self.assertRaises(ValueError):
                    train.tuning_params(incomplete)
            config["hparam_tuning"]["params"]["conv.initial_lr"] = [0.1, 0.2]
            with self.assertRaisesRegex(ValueError, "explicitly"):
                train.tuning_params(config)
            config["hparam_tuning"]["parameters"] = config["hparam_tuning"].pop(
                "params"
            )
            with self.assertRaises(ValueError):
                train.tuning_params(config)

    def test_grid_and_coordinate_still_find_best_tta(self):
        scores = {
            (0.04, 0.6): 0,
            (0.06, 0.6): 1,
            (0.08, 0.6): 0.5,
            (0.04, 0.7): 1,
            (0.06, 0.7): 2,
            (0.08, 0.7): 3,
            (0.04, 0.8): 2.5,
            (0.06, 0.8): 1.5,
            (0.08, 0.8): 4,
        }

        def fake_main(run, model, **config):
            point = (
                train.get_hparam(config, "conv.initial_lr"),
                train.get_hparam(config, "conv.momentum"),
            )
            score = scores[point] / 10
            return dict(val_acc=1 - score, tta_val_acc=score)

        for algorithm in ("grid", "coordinate"):
            with self.subTest(algorithm=algorithm):
                config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                config = train.with_hparam(config, "conv.initial_lr", 0.04)
                config = train.with_hparam(config, "conv.momentum", 0.6)
                config["hparam_tuning"] = search_config(
                    algorithm=algorithm,
                    params={
                        "conv.initial_lr": dict(
                            initial=[0.04, 0.06, 0.08][0], choices=[0.04, 0.06, 0.08]
                        ),
                        "conv.momentum": dict(
                            initial=min(
                                [0.6, 0.7, 0.8],
                                key=lambda v: abs(
                                    v - train.get_hparam(config, "conv.momentum")
                                ),
                            ),
                            choices=[0.6, 0.7, 0.8],
                        ),
                    },
                    metric="tta_val_acc",
                    max_side_steps=20,
                    interval_steps=40,
                    cooldown_steps=40,
                    initial_dropoff_margin=0.02,
                )
                output = io.StringIO()
                with (
                    patch.object(train, "main", side_effect=fake_main),
                    contextlib.redirect_stdout(output),
                ):
                    result = train.run_experiment(0, None, config)
                self.assertEqual(len(result["trials"]), 9)
                self.assertEqual(result["best_result"]["tta_val_acc"], 0.4)
                self.assertEqual(
                    train.get_hparam(result["best_config"], "conv.initial_lr"), 0.08
                )
                log = output.getvalue()
                self.assertEqual(log.count("base_config run=0\n"), 1)
                self.assertEqual(log.count("config_diff run="), 9)
                self.assertIn(
                    "conv.lr_scheduler: [[3200, 0.04, 0.0]] -> [[3200, 0.06, 0.0]]", log
                )
                self.assertIn("search_best run=0", log)
                self.assertNotIn("%", log)

    def test_tuning_initial_values_do_not_change_untuned_baseline(self):
        baseline_lrs = {125: 0.047, 500: 0.078, 2000: 0.22}
        tuning_starts = {125: 0.047, 500: 0.078, 2000: 0.22}
        for original in train.INTERVAL_RUN_CONFIGS + train.GLOBAL_NEIGHBOUR_RUN_CONFIGS:
            with self.subTest(
                batch_size=original["batch_size"],
                algorithm=original["hparam_tuning"]["algorithm"],
            ):
                baseline = baseline_lrs[original["batch_size"]]
                self.assertEqual(
                    train.get_hparam(original, "conv.initial_lr"), baseline
                )
                _, initial = train.interval_search_space(original)
                spec = original["hparam_tuning"]["params"]["conv.initial_lr"]
                self.assertEqual(spec["initial"], tuning_starts[original["batch_size"]])
                self.assertEqual(
                    initial["conv.initial_lr"],
                    train.snap_lr_to_grid(spec["initial"], spec["mult"]),
                )
                if original["hparam_tuning"]["algorithm"] == "global_neighbour":
                    self.assertEqual(initial["head.momentum"], 0.85)
                    self.assertEqual(initial["norm_bias.momentum"], 0.85)
                    for path in ("head.initial_lr", "norm_bias.initial_lr"):
                        spec = original["hparam_tuning"]["params"][path]
                        self.assertEqual(
                            initial[path],
                            train.snap_lr_to_grid(spec["initial"], spec["mult"]),
                        )
                config = copy.deepcopy(original)
                config["hparam_tuning"] = None
                with patch.object(
                    train, "main", return_value=dict(val_acc=0.9, tta_val_acc=0.94)
                ) as main:
                    train.run_experiment(0, None, config)
                passed = main.call_args.kwargs
                self.assertIsNone(passed["hparam_tuning"])
                self.assertEqual(train.get_hparam(passed, "conv.initial_lr"), baseline)
                config = copy.deepcopy(original)
                del config["hparam_tuning"]["params"]["conv.initial_lr"]
                choices, _ = train.interval_search_space(config)
                self.assertNotIn("conv.initial_lr", choices)
                self.assertEqual(train.get_hparam(config, "conv.initial_lr"), baseline)
        self.assertEqual(
            [c["batch_size"] for c in train.BASELINE_RUN_CONFIGS], [125, 500, 2000]
        )
        self.assertTrue(
            all(c["hparam_tuning"] is None for c in train.BASELINE_RUN_CONFIGS)
        )

    def test_structured_params_work_with_grid_and_coordinate(self):
        for algorithm in ("grid", "coordinate"):
            with self.subTest(algorithm=algorithm):
                config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                config["hparam_tuning"] = search_config(
                    algorithm=algorithm,
                    params={
                        "conv.initial_lr": dict(
                            initial=0.06, choices=[0.04, 0.06, 0.08]
                        ),
                        "conv.momentum": dict(initial=0.7, choices=[0.6, 0.7, 0.8]),
                    },
                    metric="tta_val_acc",
                    max_side_steps=20,
                    interval_steps=40,
                    cooldown_steps=40,
                    initial_dropoff_margin=0.02,
                )
                calls = []

                def evaluate(run, model, **candidate):
                    for name, group in config["param_groups"].items():
                        actual = candidate["param_groups"][name]
                        for field, expected in group.items():
                            if name == "conv" and field in ("lr_scheduler", "momentum"):
                                continue
                            self.assertEqual(actual[field], expected)
                    point = (
                        train.get_hparam(candidate, "conv.initial_lr"),
                        train.get_hparam(candidate, "conv.momentum"),
                    )
                    calls.append(point)
                    return dict(val_acc=0, tta_val_acc=sum(point))

                with (
                    patch.object(train, "main", side_effect=evaluate),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = train.run_experiment(0, None, config)
                best = result["best_config"]
                self.assertEqual(train.get_hparam(best, "conv.initial_lr"), 0.08)
                self.assertEqual(train.get_hparam(best, "conv.momentum"), 0.8)
                if algorithm == "coordinate":
                    self.assertEqual(calls[0], (0.06, 0.7))
                else:
                    self.assertEqual(len(calls), 9)

    def test_explicit_momentum_start_can_be_between_choices(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config["hparam_tuning"] = search_config(
            algorithm="global_neighbour",
            params={"head.momentum": dict(initial=0.85, choices=[0.8, 0.9])},
            metric="tta_val_acc",
            max_side_steps=20,
            interval_steps=40,
            cooldown_steps=40,
            initial_dropoff_margin=0.02,
        )
        choices, initial = train.interval_search_space(config)
        calls = []

        def evaluate(point):
            calls.append(point["head.momentum"])
            return -abs(point["head.momentum"] - 0.85)

        best, _ = train.directional_search(
            initial,
            choices,
            evaluate,
            initial_lr_search=False,
            factors={"conv.initial_lr": 0.6, "head.initial_lr": 0.6},
            max_side_steps=20,
            dropoff_margin=0.02,
            on_event=None,
        )
        self.assertEqual(calls, [0.85, 0.8, 0.9])
        self.assertEqual(best["head.momentum"], 0.85)

    def test_config_logging_is_readable_and_diffs_only_changed_values(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config["batch_size"] = 125
        config = train.with_hparam(config, "head.initial_lr", 84)
        config = train.with_hparam(config, "conv.initial_lr", 0.04)
        config = train.with_hparam(config, "conv.momentum", 0.6)
        changed = train.with_hparam(config, "head.initial_lr", 83.75)
        changed = train.with_hparam(changed, "conv.momentum", 0.718)
        changed = train.with_hparam(changed, "conv.initial_lr", 0.0617)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train.log_config(0, config, label="base_config")
            train.log_config_diff("0.1", 0, config, changed)
        header, body = output.getvalue().split("\n", 1)
        self.assertEqual(header, "base_config run=0")
        logged, end = json.JSONDecoder().raw_decode(body)
        self.assertEqual(logged["batch_size"], 125)
        self.assertEqual(logged["param_groups"]["conv"]["lr_scheduler"][0][1], 0.04)
        self.assertIn('\n  "batch_size": 125,\n', body)
        diff = body[end:]
        self.assertIn(
            "conv.lr_scheduler: [[3200, 0.04, 0.0]] -> [[3200, 0.062, 0.0]]", diff
        )
        self.assertIn("conv.momentum: 0.6 -> 0.72", diff)
        self.assertNotIn("head.lr_scheduler", diff)
        self.assertNotIn("param_groups", diff)

    def test_directional_sweeps_revisit_parameters_and_cache(self):
        scores = {
            (1, 1): 0,
            (2, 1): 1,
            (3, 1): 0.5,
            (1, 2): 1,
            (2, 2): 2,
            (3, 2): 3,
            (1, 3): 2.5,
            (2, 3): 1.5,
            (3, 3): 4,
        }
        calls = []

        def score(point):
            key = (point["a.initial_lr"], point["b.momentum"])
            calls.append(key)
            return scores[key]

        best, value = train.directional_search(
            {"a.initial_lr": 1, "b.momentum": 1},
            {"a.initial_lr": [1, 2, 3], "b.momentum": [1, 2, 3]},
            score,
            initial_lr_search=False,
            factors={"conv.initial_lr": 0.6, "head.initial_lr": 0.6},
            max_side_steps=20,
            dropoff_margin=0.02,
            on_event=None,
        )
        self.assertEqual(best, {"a.initial_lr": 3, "b.momentum": 3})
        self.assertEqual(value, 4)
        self.assertEqual(len(calls), len(set(calls)))

    def test_initial_search_chooses_largest_lr_within_margin(self):
        scores = {0.5: 0.81, 1: 0.8, 2: 0.805, 4: 0.6}
        best, value = train.directional_search(
            {"conv.initial_lr": 1},
            {"conv.initial_lr": list(scores)},
            lambda point: scores[point["conv.initial_lr"]],
            initial_lr_search=True,
            factors={"conv.initial_lr": 0.6, "head.initial_lr": 0.6},
            max_side_steps=20,
            dropoff_margin=0.02,
            on_event=None,
        )
        self.assertEqual(best["conv.initial_lr"], 2)
        self.assertEqual(value, 0.805)

    def test_multiplicative_probes_round_and_bound_plateaus(self):
        calls = []

        def score(point):
            lr = point["conv.initial_lr"]
            self.assertEqual(lr, float(f"{lr:.2g}"))
            calls.append(lr)
            return 0.5

        best, _ = train.directional_search(
            {"conv.initial_lr": 0.04},
            {"conv.initial_lr": None},
            score,
            max_side_steps=5,
            initial_lr_search=False,
            factors={"conv.initial_lr": 0.6, "head.initial_lr": 0.6},
            dropoff_margin=0.02,
            on_event=None,
        )
        self.assertEqual(best["conv.initial_lr"], 0.047)
        self.assertIn(0.028, calls)
        self.assertIn(0.078, calls)
        self.assertEqual(len(calls), 9)  # center, three smaller, five larger

    def test_lr_search_snaps_start_and_probes_adjacent_exponents(self):
        calls = []

        def score(point):
            calls.append(point["head.initial_lr"])
            return -abs(point["head.initial_lr"] - 84)

        best, _ = train.directional_search(
            {"head.initial_lr": 84},
            {"head.initial_lr": None},
            score,
            initial_lr_search=False,
            factors={"conv.initial_lr": 0.6, "head.initial_lr": 0.6},
            max_side_steps=20,
            dropoff_margin=0.02,
            on_event=None,
        )
        self.assertEqual(calls, [99, 60, 170])
        self.assertEqual(best["head.initial_lr"], 99)

    def test_configured_lr_starts_snap_to_each_parameter_grid(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        for algorithm in ("interval", "global_neighbour"):
            config["hparam_tuning"] = search_config(
                algorithm=algorithm,
                params={
                    "conv.initial_lr": dict(initial=0.24, mult=0.6, choices=None),
                    "head.initial_lr": dict(initial=0.24, mult=0.8, choices=None),
                    "head.momentum": dict(initial=0.85, choices=[0.8, 0.9]),
                },
                metric="tta_val_acc",
                max_side_steps=20,
                interval_steps=40,
                cooldown_steps=40,
                initial_dropoff_margin=0.02,
            )
            choices, initial = train.interval_search_space(config)
            self.assertEqual(
                initial,
                {
                    "conv.initial_lr": 0.22,
                    "head.initial_lr": 0.26,
                    "head.momentum": 0.85,
                },
            )
            calls = []
            factors = {"conv.initial_lr": 0.6, "head.initial_lr": 0.8}

            def score(point):
                calls.append(point)
                for path, mult in factors.items():
                    self.assertIn(
                        point[path],
                        {train.round_hparam(mult**k) for k in range(-20, 30)},
                    )
                return -sum(abs(point[path] - initial[path]) for path in initial)

            best, _ = train.directional_search(
                initial,
                choices,
                score,
                factors=factors,
                initial_lr_search=False,
                max_side_steps=20,
                dropoff_margin=0.02,
                on_event=None,
            )
            self.assertEqual(calls[0], initial)
            self.assertEqual(best, initial)

    def test_lr_grid_stays_fixed_across_searches(self):
        current = {"conv.initial_lr": 0.36}
        choices = {"conv.initial_lr": None}
        grid = {train.round_hparam(0.6**k) for k in range(-20, 30)}
        for target in (0.22, 0.36):

            def score(point):
                lr = point["conv.initial_lr"]
                self.assertIn(lr, grid)
                return -((lr - target) ** 2)

            current, _ = train.directional_search(
                current,
                choices,
                score,
                initial_lr_search=False,
                factors={"conv.initial_lr": 0.6, "head.initial_lr": 0.6},
                max_side_steps=20,
                dropoff_margin=0.02,
                on_event=None,
            )
            self.assertEqual(current["conv.initial_lr"], target)

    def test_validation_uses_training_batchnorm_without_gradients(self):
        clock = dict(time=0, training=0)
        model = TinyModel(clock).eval()
        loader = TinyLoader(
            "", train=False, batch_size=2, aug=dict(flip=False, translate=0)
        )
        before = model.norm.num_batches_tracked.item()
        train.evaluate(model, loader, tta_level=0)
        self.assertTrue(model.training)
        self.assertGreater(model.norm.num_batches_tracked.item(), before)
        self.assertEqual(clock["training"], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_line_schedules_allow_jumps_and_include_both_endpoints(self):
        lines = [(3, 0.04), (3, 0.02, 0.0)]
        self.assertEqual(
            [train.hparam_at_step(lines, step) for step in range(6)],
            [0.04, 0.04, 0.04, 0.02, 0.01, 0.0],
        )
        self.assertEqual(train.hparam_at_step([(1, 0.04, 0.0)], 0), 0)
        self.assertEqual(train.hparam_at_step([(1, 0.04)], 0), 0.04)
        for step in (-1, 6):
            with self.assertRaises(ValueError):
                train.hparam_at_step(lines, step)

    def test_segment_validation_and_exact_step_counts(self):
        config = train.normalize_config(
            dict(
                batch_size=125,
                num_epochs=125,
                param_groups={
                    "conv": dict(lr_scheduler=[(125, 0.1234, 0.0), (75, 0.0)])
                },
            ),
            key=None,
        )
        self.assertEqual(config["num_epochs"], 125)
        self.assertEqual(
            config["param_groups"]["conv"]["lr_scheduler"],
            [(125, 0.12, 0.0), (75, 0.0)],
        )
        for bad in (
            [],
            [3],
            [(3,)],
            [(3, 1, 0, 0)],
            [(0, 1)],
            [(-1, 1)],
            [(1.5, 1)],
            [(True, 1)],
            [(3, -1)],
            [(3, float("nan"))],
            [(3, 1, float("inf"))],
            [(3, True)],
        ):
            with self.subTest(schedule=bad), self.assertRaises(ValueError):
                train.normalize_schedule(bad, total_steps=None, name="lr_scheduler")
        for total in (199, 201):
            with self.assertRaisesRegex(ValueError, "sum to 200"):
                train.normalize_schedule(
                    [(125, 0.1), (75, 0)], total, name="lr_scheduler"
                )
        for config in (
            train.BASELINE_RUN_CONFIGS
            + train.INTERVAL_RUN_CONFIGS
            + train.GLOBAL_NEIGHBOUR_RUN_CONFIGS
            + train.RUN_CONFIGS
        ):
            total = config["num_epochs"] * (50000 // config["batch_size"])
            schedules = train.configured_hparam_schedules(config["param_groups"], total)
            for lines in schedules.values():
                self.assertEqual(sum(line[0] for line in lines), total)

    def test_all_algorithms_reject_wrong_schedule_duration_before_training(self):
        for algorithm in (None, "grid", "coordinate", "interval", "global_neighbour"):
            for duration in (4, 6):
                with self.subTest(algorithm=algorithm, duration=duration):
                    config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                    config.update(
                        batch_size=2,
                        num_epochs=5,
                        overfit=True,
                        hparam_tuning=None
                        if algorithm is None
                        else search_config(
                            algorithm=algorithm,
                            params={
                                "conv.initial_lr": dict(initial=0.001, choices=[0.001])
                            },
                            metric="tta_val_acc",
                            max_side_steps=20,
                            interval_steps=40,
                            cooldown_steps=40,
                            initial_dropoff_margin=0.02,
                        ),
                    )
                    for group in config["param_groups"].values():
                        group["lr_scheduler"] = [(5, 0.001)]
                    config["param_groups"]["whiten_bias"]["lr_scheduler"] = [
                        (duration, 0.001)
                    ]
                    model = TinyModel(dict(time=0, training=0))
                    with (
                        patch.object(train, "CifarLoader", TinyLoader),
                        patch.object(train.GroupOptimizer, "step") as update,
                        contextlib.redirect_stdout(io.StringIO()),
                        self.assertRaisesRegex(
                            ValueError,
                            "whiten_bias.lr_scheduler.*expected total_steps=5",
                        ),
                    ):
                        train.run_experiment(0, model, config)
                    update.assert_not_called()

    def test_initial_lr_tuning_preserves_piecewise_shape_and_unsearched_groups(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config["param_groups"]["conv"]["lr_scheduler"] = [
            (2, 0.1),
            (3, 0.1, 0.02),
            (2, 0),
        ]
        changed = train.with_hparam(config, "conv.initial_lr", 0.2)
        self.assertEqual(
            changed["param_groups"]["conv"]["lr_scheduler"],
            [(2, 0.2), (3, 0.2, 0.04), (2, 0)],
        )
        for name in config["param_groups"]:
            if name != "conv":
                self.assertEqual(
                    changed["param_groups"][name], config["param_groups"][name]
                )

    def test_search_splices_cover_full_run_and_keep_committed_prefix(self):
        base = {
            "conv.initial_lr": [(3, 0.1), (4, 0.08, 0)],
            "head.initial_lr": [(2, 0.2), (5, 0.1, 0.0)],
        }
        selected = train.segment_hparam_schedules(
            base, {"conv.initial_lr": 0.04}, 3, 2, global_search=False
        )
        selected = train.segment_hparam_schedules(
            selected, {"conv.initial_lr": 0.02}, 5, 2, global_search=False
        )
        self.assertEqual(selected["conv.initial_lr"], [(3, 0.1), (2, 0.04), (2, 0.02)])
        self.assertEqual(selected["head.initial_lr"], base["head.initial_lr"])
        for lines in selected.values():
            self.assertEqual(sum(line[0] for line in lines), 7)
        self.assertEqual(
            [
                train.hparam_at_step(selected["conv.initial_lr"], step)
                for step in range(7)
            ],
            [0.1, 0.1, 0.1, 0.04, 0.04, 0.02, 0.02],
        )

    def test_global_search_replays_full_configured_decay(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config.update(
            batch_size=2,
            num_epochs=5,
            overfit=True,
            hparam_tuning=search_config(
                algorithm="global_neighbour",
                params={
                    "conv.initial_lr": dict(
                        initial=[0.001, 0.002][0], choices=[0.001, 0.002]
                    ),
                    "conv.momentum": dict(
                        initial=min(
                            [0.0, 0.6],
                            key=lambda v: abs(
                                v - train.get_hparam(config, "conv.momentum")
                            ),
                        ),
                        choices=[0.0, 0.6],
                    ),
                },
                metric="tta_val_acc",
                max_side_steps=20,
                interval_steps=1,
                cooldown_steps=4,
                initial_dropoff_margin=0.02,
            ),
        )
        for name, group in config["param_groups"].items():
            group["lr_scheduler"] = [
                (config["num_epochs"], 0 if name.endswith("_weight") else 0.0012)
            ]
        config["param_groups"]["conv"]["lr_scheduler"] = [(5, 0.001, 0.0)]
        config = train.with_hparam(config, "conv.initial_lr", 0.001)
        config = train.with_hparam(config, "conv.momentum", 0.6)
        clock = dict(time=0, training=0)
        model = TinyModel(clock)
        updates, initial_states, initial_gradients = [], [], []
        original_step = train.GroupOptimizer.step

        def step(optimizer):
            if model.training_steps.item() == 1:
                initial_states.append(copy.deepcopy(model.state_dict()))
                initial_gradients.append(model.conv.weight.grad.clone())
            updates.append(
                {g["name"]: (g["lr"], g["momentum"]) for g in optimizer.param_groups}
            )
            original_step(optimizer)

        def evaluate(model, loader, tta_level):
            self.assertEqual(model.training_steps.item(), 5)
            clock["time"] += 100
            start_lr, momentum = updates[-5]["conv"]
            # The larger LR is within 0.02 of the best, but must not win here.
            return (
                0.9
                - (0.01 if start_lr > 0.001 else 0)
                + (0.001 if momentum == 0 else 0)
            )

        output = io.StringIO()
        with (
            patch.object(train, "CifarLoader", TinyLoader),
            patch.object(train.GroupOptimizer, "step", step),
            patch.object(train, "evaluate", side_effect=evaluate),
            patch.object(
                train, "directional_search", wraps=train.directional_search
            ) as search,
            patch.object(
                train, "time", SimpleNamespace(perf_counter=lambda: clock["time"])
            ),
            contextlib.redirect_stdout(output),
        ):
            result = train.run_experiment(0, model, config)["best_result"]
        self.assertEqual(search.call_count, 1)
        self.assertFalse(search.call_args.kwargs["initial_lr_search"])
        self.assertEqual(len(result["intervals"]), 1)
        self.assertEqual(result["intervals"][0]["steps"], 5)
        self.assertEqual(result["intervals"][0]["cooldown_steps"], 0)
        self.assertEqual(
            result["intervals"][0]["main_hparams"],
            {"conv.initial_lr": 0.001, "conv.momentum": 0.0},
        )
        self.assertEqual(result["seconds"], clock["time"])
        self.assertGreater(len(initial_states), 2)
        for state, gradient in zip(initial_states, initial_gradients):
            for name, value in state.items():
                torch.testing.assert_close(
                    value, initial_states[0][name], rtol=0, atol=0
                )
            torch.testing.assert_close(gradient, initial_gradients[0], rtol=0, atol=0)
        for start in range(0, len(updates), 5):
            trial = updates[start : start + 5]
            lr, momentum = trial[0]["conv"]
            self.assertEqual(
                [u["conv"][0] for u in trial],
                [train.round_hparam(lr * f) for f in (1, 0.75, 0.5, 0.25, 0)],
            )
            self.assertEqual([u["conv"][1] for u in trial], [momentum] * 5)
            self.assertEqual([u["head"][0] for u in trial], [0.0012] * 5)
        for path, lines in result["hparam_schedules"].items():
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0][0], 5)
            if path == "conv.initial_lr":
                self.assertEqual(lines[0][2], 0)
            else:
                self.assertEqual(len(lines[0]), 2)
        self.assertEqual(result["conv_lr"], 0)
        logged, _ = json.JSONDecoder().raw_decode(output.getvalue().split("\n", 1)[1])
        self.assertEqual(logged["hparam_tuning"]["interval_steps"], 5)
        self.assertEqual(logged["hparam_tuning"]["cooldown_steps"], 0)
        self.assertNotIn("initial_dropoff_margin", logged["hparam_tuning"])
        self.assertNotIn("phase=cooldown", output.getvalue())

    def test_per_lr_multipliers_and_validation(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config["hparam_tuning"] = search_config(
            algorithm="global_neighbour",
            params={
                "conv.initial_lr": dict(initial=1, mult=0.8, choices=None),
                "head.initial_lr": dict(initial=1, choices=None, mult=0.6),
            },
            metric="tta_val_acc",
            max_side_steps=20,
            interval_steps=40,
            cooldown_steps=40,
            initial_dropoff_margin=0.02,
        )
        specs = train.tuning_params(config)
        factors = {path: spec["mult"] for path, spec in specs.items()}
        self.assertEqual(factors, {"conv.initial_lr": 0.8, "head.initial_lr": 0.6})
        choices, initial = train.interval_search_space(config)
        for initial_search in (False, True):
            calls = []

            def score(point):
                calls.append(point)
                return -sum(abs(value - 1) for value in point.values())

            train.directional_search(
                initial,
                choices,
                score,
                factors=factors,
                initial_lr_search=initial_search,
                max_side_steps=1,
                dropoff_margin=0.02,
                on_event=None,
            )
            self.assertEqual(
                [p["conv.initial_lr"] for p in calls if p["head.initial_lr"] == 1],
                [1, 1.2, 0.8] if initial_search else [1, 0.8, 1.2],
            )
            self.assertEqual(
                [p["head.initial_lr"] for p in calls if p["conv.initial_lr"] == 1],
                [1, 1.7, 0.6] if initial_search else [1, 0.6, 1.7],
            )
        for invalid in (0, 1, -0.6, float("nan"), float("inf"), True, "0.8"):
            config["hparam_tuning"]["params"]["conv.initial_lr"]["mult"] = invalid
            with self.subTest(mult=invalid), self.assertRaisesRegex(ValueError, "mult"):
                train.tuning_params(config)

    def test_interval_variants_preserve_every_unsearched_update(self):
        for algorithm in ("interval", "global_neighbour"):
            with self.subTest(algorithm=algorithm):
                config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                config.update(
                    batch_size=2,
                    num_epochs=5,
                    overfit=True,
                    hparam_tuning=search_config(
                        algorithm=algorithm,
                        params={
                            "conv.initial_lr": dict(
                                initial=0.001, mult=0.8, choices=None
                            )
                        },
                        metric="tta_val_acc",
                        max_side_steps=1,
                        interval_steps=2,
                        cooldown_steps=2,
                        initial_dropoff_margin=0.02,
                    ),
                )
                for name, group in config["param_groups"].items():
                    group["lr_scheduler"] = [
                        (5, 0 if name.endswith("_weight") else 0.001)
                    ]
                config["param_groups"]["whiten_bias"]["lr_scheduler"] = [
                    (3, 0.001, 0),
                    (2, 0),
                ]
                config["param_groups"]["head"]["lr_scheduler"] = [(5, 0.001, 0)]
                clock = dict(time=0, training=0)
                model = TinyModel(clock)
                original_step = train.GroupOptimizer.step
                updates = []

                def step(optimizer):
                    current_step = model.training_steps.item() - 1
                    updates.append(current_step)
                    for group in optimizer.param_groups:
                        baseline = config["param_groups"][group["name"]]
                        if group["name"] != "conv":
                            self.assertEqual(
                                group["lr"],
                                train.hparam_at_step(
                                    baseline["lr_scheduler"], current_step
                                ),
                            )
                        for field in ("momentum", "nesterov", "algorithm"):
                            self.assertEqual(group[field], baseline[field])
                    original_step(optimizer)

                def evaluate(model, loader, tta_level):
                    return model.training_steps.item() / 100

                with (
                    patch.object(train, "CifarLoader", TinyLoader),
                    patch.object(train.GroupOptimizer, "step", step),
                    patch.object(train, "evaluate", side_effect=evaluate),
                    patch.object(
                        train, "directional_search", wraps=train.directional_search
                    ) as search,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = train.run_experiment(0, model, config)["best_result"]
                self.assertGreater(len(updates), 5)
                if algorithm == "interval":
                    self.assertTrue(
                        any(i["cooldown_steps"] > 0 for i in result["intervals"])
                    )
                for call in search.call_args_list:
                    self.assertEqual(call.kwargs["factors"], {"conv.initial_lr": 0.8})
                for interval in result["intervals"]:
                    for lines in interval["hparam_schedules"].values():
                        self.assertEqual(sum(line[0] for line in lines), 5)
                baseline = train.configured_hparam_schedules(config["param_groups"], 5)
                for path, lines in baseline.items():
                    if path != "conv.initial_lr":
                        self.assertEqual(result["hparam_schedules"][path], lines)

    def test_invalid_interval_space_and_nesterov_zero_momentum(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config["param_groups"]["conv"]["nesterov"] = True
        config["hparam_tuning"] = search_config(
            algorithm="interval",
            params={
                "conv.momentum": dict(
                    initial=min(
                        [0, 0.6, 0.9],
                        key=lambda v: abs(
                            v - train.get_hparam(config, "conv.momentum")
                        ),
                    ),
                    choices=[0, 0.6, 0.9],
                )
            },
            metric="tta_val_acc",
            max_side_steps=20,
            interval_steps=40,
            cooldown_steps=40,
            initial_dropoff_margin=0.02,
        )
        choices, _ = train.interval_search_space(config)
        self.assertEqual(choices["conv.momentum"], [0, 0.6, 0.9])
        config["hparam_tuning"]["params"] = {
            "conv.momentum": dict(initial=0, choices=[0])
        }
        choices, initial = train.interval_search_space(config)
        self.assertEqual(choices["conv.momentum"], [0])
        self.assertEqual(initial["conv.momentum"], 0)
        for parameters in (
            {"batch_size": dict(initial=125, choices=[125])},
            {"conv.initial_lr": dict(initial=0.04, choices=[])},
            {"whiten_weight.initial_lr": dict(initial=0, choices=None, mult=0.6)},
            {"head.momentum": dict(initial=0, choices=[0])},
        ):
            config["hparam_tuning"]["params"] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                train.interval_search_space(config)

    def test_muon_zero_momentum_refreshes_buffer_before_momentum_returns(self):
        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov):
                parameter = torch.nn.Parameter(torch.randn(3, 2))
                optimizer = train.GroupOptimizer(
                    [
                        dict(
                            params=[parameter],
                            algorithm="muon",
                            lr=0.001,
                            momentum=0.0,
                            nesterov=nesterov,
                            name="test",
                            ns_steps=3,
                            ns_eps=0.0,
                        )
                    ]
                )
                expected = torch.zeros_like(parameter)
                for momentum in (0.0, 0.6, 0.0, 0.1):
                    gradient = torch.randn_like(parameter)
                    parameter.grad = gradient.clone()
                    optimizer.param_groups[0]["momentum"] = momentum
                    expected.mul_(momentum).add_(gradient)
                    direction = (
                        gradient + momentum * expected if nesterov else expected.clone()
                    )
                    before = parameter.detach().clone()
                    # Inspect Muon's input independently of its matrix transform.
                    with patch.object(
                        train,
                        "zeropower_via_newtonschulz5",
                        side_effect=lambda g, steps, eps: g,
                    ) as transform:
                        optimizer.step()
                    torch.testing.assert_close(transform.call_args.args[0], direction)
                    torch.testing.assert_close(
                        parameter,
                        before * (len(before) ** 0.5 / before.norm())
                        - 0.001 * direction,
                    )
                    torch.testing.assert_close(
                        optimizer.state[parameter]["momentum_buffer"],
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_sgdh_normalizes_each_output_channel_after_momentum(self):
        for shape in ((2, 2), (2, 1, 1, 2)):
            for nesterov in (False, True):
                with self.subTest(shape=shape, nesterov=nesterov):
                    parameter = torch.nn.Parameter(
                        torch.tensor([[3.0, 4.0], [0.0, 2.0]]).reshape(shape)
                    )
                    optimizer = train.GroupOptimizer(
                        [
                            dict(
                                params=[parameter],
                                algorithm="sgdh",
                                lr=0.1,
                                momentum=0.6,
                                nesterov=nesterov,
                                name="test",
                                normalization_eps=1e-6,
                            )
                        ]
                    )
                    buffer = torch.zeros_like(parameter)
                    for momentum, grad in (
                        (0.6, [[0.0, 2.0], [3.0, 4.0]]),
                        (0.0, [[4.0, 0.0], [0.0, 5.0]]),
                        (0.6, [[2.0, 2.0], [1.0, 0.0]]),
                    ):
                        optimizer.param_groups[0]["momentum"] = momentum
                        gradient = torch.tensor(grad).reshape(shape)
                        parameter.grad = gradient.clone()
                        before = parameter.detach().clone().reshape(2, 2)
                        buffer = momentum * buffer + gradient
                        update = (
                            gradient + momentum * buffer if nesterov else buffer
                        ).reshape(2, 2)
                        expected = before / before.norm(
                            dim=1, keepdim=True
                        ) - 0.1 * update / update.norm(dim=1, keepdim=True)
                        optimizer.step()
                        torch.testing.assert_close(parameter.reshape(2, 2), expected)
                        torch.testing.assert_close(
                            optimizer.state[parameter]["momentum_buffer"], buffer
                        )
                        torch.testing.assert_close(
                            parameter.grad, gradient, rtol=0, atol=0
                        )

        # The actual convolution weights use channels-last storage: normalizing
        # a flattened reshape in place could otherwise modify only a temporary.
        parameter = torch.nn.Parameter(
            torch.randn(4, 3, 2, 2).to(memory_format=torch.channels_last)
        )
        parameter.grad = torch.randn_like(parameter)
        weights = parameter.detach().clone().reshape(4, -1)
        gradient = parameter.grad.clone().reshape(4, -1)
        expected = weights / weights.norm(
            dim=1, keepdim=True
        ) - 0.1 * gradient / gradient.norm(dim=1, keepdim=True)
        optimizer = train.GroupOptimizer(
            [
                dict(
                    params=[parameter],
                    algorithm="sgdh",
                    lr=0.1,
                    momentum=0,
                    nesterov=True,
                    name="test",
                    normalization_eps=1e-6,
                )
            ]
        )
        optimizer.step()
        torch.testing.assert_close(parameter.reshape(4, -1), expected)
        self.assertTrue(parameter.is_contiguous(memory_format=torch.channels_last))

    def test_sgdh_zero_rows_and_zero_lr_are_safe(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for dtype in (torch.float32, torch.float16):
                with self.subTest(device=device, dtype=dtype):
                    parameter = torch.nn.Parameter(
                        torch.tensor(
                            [[0.0, 0.0], [3.0, 4.0]], device=device, dtype=dtype
                        )
                    )
                    optimizer = train.GroupOptimizer(
                        [
                            dict(
                                params=[parameter],
                                algorithm="sgdh",
                                lr=0.0,
                                momentum=0.0,
                                nesterov=True,
                                name="test",
                                normalization_eps=1e-6,
                            )
                        ]
                    )
                    parameter.grad = torch.zeros_like(parameter)
                    before = parameter.detach().clone()
                    optimizer.step()
                    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                    optimizer.param_groups[0]["lr"] = 0.1
                    optimizer.step()
                    torch.testing.assert_close(
                        parameter,
                        torch.tensor(
                            [[0.0, 0.0], [0.6, 0.8]], device=device, dtype=dtype
                        ),
                    )
                    self.assertTrue(torch.isfinite(parameter).all())
        with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
            train.GroupOptimizer(
                [
                    dict(
                        params=[torch.nn.Parameter(torch.ones(2))],
                        algorithm="sgdh",
                        lr=0.1,
                        momentum=0,
                        nesterov=False,
                        name="test",
                        normalization_eps=1e-6,
                    )
                ]
            )

    def test_stream_replays_across_augmented_epoch_boundary(self):
        loader = TinyLoader(
            "", aug=dict(flip=True, translate=2), train=True, batch_size=2
        )
        loader.normalized_images()
        stream = train.TrainingBatchStream(loader, fixed_batch=None)
        stream.next_batch()
        state, rng = stream.state_dict(), torch.random.get_rng_state()
        expected = [stream.next_batch() for _ in range(5)]
        stream.load_state_dict(state)
        torch.random.set_rng_state(rng)
        actual = [stream.next_batch() for _ in range(5)]
        for first, second in zip(expected, actual):
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_input_conditioned_uses_sgd_direction_and_batch_mean_covariance(self):
        inputs = torch.tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
        covariance = inputs.T @ inputs / len(inputs)
        for nesterov in (False, True):
            parameter = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.float64))
            reference = torch.nn.Parameter(parameter.detach().clone())
            optimizer = train.GroupOptimizer(
                [
                    dict(
                        params=[parameter],
                        algorithm="input_conditioned",
                        lr=0.1,
                        momentum=0.6,
                        nesterov=nesterov,
                        svd_mean_percentage_damping=0.2,
                        name="test",
                        input_conditioner_momentum=0.0,
                        gradient_momentum_before_conditioning=True,
                    )
                ]
            )
            sgd = torch.optim.SGD([reference], lr=0.1, momentum=0.6, nesterov=nesterov)
            for step, momentum in enumerate(
                (0.6, 0.3, 0.7) if nesterov else (0.6, 0.0, 0.7)
            ):
                gradient = (
                    torch.tensor(
                        [[1.0, 2.0], [4.0, 1.0], [3.0, -2.0]], dtype=torch.float64
                    )
                    + step
                )
                parameter.grad = gradient.clone()
                reference.grad = gradient.clone()
                optimizer.param_groups[0]["momentum"] = momentum
                sgd.param_groups[0]["momentum"] = momentum
                before_sgd = reference.detach().clone()
                sgd.step()
                sgd_direction = (before_sgd - reference.detach()) / 0.1
                damped = covariance + 0.2 * covariance.trace() / 2 * torch.eye(
                    2, dtype=torch.float64
                )
                expected_direction = torch.linalg.solve(damped, sgd_direction.T).T
                before = parameter.detach().clone()
                optimizer.record_input(parameter, inputs)
                optimizer.step()
                torch.testing.assert_close(parameter, before - 0.1 * expected_direction)
                torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                torch.testing.assert_close(
                    optimizer.state[parameter]["momentum_buffer"],
                    sgd.state[reference]["momentum_buffer"],
                )
                self.assertFalse(optimizer._input_covariances)

    def test_input_conditioned_momentum_after_conditioning(self):
        # Orders only differ when the covariance changes, so vary the inputs.
        batches = [
            torch.tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
            * (step + 1)
            + torch.tensor([[0.0, step]], dtype=torch.float64)
            for step in range(3)
        ]
        for nesterov in (False, True):
            finals = {}
            for momentum_first in (False, True):
                parameter = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.float64))
                reference = torch.nn.Parameter(parameter.detach().clone())
                optimizer = train.GroupOptimizer(
                    [
                        dict(
                            params=[parameter],
                            algorithm="input_conditioned",
                            lr=0.1,
                            momentum=0.6,
                            nesterov=nesterov,
                            svd_mean_percentage_damping=0.2,
                            input_conditioner_momentum=0.0,
                            gradient_momentum_before_conditioning=momentum_first,
                            name="test",
                        )
                    ]
                )
                sgd = torch.optim.SGD(
                    [reference], lr=0.1, momentum=0.6, nesterov=nesterov
                )
                for step, inputs in enumerate(batches):
                    gradient = (
                        torch.tensor(
                            [[1.0, 2.0], [4.0, 1.0], [3.0, -2.0]], dtype=torch.float64
                        )
                        + step
                    )
                    covariance = inputs.T @ inputs / len(inputs)
                    damped = covariance + 0.2 * covariance.trace() / 2 * torch.eye(
                        2, dtype=torch.float64
                    )
                    parameter.grad = gradient.clone()
                    reference.grad = torch.linalg.solve(damped, gradient.T).T
                    optimizer.record_input(parameter, inputs)
                    optimizer.step()
                    sgd.step()
                    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                    if not momentum_first:
                        # Plain SGD momentum over the conditioned gradients.
                        torch.testing.assert_close(parameter, reference)
                        torch.testing.assert_close(
                            optimizer.state[parameter]["momentum_buffer"],
                            sgd.state[reference]["momentum_buffer"],
                        )
                finals[momentum_first] = parameter.detach().clone()
            self.assertFalse(torch.allclose(finals[False], finals[True]))

    def test_input_conditioned_singular_covariance_and_half_precision(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for dtype in (torch.float32, torch.float16):
                parameter = torch.nn.Parameter(
                    torch.zeros(2, 3, device=device, dtype=dtype)
                )
                optimizer = train.GroupOptimizer(
                    [
                        dict(
                            params=[parameter],
                            algorithm="input_conditioned",
                            lr=0.1,
                            momentum=0.0,
                            nesterov=False,
                            svd_mean_percentage_damping=0.0,
                            name="test",
                            input_conditioner_momentum=0.0,
                            gradient_momentum_before_conditioning=True,
                        )
                    ]
                )
                # N=1 < D=3: covariance is diag(4, 0, 0).
                inputs = torch.tensor([[2.0, 0.0, 0.0]], device=device, dtype=dtype)
                parameter.grad = torch.tensor(
                    [[4.0, 9.0, 2.0], [8.0, 3.0, 6.0]], device=device, dtype=dtype
                )
                optimizer.record_input(parameter, inputs)
                optimizer.step()
                torch.testing.assert_close(
                    parameter,
                    torch.tensor(
                        [[-0.1, 0, 0], [-0.2, 0, 0]], device=device, dtype=dtype
                    ),
                )
                optimizer.param_groups[0]["lr"] = 0
                before = parameter.detach().clone()
                optimizer.step()
                torch.testing.assert_close(parameter, before, rtol=0, atol=0)

    def test_conditioner_damping_and_covariance_ema(self):
        parameter = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))
        optimizer = train.GroupOptimizer(
            [
                dict(
                    params=[parameter],
                    algorithm="input_conditioned",
                    lr=0.1,
                    momentum=0,
                    nesterov=False,
                    svd_mean_percentage_damping=0.5,
                    input_conditioner_momentum=0.75,
                    gradient_momentum_before_conditioning=True,
                    name="test",
                )
            ]
        )
        gradient = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        batches = [
            torch.tensor([[2.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        ]
        # Initialize from the first batch, then 0.75*old + 0.25*new.
        for inputs, expected_diagonal in zip(batches, ([4.0, 0.0], [3.0, 1.0])):
            parameter.grad = gradient.clone()
            before = parameter.detach().clone()
            optimizer.record_input(parameter, inputs)
            optimizer.step()
            diagonal = torch.tensor(expected_diagonal, dtype=torch.float64)
            torch.testing.assert_close(
                optimizer.state[parameter]["input_covariance"], torch.diag(diagonal)
            )
            expected = before - 0.1 * gradient / (diagonal + 0.5 * diagonal.mean())
            torch.testing.assert_close(parameter, expected)
        # Zero EMA momentum immediately replaces the history; zero LR still
        # records the latest covariance without changing the weights.
        optimizer.param_groups[0].update(input_conditioner_momentum=0, lr=0)
        before = parameter.detach().clone()
        optimizer.record_input(parameter, batches[0])
        optimizer.step()
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state[parameter]["input_covariance"],
            torch.diag(torch.tensor([4.0, 0.0], dtype=torch.float64)),
        )
        optimizer.param_groups[0]["lr"] = 0.1
        optimizer.record_input(parameter, torch.zeros_like(batches[0]))
        parameter.grad.zero_()
        optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())

    def test_conditioner_ema_checkpoint_preserves_precision_and_storage(self):
        for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
            parameter = torch.nn.Parameter(
                torch.zeros(1, 2, device=device, dtype=torch.float16)
            )
            optimizer = train.GroupOptimizer(
                [
                    dict(
                        params=[parameter],
                        algorithm="input_conditioned",
                        lr=0.001,
                        momentum=0,
                        nesterov=False,
                        svd_mean_percentage_damping=0.1,
                        input_conditioner_momentum=0.9,
                        gradient_momentum_before_conditioning=True,
                        name="test",
                    )
                ]
            )
            parameter.grad = torch.ones_like(parameter)
            inputs = torch.tensor(
                [[1.234, 2.345], [0.31, 0.74]], device=device, dtype=torch.float16
            )
            optimizer.record_input(parameter, inputs)
            optimizer.step()
            checkpoint = copy.deepcopy(optimizer.state_dict())
            saved = next(iter(checkpoint["state"].values()))["input_covariance"]
            reference = saved.clone()
            weights = parameter.detach().clone()

            def trial():
                with torch.no_grad():
                    parameter.copy_(weights)
                optimizer.load_state_dict(checkpoint)
                restored = optimizer.state[parameter]["input_covariance"]
                self.assertEqual(restored.dtype, torch.float32)
                self.assertNotEqual(restored.data_ptr(), saved.data_ptr())
                torch.testing.assert_close(restored, reference, rtol=0, atol=0)
                optimizer.record_input(parameter, inputs * 2)
                optimizer.step()
                return parameter.detach().clone(), optimizer.state[parameter][
                    "input_covariance"
                ].clone()

            first, second = trial(), trial()
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(saved, reference, rtol=0, atol=0)

    def test_input_capture_ignores_evaluation_and_replaces_hooks_between_runs(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config["param_groups"]["head"].update(
            algorithm="input_conditioned",
            svd_mean_percentage_damping=0.01,
            input_conditioner_momentum=0.0,
            gradient_momentum_before_conditioning=True,
        )
        model = TinyModel(dict(time=0, training=0))
        optimizer = train.make_optimizer(model, config["param_groups"])
        inputs = torch.randn(5, 4)
        model.head(inputs)
        expected = inputs.T @ inputs / 5
        torch.testing.assert_close(
            optimizer._input_covariances[model.head.weight], expected
        )
        with torch.inference_mode():
            model.head(inputs * 10)
        torch.testing.assert_close(
            optimizer._input_covariances[model.head.weight], expected
        )
        snapshot = copy.deepcopy(optimizer.state_dict())
        model.head(inputs * 2)
        optimizer.load_state_dict(snapshot)
        self.assertFalse(optimizer._input_covariances)
        model.head(inputs)
        optimizer.zero_grad(set_to_none=True)
        self.assertFalse(optimizer._input_covariances)
        replacement = train.make_optimizer(model, config["param_groups"])
        self.assertEqual(len(model.head._forward_pre_hooks), 1)
        model.head(inputs)
        self.assertFalse(optimizer._input_covariances)
        self.assertIn(model.head.weight, replacement._input_covariances)
        config["param_groups"]["head"]["algorithm"] = "sgd"
        for field in train.INPUT_CONDITIONER_FIELDS:
            del config["param_groups"]["head"][field]
        train.make_optimizer(model, config["param_groups"])
        self.assertEqual(len(model.head._forward_pre_hooks), 0)

    def test_conditioner_options_are_validated_and_searchable_by_all_algorithms(self):
        for field, invalid_values in {
            "svd_mean_percentage_damping": [-1, float("inf"), True],
            "input_conditioner_momentum": [-0.1, 1, 0.999, float("nan"), True],
            "gradient_momentum_before_conditioning": [0, 1, 1.0, None, "True"],
        }.items():
            for value in invalid_values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    train.input_conditioner_options(
                        dict(
                            {
                                "svd_mean_percentage_damping": 0.01,
                                "input_conditioner_momentum": 0.0,
                                "gradient_momentum_before_conditioning": True,
                            },
                            **{field: value},
                        )
                    )
        for algorithm in ("grid", "coordinate", "interval", "global_neighbour"):
            with self.subTest(algorithm=algorithm):
                config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                config.update(
                    batch_size=2,
                    num_epochs=3,
                    overfit=True,
                    hparam_tuning=search_config(
                        algorithm=algorithm,
                        params={
                            "head.svd_mean_percentage_damping": dict(
                                initial=0.1, choices=[0.1, 0.2]
                            ),
                            "head.input_conditioner_momentum": dict(
                                initial=0.0, choices=[0.0, 0.5]
                            ),
                        },
                        metric="tta_val_acc",
                        max_side_steps=20,
                        interval_steps=2,
                        cooldown_steps=1,
                        initial_dropoff_margin=0.02,
                    ),
                )
                for name, group in config["param_groups"].items():
                    group["lr_scheduler"] = [
                        (3, 0 if name.endswith("_weight") else 0.001)
                    ]
                config["param_groups"]["head"].update(
                    algorithm="input_conditioned",
                    svd_mean_percentage_damping=0.1,
                    input_conditioner_momentum=0,
                    gradient_momentum_before_conditioning=True,
                )
                model = TinyModel(dict(time=0, training=0))
                options_seen = []
                original_step = train.GroupOptimizer.step

                def step(optimizer):
                    head = next(
                        g for g in optimizer.param_groups if g["name"] == "head"
                    )
                    options_seen.append(
                        (
                            head["svd_mean_percentage_damping"],
                            head["input_conditioner_momentum"],
                        )
                    )
                    original_step(optimizer)

                def evaluate(model, loader, tta_level):
                    return (
                        sum(options_seen[-1]) / 10 + model.training_steps.item() / 100
                    )

                with (
                    patch.object(train, "CifarLoader", TinyLoader),
                    patch.object(train.GroupOptimizer, "step", step),
                    patch.object(train, "evaluate", side_effect=evaluate),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = train.run_experiment(0, model, config)
                self.assertIn((0.1, 0.0), options_seen)
                self.assertIn((0.2, 0.5), options_seen)
                if algorithm in ("interval", "global_neighbour"):
                    self.assertEqual(options_seen[-1], (0.2, 0.5))
                    for field, expected in (
                        ("svd_mean_percentage_damping", 0.2),
                        ("input_conditioner_momentum", 0.5),
                    ):
                        lines = result["best_result"]["hparam_schedules"][
                            f"head.{field}"
                        ]
                        self.assertEqual(sum(line[0] for line in lines), 3)
                        self.assertTrue(
                            all(
                                len(line) == 2 and line[1] == expected for line in lines
                            )
                        )
                else:
                    best = result["best_config"]["param_groups"]["head"]
                    self.assertEqual(
                        (
                            best["svd_mean_percentage_damping"],
                            best["input_conditioner_momentum"],
                        ),
                        (0.2, 0.5),
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_interval_variants_with_real_scores_and_multiple_candidates(self):
        class CudaLoader(TinyLoader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.images = self.images.cuda()
                self.labels = self.labels.cuda()

        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
        config.update(
            batch_size=2,
            num_epochs=3,
            overfit=True,
            hparam_tuning=search_config(
                algorithm="interval",
                params={
                    "conv.initial_lr": dict(
                        initial=train.get_hparam(config, "conv.initial_lr"),
                        choices=None,
                        mult=0.6,
                    ),
                    "conv.momentum": dict(
                        initial=min(
                            [0.3, 0.6],
                            key=lambda v: abs(
                                v - train.get_hparam(config, "conv.momentum")
                            ),
                        ),
                        choices=[0.3, 0.6],
                    ),
                },
                metric="tta_val_acc",
                max_side_steps=2,
                interval_steps=2,
                cooldown_steps=2,
                initial_dropoff_margin=0.02,
            ),
        )
        for name, group in config["param_groups"].items():
            group["lr_scheduler"] = [
                (config["num_epochs"], 0 if name.endswith("_weight") else 0.0012)
            ]
        config["param_groups"]["head"].update(
            algorithm="input_conditioned",
            svd_mean_percentage_damping=0.01,
            input_conditioner_momentum=0.0,
            gradient_momentum_before_conditioning=True,
        )
        original = json.dumps(config)
        clock = dict(time=0, training=0)
        model = TinyModel(clock).cuda()
        with (
            patch.object(train, "CifarLoader", CudaLoader),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = train.run_experiment(0, model, config)["best_result"]
        self.assertEqual(model.training_steps.item(), 3)
        self.assertEqual([item["steps"] for item in result["intervals"]], [2, 1])
        self.assertEqual(
            [item["cooldown_steps"] for item in result["intervals"]], [1, 0]
        )
        self.assertGreater(clock["training"], 3)
        self.assertGreater(result["seconds"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertEqual(json.dumps(config), original)
        for interval in result["intervals"]:
            for value in interval["main_hparams"].values():
                self.assertEqual(value, float(f"{value:.2g}"))

        config["hparam_tuning"]["algorithm"] = "global_neighbour"
        config["hparam_tuning"]["params"]["conv.decay"] = dict(
            initial="linear_decay", choices=["linear_decay"]
        )
        for field in ("interval_steps", "cooldown_steps", "initial_dropoff_margin"):
            del config["hparam_tuning"][field]
        with (
            patch.object(train, "CifarLoader", CudaLoader),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = train.run_experiment(0, model, config)["best_result"]
        self.assertEqual(model.training_steps.item(), 3)
        self.assertEqual([item["steps"] for item in result["intervals"]], [3])
        self.assertEqual([item["cooldown_steps"] for item in result["intervals"]], [0])
        self.assertEqual(result["conv_lr"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_search_checkpoints_do_not_alias_live_momentum_buffers(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for algorithm in ("interval", "global_neighbour"):
                with self.subTest(device=device, algorithm=algorithm):

                    class DeviceLoader(TinyLoader):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.images = self.images.to(device)
                            self.labels = self.labels.to(device)

                    config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                    config.update(
                        batch_size=2,
                        num_epochs=9,
                        overfit=True,
                        hparam_tuning=search_config(
                            algorithm=algorithm,
                            params={
                                "conv.initial_lr": dict(
                                    initial=[0.001, 0.002][0], choices=[0.001, 0.002]
                                ),
                                "conv.momentum": dict(
                                    initial=min(
                                        [0.0, 0.6],
                                        key=lambda v: abs(
                                            v
                                            - train.get_hparam(config, "conv.momentum")
                                        ),
                                    ),
                                    choices=[0.0, 0.6],
                                ),
                                "head.momentum": dict(
                                    initial=min(
                                        [0.3, 0.85],
                                        key=lambda v: abs(
                                            v
                                            - train.get_hparam(config, "head.momentum")
                                        ),
                                    ),
                                    choices=[0.3, 0.85],
                                ),
                                "norm_bias.momentum": dict(
                                    initial=min(
                                        [0.3, 0.85],
                                        key=lambda v: abs(
                                            v
                                            - train.get_hparam(
                                                config, "norm_bias.momentum"
                                            )
                                        ),
                                    ),
                                    choices=[0.3, 0.85],
                                ),
                            },
                            metric="tta_val_acc",
                            max_side_steps=20,
                            interval_steps=4,
                            cooldown_steps=3,
                            initial_dropoff_margin=0.02,
                        ),
                    )
                    for name, group in config["param_groups"].items():
                        group["lr_scheduler"] = [
                            (9, 0 if name.endswith("_weight") else 0.0012)
                        ]
                    config["param_groups"]["head"].update(
                        algorithm="input_conditioned",
                        input_conditioner_momentum=0.9,
                        gradient_momentum_before_conditioning=True,
                        svd_mean_percentage_damping=0.01,
                    )
                    model = TinyModel(dict(time=0, training=0)).to(device)
                    checkpoints, restores = [], []
                    original_step = train.GroupOptimizer.step

                    def watch_copy(value):
                        copied = copy.deepcopy(value)
                        if (
                            isinstance(value, dict)
                            and {"state", "param_groups"} <= value.keys()
                        ):
                            if any(
                                value is checkpoint for checkpoint, _ in checkpoints
                            ):
                                restores.append(value)
                            else:
                                # Observe the actual checkpoint returned to snapshot().
                                checkpoints.append((copied, copy.deepcopy(copied)))
                        return copied

                    def checked_step(optimizer):
                        original_step(optimizer)
                        live_buffers = {
                            value.data_ptr()
                            for state in optimizer.state.values()
                            for value in state.values()
                            if torch.is_tensor(value)
                        }
                        for checkpoint, baseline in checkpoints:
                            self.assertEqual(
                                checkpoint["param_groups"], baseline["param_groups"]
                            )
                            self.assertEqual(
                                checkpoint["state"].keys(), baseline["state"].keys()
                            )
                            for index, state in checkpoint["state"].items():
                                for field, buffer in state.items():
                                    self.assertNotIn(
                                        buffer.data_ptr(),
                                        live_buffers,
                                        f"Saved {field} aliases a live optimizer buffer",
                                    )
                                    torch.testing.assert_close(
                                        buffer,
                                        baseline["state"][index][field],
                                        rtol=0,
                                        atol=0,
                                    )

                    with (
                        patch.object(train, "CifarLoader", DeviceLoader),
                        patch.object(
                            train, "copy", SimpleNamespace(deepcopy=watch_copy)
                        ),
                        patch.object(train.GroupOptimizer, "step", checked_step),
                        patch.object(
                            train,
                            "evaluate",
                            side_effect=lambda model, loader, tta_level=0: (
                                model.training_steps.item() / 100
                            ),
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        train.run_experiment(0, model, config)
                    self.assertGreater(len(restores), 1)
                    if algorithm == "interval":
                        # Cooldown and later main checkpoints must contain both optimizers' buffers.
                        populated = [
                            state for state, _ in checkpoints if state["state"]
                        ]
                        self.assertGreater(len(populated), 1)
                        self.assertEqual(
                            {g["algorithm"] for g in populated[0]["param_groups"]},
                            {"sgd", "muon", "input_conditioned"},
                        )
                    else:
                        # Full-run trials all restore the pristine optimizer, with no buffers yet.
                        self.assertTrue(
                            all(not state["state"] for state, _ in checkpoints)
                        )

    def test_interval_replay_matches_committed_training_and_includes_eval_time(self):
        # Dropout, augmentation, BatchNorm buffers and optimizer momentum all
        # must match a plain run after discarding every probe and cooldown.
        for overfit in (False, True):
            with self.subTest(overfit=overfit):
                config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[0])
                config.update(
                    batch_size=2,
                    num_epochs=9 if overfit else 3,
                    overfit=overfit,
                    hparam_tuning=None,
                )
                config = train.with_hparam(config, "conv.momentum", 0.6)
                for name, group in config["param_groups"].items():
                    group["lr_scheduler"] = [
                        (9, 0 if name.endswith("_weight") else 0.0012)
                    ]
                models, optimizers = [], []
                make_optimizer = train.make_optimizer

                def capture_optimizer(*args):
                    optimizer = make_optimizer(*args)
                    optimizers.append(optimizer)
                    return optimizer

                for interval in (False, True):
                    clock = dict(time=0, training=0)
                    model = TinyModel(clock)
                    models.append(model)

                    def evaluate(model, loader, tta_level):
                        model.train()
                        clock["time"] += 100
                        return model.training_steps.item() / 100

                    if interval:
                        config["hparam_tuning"] = search_config(
                            algorithm="interval",
                            params={
                                "conv.initial_lr": dict(
                                    initial=[0.0012][0], choices=[0.0012]
                                ),
                                "conv.momentum": dict(
                                    initial=min(
                                        [0.6],
                                        key=lambda v: abs(
                                            v
                                            - train.get_hparam(config, "conv.momentum")
                                        ),
                                    ),
                                    choices=[0.6],
                                ),
                            },
                            metric="tta_val_acc",
                            max_side_steps=20,
                            interval_steps=4,
                            cooldown_steps=3,
                            initial_dropoff_margin=0.02,
                        )
                    output = io.StringIO()
                    with (
                        patch.object(train, "CifarLoader", TinyLoader),
                        patch.object(
                            train, "make_optimizer", side_effect=capture_optimizer
                        ),
                        patch.object(train, "evaluate", side_effect=evaluate),
                        patch.object(
                            train,
                            "time",
                            SimpleNamespace(
                                perf_counter=lambda: clock["time"], monotonic=lambda: 0
                            ),
                        ),
                        contextlib.redirect_stdout(output),
                    ):
                        result = train.run_experiment(0, model, config)["best_result"]
                    self.assertEqual(model.training_steps.item(), 9)
                    self.assertEqual(result["seconds"], clock["time"])
                    self.assertGreater(result["seconds"], clock["training"])
                    log = output.getvalue()
                    header, body = log.split("\n", 1)
                    logged_config, end = json.JSONDecoder().raw_decode(body)
                    lines = body[end:].strip().splitlines()
                    self.assertEqual(logged_config["batch_size"], 2)
                    self.assertNotIn("%", log)
                    self.assertTrue(
                        lines[-1].startswith(
                            "val_acc=0.0900 tta_val_acc=0.0900 seconds="
                        )
                    )
                    if interval:
                        self.assertEqual(header, "base_config run=0")
                        self.assertEqual(log.count("train_hparams "), 3)
                        self.assertEqual(log.count("interval_boundary_eval "), 3)
                        self.assertNotIn("train_loss ", log)
                        self.assertIn(
                            "main hparams: conv.initial_lr=0.0012 conv.momentum=0.6 main=0.04",
                            log,
                        )
                        self.assertIn("conv.initial_lr=0.0012 -> tta_val_acc=0.07", log)
                        self.assertIn("search_path step=0 conv.initial_lr=0.0012", log)
                        self.assertIn("phase=cooldown", log)
                        self.assertNotIn("candidate_start ", log)
                        self.assertNotIn("candidate_result ", log)
                        self.assertNotIn("main_diff:", log)
                        self.assertNotIn("cooldown_diff:", log)
                        self.assertIn("best_cooldown=0.07", log)
                        first_end = next(
                            line
                            for line in lines
                            if line.startswith("interval_boundary_eval ")
                        )
                        self.assertIn("tta_val_acc=0.04", first_end)
                        self.assertIn("step=4", first_end)
                        self.assertGreater(clock["training"], 9)
                        self.assertEqual(
                            [i["steps"] for i in result["intervals"]], [4, 4, 1]
                        )
                        self.assertEqual(
                            [i["cooldown_steps"] for i in result["intervals"]],
                            [3, 1, 0],
                        )
                        self.assertEqual(
                            result["intervals"][0]["cooldown_hparams"],
                            {"conv.initial_lr": 0.0012},
                        )
                        for path in ("conv.initial_lr", "conv.momentum"):
                            schedule = result["hparam_schedules"][path]
                            self.assertEqual([line[0] for line in schedule], [4, 4, 1])
                            self.assertTrue(all(len(line) == 2 for line in schedule))
                    else:
                        self.assertEqual(header, "config run=0")
                        self.assertEqual(len(lines), 1)
                for name, value in models[0].state_dict().items():
                    torch.testing.assert_close(
                        value, models[1].state_dict()[name], rtol=0, atol=0
                    )
                states = [o.state_dict()["state"] for o in optimizers]
                for index in states[0]:
                    torch.testing.assert_close(
                        states[0][index]["momentum_buffer"],
                        states[1][index]["momentum_buffer"],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main(num_epochs=8, overfit=False, hparam_tuning=None, _log_config=True)
