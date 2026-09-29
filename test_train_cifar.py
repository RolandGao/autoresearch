import contextlib
import copy
import io
import json
import unittest
import ast
import inspect
from itertools import product
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

    def test_compile_warmup_preserves_state_and_honors_new_hooks(self):
        model = torch.nn.Sequential()
        model.add_module("conv", torch.nn.Conv2d(3, 2, 1))
        model.add_module("norm", torch.nn.BatchNorm2d(2))
        model.add_module("pool", torch.nn.AdaptiveAvgPool2d(1))
        model.add_module("flatten", torch.nn.Flatten())
        model.add_module("head", torch.nn.Linear(2, 10))
        model.conv.eval()
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        original_gradient = parameter.grad
        state = copy.deepcopy(model.state_dict())
        modes = {module: module.training for module in model.modules()}
        rng = torch.random.get_rng_state().clone()
        compile_model = model.compile
        configs = [dict(
            batch_size=2,
            hparam_tuning=dict(params={"batch_size": dict(choices=[2, 3])}),
        )]
        with (
            patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", True),
            patch.object(
                model, "compile",
                side_effect=lambda **kwargs: compile_model(backend="eager"),
            ) as compile_call,
            patch.object(train, "infer", wraps=train.infer) as infer_call,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train.compile_and_warmup_model(model, configs)
            compile_call.assert_called_once_with(mode=train.MODEL_COMPILE_MODE)
            self.assertEqual(
                [call.kwargs["tta_level"] for call in infer_call.call_args_list],
                [0, 2] * train.MODEL_WARMUP_STEPS,
            )
            self.assertFalse(torch._dynamo.config.skip_nnmodule_hook_guards)
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            self.assertEqual({module: module.training for module in model.modules()}, modes)
            self.assertIs(parameter.grad, original_gradient)
            captured = []
            handle = model.head.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach())
            )
            try:
                model.train()
                model(torch.randn(2, 3, 32, 32)).sum().backward()
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0].shape, (2, 2))
            finally:
                handle.remove()
                torch._dynamo.reset()

    def test_experiment_configs_and_fixed_nesterov(self):
        decays = ("linear_decay", "constant")
        exp4_names = [
            f"exp4_{decay}_momentum_{order}_conditioning"
            for decay in decays
            for order in ("before", "after")
        ]
        self.assertEqual(
            [name for name in train.EXPERIMENT_RUN_CONFIGS if name == "exp3" or name.startswith("exp4_")],
            ["exp3"] + exp4_names,
        )
        # LR decay is fixed per run instead of searched.
        expected_paths = {
            "exp3": ["conv.initial_lr", "conv.momentum"],
            "exp4": [
                "head.initial_lr",
                "head.momentum",
                "head.svd_mean_percentage_damping",
                "head.input_conditioner_momentum",
            ],
        }
        baseline = train.BASELINE_RUN_CONFIGS[2]
        for name, original in train.EXPERIMENT_RUN_CONFIGS.items():
            if name != "exp3" and not name.startswith("exp4_"):
                continue
            experiment = name.split("_")[0]
            decay = "constant" if "_constant" in name else "linear_decay"
            changed = ("conv",) if experiment == "exp3" else ("head",)
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
            if experiment == "exp3":
                self.assertEqual(choices["conv.momentum"], train.MOMENTUM_CHOICES)
                self.assertEqual(initial["conv.momentum"], 0.7)
            else:
                self.assertEqual(choices["head.momentum"], train.MOMENTUM_CHOICES)
                self.assertEqual(initial["head.momentum"], 0.85)
            model = TinyModel(dict(time=0, training=0))
            train.make_optimizer(model, config["param_groups"])
        runs = train.EXPERIMENT_RUN_CONFIGS
        conv = runs["exp3"]["param_groups"]["conv"]
        self.assertEqual(conv["algorithm"], "muon2")
        self.assertEqual(conv["momentum_version"], 3)
        self.assertNotIn("ns_steps", conv)
        self.assertNotIn("ns_eps", conv)
        for name in exp4_names:
            head = runs[name]["param_groups"]["head"]
            self.assertEqual(head["algorithm"], "input_conditioned")
            self.assertEqual(head["momentum_version"], 3)
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

    def test_head_grid_covers_all_requested_combinations(self):
        grids = {
            "exp5_head_grid": {100, 170, 280, 470, 780, 1300, 2200, 3600, 6000, 10000, 17000},
            "exp5_head_grid_high_lr": {28000, 46000, 77000, 130000, 210000},
        }
        self.assertTrue(grids["exp5_head_grid"].isdisjoint(grids["exp5_head_grid_high_lr"]))
        grids["exp6_head_sgd_grid"] = grids["exp5_head_grid"]
        grids["exp7_head_lion_grid"] = grids["exp5_head_grid"]
        grids["exp8_head_lion_zero_momentum_grid"] = {10.0 ** k for k in range(-5, 3)}
        grids["exp9_head_lion_grid"] = {train.round_hparam(0.6 ** k) for k in range(-10, 6)}
        v3_grids = ("exp10_head_sgd_v3_grid", "exp11_head_input_conditioned_v3_grid")
        for name in v3_grids:
            exponents = range(4, 9) if name == "exp10_head_sgd_v3_grid" else range(-8, 4)
            grids[name] = {train.round_hparam(6000 * 0.6 ** k) for k in exponents}
        for name, expected_lrs in grids.items():
            config = train.EXPERIMENT_RUN_CONFIGS[name]
            self.assertEqual(config["batch_size"], 2000)
            self.assertEqual(config["num_epochs"], 8)
            self.assertFalse(config["overfit"])
            self.assertEqual(config["hparam_tuning"]["algorithm"], "grid")
            self.assertEqual(config["hparam_tuning"]["metric"], "tta_val_acc")
            is_lion = name in (
                "exp7_head_lion_grid", "exp8_head_lion_zero_momentum_grid", "exp9_head_lion_grid"
            )
            versions = (3,)
            nesterov_choices = (True, False) if name in v3_grids else (True,)
            momenta = (0,) if name == "exp8_head_lion_zero_momentum_grid" else (0, 0.5, 0.7, 0.8, 0.9)
            decays = ("linear_decay",) if name in (*v3_grids, "exp9_head_lion_grid") else ("constant", "linear_decay")
            algorithm = (
                "sgd" if name in ("exp6_head_sgd_grid", "exp10_head_sgd_v3_grid") else
                "lion" if is_lion else "input_conditioned"
            )
            if algorithm == "lion":
                self.assertNotIn("head.momentum_version", config["hparam_tuning"]["params"])
            seen = set()

            def evaluate(run, model, **candidate):
                head = candidate["param_groups"]["head"]
                for name, group in candidate["param_groups"].items():
                    train.GroupOptimizer.validate_group(group, runtime=False)
                    if name == "conv":
                        self.assertEqual(group["algorithm"], "muon2")
                        self.assertEqual(group["lr_scheduler"], [(200, 0.22, 0.0)])
                        self.assertEqual(group["momentum"], 0.7)
                        self.assertEqual(group["momentum_version"], 3)
                        self.assertTrue(group["nesterov"])
                    elif name != "head":
                        self.assertEqual(
                            group, train.BASELINE_RUN_CONFIGS[2]["param_groups"][name]
                        )
                self.assertEqual(head["algorithm"], algorithm)
                if algorithm == "input_conditioned":
                    self.assertFalse(head["gradient_momentum_before_conditioning"])
                    self.assertEqual(head["svd_mean_percentage_damping"], 0.01)
                    self.assertEqual(head["input_conditioner_momentum"], 0.0)
                else:
                    self.assertTrue(train.INPUT_CONDITIONER_FIELDS.isdisjoint(head))
                self.assertIsInstance(head["nesterov"], bool)
                point = (
                    train.get_hparam(candidate, "head.initial_lr"),
                    head["momentum"],
                    train.get_hparam(candidate, "head.decay"),
                    head["momentum_version"],
                    head["nesterov"],
                )
                self.assertNotIn(point, seen)
                seen.add(point)
                return dict(val_acc=0.9, tta_val_acc=0.94)

            with (
                patch.object(train, "main", side_effect=evaluate),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = train.run_experiment(name, None, config)
            expected = {
                (lr, momentum, decay, version, nesterov)
                for lr in expected_lrs
                for momentum in momenta
                for decay in decays
                for version in versions
                for nesterov in nesterov_choices
            }
            self.assertEqual(seen, expected)
            self.assertEqual(len(result["trials"]), len(expected))

    def test_input_conditioned_zero_momentum_uses_current_gradient(self):
        for version in (3,):
            config = copy.deepcopy(train.EXPERIMENT_RUN_CONFIGS["exp5_head_grid"])
            head = config["param_groups"]["head"]
            head["momentum_version"] = version
            head["lr_scheduler"] = [(2, 0.1)]
            model = TinyModel(dict(time=0, training=0))
            optimizer = train.make_optimizer(model, config["param_groups"])
            parameter = model.head.weight
            inputs = torch.eye(parameter.shape[1])
            covariance = inputs.T @ inputs / len(inputs)
            for scale in (1.0, -2.0):
                parameter.grad = torch.full_like(parameter, scale)
                before = parameter.detach().clone()
                optimizer.record_input(parameter, inputs)
                expected = train.GroupOptimizer.condition(
                    parameter.grad, covariance, 0.01
                )
                optimizer.step()
                torch.testing.assert_close(parameter, before - 0.1 * expected)

    def test_global_decay_candidates_replay_and_preserve_unsearched_schedules(self):
        for preferred_decay in ("constant", "linear_decay"):
            with self.subTest(preferred_decay=preferred_decay):
                config = copy.deepcopy(train.EXPERIMENT_RUN_CONFIGS["exp3"])
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
                    result = train.run_experiment("exp3", model, config)["best_result"]
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
                        [0.002, 0.003, 0.002],
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
            "lion": {},
            "adam": dict(beta2=0.999, eps=1e-8),
            "muon": dict(ns_steps=3, ns_eps=0.0),
            "muon2": {},
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
                nesterov=algorithm != "adam",
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
                    self.assertEqual(
                        initial["norm_bias.momentum"],
                        0.8 if original["batch_size"] == 2000 else 0.85,
                    )
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

    def test_line_schedules_allow_jumps_and_exclude_last_endpoint(self):
        lines = [(3, 0.04), (3, 0.02, 0.0)]
        self.assertEqual(
            [train.hparam_at_step(lines, step) for step in range(6)],
            [0.04, 0.04, 0.04, 0.02, 0.013, 0.0067],
        )
        self.assertEqual(train.hparam_at_step([(1, 0.04, 0.0)], 0), 0.04)
        self.assertEqual(train.hparam_at_step([(1, 0.04)], 0), 0.04)
        decay = train.lr_decay_schedule(200, 0.2, "linear_decay")
        self.assertEqual(train.hparam_at_step(decay, 0), 0.2)
        self.assertEqual(train.hparam_at_step(decay, 199), 0.001)
        for step in (-1, 6):
            with self.assertRaises(ValueError):
                train.hparam_at_step(lines, step)

    def test_schedule_slices_preserve_linear_values(self):
        lines = [(3, 0.04), (5, 0.12, 0), (3, 0.02, 0.08)]
        total = sum(line[0] for line in lines)
        for start in range(total):
            for end in range(start + 1, total + 1):
                with self.subTest(start=start, end=end):
                    sliced = train.slice_schedule(lines, start, end)
                    self.assertEqual(sum(line[0] for line in sliced), end - start)
                    self.assertEqual(
                        [
                            train.hparam_at_step(sliced, step)
                            for step in range(end - start)
                        ],
                        [
                            train.hparam_at_step(lines, step)
                            for step in range(start, end)
                        ],
                    )

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
                [train.round_hparam(lr * f) for f in (1, 0.8, 0.6, 0.4, 0.2)],
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
        self.assertEqual(result["conv_lr"], 0.0002)
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
        ):
            config["hparam_tuning"]["params"] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                train.interval_search_space(config)

    def test_bias_corrected_momentum_versions_and_checkpoint(self):
        for version, momentum, nesterov in product((3,), (0.0, 0.6), (False, True)):
            with self.subTest(version=version, momentum=momentum, nesterov=nesterov):
                parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
                group = dict(
                    name="test",
                    params=[parameter],
                    algorithm="sgd",
                    lr=0.1,
                    momentum=momentum,
                    momentum_version=version,
                    nesterov=nesterov,
                )
                optimizer = train.GroupOptimizer([dict(group)])
                gradients = [
                    torch.tensor(g, dtype=torch.float64)
                    for g in ([1.0, -2.0], [3.0, 4.0], [-5.0, 6.0], [2.0, -1.0])
                ]
                for t, gradient in enumerate(gradients, 1):
                    # Missing gradients must not advance the parameter's counter.
                    parameter.grad = None
                    optimizer.step()
                    parameter.grad = gradient.clone()
                    weights = [momentum ** (t - i - 1) for i in range(t)]
                    corrected = sum(
                        weight * g for weight, g in zip(weights, gradients[:t])
                    ) / sum(weights)
                    direction = (
                        ((1 - momentum) if version == 3 else 1) * gradient
                        + momentum * corrected if nesterov else corrected
                    )
                    before = parameter.detach().clone()
                    optimizer.step()
                    torch.testing.assert_close(parameter, before - 0.1 * direction)
                    torch.testing.assert_close(parameter.grad, gradient)
                    torch.testing.assert_close(
                        optimizer.state[parameter]["momentum_buffer"],
                        corrected * (1 - momentum ** t),
                    )
                    self.assertEqual(optimizer.state[parameter]["momentum_step"], t)
                    if t == 2:
                        checkpoint = copy.deepcopy(optimizer.state_dict())
                        optimizer = train.GroupOptimizer([dict(group)])
                        optimizer.load_state_dict(checkpoint)

    def test_version_three_preserves_constant_conditioned_gradient_scale(self):
        for nesterov, momentum_first in product((False, True), (False, True)):
            with self.subTest(nesterov=nesterov, momentum_first=momentum_first):
                config = copy.deepcopy(train.EXPERIMENT_RUN_CONFIGS["exp5_head_grid"])
                config["param_groups"]["head"].update(
                    momentum_version=3, momentum=0.9, nesterov=nesterov,
                    gradient_momentum_before_conditioning=momentum_first,
                    lr_scheduler=[(4, 0.1)],
                )
                model = TinyModel(dict(time=0, training=0))
                optimizer = train.make_optimizer(model, config["param_groups"])
                parameter = model.head.weight
                gradient = torch.ones_like(parameter)
                inputs = 2 * torch.eye(parameter.shape[1])
                covariance = inputs.T @ inputs / len(inputs)
                direction = train.GroupOptimizer.condition(gradient, covariance, 0.01)
                for _ in range(4):
                    parameter.grad = gradient.clone()
                    optimizer.record_input(parameter, inputs)
                    before = parameter.detach().clone()
                    optimizer.step()
                    torch.testing.assert_close(parameter, before - 0.1 * direction)
                    torch.testing.assert_close(parameter.grad, gradient)

    def test_lion_takes_sign_without_changing_momentum_history(self):
        for nesterov in (False, True):
            parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
            reference = torch.nn.Parameter(parameter.detach().clone())
            optimizer = train.GroupOptimizer([dict(
                name="test", params=[parameter], algorithm="lion", lr=0.1,
                momentum=0.6, nesterov=nesterov, momentum_version=3,
            )])
            sgd = train.GroupOptimizer([dict(
                name="reference", params=[reference], algorithm="sgd", lr=0.1,
                momentum=0.6, nesterov=nesterov, momentum_version=3,
            )])
            for values in ([1.0, -2.0, 0.0], [-0.2, 0.1, 0.0], [-2.0, 3.0, 0.0]):
                gradient = torch.tensor(values, dtype=torch.float64)
                parameter.grad = gradient.clone()
                reference.grad = gradient.clone()
                before_reference = reference.detach().clone()
                sgd.step()
                direction = (before_reference - reference).sign()
                before = parameter.detach().clone()
                optimizer.step()
                torch.testing.assert_close(parameter, before - 0.1 * direction)
                torch.testing.assert_close(parameter.grad, gradient)
                torch.testing.assert_close(
                    optimizer.state[parameter]["momentum_buffer"],
                    sgd.state[reference]["momentum_buffer"],
                )
            optimizer.param_groups[0]["momentum"] = 0
            before = parameter.detach().clone()
            optimizer.step()
            torch.testing.assert_close(parameter, before - 0.1 * gradient.sign())
            torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], gradient)
            optimizer.param_groups[0]["lr"] = 0
            before = parameter.detach().clone()
            optimizer.step()
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)

    def test_adam_matches_torch_and_restores_checkpoint(self):
        for beta1, beta2 in ((0.0, 0.0), (0.9, 0.999), (0.6, 0.95)):
            with self.subTest(beta1=beta1, beta2=beta2):
                parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
                reference = torch.nn.Parameter(parameter.detach().clone())
                group = dict(
                    name="test", params=[parameter], algorithm="adam", lr=0.01,
                    momentum=beta1, beta2=beta2, eps=1e-8, nesterov=False,
                )
                optimizer = train.GroupOptimizer([dict(group)])
                adam = torch.optim.Adam([reference], lr=0.01, betas=(beta1, beta2), eps=1e-8)
                for t, values in enumerate((
                    [1.0, -2.0, 0.0], [0.5, 1.5, 0.0], [-3.0, 0.0, 0.0],
                    [1e-10, 1e-10, 0.0],
                ), 1):
                    parameter.grad = reference.grad = None
                    optimizer.step()
                    adam.step()
                    gradient = torch.tensor(values, dtype=torch.float64)
                    parameter.grad = gradient.clone()
                    reference.grad = gradient.clone()
                    # Frozen weights still accumulate first and second moments.
                    lr = 0.0 if t == 2 else 0.01
                    optimizer.param_groups[0]["lr"] = adam.param_groups[0]["lr"] = lr
                    optimizer.step()
                    adam.step()
                    torch.testing.assert_close(parameter, reference, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(parameter.grad, gradient)
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizer.state[parameter][key], adam.state[reference][key]
                        )
                    self.assertEqual(optimizer.state[parameter]["adam_step"], t)
                    if t == 2:
                        checkpoint = copy.deepcopy(optimizer.state_dict())
                        optimizer = train.GroupOptimizer([dict(group)])
                        optimizer.load_state_dict(checkpoint)

    def test_adam_beta2_precision_and_half_precision_checkpoint(self):
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[2])
        config["param_groups"]["head"].update(
            algorithm="adam", momentum=0.9, nesterov=False, beta2=0.999, eps=1e-8,
        )
        config = train.normalize_config(config, key=None)
        self.assertEqual(config["param_groups"]["head"]["beta2"], 0.999)
        config = train.with_hparam(config, "head.beta2", 0.9999)
        model = TinyModel(dict(time=0, training=0)).half()
        optimizer = train.make_optimizer(model, config["param_groups"])
        schedules = train.configured_hparam_schedules(config["param_groups"], 200)
        train.apply_hparam_schedules(optimizer, schedules, 0)
        group = next(g for g in optimizer.param_groups if g["name"] == "head")
        self.assertEqual(group["beta2"], 0.9999)
        group["lr"] = 0.01
        parameter = model.head.weight
        parameter.grad = torch.full_like(parameter, 1e-4)
        optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        checkpoint = copy.deepcopy(optimizer.state_dict())
        before = {key: value.clone() for key, value in optimizer.state[parameter].items() if torch.is_tensor(value)}
        optimizer.load_state_dict(checkpoint)
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(optimizer.state[parameter][key].dtype, torch.float32)
            torch.testing.assert_close(optimizer.state[parameter][key], before[key], rtol=0, atol=0)
        optimizer.step()
        for state in checkpoint["state"].values():
            if "exp_avg" in state:
                torch.testing.assert_close(state["exp_avg"], before["exp_avg"], rtol=0, atol=0)

    def test_adam_and_lion_reject_unsupported_options(self):
        group = dict(
            name="test", params=[torch.nn.Parameter(torch.ones(2))],
            algorithm="adam", lr=0.1, momentum=0.9, beta2=0.999, eps=1e-8,
            nesterov=False,
        )
        for changes in (
            dict(momentum=1), dict(beta2=1), dict(beta2=-0.1), dict(eps=0),
            dict(nesterov=True), dict(momentum_version=2),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train.GroupOptimizer([dict(group, **changes)])
        del group["beta2"], group["eps"]
        group.update(algorithm="lion", momentum_version=1)
        with self.assertRaisesRegex(ValueError, "lion requires momentum_version=3"):
            train.GroupOptimizer([group])

    def test_momentum_versions_follow_algorithm_policy(self):
        options = {
            "muon": dict(ns_steps=3, ns_eps=0.0),
            "sgdh": dict(normalization_eps=1e-6),
            "adam": dict(beta2=0.999, eps=1e-8),
            "input_conditioned": dict(
                svd_mean_percentage_damping=0.01, input_conditioner_momentum=0.0,
                gradient_momentum_before_conditioning=False,
            ),
        }
        for algorithm in train.KFAC_ALGORITHMS:
            options[algorithm] = dict(
                kfac_damping=0.03, kfac_input_damping=0.03,
                kfac_factor_momentum=0.0, kfac_probes=1,
                gradient_momentum_before_conditioning=False,
            )
        for algorithm in train.ALGORITHM_FIELDS:
            group = dict(
                name="test", params=[torch.nn.Parameter(torch.ones(2, 2))],
                algorithm=algorithm, lr=0.1, momentum=0.6, nesterov=False,
                **options.get(algorithm, {}),
            )
            allowed = (1, 3) if algorithm in ("muon", "muon2") else (3,)
            optimizer = train.GroupOptimizer([dict(group)])
            self.assertEqual(optimizer.param_groups[0]["momentum_version"], allowed[0])
            for version in (1, 2, 3):
                with self.subTest(algorithm=algorithm, version=version):
                    candidate = dict(group, momentum_version=version)
                    config_group = {k: v for k, v in candidate.items() if k not in ("name", "params", "lr")}
                    config_group["lr_scheduler"] = [(2, 0.1)]
                    if version in allowed:
                        train.GroupOptimizer.validate_group(config_group, runtime=False)
                        train.GroupOptimizer([candidate])
                    else:
                        with self.assertRaisesRegex(ValueError, "momentum_version"):
                            train.GroupOptimizer.validate_group(config_group, runtime=False)
                        with self.assertRaisesRegex(ValueError, "momentum_version"):
                            train.GroupOptimizer([candidate])
                        checkpoint = copy.deepcopy(optimizer.state_dict())
                        checkpoint["param_groups"][0]["momentum_version"] = version
                        with self.assertRaisesRegex(ValueError, "momentum_version"):
                            optimizer.load_state_dict(checkpoint)

    def test_bs2000_all_active_v3_search_preserves_schedules(self):
        config = train.EXPERIMENT_RUN_CONFIGS["exp12_bs2000_all_active_v3"]
        self.assertEqual(train.experiments_to_run, ["exp12_bs2000_all_active_v3"])
        self.assertEqual(config["batch_size"], 2000)
        self.assertEqual(config["num_epochs"], 8)
        self.assertEqual(config["hparam_tuning"]["algorithm"], "global_neighbour")
        self.assertEqual(config["hparam_tuning"]["metric"], "tta_val_acc")
        active = {"whiten_bias", "norm_bias", "head", "conv"}
        self.assertEqual(
            set(config["hparam_tuning"]["params"]),
            {f"{name}.{field}" for name in active for field in ("initial_lr", "momentum")},
        )
        baseline = train.BASELINE_RUN_CONFIGS[2]
        self.assertEqual(baseline["param_groups"]["head"]["momentum_version"], 3)
        self.assertEqual(baseline["param_groups"]["conv"]["momentum_version"], 3)
        for name, group in config["param_groups"].items():
            self.assertEqual(group["momentum_version"], 3)
            self.assertEqual(group["lr_scheduler"], baseline["param_groups"][name]["lr_scheduler"])
        choices, initial = train.interval_search_space(config)
        schedules = train.configured_hparam_schedules(config["param_groups"], 200)
        schedules = train.segment_hparam_schedules(schedules, initial, 0, 200, True)
        for name in active:
            self.assertEqual(choices[f"{name}.momentum"], train.MOMENTUM_CHOICES)
            self.assertGreater(train.hparam_at_step(schedules[f"{name}.initial_lr"], 0), 0)
        for step in range(75, 200):
            self.assertEqual(train.hparam_at_step(schedules["whiten_bias.initial_lr"], step), 0)
        for name in ("whiten_weight", "norm_weight"):
            self.assertTrue(train.is_zero_lr_schedule(schedules[f"{name}.initial_lr"]))
        for original in (*train.BASE_RUN_CONFIGS, *train.BASELINE_RUN_CONFIGS,
                         *train.INTERVAL_RUN_CONFIGS, *train.GLOBAL_NEIGHBOUR_RUN_CONFIGS,
                         *train.EXPERIMENT_RUN_CONFIGS.values()):
            for group in original["param_groups"].values():
                train.GroupOptimizer.validate_group(group, runtime=False)

    def test_momentum_version_validation(self):
        group = dict(
            name="test",
            params=[torch.nn.Parameter(torch.ones(2))],
            algorithm="sgd",
            lr=0.1,
            momentum=0.6,
            nesterov=False,
        )
        for version in (0, 1, 2, 4, True, 1.0, "2"):
            with (
                self.subTest(version=version),
                self.assertRaisesRegex(ValueError, "momentum_version"),
            ):
                train.GroupOptimizer([dict(group, momentum_version=version)])
        for version, momentum in product((3,), (1.0, 1.1, 0.999)):
            with (
                self.subTest(version=version, momentum=momentum),
                self.assertRaisesRegex(ValueError, f"momentum_version {version}"),
            ):
                train.GroupOptimizer([
                    dict(group, momentum_version=version, momentum=momentum)
                ])

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

    def test_muon2_gram_norm_and_orthogonalization(self):
        generator = torch.Generator().manual_seed(42)
        for shape in ((3, 5), (5, 3), (4, 4), (2, 3, 5), (2, 5, 3)):
            with self.subTest(shape=shape):
                gradient = torch.randn(shape, generator=generator)
                singular_values = torch.linalg.svdvals(gradient)
                expected_norm = singular_values.pow(4).sum(-1).pow(0.25)
                for keepdim in (False, True):
                    expected = (
                        expected_norm[..., None, None] if keepdim else expected_norm
                    )
                    torch.testing.assert_close(
                        train.gram_frobenius_norm_estimate(gradient, keepdim, 1e-7),
                        expected,
                    )
                for dtype in (torch.float32, torch.bfloat16):
                    source = gradient.to(dtype)
                    before = source.clone()
                    result = train.zeropower_via_newtonschulz5_muon2(source)
                    u, _, vh = torch.linalg.svd(source.float(), full_matrices=False)
                    self.assertEqual(result.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        result.float(), u @ vh, atol=0.035, rtol=0.035
                    )
                    torch.testing.assert_close(source, before, atol=0, rtol=0)
                zeros = torch.zeros(shape)
                self.assertTrue(
                    torch.all(
                        train.gram_frobenius_norm_estimate(zeros, False, 1e-7) == 1e-7
                    )
                )
                torch.testing.assert_close(
                    train.zeropower_via_newtonschulz5_muon2(zeros), zeros.bfloat16()
                )

    def test_muon2_weight_normalization_and_momentum(self):
        for shape in ((3, 2), (3, 2, 2, 2)):
            for version in (1, 3):
                for nesterov in (False, True):
                    with self.subTest(shape=shape, version=version, nesterov=nesterov):
                        parameter = torch.nn.Parameter(torch.randn(shape))
                        optimizer = train.GroupOptimizer(
                            [
                                dict(
                                    name="test",
                                    params=[parameter],
                                    algorithm="muon2",
                                    lr=0.1,
                                    momentum=0.6,
                                    momentum_version=version,
                                    nesterov=nesterov,
                                )
                            ]
                        )
                        buffer = torch.zeros_like(parameter)
                        for step, momentum in enumerate((0.6, 0.0, 0.6), 1):
                            gradient = torch.randn_like(parameter)
                            parameter.grad = gradient.clone()
                            optimizer.param_groups[0]["momentum"] = momentum
                            buffer = momentum * buffer + (
                                (1 - momentum) if version == 3 else 1
                            ) * gradient
                            corrected = (
                                buffer / (1 - momentum ** step)
                                if version == 3
                                else buffer
                            )
                            direction = (
                                ((1 - momentum) if version == 3 else 1) * gradient
                                + momentum * corrected
                                if nesterov
                                else corrected
                            )
                            before = parameter.detach().clone()
                            with patch.object(
                                train,
                                "zeropower_via_newtonschulz5_muon2",
                                side_effect=lambda g: g,
                            ) as transform:
                                optimizer.step()
                            torch.testing.assert_close(
                                transform.call_args.args[0],
                                direction.reshape(shape[0], -1),
                            )
                            torch.testing.assert_close(
                                parameter,
                                before * (shape[0] ** 0.5 / before.norm())
                                - 0.1 * direction,
                            )
                            torch.testing.assert_close(
                                optimizer.state[parameter]["momentum_buffer"], buffer
                            )
                            torch.testing.assert_close(parameter.grad, gradient)
                        optimizer.param_groups[0]["lr"] = 0
                        before = parameter.detach().clone()
                        optimizer.step()
                        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
            train.GroupOptimizer(
                [
                    dict(
                        name="test",
                        params=[torch.nn.Parameter(torch.ones(2))],
                        algorithm="muon2",
                        lr=0.1,
                        momentum=0.6,
                        nesterov=False,
                    )
                ]
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
                    for step, (momentum, grad) in enumerate((
                        (0.6, [[0.0, 2.0], [3.0, 4.0]]),
                        (0.0, [[4.0, 0.0], [0.0, 5.0]]),
                        (0.6, [[2.0, 2.0], [1.0, 0.0]]),
                    ), 1):
                        optimizer.param_groups[0]["momentum"] = momentum
                        gradient = torch.tensor(grad).reshape(shape)
                        parameter.grad = gradient.clone()
                        before = parameter.detach().clone().reshape(2, 2)
                        buffer = momentum * buffer + (1 - momentum) * gradient
                        corrected = buffer / (1 - momentum ** step)
                        update = (
                            (1 - momentum) * gradient + momentum * corrected
                            if nesterov else corrected
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
            sgd = train.GroupOptimizer([dict(
                name="reference", params=[reference], algorithm="sgd", lr=0.1,
                momentum=0.6, nesterov=nesterov, momentum_version=3,
            )])
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
                sgd = train.GroupOptimizer([dict(
                    name="reference", params=[reference], algorithm="sgd", lr=0.1,
                    momentum=0.6, nesterov=nesterov, momentum_version=3,
                )])
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
        self.assertEqual(
            result["conv_lr"],
            train.round_hparam(
                result["intervals"][0]["main_hparams"]["conv.initial_lr"] / 3
            ),
        )
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
                                    if torch.is_tensor(buffer):
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


class KFACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def config(algorithm):
        """Small K-FAC fixture, independent of the runnable experiment registry."""
        config = copy.deepcopy(train.BASELINE_RUN_CONFIGS[2])
        config["param_groups"] = {
            name: dict(
                algorithm=algorithm,
                lr_scheduler=(
                    [(200, 0.0)] if train.is_zero_lr_schedule(group["lr_scheduler"])
                    else [(200, 0.15, 0.0)]
                ),
                momentum=0.9, momentum_version=3, nesterov=True,
                kfac_damping=0.0, kfac_input_damping=0.03,
                kfac_factor_momentum=0.0, kfac_probes=1,
                gradient_momentum_before_conditioning=False,
            )
            for name, group in config["param_groups"].items()
        }
        config["hparam_tuning"] = dict(
            algorithm="global_neighbour", metric="tta_val_acc", max_side_steps=20,
            params={
                "all.initial_lr": dict(
                    initial=0.15,
                    choices=sorted(train.round_hparam(0.15 * 0.6 ** k) for k in range(-7, 8)),
                ),
                "all.momentum": dict(initial=0.9, choices=[0.0, 0.5, 0.7, 0.8, 0.9]),
                "all.kfac_damping": dict(initial=0.0, choices=[0.0, 0.5]),
            },
        )
        return config

    def test_grid_shares_each_candidate_across_every_group(self):
        expected = set(product(
            (train.round_hparam(0.15 * 0.6 ** k) for k in range(-7, 8)),
            (0.0, 0.5, 0.7, 0.8, 0.9), (0.0, 0.5),
        ))
        for algorithm in train.KFAC_ALGORITHMS:
            config = self.config(algorithm)
            self.assertEqual(config["hparam_tuning"]["algorithm"], "global_neighbour")
            choices, initial = train.interval_search_space(config)
            self.assertEqual(set(product(*choices.values())), expected)
            self.assertEqual(initial, {
                "all.initial_lr": 0.15, "all.momentum": 0.9,
                "all.kfac_damping": 0.0,
            })
            config["hparam_tuning"]["algorithm"] = "grid"
            del config["hparam_tuning"]["max_side_steps"]
            original = copy.deepcopy(config)
            seen = set()

            def fake_main(run, model, **candidate):
                points = set()
                for name, group in candidate["param_groups"].items():
                    train.GroupOptimizer.validate_group(group, runtime=False)
                    self.assertEqual(group["algorithm"], algorithm)
                    self.assertTrue(group["nesterov"])
                    if name in ("whiten_weight", "norm_weight"):
                        self.assertEqual(group["lr_scheduler"], [(200, 0.0)])
                        continue
                    points.add((group["lr_scheduler"][0][1], group["momentum"], group["kfac_damping"]))
                self.assertEqual(len(points), 1)
                point = points.pop()
                seen.add(point)
                schedules = train.configured_hparam_schedules(candidate["param_groups"], 200)
                optimizer = SimpleNamespace(param_groups=[
                    dict(name=name, **group) for name, group in candidate["param_groups"].items()
                ])
                train.apply_hparam_schedules(optimizer, schedules, 100)
                for group in optimizer.param_groups:
                    expected_lr = 0.0 if group["name"] in ("whiten_weight", "norm_weight") else train.round_hparam(point[0] / 2)
                    self.assertEqual(group["lr"], expected_lr)
                    self.assertEqual(group["momentum"], point[1])
                    self.assertEqual(group["kfac_damping"], point[2])
                    self.assertTrue(group["nesterov"])
                return dict(val_acc=0.5, tta_val_acc=0.5)

            with patch.object(train, "main", side_effect=fake_main), contextlib.redirect_stdout(io.StringIO()):
                result = train.run_experiment("shared_grid", None, config)
            self.assertEqual(len(result["trials"]), 150)
            self.assertEqual(seen, expected)
            self.assertEqual(config, original)

    def test_shared_search_rejects_ambiguous_or_unsupported_options(self):
        config = self.config("kfac")
        config["param_groups"]["head"]["momentum"] = 0.7
        with self.assertRaisesRegex(ValueError, "must share"):
            train.tuning_params(config)
        config["param_groups"]["head"]["momentum"] = 0.9
        config["hparam_tuning"]["params"]["head.momentum"] = dict(initial=0.9, choices=[0.9])
        with self.assertRaisesRegex(ValueError, "individual group"):
            train.tuning_params(config)
        del config["hparam_tuning"]["params"]["head.momentum"]
        config["hparam_tuning"]["algorithm"] = "interval"
        with self.assertRaisesRegex(ValueError, "require grid, coordinate, or global_neighbour"):
            train.tuning_params(config)

    def test_global_neighbour_applies_shared_values_to_every_training_step(self):
        for algorithm in train.KFAC_ALGORITHMS:
            config = self.config(algorithm)
            config.update(batch_size=2, num_epochs=2, overfit=True)
            for name, group in config["param_groups"].items():
                lines = [(2, 0.0)] if name in ("whiten_weight", "norm_weight") else [(2, 0.001, 0.0)]
                group.update(lr_scheduler=lines, momentum=0.0)
            config["hparam_tuning"]["params"] = {
                "all.initial_lr": dict(initial=0.001, choices=[0.001, 0.002]),
                "all.momentum": dict(initial=0.0, choices=[0.0, 0.5]),
                "all.kfac_damping": dict(initial=0.0, choices=[0.0, 0.5]),
            }
            original = copy.deepcopy(config)
            model = TinyModel(dict(time=0, training=0))
            optimizers, applied = [], []
            make_optimizer, step = train.make_optimizer, train.GroupOptimizer.step

            def capture_optimizer(*args):
                optimizer = make_optimizer(*args)
                optimizers.append(optimizer)
                return optimizer

            def checked_step(optimizer):
                settings = {
                    (g["lr"], g["momentum"], g["kfac_damping"], g["nesterov"])
                    for g in optimizer.param_groups
                    if g["name"] not in ("whiten_weight", "norm_weight")
                }
                self.assertEqual(len(settings), 1)
                point = settings.pop()
                self.assertTrue(point[3])
                applied.append(point)
                frozen = {}
                for group in optimizer.param_groups:
                    if group["name"] in ("whiten_weight", "norm_weight"):
                        self.assertEqual(group["lr"], 0.0)
                        for parameter in group["params"]:
                            frozen[parameter] = parameter.detach().clone()
                result = step(optimizer)
                for parameter, before in frozen.items():
                    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                return result

            def score(*args, **kwargs):
                group = optimizers[0].param_groups[0]
                return group["lr"] + 0.2 * (group["momentum"] + group["kfac_damping"])

            with (
                patch.object(train, "CifarLoader", TinyLoader),
                patch.object(train, "make_optimizer", side_effect=capture_optimizer),
                patch.object(train.GroupOptimizer, "step", checked_step),
                patch.object(train, "evaluate", side_effect=score),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = train.run_experiment("shared_neighbour", model, config)["best_result"]
            self.assertEqual(model.training_steps.item(), 2)
            self.assertEqual(len(result["intervals"]), 1)
            interval = result["intervals"][0]
            self.assertEqual(interval["cooldown_steps"], 0)
            self.assertEqual(interval["main_hparams"], {
                "all.initial_lr": 0.002, "all.momentum": 0.5,
                "all.kfac_damping": 0.5,
            })
            self.assertEqual(applied[-2:], [(0.002, 0.5, 0.5, True), (0.001, 0.5, 0.5, True)])
            for name in config["param_groups"]:
                expected_lr = [(2, 0.0)] if name in ("whiten_weight", "norm_weight") else [(2, 0.002, 0.0)]
                self.assertEqual(result["hparam_schedules"][f"{name}.initial_lr"], expected_lr)
                self.assertEqual(result["hparam_schedules"][f"{name}.kfac_damping"], [(2, 0.5)])
            self.assertEqual(config, original)

    def test_shared_lr_and_decay_preserve_frozen_schedules(self):
        config = self.config("kfac")
        self.assertEqual(train.get_hparam(config, "all.initial_lr"), 0.15)
        changed = train.with_hparam(config, "all.initial_lr", 5.4)
        changed = train.with_hparam(changed, "all.decay", "constant")
        schedules = train.configured_hparam_schedules(config["param_groups"], 200)
        selected = train.segment_hparam_schedules(
            schedules, {"all.initial_lr": 5.4, "all.decay": "constant"},
            0, 200, global_search=True,
        )
        for name, group in changed["param_groups"].items():
            expected = [(200, 0.0)] if name in ("whiten_weight", "norm_weight") else [(200, 5.4)]
            self.assertEqual(group["lr_scheduler"], expected)
            self.assertEqual(selected[f"{name}.initial_lr"], expected)

    def optimizer(self, model, algorithm, **options):
        defaults = dict(
            algorithm=algorithm, lr=0.02, momentum=0.0, nesterov=False,
            kfac_damping=0.0, kfac_factor_momentum=0.0, kfac_probes=1,
            gradient_momentum_before_conditioning=False,
        )
        defaults.update(options)
        defaults.setdefault("kfac_input_damping", defaults["kfac_damping"])
        optimizer = train.GroupOptimizer([
            dict(name=name, params=[p], **defaults) for name, p in model.named_parameters()
            if p.requires_grad
        ])
        optimizer.bind_kfac_layers(model)
        return optimizer

    def exact_probes(self, logits, algorithm):
        """Enumerate a square root of the full logits metric, eliminating MC noise."""
        batch, classes = logits.shape
        if algorithm == "kfac":
            probabilities = logits.detach().softmax(-1)
            hessian = torch.diag_embed(probabilities) - probabilities[:, :, None] * probabilities[:, None, :]
            eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
            root = eigenvectors * eigenvalues.clamp_min(0).sqrt()[:, None, :]
        else:
            root = torch.eye(classes, dtype=logits.dtype, device=logits.device).expand(batch, -1, -1)
        seeds = []
        for example, column in product(range(batch), range(classes)):
            seed = torch.zeros_like(logits)
            seed[example] = root[example, :, column] * (batch * classes) ** 0.5
            seeds.append(seed)
        return seeds

    def prepare_exact(self, optimizer, logits, algorithm):
        seeds = self.exact_probes(logits, algorithm)
        for group in optimizer.param_groups:
            group["kfac_probes"] = len(seeds)
        with patch.object(optimizer, "_kfac_probe", side_effect=seeds):
            optimizer.prepare_kfac(logits)

    def test_probe_covariance_is_ce_hessian_or_identity(self):
        # Exhaust the Rademacher distribution for three logits.
        bits = torch.tensor(list(product((0., 1.), repeat=3)), dtype=torch.float64)
        logits = torch.tensor([1., -2., 0.5], dtype=torch.float64).expand(8, -1)
        for algorithm in train.KFAC_ALGORITHMS:
            with patch.object(torch.Tensor, "bernoulli_", lambda tensor, _: tensor.copy_(bits)):
                seeds = train.GroupOptimizer._kfac_probe(logits, algorithm)
            covariance = seeds.T @ seeds / 8
            p = logits[0].softmax(-1)
            expected = torch.diag(p) - p[:, None] * p[None, :] if algorithm == "kfac" else torch.eye(3, dtype=p.dtype)
            torch.testing.assert_close(covariance, expected)

    def test_linear_factors_and_independent_bias_updates(self):
        x = torch.tensor([[1., 2.], [-1., 3.], [2., -1.]], dtype=torch.float64)
        labels = torch.tensor([0, 1, 2])
        for algorithm in train.KFAC_ALGORITHMS:
            torch.manual_seed(4)
            model = torch.nn.Linear(2, 3).double()
            optimizer = self.optimizer(model, algorithm)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            logits = model(x) / 2  # Include the head's downstream logits scaling.
            self.prepare_exact(optimizer, logits, algorithm)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertFalse(optimizer._kfac_pending)
            loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.2)
            loss.backward()
            a = x.T @ x / len(x)
            if algorithm == "kfac":
                p = logits.detach().softmax(-1)
                g = (torch.diag_embed(p) - p[:, :, None] * p[:, None, :]).mean(0) / 4
            else:
                g = torch.eye(3, dtype=x.dtype) / 4
            expected_weight = torch.linalg.pinv(g, hermitian=True) @ model.weight.grad @ torch.linalg.inv(a)
            expected_bias = torch.linalg.pinv(g, hermitian=True) @ model.bias.grad
            optimizer.step()
            torch.testing.assert_close(model.weight, before["weight"] - 0.02 * expected_weight)
            torch.testing.assert_close(model.bias, before["bias"] - 0.02 * expected_bias)
            torch.testing.assert_close(optimizer.state[model.weight]["kfac_input_covariance"], a)
            torch.testing.assert_close(optimizer.state[model.bias]["kfac_output_covariance"], g)
            with self.assertRaisesRegex(RuntimeError, "prepare_kfac"):
                optimizer.step()

    def test_jacobian_update_matches_explicit_pseudoinverse(self):
        model = torch.nn.Linear(2, 3, bias=False).double()
        optimizer = self.optimizer(model, "kfac-jacobian")
        x = torch.tensor([[1., 2.]], dtype=torch.float64)
        weight = model.weight.detach().clone()
        logits = model(x) / 3
        self.prepare_exact(optimizer, logits, "kfac-jacobian")
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        residual = torch.autograd.grad(loss, logits, retain_graph=True)[0].flatten()
        jacobian = torch.autograd.functional.jacobian(lambda w: (x @ w.T / 3).flatten(), weight).reshape(3, -1)
        expected = (torch.linalg.pinv(jacobian) @ residual).reshape_as(weight)
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(model.weight, weight - 0.02 * expected)

    def test_conv_patch_and_spatial_factors(self):
        for padding in (0, "same"):
            model = torch.nn.Conv2d(1, 2, 2, padding=padding).double()
            optimizer = self.optimizer(model, "kfac-jacobian", kfac_damping=0.03)
            x = torch.arange(18, dtype=torch.float64).reshape(2, 1, 3, 3) / 10
            output = model(x)
            sites = output.shape[2] * output.shape[3]
            logits = output.mean((2, 3))
            self.prepare_exact(optimizer, logits, "kfac-jacobian")
            padded = torch.nn.functional.pad(x, (0, 1, 0, 1)) if padding == "same" else x
            patches = torch.nn.functional.unfold(padded, 2).transpose(1, 2).reshape(-1, 4)
            a, g, _ = optimizer._kfac_factors[model.weight]
            torch.testing.assert_close(a, patches.T @ patches / len(patches))
            torch.testing.assert_close(g, torch.eye(2, dtype=x.dtype) / sites)
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
            optimizer.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_batchnorm_scale_restricts_affine_kronecker_block(self):
        model = train.BatchNorm(2, momentum=0.6, eps=1e-5).double()
        optimizer = self.optimizer(model, "kfac-jacobian", kfac_damping=0.1)
        x = torch.tensor([[[[1., 3.]], [[2., 4.]]], [[[4., 8.]], [[-1., 2.]]]], dtype=torch.float64)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        output = model(x)
        logits = output[:, :, 0, 0] + 2 * output[:, :, 0, 1]
        self.prepare_exact(optimizer, logits, "kfac-jacobian")
        variance, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0)
        normalized = (x - mean[None, :, None, None]) / (variance[None, :, None, None] + model.eps).sqrt()
        rows = normalized.permute(0, 2, 3, 1).reshape(-1, 2)
        a = rows.T @ rows / len(rows)
        g = 5 * torch.eye(2, dtype=x.dtype)
        torch.testing.assert_close(optimizer._kfac_factors[model.weight][0], a)
        torch.testing.assert_close(optimizer._kfac_factors[model.weight][1], g)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        directions = {}
        for name, parameter in model.named_parameters():
            block = a * g if name == "weight" else g
            damped = block + 0.1 * block.diagonal().mean() * torch.eye(2, dtype=x.dtype)
            directions[name] = torch.linalg.solve(damped, parameter.grad)
        optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name] - 0.02 * directions[name])

    def test_all_group_configs_hooks_and_checkpoint(self):
        for algorithm in train.KFAC_ALGORITHMS:
            config = train.normalize_config(self.config(algorithm), None)
            train.validate_tuning_config(config["hparam_tuning"])
            train.tuning_params(config)
            train.configured_hparam_schedules(config["param_groups"], 200)
            model = TinyModel(dict(time=0, training=0))
            optimizer = train.make_optimizer(model, config["param_groups"])
            self.assertEqual({g["algorithm"] for g in optimizer.param_groups}, {algorithm})
            self.assertEqual({p for g in optimizer.param_groups for p in g["params"]}, set(model.parameters()))
            x = torch.randn(3, 3, 8, 8)
            before = {p: p.detach().clone() for p in model.parameters()}
            logits = model(x)
            optimizer.prepare_kfac(logits)
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
            optimizer.step()
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    self.assertTrue(torch.isfinite(parameter).all())
                    if group["name"] in ("whiten_weight", "norm_weight"):
                        torch.testing.assert_close(parameter, before[parameter], rtol=0, atol=0)
                    else:
                        self.assertFalse(torch.equal(parameter, before[parameter]))
                    self.assertIn("kfac_output_covariance", optimizer.state[parameter])
            checkpoint = copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                model(x)
            self.assertFalse(optimizer._kfac_pending)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    optimizer.state[parameter]["kfac_output_covariance"].zero_()
            optimizer.load_state_dict(checkpoint)
            for group, saved_group in zip(optimizer.param_groups, checkpoint["param_groups"]):
                for parameter, index in zip(group["params"], saved_group["params"]):
                    restored = optimizer.state[parameter]["kfac_output_covariance"]
                    saved = checkpoint["state"][index]["kfac_output_covariance"]
                    torch.testing.assert_close(restored, saved)
                    self.assertNotEqual(restored.data_ptr(), saved.data_ptr())
            replacement = train.make_optimizer(model, config["param_groups"])
            self.assertEqual(len(model.head._forward_hooks), 1)
            model(x)
            self.assertFalse(optimizer._kfac_pending)
            self.assertTrue(replacement._kfac_pending)
            train.make_optimizer(model, train.BASELINE_RUN_CONFIGS[2]["param_groups"])
            self.assertEqual(len(model.head._forward_hooks), 0)

    def test_kfac_validation_and_missing_capture(self):
        for algorithm in train.KFAC_ALGORITHMS:
            model = torch.nn.Linear(2, 2)
            for options in (
                {"kfac_damping": -1}, {"kfac_input_damping": -1},
                {"kfac_factor_momentum": 1}, {"kfac_factor_momentum": 0.5},
                {"kfac_probes": 0}, {"kfac_probes": 1.5}, {"kfac_probes": True},
                {"gradient_momentum_before_conditioning": 1}, {"kfac_filter_normalization": 0},
            ):
                with self.assertRaises(ValueError):
                    self.optimizer(model, algorithm, **options)
            optimizer = self.optimizer(model, algorithm)
            model(torch.ones(1, 2)).sum().backward()
            with self.assertRaisesRegex(RuntimeError, "prepare_kfac"):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with self.assertRaisesRegex(RuntimeError, "current training forward"):
                optimizer.prepare_kfac(torch.ones(1, 2, requires_grad=True))

    def test_current_batch_factors_and_half_precision_checkpoint_replay(self):
        model = torch.nn.Linear(2, 2).half()
        optimizer = self.optimizer(
            model, "kfac-jacobian", kfac_damping=0.1,
            kfac_factor_momentum=0.0, momentum=0.9, momentum_version=3,
        )
        x = torch.tensor([[1., 2.], [3., -1.]], dtype=torch.float16)

        def step(inputs):
            logits = model(inputs)
            optimizer.prepare_kfac(logits)
            factors = {
                parameter: (a.clone() if a is not None else None, g.clone())
                for parameter, (a, g, _) in optimizer._kfac_factors.items()
            }
            torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1])).backward()
            optimizer.step()
            for parameter, (a, g) in factors.items():
                # Both factors must match this batch, including after a checkpoint restore.
                state = optimizer.state[parameter]
                if a is not None:
                    torch.testing.assert_close(state["kfac_input_covariance"], a, rtol=0, atol=0)
                torch.testing.assert_close(state["kfac_output_covariance"], g, rtol=0, atol=0)
            optimizer.zero_grad(set_to_none=True)

        step(x)
        checkpoint = copy.deepcopy(optimizer.state_dict())
        weights = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state()
        initial_a = optimizer.state[model.weight]["kfac_input_covariance"].clone()
        expected_a = 4 * initial_a  # Only (2X)'(2X), with no contribution from X'X.
        results = []
        for _ in range(2):
            model.load_state_dict(weights)
            optimizer.load_state_dict(checkpoint)
            torch.set_rng_state(rng)
            for group, saved_group in zip(optimizer.param_groups, checkpoint["param_groups"]):
                parameter = group["params"][0]
                for key, saved in checkpoint["state"][saved_group["params"][0]].items():
                    if not isinstance(saved, torch.Tensor):
                        continue
                    restored = optimizer.state[parameter][key]
                    self.assertEqual(restored.dtype, torch.float32)
                    self.assertNotEqual(restored.data_ptr(), saved.data_ptr())
                    torch.testing.assert_close(restored, saved, rtol=0, atol=0)
            step(2 * x)
            torch.testing.assert_close(optimizer.state[model.weight]["kfac_input_covariance"], expected_a)
            results.append(copy.deepcopy(model.state_dict()))
        for key in results[0]:
            torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
        optimizer.load_state_dict(checkpoint)
        for group in optimizer.param_groups:
            group["kfac_probes"] = 2
        logits = model(x)
        with patch.object(optimizer, "_kfac_probe", wraps=optimizer._kfac_probe) as probe:
            optimizer.prepare_kfac(logits)
            self.assertEqual(probe.call_count, 2)

    def test_input_damping_applies_only_to_the_input_factor(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 2, bias=False).double()
        optimizer = self.optimizer(model, "kfac-jacobian", kfac_damping=0.1, kfac_input_damping=2.0)
        x = torch.randn(5, 3, dtype=torch.float64)
        logits = model(x)
        self.prepare_exact(optimizer, logits, "kfac-jacobian")
        a, g, _ = optimizer._kfac_factors[model.weight]
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1, 1])).backward()
        gradient = model.weight.grad.clone()
        before = model.weight.detach().clone()
        optimizer.step()

        def damped(matrix, alpha):
            return matrix + alpha * matrix.diagonal().mean() * torch.eye(len(matrix), dtype=matrix.dtype)

        expected = torch.linalg.solve(damped(g, 0.1), gradient) @ torch.linalg.inv(damped(a, 2.0))
        torch.testing.assert_close(before - model.weight, 0.02 * expected)

    def test_momentum_before_conditioning_without_filter_normalization(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 4, bias=False).double()
        optimizer = self.optimizer(
            model, "kfac-jacobian", lr=0.1, kfac_damping=0.5, momentum=0.5, momentum_version=3,
            nesterov=True, gradient_momentum_before_conditioning=True,
        )
        raw, buffer = [], torch.zeros_like(model.weight)
        for step in range(2):
            x = torch.randn(6, 3, dtype=torch.float64)
            logits = model(x)
            self.prepare_exact(optimizer, logits, "kfac-jacobian")
            a, g, _ = optimizer._kfac_factors[model.weight]
            torch.nn.functional.cross_entropy(logits, torch.arange(6) % 4).backward()
            raw.append(model.weight.grad.clone())
            before = model.weight.detach().clone()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Nesterov direction of the bias-corrected EMA of raw gradients, then K-FAC.
            buffer = 0.5 * buffer + 0.5 * raw[-1]
            direction = 0.5 * raw[-1] + 0.5 * buffer / (1 - 0.5 ** (step + 1))
            # Factor momentum 0 uses the current step's factors.
            eye = torch.eye(len(a), dtype=torch.float64)
            update = torch.linalg.solve(
                g + 0.5 * g.diagonal().mean() * torch.eye(len(g), dtype=torch.float64), direction
            ) @ torch.linalg.inv(a + 0.5 * a.diagonal().mean() * eye)
            torch.testing.assert_close(before - model.weight, 0.1 * update)

    def test_removed_filter_normalization_option_is_rejected(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "kfac_filter_normalization"):
            self.optimizer(model, "kfac", kfac_filter_normalization=True)

    def test_kfac_experiments_are_retained_but_not_selected(self):
        for algorithm, prefix in product(train.KFAC_ALGORITHMS, ("all", "tuned")):
            name = f"{prefix}_{algorithm}"
            config = train.EXPERIMENT_RUN_CONFIGS[name]
            self.assertNotIn(name, train.experiments_to_run)
            self.assertNotIn(name, [run for run, _ in train.RUNS])
            for group in config["param_groups"].values():
                train.GroupOptimizer.validate_group(group, runtime=False)
                self.assertEqual(group["algorithm"], algorithm)
                self.assertEqual(group["kfac_factor_momentum"], 0)
                self.assertNotIn("kfac_filter_normalization", group)
                for segment in group["lr_scheduler"]:
                    if len(segment) == 3:
                        self.assertGreaterEqual(segment[1], segment[2])
                if not train.is_zero_lr_schedule(group["lr_scheduler"]):
                    self.assertGreater(group["lr_scheduler"][0][1], 0)
            if prefix == "all":
                train.interval_search_space(config)
                self.assertNotIn("all.kfac_factor_momentum", config["hparam_tuning"]["params"])
            else:
                self.assertIsNone(config["hparam_tuning"])
                self.assertEqual(config["param_groups"]["whiten_bias"]["lr_scheduler"][-1], (125, 0.0))
        self.assertFalse(hasattr(train.GroupOptimizer, "normalize_filters"))

    def test_both_training_paths_with_all_kfac_groups(self):
        for algorithm, search in product(train.KFAC_ALGORITHMS, ("grid", "global_neighbour")):
            with self.subTest(algorithm=algorithm, search=search):
                config = self.config(algorithm)
                config.update(batch_size=2, num_epochs=2, overfit=True)
                for group in config["param_groups"].values():
                    group["lr_scheduler"] = [(2, 0.001)]
                config["hparam_tuning"] = dict(
                    algorithm=search, metric="tta_val_acc",
                    params={"head.initial_lr": dict(initial=0.001, choices=[0.001])},
                )
                if search == "global_neighbour":
                    config["hparam_tuning"]["max_side_steps"] = 1
                model = TinyModel(dict(time=0, training=0))
                with patch.object(train, "CifarLoader", TinyLoader), contextlib.redirect_stdout(io.StringIO()):
                    result = train.run_experiment("kfac_test", model, config)
                self.assertTrue(torch.isfinite(torch.tensor(result["best_result"]["val_acc"])))
                self.assertEqual(model.training_steps.item(), 2)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main(num_epochs=8, overfit=False, hparam_tuning=None, _log_config=True)
