import contextlib
import copy
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train_cifar as train


class TinyLoader(train.CifarLoader):
    def __init__(self, path, train=True, batch_size=2, aug=None):
        self.images = torch.rand(6, 3, 8, 8)
        self.labels = torch.arange(6) % 2
        self.normalize = torch.nn.Identity()
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = self.shuffle = train


class TinyModel(torch.nn.Module):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.whiten = torch.nn.Conv2d(3, 4, 1)
        self.norm = train.BatchNorm(4)
        self.conv = torch.nn.Conv2d(4, 4, 1, bias=False)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(4, 2, bias=False)
        self.register_buffer("training_steps", torch.tensor(0))

    def reset(self):
        for module in (self.whiten, self.norm, self.conv, self.head):
            module.reset_parameters()
        self.training_steps.zero_()

    def init_whiten(self, images):
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

    def test_grid_and_coordinate_still_find_best_tta(self):
        scores = {(0.04, 0.6): 0, (0.06, 0.6): 1, (0.08, 0.6): 0.5,
                  (0.04, 0.7): 1, (0.06, 0.7): 2, (0.08, 0.7): 3,
                  (0.04, 0.8): 2.5, (0.06, 0.8): 1.5, (0.08, 0.8): 4}

        def fake_main(run, model, **config):
            point = (train.get_hparam(config, "conv.initial_lr"),
                     train.get_hparam(config, "conv.momentum"))
            score = scores[point] / 10
            return dict(val_acc=1 - score, tta_val_acc=score)

        for algorithm in ("grid", "coordinate"):
            with self.subTest(algorithm=algorithm):
                config = copy.deepcopy(train.RUN_CONFIGS[0])
                config = train.with_hparam(config, "conv.initial_lr", 0.04)
                config = train.with_hparam(config, "conv.momentum", 0.6)
                config["hparam_tuning"] = dict(algorithm=algorithm, parameters={
                    "conv.initial_lr": [0.04, 0.06, 0.08],
                    "conv.momentum": [0.6, 0.7, 0.8]})
                output = io.StringIO()
                with patch.object(train, "main", side_effect=fake_main), \
                     contextlib.redirect_stdout(output):
                    result = train.run_experiment(0, None, config)
                self.assertEqual(len(result["trials"]), 9)
                self.assertEqual(result["best_result"]["tta_val_acc"], 0.4)
                self.assertEqual(train.get_hparam(result["best_config"], "conv.initial_lr"), 0.08)
                log = output.getvalue()
                self.assertEqual(log.count("base_config run=0\n"), 1)
                self.assertEqual(log.count("config_diff run="), 9)
                self.assertIn("conv.initial_lr: 0.04 -> 0.06", log)
                self.assertIn("search_best run=0", log)
                self.assertNotIn("%", log)

    def test_tuning_initial_values_do_not_change_untuned_baseline(self):
        baseline_lrs = {125: 0.04, 500: 0.079, 2000: 0.19}
        for original in train.INTERVAL_RUN_CONFIGS + train.INTERVAL_LINEAR_RUN_CONFIGS:
            with self.subTest(batch_size=original["batch_size"],
                              algorithm=original["hparam_tuning"]["algorithm"]):
                baseline = baseline_lrs[original["batch_size"]]
                self.assertEqual(train.get_hparam(original, "conv.initial_lr"), baseline)
                _, initial = train.interval_search_space(original)
                self.assertEqual(initial["conv.initial_lr"], 1.0)
                if original["hparam_tuning"]["algorithm"] == "interval_linear":
                    self.assertEqual(initial["head.momentum"], 0.85)
                    self.assertEqual(initial["norm_bias.momentum"], 0.85)
                    self.assertEqual(initial["head.initial_lr"], train.get_hparam(original, "head.initial_lr"))
                    self.assertEqual(initial["norm_bias.initial_lr"], train.get_hparam(original, "norm_bias.initial_lr"))
                config = copy.deepcopy(original)
                config["hparam_tuning"] = None
                with patch.object(train, "main", return_value=dict(val_acc=0.9, tta_val_acc=0.94)) as main:
                    train.run_experiment(0, None, config)
                passed = main.call_args.kwargs
                self.assertIsNone(passed["hparam_tuning"])
                self.assertEqual(train.get_hparam(passed, "conv.initial_lr"), baseline)
                config = copy.deepcopy(original)
                del config["hparam_tuning"]["params"]["conv.initial_lr"]
                choices, _ = train.interval_search_space(config)
                self.assertNotIn("conv.initial_lr", choices)
                self.assertEqual(train.get_hparam(config, "conv.initial_lr"), baseline)
        self.assertEqual([c["batch_size"] for c in train.BASELINE_RUN_CONFIGS], [125, 500, 2000])
        self.assertTrue(all(c["hparam_tuning"] is None for c in train.BASELINE_RUN_CONFIGS))

    def test_structured_params_work_with_grid_and_coordinate(self):
        for algorithm in ("grid", "coordinate"):
            with self.subTest(algorithm=algorithm):
                config = copy.deepcopy(train.RUN_CONFIGS[0])
                config["hparam_tuning"] = dict(algorithm=algorithm, params={
                    "conv.initial_lr": dict(initial=0.06, choices=[0.04, 0.06, 0.08]),
                    "conv.momentum": dict(initial=0.7, choices=[0.6, 0.7, 0.8])})
                calls = []

                def evaluate(run, model, **candidate):
                    point = (train.get_hparam(candidate, "conv.initial_lr"),
                             train.get_hparam(candidate, "conv.momentum"))
                    calls.append(point)
                    return dict(val_acc=0, tta_val_acc=sum(point))

                with patch.object(train, "main", side_effect=evaluate), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = train.run_experiment(0, None, config)
                best = result["best_config"]
                self.assertEqual(train.get_hparam(best, "conv.initial_lr"), 0.08)
                self.assertEqual(train.get_hparam(best, "conv.momentum"), 0.8)
                if algorithm == "coordinate":
                    self.assertEqual(calls[0], (0.06, 0.7))
                else:
                    self.assertEqual(len(calls), 9)

    def test_explicit_momentum_start_can_be_between_choices(self):
        config = copy.deepcopy(train.RUN_CONFIGS[0])
        config["hparam_tuning"] = dict(algorithm="interval_linear", params={
            "head.momentum": dict(initial=0.85, choices=[0.8, 0.9])})
        choices, initial = train.interval_search_space(config)
        calls = []

        def evaluate(point):
            calls.append(point["head.momentum"])
            return -abs(point["head.momentum"] - 0.85)

        best, _ = train.directional_search(initial, choices, evaluate)
        self.assertEqual(calls, [0.85, 0.8, 0.9])
        self.assertEqual(best["head.momentum"], 0.85)

    def test_config_logging_is_readable_and_diffs_only_changed_values(self):
        config = copy.deepcopy(train.RUN_CONFIGS[0])
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
        self.assertEqual(logged["param_groups"]["conv"]["lr_scheduler"]["initial_lr"], 0.04)
        self.assertIn('\n  "batch_size": 125,\n', body)
        diff = body[end:]
        self.assertIn("conv.initial_lr: 0.04 -> 0.062", diff)
        self.assertIn("conv.momentum: 0.6 -> 0.72", diff)
        self.assertNotIn("head.initial_lr", diff)
        self.assertNotIn("param_groups", diff)
        self.assertNotIn("lr_scheduler", diff)

    def test_directional_sweeps_revisit_parameters_and_cache(self):
        scores = {(1, 1): 0, (2, 1): 1, (3, 1): 0.5,
                  (1, 2): 1, (2, 2): 2, (3, 2): 3,
                  (1, 3): 2.5, (2, 3): 1.5, (3, 3): 4}
        calls = []

        def score(point):
            key = (point["a.initial_lr"], point["b.momentum"])
            calls.append(key)
            return scores[key]

        best, value = train.directional_search(
            {"a.initial_lr": 1, "b.momentum": 1},
            {"a.initial_lr": [1, 2, 3], "b.momentum": [1, 2, 3]}, score)
        self.assertEqual(best, {"a.initial_lr": 3, "b.momentum": 3})
        self.assertEqual(value, 4)
        self.assertEqual(len(calls), len(set(calls)))

    def test_initial_search_chooses_largest_lr_within_margin(self):
        scores = {0.5: 0.81, 1: 0.8, 2: 0.805, 4: 0.6}
        best, value = train.directional_search(
            {"conv.initial_lr": 1}, {"conv.initial_lr": list(scores)},
            lambda point: scores[point["conv.initial_lr"]], initial_lr_search=True)
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
            {"conv.initial_lr": 0.04}, {"conv.initial_lr": None}, score,
            max_side_steps=5)
        self.assertEqual(best["conv.initial_lr"], 0.04)
        self.assertIn(0.028, calls)
        self.assertIn(0.047, calls)
        self.assertEqual(len(calls), 9)  # center, three smaller, five larger

    def test_lr_search_starts_at_baseline_without_snapping_or_skipping_neighbors(self):
        calls = []

        def score(point):
            calls.append(point["head.initial_lr"])
            return -abs(point["head.initial_lr"] - 84)

        best, _ = train.directional_search(
            {"head.initial_lr": 84}, {"head.initial_lr": None}, score)
        self.assertEqual(calls, [84, 60, 99])
        self.assertEqual(best["head.initial_lr"], 84)

    def test_lr_grid_stays_fixed_across_searches(self):
        current = {"conv.initial_lr": 0.36}
        choices = {"conv.initial_lr": None}
        grid = {train.round_hparam(0.6 ** k) for k in range(-20, 30)}
        for target in (0.22, 0.36):
            def score(point):
                lr = point["conv.initial_lr"]
                self.assertIn(lr, grid)
                return -(lr - target) ** 2

            current, _ = train.directional_search(current, choices, score)
            self.assertEqual(current["conv.initial_lr"], target)

    def test_validation_uses_training_batchnorm_without_gradients(self):
        clock = dict(time=0, training=0)
        model = TinyModel(clock).eval()
        loader = TinyLoader("", train=False)
        before = model.norm.num_batches_tracked.item()
        train.evaluate(model, loader)
        self.assertTrue(model.training)
        self.assertGreater(model.norm.num_batches_tracked.item(), before)
        self.assertEqual(clock["training"], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_line_schedules_allow_jumps_and_include_both_endpoints(self):
        lines = [train.hparam_line(0, 2, 0.04, 0.04),
                 train.hparam_line(3, 5, 0.02, 0.0)]
        self.assertEqual([train.hparam_at_step(lines, step) for step in range(6)],
                         [0.04, 0.04, 0.04, 0.02, 0.01, 0.0])
        self.assertEqual(train.hparam_at_step([train.hparam_line(0, 0, 0.04, 0.0)], 0), 0)

    def test_linear_interval_searches_starts_and_replays_one_full_decay(self):
        config = copy.deepcopy(train.RUN_CONFIGS[0])
        config.update(batch_size=2, num_epochs=5, overfit=True, hparam_tuning=dict(
            algorithm="interval_linear", interval_steps=1, cooldown_steps=4,
            parameters={"conv.initial_lr": [0.001, 0.002], "conv.momentum": [0.0, 0.6]}))
        for name, group in config["param_groups"].items():
            group["lr_scheduler"] = train.constant_lr_scheduler(
                0 if name.endswith("_weight") else 0.0012)
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
            updates.append({g["name"]: (g["lr"], g["momentum"])
                            for g in optimizer.param_groups})
            original_step(optimizer)

        def evaluate(model, loader, tta_level=0):
            self.assertEqual(model.training_steps.item(), 5)
            clock["time"] += 100
            start_lr, momentum = updates[-5]["conv"]
            # The larger LR is within 0.02 of the best, but must not win here.
            return 0.9 - (0.01 if start_lr > 0.001 else 0) + (0.001 if momentum == 0 else 0)

        output = io.StringIO()
        with patch.object(train, "CifarLoader", TinyLoader), \
             patch.object(train.GroupOptimizer, "step", step), \
             patch.object(train, "evaluate", side_effect=evaluate), \
             patch.object(train, "directional_search", wraps=train.directional_search) as search, \
             patch.object(train, "time", SimpleNamespace(perf_counter=lambda: clock["time"])), \
             contextlib.redirect_stdout(output):
            result = train.run_experiment(0, model, config)["best_result"]
        self.assertEqual(search.call_count, 1)
        self.assertFalse(search.call_args.kwargs["initial_lr_search"])
        self.assertEqual(len(result["intervals"]), 1)
        self.assertEqual(result["intervals"][0]["steps"], 5)
        self.assertEqual(result["intervals"][0]["cooldown_steps"], 0)
        self.assertEqual(result["intervals"][0]["main_hparams"],
                         {"conv.initial_lr": 0.001, "conv.momentum": 0.0})
        self.assertEqual(result["seconds"], clock["time"])
        self.assertGreater(len(initial_states), 2)
        for state, gradient in zip(initial_states, initial_gradients):
            for name, value in state.items():
                torch.testing.assert_close(value, initial_states[0][name], rtol=0, atol=0)
            torch.testing.assert_close(gradient, initial_gradients[0], rtol=0, atol=0)
        for start in range(0, len(updates), 5):
            trial = updates[start:start + 5]
            lr, momentum = trial[0]["conv"]
            self.assertEqual([u["conv"][0] for u in trial],
                             [train.round_hparam(lr * f) for f in (1, 0.75, 0.5, 0.25, 0)])
            self.assertEqual([u["conv"][1] for u in trial], [momentum] * 5)
            self.assertEqual([u["head"][0] for u in trial], [0.0012, 0.0009, 0.0006, 0.0003, 0])
        for path, lines in result["hparam_schedules"].items():
            self.assertEqual(len(lines), 1)
            self.assertEqual((lines[0]["start_step"], lines[0]["end_step"]), (0, 4))
            if path.endswith(".initial_lr"):
                self.assertEqual(lines[0]["end"], 0)
            else:
                self.assertEqual(lines[0]["start"], lines[0]["end"])
        self.assertEqual(result["conv_lr"], 0)
        logged, _ = json.JSONDecoder().raw_decode(output.getvalue().split("\n", 1)[1])
        self.assertEqual(logged["hparam_tuning"]["interval_steps"], 5)
        self.assertEqual(logged["hparam_tuning"]["cooldown_steps"], 0)
        self.assertNotIn("initial_dropoff_margin", logged["hparam_tuning"])
        self.assertNotIn("phase=cooldown", output.getvalue())

    def test_invalid_interval_space_and_nesterov_zero_momentum(self):
        config = copy.deepcopy(train.RUN_CONFIGS[0])
        config["param_groups"]["conv"]["nesterov"] = True
        config["hparam_tuning"] = dict(algorithm="interval", parameters={
            "conv.momentum": [0, 0.6, 0.9]})
        choices, _ = train.interval_search_space(config)
        self.assertEqual(choices["conv.momentum"], [0, 0.6, 0.9])
        config["hparam_tuning"]["parameters"] = {"conv.momentum": [0]}
        choices, initial = train.interval_search_space(config)
        self.assertEqual(choices["conv.momentum"], [0])
        self.assertEqual(initial["conv.momentum"], 0)
        for parameters in ({"batch_size": [125]}, {"conv.initial_lr": []},
                           {"whiten_weight.initial_lr": None}, {"head.momentum": [0]}):
            config["hparam_tuning"]["parameters"] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                train.interval_search_space(config)

    def test_muon_zero_momentum_refreshes_buffer_before_momentum_returns(self):
        for nesterov in (False, True):
            with self.subTest(nesterov=nesterov):
                parameter = torch.nn.Parameter(torch.randn(3, 2))
                optimizer = train.GroupOptimizer([dict(params=[parameter], algorithm="muon",
                                                      lr=0.001, momentum=0.0, nesterov=nesterov)])
                expected = torch.zeros_like(parameter)
                for momentum in (0.0, 0.6, 0.0, 0.1):
                    gradient = torch.randn_like(parameter)
                    parameter.grad = gradient.clone()
                    optimizer.param_groups[0]["momentum"] = momentum
                    expected.mul_(momentum).add_(gradient)
                    direction = gradient + momentum * expected if nesterov else expected.clone()
                    before = parameter.detach().clone()
                    # Inspect Muon's input independently of its matrix transform.
                    with patch.object(train, "zeropower_via_newtonschulz5",
                                      side_effect=lambda g: g) as transform:
                        optimizer.step()
                    torch.testing.assert_close(transform.call_args.args[0], direction)
                    torch.testing.assert_close(parameter, before * (len(before) ** 0.5 / before.norm())
                                               - 0.001 * direction)
                    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"],
                                               expected, rtol=0, atol=0)

    def test_stream_replays_across_augmented_epoch_boundary(self):
        loader = TinyLoader("", aug=dict(flip=True, translate=2))
        loader.normalized_images()
        stream = train.TrainingBatchStream(loader)
        stream.next_batch()
        state, rng = stream.state_dict(), torch.random.get_rng_state()
        expected = [stream.next_batch() for _ in range(5)]
        stream.load_state_dict(state)
        torch.random.set_rng_state(rng)
        actual = [stream.next_batch() for _ in range(5)]
        for first, second in zip(expected, actual):
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_interval_variants_with_real_scores_and_multiple_candidates(self):
        class CudaLoader(TinyLoader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.images = self.images.cuda()
                self.labels = self.labels.cuda()

        config = copy.deepcopy(train.RUN_CONFIGS[0])
        config.update(batch_size=2, num_epochs=3, overfit=True,
                      hparam_tuning=dict(algorithm="interval", interval_steps=2,
                                         cooldown_steps=2, max_side_steps=2,
                                         parameters={"conv.initial_lr": None,
                                                     "conv.momentum": [0.3, 0.6]}))
        for name, group in config["param_groups"].items():
            group["lr_scheduler"] = train.constant_lr_scheduler(
                0 if name.endswith("_weight") else 0.0012)
        original = json.dumps(config, default=lambda scheduler: scheduler.config)
        clock = dict(time=0, training=0)
        model = TinyModel(clock).cuda()
        with patch.object(train, "CifarLoader", CudaLoader), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train.run_experiment(0, model, config)["best_result"]
        self.assertEqual(model.training_steps.item(), 3)
        self.assertEqual([item["steps"] for item in result["intervals"]], [2, 1])
        self.assertEqual([item["cooldown_steps"] for item in result["intervals"]], [1, 0])
        self.assertGreater(clock["training"], 3)
        self.assertGreater(result["seconds"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertEqual(json.dumps(config, default=lambda scheduler: scheduler.config), original)
        for interval in result["intervals"]:
            for value in interval["main_hparams"].values():
                self.assertEqual(value, float(f"{value:.2g}"))

        config["hparam_tuning"]["algorithm"] = "interval_linear"
        with patch.object(train, "CifarLoader", CudaLoader), \
             contextlib.redirect_stdout(io.StringIO()):
            result = train.run_experiment(0, model, config)["best_result"]
        self.assertEqual(model.training_steps.item(), 3)
        self.assertEqual([item["steps"] for item in result["intervals"]], [3])
        self.assertEqual([item["cooldown_steps"] for item in result["intervals"]], [0])
        self.assertEqual(result["conv_lr"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_search_checkpoints_do_not_alias_live_momentum_buffers(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for algorithm in ("interval", "interval_linear"):
                with self.subTest(device=device, algorithm=algorithm):
                    class DeviceLoader(TinyLoader):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.images = self.images.to(device)
                            self.labels = self.labels.to(device)

                    config = copy.deepcopy(train.RUN_CONFIGS[0])
                    config.update(batch_size=2, num_epochs=9, overfit=True,
                                  hparam_tuning=dict(algorithm=algorithm,
                                      interval_steps=4, cooldown_steps=3, parameters={
                                          "conv.initial_lr": [0.001, 0.002],
                                          "conv.momentum": [0.0, 0.6],
                                          "head.momentum": [0.3, 0.85],
                                          "norm_bias.momentum": [0.3, 0.85]}))
                    for name, group in config["param_groups"].items():
                        group["lr_scheduler"] = train.constant_lr_scheduler(
                            0 if name.endswith("_weight") else 0.0012)
                    model = TinyModel(dict(time=0, training=0)).to(device)
                    checkpoints, restores = [], []
                    original_step = train.GroupOptimizer.step

                    def watch_copy(value):
                        copied = copy.deepcopy(value)
                        if isinstance(value, dict) and {"state", "param_groups"} <= value.keys():
                            if any(value is checkpoint for checkpoint, _ in checkpoints):
                                restores.append(value)
                            else:
                                # Observe the actual checkpoint returned to snapshot().
                                checkpoints.append((copied, copy.deepcopy(copied)))
                        return copied

                    def checked_step(optimizer):
                        original_step(optimizer)
                        live_buffers = {state["momentum_buffer"].data_ptr()
                                        for state in optimizer.state.values()
                                        if "momentum_buffer" in state}
                        for checkpoint, baseline in checkpoints:
                            self.assertEqual(checkpoint["param_groups"], baseline["param_groups"])
                            self.assertEqual(checkpoint["state"].keys(), baseline["state"].keys())
                            for index, state in checkpoint["state"].items():
                                buffer = state["momentum_buffer"]
                                self.assertNotIn(buffer.data_ptr(), live_buffers,
                                                 "Saved momentum aliases a live optimizer buffer")
                                torch.testing.assert_close(buffer,
                                    baseline["state"][index]["momentum_buffer"], rtol=0, atol=0)

                    with patch.object(train, "CifarLoader", DeviceLoader), \
                         patch.object(train, "copy", SimpleNamespace(deepcopy=watch_copy)), \
                         patch.object(train.GroupOptimizer, "step", checked_step), \
                         patch.object(train, "evaluate", side_effect=lambda model, loader, tta_level=0:
                                      model.training_steps.item() / 100), \
                         contextlib.redirect_stdout(io.StringIO()):
                        train.run_experiment(0, model, config)
                    self.assertGreater(len(restores), 1)
                    if algorithm == "interval":
                        # Cooldown and later main checkpoints must contain both optimizers' buffers.
                        populated = [state for state, _ in checkpoints if state["state"]]
                        self.assertGreater(len(populated), 1)
                        self.assertEqual({g["algorithm"] for g in populated[0]["param_groups"]},
                                         {"sgd", "muon"})
                    else:
                        # Full-run trials all restore the pristine optimizer, with no buffers yet.
                        self.assertTrue(all(not state["state"] for state, _ in checkpoints))

    def test_interval_replay_matches_committed_training_and_includes_eval_time(self):
        # Dropout, augmentation, BatchNorm buffers and optimizer momentum all
        # must match a plain run after discarding every probe and cooldown.
        for overfit in (False, True):
            with self.subTest(overfit=overfit):
                config = copy.deepcopy(train.RUN_CONFIGS[0])
                config.update(batch_size=2, num_epochs=9 if overfit else 3,
                              overfit=overfit, hparam_tuning=None)
                for name, group in config["param_groups"].items():
                    group["lr_scheduler"] = train.constant_lr_scheduler(
                        0 if name.endswith("_weight") else 0.0012)
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

                    def evaluate(model, loader, tta_level=0):
                        model.train()
                        clock["time"] += 100
                        return model.training_steps.item() / 100

                    if interval:
                        config["hparam_tuning"] = dict(
                            algorithm="interval", interval_steps=4, cooldown_steps=3,
                            parameters={"conv.initial_lr": [0.0012], "conv.momentum": [0.6]})
                    output = io.StringIO()
                    with patch.object(train, "CifarLoader", TinyLoader), \
                         patch.object(train, "make_optimizer", side_effect=capture_optimizer), \
                         patch.object(train, "evaluate", side_effect=evaluate), \
                         patch.object(train, "time", SimpleNamespace(
                             perf_counter=lambda: clock["time"], monotonic=lambda: 0)), \
                         contextlib.redirect_stdout(output):
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
                    self.assertTrue(lines[-1].startswith("val_acc=0.0900 tta_val_acc=0.0900 seconds="))
                    if interval:
                        self.assertEqual(header, "base_config run=0")
                        self.assertEqual(log.count("train_hparams "), 3)
                        self.assertEqual(log.count("interval_boundary_eval "), 3)
                        self.assertEqual(log.count("train_loss "), 9)
                        self.assertIn("main hparams: conv.initial_lr=0.0012 conv.momentum=0.6 main=0.04", log)
                        self.assertIn("conv.initial_lr=0.0012 -> tta_val_acc=0.07", log)
                        self.assertIn("search_path step=0 conv.initial_lr=0.0012", log)
                        self.assertIn("phase=cooldown", log)
                        self.assertNotIn("candidate_start ", log)
                        self.assertNotIn("candidate_result ", log)
                        self.assertNotIn("main_diff:", log)
                        self.assertNotIn("cooldown_diff:", log)
                        self.assertIn("best_cooldown=0.07", log)
                        first_end = next(line for line in lines if line.startswith("interval_boundary_eval "))
                        self.assertIn("tta_val_acc=0.04", first_end)
                        self.assertIn("step=4", first_end)
                        self.assertGreater(clock["training"], 9)
                        self.assertEqual([i["steps"] for i in result["intervals"]], [4, 4, 1])
                        self.assertEqual([i["cooldown_steps"] for i in result["intervals"]], [3, 1, 0])
                        self.assertEqual(result["intervals"][0]["cooldown_hparams"],
                                         {"conv.initial_lr": 0.0012})
                        for path in ("conv.initial_lr", "conv.momentum"):
                            schedule = result["hparam_schedules"][path]
                            self.assertEqual([line["start_step"] for line in schedule], [0, 4, 8])
                            self.assertEqual([line["end_step"] for line in schedule], [3, 7, 8])
                            self.assertTrue(all(line["start"] == line["end"] for line in schedule))
                    else:
                        self.assertEqual(header, "config run=0")
                        self.assertEqual(len(lines), 1)
                for name, value in models[0].state_dict().items():
                    torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
                states = [o.state_dict()["state"] for o in optimizers]
                for index in states[0]:
                    torch.testing.assert_close(states[0][index]["momentum_buffer"],
                                               states[1][index]["momentum_buffer"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
