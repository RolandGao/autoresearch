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
        if self.training:
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
                config["hparam_tuning"] = dict(algorithm=algorithm, parameters={
                    "conv.initial_lr": [0.04, 0.06, 0.08],
                    "conv.momentum": [0.6, 0.7, 0.8]})
                with patch.object(train, "main", side_effect=fake_main):
                    result = train.run_experiment(0, None, config)
                self.assertEqual(len(result["trials"]), 9)
                self.assertEqual(result["best_result"]["tta_val_acc"], 0.4)
                self.assertEqual(train.get_hparam(result["best_config"], "conv.initial_lr"), 0.08)

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
        self.assertIn(0.024, calls)
        self.assertIn(0.067, calls)
        self.assertEqual(len(calls), 9)  # center, three smaller, five larger

    def test_invalid_interval_space_and_nesterov_zero_momentum(self):
        config = copy.deepcopy(train.RUN_CONFIGS[0])
        config["param_groups"]["conv"]["nesterov"] = True
        config["hparam_tuning"] = dict(algorithm="interval", parameters={
            "conv.momentum": [0, 0.6, 0.9]})
        choices, _ = train.interval_search_space(config)
        self.assertEqual(choices["conv.momentum"], [0.6, 0.9])
        for parameters in ({"batch_size": [125]}, {"conv.initial_lr": []},
                           {"whiten_weight.initial_lr": None}, {"conv.momentum": [0]}):
            config["hparam_tuning"]["parameters"] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                train.interval_search_space(config)

    def test_muon_zero_momentum_refreshes_buffer_before_momentum_returns(self):
        parameter = torch.nn.Parameter(torch.randn(3, 2))
        optimizer = train.GroupOptimizer([dict(params=[parameter], algorithm="muon",
                                              lr=0.001, momentum=0.6, nesterov=False)])
        expected = torch.zeros_like(parameter)
        for momentum in (0.6, 0.0, 0.1):
            gradient = torch.randn_like(parameter)
            parameter.grad = gradient.clone()
            optimizer.param_groups[0]["momentum"] = momentum
            expected.mul_(momentum).add_(gradient)
            optimizer.step()
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
    def test_cuda_interval_with_real_scores_and_multiple_candidates(self):
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

    def test_interval_replay_matches_committed_training_and_excludes_eval_time(self):
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
                        model.eval()
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
                    self.assertEqual(result["seconds"], clock["training"])
                    lines = output.getvalue().splitlines()
                    self.assertEqual(len(lines), 2)
                    self.assertEqual(json.loads(lines[0][7:])["batch_size"], 2)
                    self.assertTrue(lines[-1].startswith("val_acc=0.0900 tta_val_acc=0.0900 seconds="))
                    if interval:
                        self.assertGreater(clock["training"], 9)
                        self.assertEqual([i["steps"] for i in result["intervals"]], [4, 4, 1])
                        self.assertEqual([i["cooldown_steps"] for i in result["intervals"]], [3, 1, 0])
                        self.assertEqual(result["intervals"][0]["cooldown_hparams"],
                                         {"conv.initial_lr": 0.0012})
                for name, value in models[0].state_dict().items():
                    torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
                states = [o.state_dict()["state"] for o in optimizers]
                for index in states[0]:
                    torch.testing.assert_close(states[0][index]["momentum_buffer"],
                                               states[1][index]["momentum_buffer"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
