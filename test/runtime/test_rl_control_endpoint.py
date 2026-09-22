"""The in-engine RL control app as an external gateway sees it: where it
listens, what it advertises, and how it authenticates.

Run with: pytest test/runtime/test_rl_control_endpoint.py
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.entrypoints import rl_control  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _args(**overrides):
    base = {"host": "127.0.0.1", "rl_control_port": 40100, "rl_control_host": None}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestControlEndpointHelpers(unittest.TestCase):
    def test_bind_host_defaults_to_engine_host(self):
        self.assertEqual(rl_control.control_bind_host(_args()), "127.0.0.1")
        self.assertEqual(
            rl_control.control_bind_host(_args(rl_control_host="0.0.0.0")), "0.0.0.0"
        )

    def test_control_url_uses_bound_host_and_port(self):
        self.assertEqual(rl_control.control_url(_args()), "http://127.0.0.1:40100")
        self.assertEqual(
            rl_control.control_url(_args(rl_control_host="10.0.0.5")),
            "http://10.0.0.5:40100",
        )
        self.assertEqual(
            rl_control.control_url(_args(host="::1")), "http://[::1]:40100"
        )

    def test_control_url_is_none_without_a_port(self):
        self.assertIsNone(rl_control.control_url(_args(rl_control_port=None)))
        self.assertIsNone(rl_control.control_url(_args(rl_control_port=0)))

    def test_capabilities_match_the_smg_label_contract(self):
        caps = rl_control.capabilities()
        self.assertEqual(caps["rl.pause_modes"], "wait,abort,keep")
        self.assertEqual(caps["rl.update_from"], "disk,distributed")
        for key in (
            "rl.abort",
            "rl.flush_cache",
            "rl.sleep_wake",
            "rl.reports_weight_version",
        ):
            self.assertEqual(caps[key], "true")
        self.assertNotIn("tensor", caps["rl.update_from"])

    def test_advertisement_carries_url_and_capabilities(self):
        adv = rl_control.advertisement(_args())
        self.assertEqual(adv["rl.control_url"], "http://127.0.0.1:40100")
        self.assertEqual(adv["rl.abort"], "true")
        self.assertNotIn(
            "rl.control_url", rl_control.advertisement(_args(rl_control_port=None))
        )


class TestServerArgsFlags(unittest.TestCase):
    def test_flags_parse_and_default_to_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(["--model", "m"])
        self.assertIsNone(ns.rl_control_host)
        self.assertIsNone(ns.rl_control_api_key)
        ns = parser.parse_args(
            [
                "--model",
                "m",
                "--rl-control-host",
                "0.0.0.0",
                "--rl-control-api-key",
                "k",
            ]
        )
        self.assertEqual(ns.rl_control_host, "0.0.0.0")
        self.assertEqual(ns.rl_control_api_key, "k")


if __name__ == "__main__":
    unittest.main()
