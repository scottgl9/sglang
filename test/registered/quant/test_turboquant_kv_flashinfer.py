import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-large")


class TestTurboQuantKVCacheFlashInfer(CustomTestCase):
    """End-to-end: TurboQuant KV cache + FlashInfer MHA backend.

    Exercises the encode→decode round-trip on every forward pass across a
    short gsm8k run. The accuracy bar is intentionally loose — this test is
    a "does it run and produce plausible output" gate, not an accuracy
    benchmark. Tune threshold after collecting baseline numbers.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--kv-cache-dtype",
                "turboquant",
                "--attention-backend",
                "flashinfer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        urlparse(self.base_url)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=50,
            num_threads=50,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # Loose bar: TurboQuant introduces lossy quantization noise; the aim
        # here is to catch hard breakage (empty outputs, NaNs), not to bound
        # the accuracy delta.
        self.assertGreater(metrics["score"], 0.10)


if __name__ == "__main__":
    unittest.main()
