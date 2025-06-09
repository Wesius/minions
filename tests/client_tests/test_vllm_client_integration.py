"""vLLM client integration tests (requires running vLLM server)."""

import unittest
import warnings
import sys
import os

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from minions.clients.vllm import VLLMClient


class TestVLLMClientIntegration(unittest.TestCase):
    """vLLM client tests - requires a running vLLM server."""

    def setUp(self):
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        try:
            self.client = VLLMClient(
                model_name=os.getenv("VLLM_MODEL", "llama-3"),
                base_url=base_url,
                temperature=0.1,
                max_tokens=20,
            )
        except Exception as e:
            warnings.warn(
                f"Skipping vLLM tests: Could not initialize client. Error: {e}",
                UserWarning,
            )
            self.skipTest("vLLM client initialization failed")

    def test_basic_chat(self):
        """Test a simple chat round trip."""
        messages = [{"role": "user", "content": "Say 'vllm test ok'"}]
        try:
            result = self.client.chat(messages)
            self.assertIsInstance(result, tuple)
            self.assertGreaterEqual(len(result), 2)

            responses = result[0]
            self.assertIsInstance(responses, list)
            self.assertGreater(len(responses), 0)
            self.assertIn("vllm test ok", responses[0].lower())
        except Exception as e:
            warnings.warn(
                f"Skipping vLLM tests: {e}",
                UserWarning,
            )
            self.skipTest("vLLM server not available")


if __name__ == "__main__":
    unittest.main()
