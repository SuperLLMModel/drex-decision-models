"""Small CPU checkpoints exercise loading without trusting checkpoint code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import read_meta
from kev.edlm import EfficientDLM, EfficientDLMConfig
from kev.model import is_hybrid, load_backbone


class CheckpointTests(unittest.TestCase):
    def test_head_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {"head": {"q.weight": torch.ones(2, 4)}, "temperature": 1.2,
                        "head_dim": 2, "state_bidir": False}
            torch.save(expected, Path(directory) / "head.pt")
            actual = read_meta(directory)
            torch.testing.assert_close(actual["head"]["q.weight"], expected["head"]["q.weight"])
            self.assertEqual(actual["temperature"], 1.2)

    def test_pickle_payload_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"

            class Payload:
                def __reduce__(self):
                    return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

            torch.save({"head": {}, "payload": Payload()}, Path(directory) / "head.pt")
            with self.assertRaisesRegex(ValueError, "unsafe pickle"):
                read_meta(directory)
            self.assertFalse(marker.exists())

    def test_builtin_and_edlm_ignore_checkpoint_python(self):
        for config_class, model_class in ((Qwen3Config, Qwen3ForCausalLM),
                                          (EfficientDLMConfig, EfficientDLM)):
            with self.subTest(model=model_class.__name__), tempfile.TemporaryDirectory() as directory:
                config = config_class(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2,
                                      num_key_value_heads=2, head_dim=8, use_cache=False,
                                      **({"mask_token_id": 0} if config_class is EfficientDLMConfig else {}))
                original = model_class(config).eval()
                original.save_pretrained(directory)
                root = Path(directory)
                marker = root / "executed"
                payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('checkpoint code executed')\n"
                for filename in ("configuration_edlm.py", "modeling_edlm.py"):
                    (root / filename).write_text(payload)
                config_data = json.loads((root / "config.json").read_text())
                config_data["auto_map"] = {"AutoConfig": "configuration_edlm.EfficientDLMConfig",
                                           "AutoModelForCausalLM": "modeling_edlm.EfficientDLM"}
                (root / "config.json").write_text(json.dumps(config_data))
                before_path = list(sys.path)
                loaded = load_backbone(directory, None, torch.float32, "eager").eval()
                self.assertEqual(sys.path, before_path)
                self.assertFalse(marker.exists())
                ids = torch.tensor([[1, 2, 3]])
                with torch.no_grad():
                    torch.testing.assert_close(loaded(input_ids=ids).last_hidden_state,
                                               original.model(input_ids=ids).last_hidden_state)

    def test_moe_backbone_keeps_only_the_language_model(self):
        text = dict(vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=16, moe_intermediate_size=16,
                    shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
                    linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                    linear_num_value_heads=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
        vision = dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=32)
        full = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=text, vision_config=vision)).eval()
        with tempfile.TemporaryDirectory() as directory:
            full.save_pretrained(directory)
            backbone = load_backbone(directory, None, torch.float32, "eager").eval()
        self.assertTrue(is_hybrid(backbone.config))
        self.assertFalse(hasattr(backbone, "visual"))
        ids = torch.arange(10)[None]
        torch.testing.assert_close(backbone(input_ids=ids).last_hidden_state,
                                   full.model.language_model(input_ids=ids).last_hidden_state)

    def test_unknown_custom_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            (root / "custom.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
            (root / "config.json").write_text(json.dumps({
                "model_type": "untrusted_custom", "auto_map": {"AutoConfig": "custom.Config"}}))
            with self.assertRaises(ValueError):
                load_backbone(directory, None, torch.float32, "eager")
            self.assertFalse(marker.exists())

    def test_pickle_backbone_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Qwen3Config().save_pretrained(directory)
            torch.save({}, Path(directory) / "pytorch_model.bin")
            with self.assertRaises(OSError):
                load_backbone(directory, None, torch.float32, "eager")


if __name__ == "__main__":
    unittest.main()
