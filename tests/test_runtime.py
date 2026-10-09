"""Weight-free tests of the single-decision System One API."""
import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference
from kev.api import SystemOneRequest, to_record
from serve import make_handler


REQUEST = {"state": "ticket", "questions": {"level": {"type": "score", "criteria": ["Low", "High"]}}}


class Tokenizer:
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(range(len(text))))


class Scorer:
    def encode(self, tok, record, **kwargs):
        return {"ids": [0, 1, 2], "record": record}

    def probs(self, enc):
        return [torch.tensor([0.25, 0.75]) for _ in enc["record"]["questions"]]


class RuntimeTests(unittest.TestCase):
    def test_ordered_legend(self):
        parsed = SystemOneRequest.model_validate(REQUEST)
        self.assertEqual(to_record(parsed)[1][0]["legend"], ["Low", "High"])

    def test_reject_batch_wrapper(self):
        for body in ({"requests": [REQUEST]}, {**REQUEST, "requests": [REQUEST]}):
            with self.assertRaises(ValueError):
                inference.decide(Tokenizer(), Scorer(), body)

    def test_incomplete_and_nonfinite_readouts(self):
        scorer = Scorer()
        for rows in ([], [torch.tensor([float("nan"), 0.0])], [torch.tensor([1.0])]):
            with mock.patch.object(scorer, "probs", return_value=rows):
                with self.assertRaisesRegex(RuntimeError, "readout"):
                    inference.decide(Tokenizer(), scorer, REQUEST)

    def test_http_contract_and_internal_errors(self):
        scorer = Scorer()
        handler = make_handler(Tokenizer(), scorer, "drex-v1.5")
        handler.log_message = lambda *args: None
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for body, expected in ((REQUEST, 200), ({"requests": [REQUEST]}, 400),
                                       ({**REQUEST, "state": float("nan")}, 400),
                                       ({"state": "x", "questions": {}}, 400)):
                    with self.subTest(body=body):
                        conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                        try:
                            conn.request("POST", "/v1/systemone", json.dumps(body), {"Content-Type": "application/json"})
                            response = conn.getresponse()
                            self.assertEqual(response.status, expected)
                            response.read()
                        finally:
                            conn.close()
                with mock.patch.object(scorer, "probs", side_effect=ValueError("private detail")):
                    conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                    try:
                        conn.request("POST", "/v1/systemone", json.dumps(REQUEST))
                        response = conn.getresponse()
                        self.assertEqual(response.status, 500)
                        self.assertEqual(json.loads(response.read()), {"error": "internal scoring error"})
                    finally:
                        conn.close()
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
