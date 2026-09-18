import unittest
from author_tool_log import image_evidence


class ToolLogTests(unittest.TestCase):
    def test_code_mode_image_requires_matching_successful_output(self):
        rows = [
            {"type": "turn_context", "payload": {"turn_id": "a"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "call_id": "c",
                    "input": 'const r = await tools.view_image({path:"/workspace/crop.png"}); image(r.image_url);',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "c",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,test",
                        }
                    ],
                },
            },
        ]
        self.assertEqual(
            image_evidence(rows),
            [
                {
                    "path": "/workspace/crop.png",
                    "call_id": "c",
                    "turn_id": "a",
                    "returned_image": True,
                }
            ],
        )
        self.assertEqual(image_evidence(rows[:-1])[0]["returned_image"], False)
        rows.append({"type": "turn_context", "payload": {"turn_id": "b"}})
        self.assertEqual(image_evidence(rows), [])

    def test_multiple_or_aliased_image_calls_rejected(self):
        for source in (
            'await tools.view_image({path:"/ok"}); await tools.view_image({path:"/outside"});',
            'const view = tools.view_image; await view({path:"/outside"});',
            'await tools["view_image"]({path:"/outside"});',
        ):
            rows = [
                {"type": "turn_context", "payload": {"turn_id": "a"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "custom_tool_call",
                        "name": "exec",
                        "call_id": "c",
                        "input": source,
                    },
                },
            ]
            with self.assertRaises(ValueError):
                image_evidence(rows)

    def test_unknown_image_output_rejected(self):
        with self.assertRaises(ValueError):
            image_evidence(
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call_output",
                            "call_id": "unknown",
                            "output": [{"type": "input_image"}],
                        },
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
