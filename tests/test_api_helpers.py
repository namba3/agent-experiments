"""Tests for API request validation and message conversion."""

from __future__ import annotations

import math
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import agent_api_client
import agent_framework_agent
import langgraph_agent


class MessageConversionTests(unittest.TestCase):
    def test_langgraph_converts_plain_user_text(self) -> None:
        converted = langgraph_agent.api_messages_to_langchain(
            [{"role": "user", "content": "hello"}]
        )

        self.assertEqual(len(converted), 1)
        self.assertIsInstance(converted[0], HumanMessage)
        self.assertEqual(converted[0].content, "hello")

    def test_langgraph_converts_openai_roles(self) -> None:
        converted = langgraph_agent.api_messages_to_langchain(
            [
                {"role": "system", "content": "system"},
                {"role": "developer", "content": "developer"},
                {"role": "assistant", "content": "assistant"},
                {"role": "user", "content": "user"},
            ]
        )

        self.assertIsInstance(converted[0], SystemMessage)
        self.assertIsInstance(converted[1], SystemMessage)
        self.assertIsInstance(converted[2], AIMessage)
        self.assertIsInstance(converted[3], HumanMessage)
        self.assertEqual(converted[1].content, "developer")

    def test_framework_converts_plain_user_text_and_developer_role(self) -> None:
        converted = agent_framework_agent.api_messages_to_agent(
            [
                {"role": "developer", "content": "developer"},
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertEqual([message.role for message in converted], ["system", "user"])
        self.assertEqual(
            [message.text for message in converted], ["developer", "hello"]
        )

    def test_both_converters_accept_a_base64_image(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            }
        ]

        graph_message = langgraph_agent.api_messages_to_langchain(messages)[0]
        framework_message = agent_framework_agent.api_messages_to_agent(messages)[0]
        self.assertIsInstance(graph_message, HumanMessage)
        self.assertEqual(graph_message.content[0], {"type": "text", "text": "describe"})
        self.assertEqual(framework_message.role, "user")
        self.assertEqual(framework_message.text, "describe")

    def test_both_converters_reject_invalid_image_data(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,not base64!"},
                    }
                ],
            }
        ]

        for converter in (
            langgraph_agent.api_messages_to_langchain,
            agent_framework_agent.api_messages_to_agent,
        ):
            with (
                self.subTest(converter=converter.__module__),
                self.assertRaises(ValueError),
            ):
                converter(messages)

    def test_converters_reject_unsupported_roles(self) -> None:
        for converter in (
            langgraph_agent.api_messages_to_langchain,
            agent_framework_agent.api_messages_to_agent,
        ):
            with (
                self.subTest(converter=converter.__module__),
                self.assertRaises(ValueError),
            ):
                converter([{"role": "tool", "content": "result"}])


class GenerationOptionTests(unittest.TestCase):
    FUNCTIONS = (
        langgraph_agent.api_generation_options,
        agent_framework_agent.api_generation_options,
    )
    DEFAULTS = {
        "temperature": 0.7,
        "seed": 11,
        "top_p": 0.8,
        "num_predict": 32,
    }

    def convert(self, function, request: dict[str, object]) -> dict[str, object]:
        return function(
            request,
            temperature=self.DEFAULTS["temperature"],
            seed=self.DEFAULTS["seed"],
            top_p=self.DEFAULTS["top_p"],
            num_predict=self.DEFAULTS["num_predict"],
        )

    def test_defaults_and_request_overrides_map_to_ollama_options(self) -> None:
        expected_defaults = {
            "temperature": 0.7,
            "seed": 11,
            "top_p": 0.8,
            "num_predict": 32,
        }
        expected_overrides = {
            "temperature": 0.2,
            "seed": 5,
            "top_p": 1.0,
            "num_predict": 64,
        }

        for function in self.FUNCTIONS:
            with self.subTest(function=function.__module__):
                self.assertEqual(self.convert(function, {}), expected_defaults)
                self.assertEqual(
                    self.convert(
                        function,
                        {
                            "temperature": 0.2,
                            "seed": 5,
                            "top_p": 1.0,
                            "max_tokens": 64,
                        },
                    ),
                    expected_overrides,
                )

    def test_invalid_generation_options_are_rejected_by_both_servers(self) -> None:
        invalid_requests = (
            {"temperature": True},
            {"temperature": math.nan},
            {"temperature": math.inf},
            {"seed": True},
            {"top_p": -0.1},
            {"top_p": 1.1},
            {"max_tokens": 0},
            {"max_tokens": True},
        )

        for function in self.FUNCTIONS:
            for request in invalid_requests:
                with self.subTest(function=function.__module__, request=request):
                    with self.assertRaises(ValueError):
                        self.convert(function, request)


class ClientMessageTests(unittest.TestCase):
    def test_user_message_supports_text_and_image_parts(self) -> None:
        self.assertEqual(
            agent_api_client.user_message("hello"),
            {"role": "user", "content": "hello"},
        )
        self.assertEqual(
            agent_api_client.user_message("describe", "data:image/png;base64,aGVsbG8="),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    },
                ],
            },
        )

    def test_history_trimming_keeps_system_messages_and_recent_turns(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]

        agent_api_client.trim_history(messages, keep_turns=2)

        self.assertEqual(
            [message["content"] for message in messages],
            ["system", "u2", "a2", "u3", "a3"],
        )


if __name__ == "__main__":
    unittest.main()
