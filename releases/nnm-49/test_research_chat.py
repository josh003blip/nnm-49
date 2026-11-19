import json
import unittest
from unittest.mock import Mock, patch

from local_language import LocalLanguage
from research_chat import ResearchChat


class ResearchChatTests(unittest.TestCase):
    def setUp(self):
        self.chat = ResearchChat()

    def test_uses_original_checkpoint_for_prediction(self):
        reply = self.chat.reply("What would the model predict for 0.5?")
        self.assertIn("NNM-49 prediction", reply)
        self.assertAlmostEqual(float(reply.split("→")[1]), self.chat.model.predict(0.5), places=11)
        self.assertIn("outside", self.chat.reply("Predict 1.1"))
        self.assertIn("3.", self.chat.reply("Run 3 times from 0.5"))

    def test_compiler_and_followup(self):
        self.assertIn("Bytecode", self.chat.reply("Compile x * (2 + 3 * 4) - 5"))
        self.assertIn("→ 37", self.chat.reply("Try that at 3"))
        self.assertIn("→ 51", self.chat.reply("Run x * (2 + 3 * 4) - 5 @ 4"))
        self.assertIn("Couldn’t complete", self.chat.reply("Compile __import__('os').getcwd()"))
        self.assertIn("→ 65", self.chat.reply("Try that at 5"))
        self.chat.reset()
        self.assertIn("first", self.chat.reply("Try that at 5"))

    def test_fit_data_is_separate_from_holdout(self):
        base = "Fit an equation to [(0,1), (1,3), (2,5), (3,7)]; test on "
        exact = self.chat.reply(base + "[(4,9), (5,11)]")
        shifted = self.chat.reply(base + "[(4,10), (5,12)]")
        self.assertEqual(exact.splitlines()[0], "y = 1 + 2*x")
        self.assertEqual(exact.splitlines()[0], shifted.splitlines()[0])
        self.assertIn("Holdout MSE: 0", exact)
        self.assertIn("Holdout MSE: 1", shifted)
        self.assertIn("different x", self.chat.reply(base + "[(3,7)]"))

    def test_event_translation_and_clarification(self):
        question = "Maya started painting at 2:00. At 2:12, was she painting?"
        self.assertIn("stop painting", self.chat.reply(question))
        self.assertIn("Yes.", self.chat.reply("2:30"))
        self.assertIn("No.", self.chat.reply("Maya was painting from 2:00 to 2:30. "
                                              "At 2:30, was she painting?"))

    def test_local_language_can_propose_but_cannot_bypass_compiler(self):
        language = Mock(side_effect=[
            '{"action":"reply","text":"Start with three examples."}',
            '{"action":"run","expression":"x*x + 2","x":3}',
            '{"action":"run","expression":"__import__(\\"os\\")","x":3}',
        ])
        chat = ResearchChat(language=language)
        self.assertIn("Local language model", chat.reply("Help me plan an experiment"))
        self.assertIn("→ 11", chat.reply("Square its input and add two, with x equal to three"))
        self.assertEqual(language.call_args.args[0][0]["content"], "Help me plan an experiment")
        self.assertIn("Couldn’t complete", chat.reply("Run a system function"))

    @patch("local_language.HTTPConnection")
    def test_language_transport_is_loopback_only(self, connection):
        response = connection.return_value.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps({"choices": [{"message": {
            "content": '<think>draft</think>\n```json\n{"action":"reply","text":"Hi"}\n```'
        }}]}).encode()
        language = LocalLanguage(port=8081, model="local")
        result = language([{"role": "user", "content": "Hello"}])
        self.assertEqual(result, '{"action":"reply","text":"Hi"}')
        connection.assert_called_once_with("127.0.0.1", 8081, timeout=30)
        request = connection.return_value.request
        self.assertEqual(request.call_args.args[:2], ("POST", "/v1/chat/completions"))
        self.assertEqual(json.loads(request.call_args.kwargs["body"])["model"], "local")
        connection.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
