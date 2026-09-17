"""Chat memory: persistence, and telling a follow-up from a standalone question.

The second half is the part with a real failure mode. A word count cannot tell
"Which recipes use coconut?" from "and the second one?" -- both are four words,
one carries its own subject and one is meaningless alone.
"""
from __future__ import annotations

import pytest

from cognitive_kitchen.rag import memory as M


@pytest.fixture
def conversation():
    convo = M.Conversation(pipeline="test")
    yield convo
    M.delete(convo.conversation_id)


class TestPersistence:
    def test_a_conversation_survives_a_round_trip(self, conversation):
        conversation.add("user", "How do I make dosa?")
        conversation.add("assistant", "Soak rice and urad dal", meta="5 sources")
        M.save(conversation)

        back = M.load(conversation.conversation_id)
        assert back is not None
        assert len(back.turns) == 2
        assert back.turns[0].content == "How do I make dosa?"
        assert back.turns[1].meta == "5 sources"
        assert back.pipeline == "test"

    def test_title_comes_from_the_first_question(self, conversation):
        assert conversation.title == "New conversation"
        conversation.add("user", "What can I cook with coconut?")
        assert conversation.title.startswith("What can I cook")

    def test_title_is_not_overwritten_by_later_turns(self, conversation):
        conversation.add("user", "first question")
        conversation.add("user", "second question")
        assert conversation.title == "first question"

    def test_loading_an_unknown_id_returns_none(self):
        assert M.load("does-not-exist-at-all") is None

    def test_transcript_is_capped(self, conversation):
        for index in range(20):
            conversation.add("user", f"question {index}")
        assert len(conversation.recent()) <= M.CONTEXT_TURNS
        assert "question 19" in conversation.transcript()
        assert "question 0\n" not in conversation.transcript()


class TestFollowUpDetection:
    @pytest.mark.parametrize("question", [
        "Which recipes use coconut?",
        "How do I make Aviyal - Mixed Vegetables?",
        "What can I cook with cumin?",
        "I am avoiding nuts. What can I make?",
    ])
    def test_a_question_with_its_own_subject_is_not_a_follow_up(self, question):
        assert not M.looks_like_follow_up(question)

    @pytest.mark.parametrize("question", [
        "and the second one?",
        "what about without dairy?",
        "can I use butter instead?",
        "show me more",
        "the first one",
    ])
    def test_a_question_leaning_on_history_is_a_follow_up(self, question):
        assert M.looks_like_follow_up(question)

    def test_subject_detection_uses_the_vocabulary(self):
        # A word count cannot separate these two: both are four words.
        assert M.has_own_subject("Which recipes use coconut?")
        assert not M.has_own_subject("and the second one?")

    def test_resolution_is_skipped_on_an_empty_conversation(self, conversation):
        # Nothing to resolve against, so no model call should be attempted.
        resolved, rewritten = M.resolve("what about without dairy?", conversation)
        assert resolved == "what about without dairy?"
        assert rewritten is False

    def test_resolution_falls_back_when_the_rewriter_fails(self, conversation):
        conversation.add("user", "How do I make dosa?")

        class Broken:
            def complete(self, *args, **kwargs):
                raise RuntimeError("no api key")

        resolved, rewritten = M.resolve("what about without dairy?",
                                        conversation, generator=Broken())
        assert resolved == "what about without dairy?"
        assert rewritten is False

    def test_resolution_rewrites_a_follow_up(self, conversation):
        conversation.add("user", "Which rice dishes do you have?")
        conversation.add("assistant", "Lemon Rice and Tomato Rice")

        class Stub:
            def complete(self, prompt, max_tokens=None):
                assert "Which rice dishes" in prompt, "history not passed"
                return "a rice dish without dairy"

        resolved, rewritten = M.resolve("what about without dairy?",
                                        conversation, generator=Stub())
        assert resolved == "a rice dish without dairy"
        assert rewritten is True

    def test_an_unchanged_rewrite_is_not_flagged(self, conversation):
        conversation.add("user", "anything")

        class Echo:
            def complete(self, prompt, max_tokens=None):
                return "show me more"

        _, rewritten = M.resolve("show me more", conversation, generator=Echo())
        assert rewritten is False