# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Unit tests for recursion limit fallback functionality in graph nodes.

Tests the graceful fallback behavior when agents hit the recursion limit,
including the _handle_recursion_limit_fallback function and its key behaviors.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from src.config.configuration import Configuration
from src.graph.nodes import _handle_recursion_limit_fallback
from src.graph.types import State


class TestHandleRecursionLimitFallback:
    """Test suite for _handle_recursion_limit_fallback() function.

    This function is called when an agent hits the recursion limit during execution.
    It should:
    1. Return early with empty messages if no messages accumulated
    2. Strip trailing system messages from accumulated messages
    3. Get the agent-specific and recursion_fallback system prompts
    4. Invoke the LLM with the cleaned messages + system prompts
    5. Sanitize the LLM response
    6. Set current_step.execution_res with the sanitized content
    7. Return accumulated messages + new AIMessage with the fallback content
    """

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_messages(self):
        """Test that fallback returns empty list when no messages accumulated.

        This is an early-return optimization - if there's nothing to summarize,
        don't waste resources calling the LLM.
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()
        # Initialize execution_res as None to properly test early return
        current_step.execution_res = None
        partial_agent_messages = []

        result = await _handle_recursion_limit_fallback(
            messages=partial_agent_messages,
            agent_name="researcher",
            current_step=current_step,
            state=state,
        )

        assert result == []
        # With early return, execution_res should not have been set
        assert current_step.execution_res is None

    @pytest.mark.asyncio
    async def test_strips_trailing_system_messages(self):
        """Test that trailing system messages are removed before generating fallback.

        The function should remove any system messages at the end of the accumulated
        messages since these will be replaced with the fallback system prompts.
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()

        # Create messages with trailing system messages
        partial_agent_messages = [
            HumanMessage(content="User input"),
            AIMessage(content="AI response"),
            SystemMessage(content="System message 1"),  # Should be stripped
            SystemMessage(content="System message 2"),  # Should be stripped
        ]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Fallback summary"

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="Fallback system prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify trailing system messages were stripped from the result
            # Result should have HumanMessage, AIMessage, and new AIMessage (fallback)
            # but NOT the trailing SystemMessages
            assert len(result) == 3
            assert isinstance(result[0], HumanMessage)
            assert isinstance(result[1], AIMessage)
            assert isinstance(result[2], AIMessage)
            assert result[2].name == "researcher"

            # Verify LLM was called with cleaned messages (no trailing system messages)
            llm_call_args = mock_llm.invoke.call_args[0][0]
            # Should have original messages without trailing system messages
            # plus the two new system prompts
            assert len(llm_call_args) == 4  # 2 original + 2 system prompts

    @pytest.mark.asyncio
    async def test_gets_system_prompts_for_agent_and_fallback(self):
        """Test that fallback gets both agent-specific and recursion_fallback system prompts.

        The function should call get_system_prompt_template twice:
        1. Once with the agent_name to get the agent's system prompt
        2. Once with "recursion_fallback" to get the fallback instructions
        """
        state = State(messages=[], locale="zh-CN")
        current_step = MagicMock()
        partial_agent_messages = [HumanMessage(content="Test")]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Summary in Chinese"

        mock_agent_system_prompt = SystemMessage(content="Agent system prompt")
        mock_fallback_system_prompt = SystemMessage(content="Fallback system prompt")

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template") as mock_get_prompt_template, \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            # Set up different return values for each call
            mock_get_prompt_template.side_effect = [
                mock_agent_system_prompt,  # First call for agent_name
                mock_fallback_system_prompt,  # Second call for "recursion_fallback"
            ]

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify get_system_prompt_template was called twice
            assert mock_get_prompt_template.call_count == 2

            # Verify first call was for agent
            first_call = mock_get_prompt_template.call_args_list[0]
            assert first_call[0][0] == "researcher"
            assert first_call[0][1]["locale"] == "zh-CN"

            # Verify second call was for recursion_fallback
            second_call = mock_get_prompt_template.call_args_list[1]
            assert second_call[0][0] == "recursion_fallback"
            assert second_call[0][1]["locale"] == "zh-CN"

    @pytest.mark.asyncio
    async def test_invokes_llm_with_correct_message_structure(self):
        """Test that LLM is invoked with cleaned messages + system prompts.

        The LLM should receive:
        1. The accumulated messages (with trailing system messages removed)
        2. The agent-specific system prompt
        3. The recursion_fallback system prompt
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()

        partial_agent_messages = [
            HumanMessage(content="Research topic: AI safety"),
            AIMessage(content="I'll research this for you."),
        ]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Research summary: AI safety is important..."

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify LLM was called
            mock_llm.invoke.assert_called_once()

            # Verify the message structure passed to LLM
            llm_call_args = mock_llm.invoke.call_args[0][0]
            # Should have: 2 original messages + 2 system prompts = 4 messages
            assert len(llm_call_args) == 4
            assert llm_call_args[0] == partial_agent_messages[0]
            assert llm_call_args[1] == partial_agent_messages[1]
            # Last two should be the system prompts
            assert isinstance(llm_call_args[2], SystemMessage)
            assert isinstance(llm_call_args[3], SystemMessage)

    @pytest.mark.asyncio
    async def test_sanitizes_llm_response(self):
        """Test that the LLM response is sanitized before being used.

        The fallback response from the LLM should be sanitized to remove
        any extra tokens or malformed content.
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()
        partial_agent_messages = [HumanMessage(content="Test")]

        # Mock unsanitized response with extra tokens
        mock_llm_response = MagicMock()
        mock_llm_response.content = "<extra_tokens>This is the summary</extra_tokens>"

        sanitized_content = "This is the summary"

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=sanitized_content) as mock_sanitize:

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify sanitize was called with the LLM response content
            mock_sanitize.assert_called_once_with(mock_llm_response.content)

            # Verify sanitized content was used in the result
            assert result[-1].content == sanitized_content
            assert current_step.execution_res == sanitized_content

    @pytest.mark.asyncio
    async def test_sets_current_step_execution_res(self):
        """Test that current_step.execution_res is set with the fallback content.

        This is important for the workflow to have access to the fallback result.
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()
        partial_agent_messages = [HumanMessage(content="Test")]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Fallback result"

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="coder",
                current_step=current_step,
                state=state,
            )

            # Verify current_step.execution_res was set
            assert current_step.execution_res == mock_llm_response.content

    @pytest.mark.asyncio
    async def test_returns_messages_with_fallback_ai_message(self):
        """Test that the function returns accumulated messages + new AIMessage.

        The result should include all the original messages (minus trailing system messages)
        plus a new AIMessage containing the fallback content with the agent's name.
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()

        partial_agent_messages = [
            HumanMessage(content="Research AI"),
            AIMessage(content="I'll search for information."),
        ]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Here's what I found about AI..."

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify result structure
            assert isinstance(result, list)
            assert len(result) == len(partial_agent_messages) + 1

            # Verify original messages are preserved
            assert result[0] == partial_agent_messages[0]
            assert result[1] == partial_agent_messages[1]

            # Verify new AIMessage with fallback content
            assert isinstance(result[-1], AIMessage)
            assert result[-1].content == mock_llm_response.content
            assert result[-1].name == "researcher"

    @pytest.mark.asyncio
    async def test_propagates_llm_exceptions(self):
        """Test that exceptions from LLM calls are propagated to the caller.

        If the fallback LLM call fails, the exception should not be caught
        but should propagate up so the caller can handle it.
        """
        state = State(messages=[], locale="en-US")
        current_step = MagicMock()
        partial_agent_messages = [HumanMessage(content="Test")]

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(side_effect=Exception("LLM API error"))
            mock_get_llm.return_value = mock_llm

            # Should raise the exception
            with pytest.raises(Exception, match="LLM API error"):
                await _handle_recursion_limit_fallback(
                    messages=partial_agent_messages,
                    agent_name="researcher",
                    current_step=current_step,
                    state=state,
                )

    @pytest.mark.asyncio
    async def test_handles_messages_with_tool_calls(self):
        """Test fallback behavior with messages containing tool calls.

        When the recursion limit is hit during tool execution, the accumulated
        messages may include tool calls and their results. The fallback should
        handle this correctly.
        """
        from langchain_core.messages import ToolCall

        state = State(messages=[], locale="en-US")
        current_step = MagicMock()

        # Create messages with tool calls
        tool_call = ToolCall(
            name="web_search",
            args={"query": "AI safety"},
            id="123"
        )

        partial_agent_messages = [
            HumanMessage(content="Research AI safety"),
            AIMessage(content="", tool_calls=[tool_call]),
            HumanMessage(content="Tool result: Found 5 articles"),
        ]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Based on the search results, AI safety is crucial..."

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify all messages are included
            assert len(result) == len(partial_agent_messages) + 1
            assert result[0] == partial_agent_messages[0]
            assert result[1] == partial_agent_messages[1]
            assert result[2] == partial_agent_messages[2]
            assert isinstance(result[-1], AIMessage)
            assert result[-1].name == "researcher"

    @pytest.mark.asyncio
    async def test_uses_default_locale_when_missing(self):
        """Test that fallback uses 'en-US' as default locale when not provided in state.

        This ensures the function works correctly even when locale is not set.
        Note: When state has locale=None, the fallback_state will have locale=None,
        and the default 'en-US' is used via .get() fallback when retrieving locale.
        """
        # Create state with locale missing entirely (not None, but missing key)
        state = State(messages=[])  # No locale key at all
        current_step = MagicMock()
        partial_agent_messages = [HumanMessage(content="Test")]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Summary"

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template") as mock_get_prompt_template, \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm
            mock_get_prompt_template.return_value = SystemMessage(content="System prompt")

            result = await _handle_recursion_limit_fallback(
                messages=partial_agent_messages,
                agent_name="researcher",
                current_step=current_step,
                state=state,
            )

            # Verify get_system_prompt_template was called with default locale "en-US"
            # The 2nd positional arg (fallback_state) should have locale="en-US"
            call_args = mock_get_prompt_template.call_args_list
            for call in call_args:
                fallback_state = call[0][1]  # 2nd positional arg
                assert fallback_state.get("locale") == "en-US"

                # 4th positional arg is also locale, should be "en-US"
                locale_arg = call[0][3]
                assert locale_arg == "en-US"

    @pytest.mark.asyncio
    async def test_handles_various_agent_names(self):
        """Test that fallback works correctly with different agent types.

        The agent_name should be passed correctly to get the right LLM and
        should be set as the name of the fallback AIMessage.
        """
        state = State(messages=[], locale="en-US")
        partial_agent_messages = [HumanMessage(content="Test")]

        mock_llm_response = MagicMock()
        mock_llm_response.content = "Agent summary"

        with patch("src.graph.nodes.get_llm_by_type") as mock_get_llm, \
             patch("src.graph.nodes.get_system_prompt_template", return_value=SystemMessage(content="System prompt")), \
             patch("src.graph.nodes.sanitize_tool_response", return_value=mock_llm_response.content):

            mock_llm = MagicMock()
            mock_llm.invoke = MagicMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            # Test valid agent names from AGENT_LLM_MAP
            for agent_name in ["researcher", "coder", "analyst", "reporter"]:
                current_step = MagicMock()

                result = await _handle_recursion_limit_fallback(
                    messages=partial_agent_messages,
                    agent_name=agent_name,
                    current_step=current_step,
                    state=state,
                )

                # Verify agent name is set on the fallback AIMessage
                assert result[-1].name == agent_name

                # Verify LLM was retrieved for the correct agent type
                mock_get_llm.assert_called()
