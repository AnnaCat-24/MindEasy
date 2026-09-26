"""OpenAI-backed conversational companion for MindEasy."""

import logging
import os
import hashlib

try:
	from dotenv import load_dotenv

except ImportError:  # The local fallback still works without python-dotenv.
	load_dotenv = None


if load_dotenv:
	load_dotenv()


class CompanionError(RuntimeError):
	"""A visible configuration or provider failure for the Flask layer."""


COMPANION_SYSTEM_PROMPT = """You are MindEasy, a warm, friendly conversational wellness companion.

Always respond to the user's latest message. Use previous conversation only to maintain context.
Never repeat your previous response. If the user changes topic, follow the new topic naturally.
Acknowledge specific details from the latest message. Keep responses short and conversational.
Ask at most one natural follow-up question. Do not diagnose mental health conditions or claim to be
a therapist. Do not pretend to have personal experiences or personal preferences. Help the user feel
heard, relaxed, and comfortable talking."""


def _fallback_response(user_message: str, sentiment: str) -> str:
	"""Give a current-message-specific response when OpenAI is unavailable."""
	lower = user_message.lower()
	def choose(options: tuple[str, ...]) -> str:
		index = hashlib.sha256(user_message.encode("utf-8")).digest()[0] % len(options)
		return options[index]

	if "biryani" in lower or "hungry" in lower:
		return choose((
			"Biryani sounds like a comforting idea. I hope you get a chance to enjoy a good meal soon. What kind of biryani are you in the mood for?",
			"That sounds like a good moment to think about food and comfort. What would make the meal feel especially satisfying today?",
		))
	if any(word in lower for word in ("tired", "exhausted", "drained")):
		return choose((
			"It sounds like your energy is running low. A small pause, some water, or a little rest might help before you decide what comes next.",
			"You sound worn out. Which would help most right now: stopping briefly, getting something to drink, or finishing one small task?",
			"That kind of tiredness can make everything feel heavier. What is one thing you could make easier for yourself this evening?",
		))
	if any(word in lower for word in ("work", "assignment", "exam", "deadline")):
		return choose((
			"That sounds like a lot to carry. Try choosing just one task for the next short stretch instead of holding the whole workload at once. What feels most urgent?",
			"There is a lot competing for your attention. Naming the next concrete step may make the workload feel less sprawling. What needs attention first?",
			"That sounds demanding. You might get some breathing room by separating what must happen today from what can wait until tomorrow.",
		))
	if sentiment == "positive":
		return choose((
			"That sounds like a genuinely happy moment. What part of it are you looking forward to most?",
			"There is a nice lift in what you shared. What made this moment stand out to you?",
			"It sounds like something went well. What would you like to hold onto from that feeling?",
		))
	return choose((
		"Thanks for sharing that with me. What part of it would feel most helpful to talk through right now?",
		"I hear you. Is there a particular detail from today that keeps coming back to you?",
		"Thanks for putting that into words. Would you rather explore the feeling itself or what happened around it?",
	))


def _conversation_messages(history: list[dict] | None, user_message: str) -> list[dict]:
	"""Convert the small Flask session history to valid Responses API messages."""
	messages = []
	for item in (history or [])[-6:]:
		role = "assistant" if item.get("role") == "mindeasy" else "user"
		content = item.get("text", "")
		if content:
			messages.append({"role": role, "content": content})
	messages.append({"role": "user", "content": user_message})
	return messages


def get_companion_response(user_message: str, sentiment: str, history: list[dict] | None = None) -> str:
	"""Call OpenAI for every request and return ordinary conversational text."""
	key = os.getenv("OPENAI_API_KEY")
	if not key:
		error = CompanionError("OPENAI_API_KEY is not configured. Add it to .env and restart Flask.")
		logging.error("Companion error: %s", error)
		return _fallback_response(user_message, sentiment)

	try:
		from openai import OpenAI
		client = OpenAI(api_key=key)
		model = os.getenv("OPENAI_MODEL")
		if not model:
			raise CompanionError("OPENAI_MODEL is not configured in .env.")
		response = client.responses.create(
			model=model,
			instructions=COMPANION_SYSTEM_PROMPT,
			input=_conversation_messages(history, user_message),
		)
		if not response.output_text or not response.output_text.strip():
			raise CompanionError("OpenAI returned an empty companion response.")
		return response.output_text.strip()
	except CompanionError as error:
		logging.error("Companion error: %s", error)
		return _fallback_response(user_message, sentiment)
	except Exception as error:
		logging.exception("OpenAI companion request failed: %s", error)
		return _fallback_response(user_message, sentiment)