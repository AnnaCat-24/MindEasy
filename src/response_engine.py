"""Message-aware supportive resources, kept separate from the ML classifier."""

from urllib.parse import quote


RESOURCE_CATALOG = {
    "positive": {
        "quotes": [
            "Joy is often found in the moments we choose to notice.",
            "Let a good moment be good without needing to explain it.",
            "Small celebrations can give an ordinary day a little more color.",
        ],
        "music": [
            ("Uplifting acoustic", "upbeat feel good acoustic music", "A bright option for holding onto a good moment."),
            ("Energetic instrumental", "upbeat instrumental music", "A lively background for positive energy."),
            ("Gentle happy playlist", "gentle happy playlist", "A warm soundtrack for a celebratory mood."),
        ],
        "exercises": [
            ("Savoring pause", "Name three details from this moment that you appreciate."),
            ("Gratitude note", "Write one short sentence about something or someone you are glad to have."),
            ("Energizing stretch", "Stand up, roll your shoulders, and take three comfortable breaths."),
        ],
    },
    "neutral": {
        "quotes": [
            "A quiet moment is still part of the day.",
            "You can be curious about a day without needing to judge it.",
            "One ordinary step can still move you forward.",
        ],
        "music": [
            ("Peaceful piano", "peaceful soft piano", "Quiet background sound for an ordinary moment."),
            ("Ambient focus", "ambient instrumental background music", "A low-key option for letting your thoughts settle."),
            ("Calm acoustic", "calm acoustic instrumental", "A gentle soundtrack for a steady pace."),
        ],
        "exercises": [
            ("One-minute pause", "Put both feet on the floor and notice five slow, comfortable breaths."),
            ("Hand relaxation", "Unclench your hands, soften your jaw, and let your shoulders drop."),
            ("Gentle stretch", "Reach your arms overhead, then release them slowly at your own pace."),
        ],
    },
    "negative": {
        "quotes": [
            "You do not have to solve everything in one moment.",
            "A difficult moment is worth meeting with patience, not judgment.",
            "It is okay to make the next step smaller.",
        ],
        "music": [
            ("Calming piano", "calming piano instrumental music", "A gentle background option for taking a quiet pause."),
            ("Soft ambient space", "soft ambient relaxing music", "A spacious sound for slowing the pace."),
            ("Peaceful nature sounds", "peaceful nature sounds", "A quiet nature-inspired option for a reset."),
        ],
        "exercises": [
            ("Slow breathing", "Breathe in gently for four counts, then exhale for six counts. Repeat three times."),
            ("5-4-3-2-1 grounding", "Notice five things you see, four you feel, three you hear, two you smell, and one you taste."),
            ("Shoulder release", "Lift your shoulders gently, hold for two seconds, and let them drop. Repeat twice."),
        ],
    },
    "focus": {
        "music": [
            ("Study instrumentals", "instrumental study music", "A steady, lyric-light option for focused work."),
            ("Lo-fi focus", "lofi beats for studying", "A gentle rhythm for settling into one task."),
            ("Concentration piano", "concentration piano music", "Soft piano for a quieter work session."),
        ],
    },
    "tired": {
        "music": [
            ("Quiet evening", "quiet evening acoustic music", "A low-key option for winding down."),
            ("Slow ambient", "slow relaxing ambient music", "A soft background for a low-energy moment."),
            ("Peaceful rest", "peaceful rest music", "A calm soundtrack for taking things gently."),
        ],
    },
}


def _themes(text: str) -> set[str]:
    """Detect broad, non-clinical topics for resource selection."""
    lower = text.lower()
    groups = {
        "stress": ("stress", "overwhelmed", "pressure", "too much", "can't manage", "cannot manage"),
        "work": ("work", "project", "assignment", "deadline", "task"),
        "study": ("study", "studying", "exam", "class", "college", "school"),
        "tired": ("tired", "exhausted", "drained", "sleepy", "sleep", "rest"),
        "relaxation": ("relax", "calm", "quiet", "pause", "break"),
        "celebration": ("birthday", "party", "celebrate", "excited", "happy", "joy"),
        "loneliness": ("lonely", "alone", "isolated"),
        "motivation": ("motivated", "proud", "hopeful", "confident"),
        "frustration": ("angry", "frustrated", "upset", "annoyed"),
        "food": ("hungry", "food", "biryani", "eat", "meal"),
        "social": ("friend", "friends", "family", "party", "people"),
    }
    return {theme for theme, words in groups.items() if any(word in lower for word in words)}


def _choose(items: list, state: dict, key: str):
    """Cycle through a catalog, avoiding the item shown most recently."""
    if not items:
        return None
    next_index = (int(state.get(key, -1)) + 1) % len(items)
    state[key] = next_index
    return items[next_index]


def _music(sentiment: str, themes: set[str], state: dict) -> list[dict]:
    key = "tired" if "tired" in themes else "focus" if "study" in themes or "work" in themes else sentiment
    if key not in RESOURCE_CATALOG:
        key = sentiment
    item = _choose(RESOURCE_CATALOG[key]["music"], state, "music_index")
    return [{
        "title": item[0],
        "artist": "Spotify search recommendation",
        "mood": key,
        "spotify_url": f"https://open.spotify.com/search/{quote(item[1])}",
        "reason": item[2],
    }]


def get_supportive_content(sentiment: str, text: str, safety: bool = False, resource_state: dict | None = None) -> dict:
    """Build supportive content and rotate resources within the current session."""
    state = resource_state if resource_state is not None else {}
    themes = _themes(text)
    category = RESOURCE_CATALOG.get(sentiment, RESOURCE_CATALOG["neutral"])
    if safety:
        return {
            "acknowledgement": "Thanks for telling me something this serious.",
            "message": "Please contact someone you trust and local emergency or crisis support if you may be in immediate danger. MindEasy is not a crisis service.",
            "follow_up": "Can you move closer to another person or call someone you trust right now?",
            "quote": "You deserve immediate, human support and care.",
            "exercise": {"title": "Stay connected", "instructions": "Move toward another person or make a call while you seek immediate human support."},
            "breathing": True,
            "music": _music("negative", themes, state),
            "safety": True,
            "resource_state": state,
        }

    workload = "work" in themes or "study" in themes or "stress" in themes
    lonely = "loneliness" in themes
    positive_theme = "celebration" in themes or "motivation" in themes
    if workload:
        acknowledgement = "That sounds like a really packed day."
        message = "When tasks keep competing for your attention, it can feel exhausting. You do not have to solve everything at once."
        follow_up = "What is the biggest thing on your mind right now?"
    elif lonely:
        acknowledgement = "It sounds like you have been carrying a lonely feeling."
        message = "That can make an ordinary day feel much heavier. A small point of connection may be worth considering."
        follow_up = "Would you like to talk about when that feeling has been strongest?"
    elif sentiment == "positive" or positive_theme:
        acknowledgement = "There is a warm, positive note in what you shared."
        message = "Thanks for sharing that moment. It sounds like something meaningful or energizing found its way into your day."
        follow_up = "What part of today would you like to remember?"
    elif sentiment == "negative":
        acknowledgement = "It sounds like today has been a lot to handle."
        message = "That sounds tiring. You do not have to figure everything out in one moment; a small pause can be enough for now."
        follow_up = "Would it help to talk through what feels heaviest, or take a short break first?"
    else:
        acknowledgement = "Thanks for sharing that check-in."
        message = "Your words have a fairly matter-of-fact tone. There is room to keep exploring if something else is on your mind."
        follow_up = "Is there anything about today you would like to unpack?"

    quote = _choose(category["quotes"], state, "quote_index")
    exercise = _choose(category["exercises"], state, "exercise_index")
    return {
        "acknowledgement": acknowledgement,
        "message": message,
        "follow_up": follow_up,
        "quote": quote,
        "exercise": {"title": exercise[0], "instructions": exercise[1]},
        "breathing": True,
        "music": _music(sentiment, themes, state),
        "safety": False,
        "resource_state": state,
    }
