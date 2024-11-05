import re


def find_flag(text):
    pattern = r"\{\{FLG:.*?\}\}"
    # Find all matches
    matches = re.findall(pattern, text)

    # Print matches
    return matches


def create_message(role, text):
    """Creates a dictionary for a chat message with a role and text."""
    if role not in ['system', 'user', 'assistant', 'tool']:
        raise ValueError("Role must be one of 'system', 'user', or 'assistant'")
    return {"role": role, "content": text}
