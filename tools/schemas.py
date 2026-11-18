"""OpenAI-compatible function schemas for the tools provided to Nemotron."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Calculate mathematical expressions. "
                "Use this for accurate arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": (
                "Get the current date and time."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_counter",
            "description": (
                "Count words and characters in text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyse"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

