from typing import Any, Iterable, Optional
from openai import OpenAI


class LLMClient:
    """
    Encapsula el cliente OpenAI-compatible detrás de una interfaz simple.
    Permite cambiar de proveedor o de SDK sin tocar la UI ni la lógica de negocio.
    """

    def __init__(self, api_key: str, base_url: str, timeout: int = 3600) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat_completions_create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ):
        """Wrapper directo sobre chat.completions.create del SDK."""
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
        )