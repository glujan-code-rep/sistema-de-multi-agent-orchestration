# /utils/llm_providers.py

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Clase base abstracta para proveedores de LLM."""

    @abstractmethod
    def get_client(self):
        """Retorna el cliente del LLM."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Retorna el nombre del modelo."""
        pass


class LMStudioProvider(LLMProvider):
    """Proveedor para LM Studio (local)."""

    def __init__(
        self,
        model: str = "gemma-4-26b-a4b-it",
        base_url: str = "http://127.0.0.1:1234/v1",
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def get_client(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                request_timeout=self.timeout,
                max_retries=2,
            )
        return self._client

    def get_model_name(self) -> str:
        return f"lmstudio:{self.model}"


class OpenAIProvider(LLMProvider):
    """Proveedor para OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        timeout: float = 60.0,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def get_client(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY no está configurada")

            self._client = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                temperature=self.temperature,
                request_timeout=self.timeout,
            )
        return self._client

    def get_model_name(self) -> str:
        return f"openai:{self.model}"


class GeminiProvider(LLMProvider):
    """Proveedor para Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        timeout: float = 60.0,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def get_client(self):
        if self._client is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "Instala langchain-google-genai: pip install langchain-google-genai"
                )

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY no está configurada")

            self._client = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=api_key,
                temperature=self.temperature,
                timeout=self.timeout,
            )
        return self._client

    def get_model_name(self) -> str:
        return f"gemini:{self.model}"


class ProviderFactory:
    """Fábrica para crear proveedores de LLM."""

    PROVIDERS = {
        "lmstudio": LMStudioProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "local": LMStudioProvider,  # Alias para lmstudio
    }

    @classmethod
    def create(cls, provider: str = "lmstudio", **kwargs) -> LLMProvider:
        """
        Crea un proveedor de LLM.

        Args:
            provider: Nombre del proveedor (lmstudio, openai, gemini)
            **kwargs: Parámetros específicos del proveedor

        Returns:
            Instancia del proveedor de LLM

        Raises:
            ValueError: Si el proveedor no es válido
        """
        provider = provider.lower()

        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Proveedor '{provider}' no válido. "
                f"Disponibles: {', '.join(cls.PROVIDERS.keys())}"
            )

        # Valores por defecto para cada proveedor
        defaults = {
            "lmstudio": {
                "model": os.getenv("LM_STUDIO_MODEL", "gemma-4-26b-a4b-it"),
                "base_url": os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1"),
            },
            "openai": {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            },
            "gemini": {
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            },
        }

        # Combinar valores por defecto con los proporcionados
        config = defaults.get(provider, {})
        config.update(kwargs)

        return cls.PROVIDERS[provider](**config)

    @classmethod
    def get_available_providers(cls) -> list:
        """Retorna lista de proveedores disponibles."""
        return list(cls.PROVIDERS.keys())


def get_llm_provider(provider: str = None, **kwargs) -> LLMProvider:
    """
    Función de conveniencia para obtener un proveedor de LLM.

    Si no se especifica provider, usa el valor de LLM_PROVIDER del entorno,
    o 'lmstudio' por defecto.

    Args:
        provider: Nombre del proveedor (lmstudio, openai, gemini)
        **kwargs: Parámetros adicionales

    Returns:
        Instancia del proveedor de LLM
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "lmstudio")

    return ProviderFactory.create(provider, **kwargs)


# Verificación automática de proveedores disponibles
def check_provider_available(provider: str) -> bool:
    """
    Verifica si un proveedor está disponible/configurado.

    Args:
        provider: Nombre del proveedor

    Returns:
        True si el proveedor está disponible
    """
    provider = provider.lower()

    if provider == "lmstudio":
        import urllib.request

        try:
            req = urllib.request.Request("http://127.0.0.1:1234/v1/models")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    elif provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))

    elif provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY"))

    return False


if __name__ == "__main__":
    print("=== PROVEEDORES DE LLM DISPONIBLES ===\n")

    # Listar proveedores
    print("Proveedores configurados:", ProviderFactory.get_available_providers())

    # Probar cada proveedor
    for provider_name in ["lmstudio", "openai", "gemini"]:
        available = check_provider_available(provider_name)
        status = "[OK] Disponible" if available else "[X] No disponible"
        print(f"{provider_name}: {status}")

    print("\n--- Uso desde código ---")
    print("""
# Usar LM Studio (local, defecto)
llm = get_llm_provider()

# Usar OpenAI
llm = get_llm_provider("openai")

# Usar Gemini
llm = get_llm_provider("gemini")

# Con variables de entorno:
# LLM_PROVIDER=openai python main.py
""")
