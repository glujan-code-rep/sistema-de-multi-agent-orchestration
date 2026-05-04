# /utils/content_moderator.py

import re
from typing import Tuple, List, Optional


class ContentModerator:
    """
    Sistema de moderación de contenido para filtrar prompts ofensivos o dañinos.
    Usa una combinación de patrones de palabras clave y reglas heurísticas.
    """

    # Categorías de contenido sensible
    BLOCKED_PATTERNS = {
        "violence": [
            r"\b(matar|asesinar|muerte|morir)\s+(a|al|por)\s+",
            r"\b(atacar|golpear|herir|violencia)\s+(a|contra)\s+",
            r"\b(bomba|arma de fuego|disparar|apunalam)\b",
            r"\b(suicidio|amenazar\s+de\s+muerte)\b",
        ],
        "discrimination": [
            r"\b(racismo|racista|discriminación|sexista|homofob)\b",
            r"\b(xenofob|antisemita|nazi|supremac)\b",
            r"\b(machista|feminazi|discriminar)\s+(por|a)\s+",
        ],
        "sexual": [
            r"\b(pornograf|sexo explícito|violación sexual)\b",
            r"\b(incest|pedofil|menor de edad)\b",
        ],
        "illegal": [
            r"\b(cocaína|heroína|marihuana| наркотик)\b",
            r"\b(hackear\s+(el|la)|robar\s+(dinero|información))\b",
            r"\b(estafa|phishing|malware|crear virus)\b",
            r"\b(pirat|soborno|corrupción)\b",
        ],
        "self_harm": [
            r"\b(autolesion|suicid|cortarse|ahorcarse)\b",
        ],
    }

    # Palabras que requieren atención (no bloquean pero registran)
    WARNING_PATTERNS = [
        r"\b(problema|queja|denuncia|demanda)\b",
        r"\b(empleo|despido|abuso laboral|acoso)\b",
        r"\b(legal|abogado|demanda|juicio)\b",
    ]

    # Respuestas safe para cada categoría
    SAFE_RESPONSES = {
        "violence": "No puedo ayudarte con contenido relacionado con violencia. Si tienes una emergencia, por favor contacta a las autoridades locales.",
        "discrimination": "No puedo procesar contenido discriminatorio. Nuestra empresa promueve un ambiente de trabajo inclusivo y respetuoso.",
        "sexual": "No puedo ayudarte con contenido de naturaleza sexual explícita.",
        "illegal": "No puedo ayudarte con actividades ilegales. Si necesitas información legal, consulta con un profesional adecuado.",
        "personal_data": "Por seguridad, no puedo procesar solicitudes que incluyan datos personales sensibles. Por favor, contacta al departamento de IT.",
        "self_harm": "Si estás experimentando pensamientos de hacerte daño, por favor contacta a un profesional de salud mental o a líneas de emergencia.",
    }

    def __init__(self, strict_mode: bool = False):
        """
        Inicializa el moderador de contenido.

        Args:
            strict_mode: Si True, bloquea cualquier contenido que coincida con los patrones.
        """
        self.strict_mode = strict_mode
        self._compile_patterns()

    def _compile_patterns(self):
        """Compila los patrones regex para mejor rendimiento."""
        self._blockedCompiled = {}
        for category, patterns in self.BLOCKED_PATTERNS.items():
            self._blockedCompiled[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._warningCompiled = [
            re.compile(p, re.IGNORECASE) for p in self.WARNING_PATTERNS
        ]

    def check_content(self, text: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        Verifica el contenido y retorna el resultado de la moderación.

        Args:
            text: El texto a verificar.

        Returns:
            Tuple de (es_seguro, respuesta_segura, advertencias)
        """
        if not text or not text.strip():
            return True, None, []

        text_lower = text.lower()

        # Verificar contenido bloqueado
        detected_categories = self._check_blocked_categories(text_lower)

        if detected_categories:
            if self.strict_mode:
                # En modo estricto, devolver la respuesta segura
                return False, self.SAFE_RESPONSES[detected_categories[0]], []
            else:
                # En modo no estricto, solo registrar las advertencias
                return True, None, detected_categories

        # Verificar advertencias
        warnings = self._check_warnings(text_lower)

        return True, None, warnings

    def _check_blocked_categories(self, text: str) -> List[str]:
        """Verifica si el texto contiene contenido bloqueado."""
        detected = []

        for category, patterns in self._blockedCompiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected.append(category)
                    break  # Solo registrar una vez por categoría

        return detected

    def _check_warnings(self, text: str) -> List[str]:
        """Verifica si el texto contiene palabras de advertencia."""
        warnings = []

        for pattern in self._warningCompiled:
            match = pattern.search(text)
            if match:
                warnings.append(match.group())

        return warnings

    def is_safe(self, text: str) -> bool:
        """
        Verifica rápidamente si el contenido es seguro.

        Args:
            text: El texto a verificar.

        Returns:
            True si el contenido es seguro, False si no lo es.
        """
        safe, _, _ = self.check_content(text)
        return safe

    def get_safe_response(self, text: str) -> Optional[str]:
        """
        Retorna una respuesta segura si el contenido no es apropiado.

        Args:
            text: El texto a verificar.

        Returns:
            Respuesta segura o None si el contenido es apropiado.
        """
        safe, response, _ = self.check_content(text)
        return response if not safe else None

    def add_custom_pattern(self, category: str, pattern: str, response: str = None):
        """
        Añade un patrón personalizado de bloqueo.

        Args:
            category: Categoría del patrón.
            pattern: Pattern regex a añadir.
            response: Respuesta segura opcional para esta categoría.
        """
        if category not in self.BLOCKED_PATTERNS:
            self.BLOCKED_PATTERNS[category] = []
            self.SAFE_RESPONSES[category] = (
                response or "Contenido no apropiado para procesamiento."
            )

        self.BLOCKED_PATTERNS[category].append(pattern)

        # Recompilar patrones
        self._compile_patterns()


def moderate_input(text: str, strict: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Función de conveniencia para moderar contenido.

    Args:
        text: Texto a moderar.
        strict: Si True, bloquea contenido problemático.

    Returns:
        Tuple de (es_seguro, respuesta_segura)
    """
    moderator = ContentModerator(strict_mode=strict)
    is_safe, response, _ = moderator.check_content(text)
    return is_safe, response


if __name__ == "__main__":
    # Pruebas
    moderator = ContentModerator()

    test_cases = [
        "Quiero información sobre trabajo remoto y mi laptop está lenta",
        "Cómo puedo matar a mi vecino",
        "Tengo una queja sobre mi salario",
        "Necesito información sobre discriminación laboral",
        "Cómo puedo hackear el sistema de la empresa",
    ]

    print("=== PRUEBAS DEL MODERADOR DE CONTENIDO ===\n")

    for text in test_cases:
        safe, response, warnings = moderator.check_content(text)
        print(f"Texto: {text}")
        print(f"  Seguro: {safe}")
        if response:
            print(f"  Respuesta: {response}")
        if warnings:
            print(f"  Advertencias: {warnings}")
        print()
