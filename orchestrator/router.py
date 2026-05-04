# /orchestrator/router.py

from typing import Dict, Any
import json
import re


class Router:
    """
    Clasificador inteligente que determina el destino de la consulta.
    Utiliza un LLM para clasificar la intención y segmentar la consulta.
    """

    def __init__(self, llm):
        self.llm = llm
        self.categories = ["RRH", "IT", "FINANCE", "FACILITIES", "GENERAL"]

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Analiza la consulta y devuelve una clasificación y segmentación.
        """
        print(f"Router analizando consulta: '{query}'")

        system_prompt = f"""
Eres un clasificador de consultas experto para un sistema de soporte interno.
Tu tarea es analizar la siguiente consulta y determinar las categorías relevantes entre: {", ".join(self.categories)}.
Si hay múltiples temas, debes segmentarlos claramente separando cada parte de la pregunta con su categoría correspondiente.

Formato de salida deseado: Un objeto JSON EXACTAMENTE con esta estructura (sin texto adicional):
{{
  "categories": ["RRH", "IT"],
  "segments": [
    {{"topic": "RRH", "details": "Consulta sobre el monto del sueldo"}},
    {{"topic": "IT", "details": "Consulta sobre la laptop lenta"}}
  ]
}}

Si solo hay un tema, incluye solo esa categoría. Nunca uses "GENERAL" a menos que no puedas clasificarla en ninguna otra categoría.

Consulta: "{query}"
"""

        response = self.llm.invoke(system_prompt)

        parsed = _parse_json_response(response.content)
        return parsed


def _parse_json_response(content: str) -> Dict[str, Any]:
    """
    Intenta parsear un JSON del LLM con múltiples estrategias de fallback.
    """
    # Estrategia 1: Parseo directo
    try:
        result = json.loads(content.strip())
        if "categories" in result and isinstance(result["categories"], list):
            return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Estrategia 2: Buscar JSON dentro del texto (entre llaves o bloques ```json)
    json_match = re.search(r'\{[^{}]*"categories"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return _validate_result(result)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Estrategia 3: Extraer categorías con regex si el JSON está roto pero tiene la estructura visible
    cats_found = []
    for cat in ["RRH", "IT", "FINANCE", "FACILITIES"]:
        if re.search(rf'["\']?{cat}["\']?\s*[:,]', content) or re.search(
            rf'"{cat}"', content
        ):
            cats_found.append(cat)

    # Estrategia 4: Fallback total
    if not cats_found:
        print(
            "Advertencia: No se pudo extraer clasificación del LLM. Usando GENERAL."
        )
        return {
            "categories": ["GENERAL"],
            "segments": [{"topic": "General", "details": content}],
        }

    return {
        "categories": cats_found,
        "segments": [
            {"topic": cat, "details": f"Consulta clasificada como {cat}"}
            for cat in cats_found
        ],
    }


def _validate_result(result: Dict) -> Dict[str, Any]:
    """Asegura que el resultado tenga la estructura correcta."""
    if not isinstance(result.get("categories"), list):
        result["categories"] = ["GENERAL"]
    if "segments" not in result or not isinstance(result.get("segments"), list):
        result["segments"] = []
    return result
