# Instrucciones de Uso - Sistema de Soporte Interno Multi-Agente

## Requisitos Previos

### 1. Python 3.12+
```bash
python3 --version
```

### 2. LM Studio (para LLM local)
- Descargar desde: https://lmstudio.ai
- Instalar y ejecutar
- Cargar un modelo de chat (ej: gemma-4-26b-a4b-it)
- El servidor debe estar en: http://127.0.0.1:1234

### 3. Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```bash
# LLM Provider (lmstudio, openai, gemini)
LLM_PROVIDER=lmstudio

# LM Studio
LM_STUDIO_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=gemma-4-26b-a4b-it

# Langfuse (opcional - para tracing)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

## Instalación

```bash
cd Internal_Support_System

# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
python3 main.py
```

## Flujo de Ejecución Automático

1. **Verificación de seguridad**: El contenido de la consulta se filtra por moderación y seguridad
2. **Construcción de bases vectoriales**: Si no existen, se crean automáticamente desde `data/`
3. **Inicialización del LLM**: Detecta el provider configurado y selecciona el mejor modelo disponible
4. **Clasificación de intención**: El Router analiza la consulta y determina categorías (RRH, IT, FINANCE, etc.)
5. **Delegación a agentes**: Se ejecutan los agentes especializados según las categorías detectadas
6. **Consolidación de respuestas**: Las respuestas de múltiples agentes se unifican
7. **Evaluación de calidad**: El ResponseEvaluator puntúa cada respuesta (relevance, completeness, accuracy)
8. **Tracing**: Si Langfuse está configurado, el workflow completo se registra

## Configuración de Agentes

### Agentes Disponibles

| Agente | Dominio | Archivos de Conocimiento |
|--------|---------|-------------------------|
| RRHAgent | Recursos Humanos | `data/rrh_policies.md` |
| ITAgent | Soporte Técnico | `data/it_faqs.txt` |
| FinanceAgent | Finanzas | `data/finance_rules.csv` |

### Agregar Nuevo Agente

1. Crear archivo en `agents/nuevo_agente.py`
2. Heredar de la estructura de agentes existentes
3. Especificar base de conocimiento en `data/`
4. Registrar en `main.py` y `orchestrator/graph_builder.py`

## Configuración de Langfuse (Opcional)

1. Crear cuenta en https://cloud.langfuse.com
2. Crear nuevo proyecto
3. Obtener Public Key y Secret Key
4. Agregar al `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
   ```

En el dashboard de Langfuse verás:
- Trace completo del workflow
- Input/output de cada paso
- Scores de calidad (relevance, completeness, accuracy)

## Pruebas

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Ejecutar tests específicos
pytest tests/test_rrh_agent.py -v
pytest tests/test_it_agent.py -v
```

## Estructura de Archivos

```
Internal_Support_System/
├── agents/              # Agentes especializados
├── orchestrator/        # Orquestación LangGraph
├── rag_utils/          # Utilidades RAG
├── utils/              # Utilities (LLM, security, tracing)
├── data/               # Datos fuente para RAG
├── knowledge_bases/    # Bases vectoriales persistidas
├── tests/              # Pruebas unitarias
├── main.py             # Punto de entrada
├── requirements.txt    # Dependencias
└── .env                # Configuración
```

## Solución de Problemas

### LM Studio no conecta
- Verificar que LM Studio esté ejecutándose
- Confirmar que el modelo esté cargado
- Revisar que la URL sea http://127.0.0.1:1234/v1

### Errores de embedding
- Verificar que `sentence-transformers` esté instalado
- La primera ejecución puede tardar en descargar el modelo de embeddings

### Langfuse no muestra traces
- Verificar que las API keys sean correctas
- Confirmar que el proyecto esté activo en Langfuse

### Respuestas de baja calidad
- Revisar el reporte de calidad en la salida
- Los scores menores a 0.5 indican problemas
- Verificar que las bases de conocimiento tengan información relevante

## Métricas de Calidad

El sistema evalúa cada respuesta con:
- **Relevance**: Qué tan pertinente es a la consulta (0-1)
- **Completeness**: Cubrimiento completo de la solicitud (0-1)
- **Accuracy**: Exactitud de la información (0-1)
- **Overall**: Promedio de las tres métricas

Status:
- >= 0.7: [OK] BUENA
- >= 0.5: [!] REGULAR
- < 0.5: [X] MALA

## Personalización

### Agregar más documentos RAG
1. Agregar archivos a `data/`
2. Ejecutar `python3 -c "from rag_utils.embedder import build_all; build_all()"`
3. Los nuevos documentos se indexarán automáticamente

### Cambiar modelo LLM
Editar `LM_STUDIO_MODEL` en `.env` o esperar a que el sistema seleccione automáticamente.

### Agregar más categorías de clasificación
Editar `self.categories` en `orchestrator/router.py:16`