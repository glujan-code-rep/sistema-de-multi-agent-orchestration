# Sistema de Soporte Interno Multi-Agente con RAG

Sistema automatizado de soporte interno empresarial usando múltiples agentes especializados orquestados con LangGraph y Recuperación-Augmented Generation (RAG).

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRADA DEL USUARIO                             │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTENT MODERATOR + SECURITY                         │
│              (Content filtering, Prompt injection, Rate limiting)     │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ORQUESTADOR (LangGraph)                           │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐             │
│  │   ROUTER    │───▶│    AGENTES    │───▶│  CONSOLIDADOR  │             │
│  │ (Clasifica) │    │ (Ejecutan)   │    │   (Unifica)    │             │
│  └─────────────┘    └──────────────┘    └────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
          │                         │                       │
          ▼                         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RRH Agent    │    │    IT Agent    │    │  Finance Agent │
│  (Recursos     │    │  (Soporte       │    │   (Finanzas)   │
│   Humanos)     │    │   Técnico)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                         │                       │
          ▼                         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector Store  │    │  Vector Store  │    │  Vector Store  │
│   (ChromaDB)   │    │   (ChromaDB)   │    │   (ChromaDB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LANGFUSE TRACING                               │
│          (Workflow tracing, Quality scores, Debugging)                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Estructura

```
Internal_Support_System/
├── agents/                      # Agentes especializados
│   ├── rrh_agent.py            # Recursos Humanos (RAG + few-shot)
│   ├── it_agent.py             # Soporte Técnico (RAG + few-shot)
│   └── finance_agent.py       # Finanzas (RAG + few-shot)
│
├── orchestrator/               # Orquestación LangGraph
│   ├── router.py               # Clasificador de intención (LLM)
│   └── graph_builder.py        # Grafo: router → delegate → consolidate
│
├── rag_utils/                  # Utilidades RAG
│   ├── retriever.py            # VectorRetriever (ChromaDB)
│   └── embedder.py             # Constructor bases vectoriales
│
├── utils/
│   ├── llm_providers.py        # Multi-provider (LM Studio, OpenAI, Gemini)
│   ├── content_moderator.py    # Filtro de contenido ofensivo
│   ├── security.py             # Prompt injection, SQL/XSS, rate limiting
│   └── tracing.py              # Langfuse integration + ResponseEvaluator
│
├── data/                       # Datos fuente para RAG
│   ├── rrh_policies.md         # Políticas de RRHH
│   ├── it_faqs.txt             # FAQs de Soporte IT
│   └── finance_rules.csv       # Reglas financieras
│
├── knowledge_bases/             # Bases vectoriales persistidas
│   ├── rrh_vector_db/
│   ├── it_vector_db/
│   └── finance_vector_db/
│
├── tests/                      # 22 tests unitarios
│   ├── test_rrh_agent.py
│   └── test_it_agent.py
│
├── main.py                     # Punto de entrada
├── requirements.txt
└── .env                        # Configuración (API keys, provider)
```

## Componentes

### 1. Agentes Especializados (RAG)

| Agente | Dominio | Base de Conocimiento | Few-Shot |
|--------|---------|---------------------|----------|
| RRHAgent | Recursos Humanos | `rrh_policies.md` | [OK] |
| ITAgent | Soporte Técnico | `it_faqs.txt` | [OK] |
| FinanceAgent | Finanzas | `finance_rules.csv` | [OK] |

Cada agente implementa: **Query → Retrieve → Generate → Response**

### 2. Orquestador (LangGraph)

**Router** (`orchestrator/router.py`):
- Clasificación usando LLM
- Categorías: RRH, IT, FINANCE, FACILITIES, GENERAL
- Segmentación multi-tópico

**SupportGraphBuilder** (`orchestrator/graph_builder.py`):
- Nodos: `router` → `delegate` → `consolidate` → END
- Estado tipado con `GraphState`
- Routing condicional por dominio

### 3. Utilidades RAG

**VectorRetriever**:
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
- Vector store: ChromaDB
- Búsqueda: similarity top-k=4

**Embedder**:
- Carga: TextLoader, CSVLoader
- Chunking: RecursiveCharacterTextSplitter (500 chars, 50 overlap)

### 4. Seguridad

| Módulo | Función |
|--------|---------|
| ContentModerator | Filtra contenido ofensivo |
| SecurityManager | Prompt injection, SQL/XSS, rate limiting |

### 5. Multi-Provider LLM

Proveedores soportados (configurable via `LLM_PROVIDER`):
- **LM Studio** (default): Modelo local `gemma-4-26b-a4b-it`
- **OpenAI**: gpt-4, gpt-3.5-turbo
- **Google Gemini**: gemini-pro, gemini-flash

Selección automática de modelo disponible.

### 6. Langfuse Tracing

Workflow tracing completo:
- Clasificación de intención
- Decisiones de routing
- Scores de calidad (relevance, completeness, accuracy)
- Métricas por query

## Decisiones Técnicas

### LangGraph vs Custom Code
**Elección**: LangGraph
**Justificación**: Flujo declarativo con nodos y aristas definidos, estado tipado, recuperación automática ante errores, integración nativa con LangChain.

### ChromaDB vs FAISS/Weaviate
**Elección**: ChromaDB
**Justificación**: Simplicidad de setup, persistencia automática, API nativa LangChain, sin dependencias externas.

### sentence-transformers vs OpenAI Embeddings
**Elección**: sentence-transformers (`all-MiniLM-L6-v2`)
**Justificación**: Ejecuta en CPU (evita conflictos GPU), gratuito, suficiente calidad para texto corto.

### Few-Shot Prompts
**Elección**: Incluir ejemplos en cada agente
**Justificación**: Mejora consistencia de respuestas, guía tono y formato, reduce variaciones no deseadas.

## Configuración

Variables en `.env`:

```bash
# LLM Provider (lmstudio, openai, gemini)
LLM_PROVIDER=lmstudio

# LM Studio (local)
LM_STUDIO_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=gemma-4-26b-a4b-it

# OpenAI (cloud)
OPENAI_API_KEY=sk-...

# Google Gemini (cloud)
GOOGLE_API_KEY=...

# Langfuse (tracing)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

## Ejecución

```bash
cd Internal_Support_System

# Asegurar que LM Studio esté corriendo en http://127.0.0.1:1234

# Ejecutar
python3 main.py
```

## Testing

```bash
pytest tests/ -v
```

**22 tests passing**: Agentes RRH e IT, integración LangGraph, RAG, seguridad.

## Flujo de Ejecución

```
1. Verificar seguridad del contenido (moderator + security)
2. Construir bases vectoriales si no existen
3. Inicializar LLM (detectar provider disponible)
4. Crear agentes especializados con RAG
5. Construir grafo LangGraph
6. Clasificar query → routing condicional
7. Ejecutar agentes relevantes en paralelo
8. Consolidar respuestas
9. Evaluar calidad (ResponseEvaluator)
10. Trazar en Langfuse
```

## Métricas de Calidad

El `ResponseEvaluator` puntúa cada respuesta:
- **relevance**: 0-1 (qué tan pertinente a la consulta)
- **completeness**: 0-1 (cubrimiento de la solicitud)
- **accuracy**: 0-1 (exactitud de la información)

Reporte generado automaticamente con status:
- [OK] >= 0.7: BUENA
- [!] >= 0.5: REGULAR  
- [X] < 0.5: MALA