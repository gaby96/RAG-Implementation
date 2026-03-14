from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from app.core.embedding_settings import EmbeddingSettings
from app.models.enums import EmbeddingType, EmbeddingStoreType
from app.integrations.qdrant.tf_sparse import TfSparseVectorizer
from app.services.text_tokenizer import TextTokenizer
from app.plugins.rag_plugin import RagPlugin
from app.cli.rag_tools import rag_tool_schemas, dispatch_rag_tool
from app.services.rag_service import RagService
from app.integrations.hybrid_idf_embedding_store import QdrantHybridIdfEmbeddingStore
from app.integrations.azure_openai import AzureOpenAIEmbeddingGenerator
from app.core.document_extraction import DocumentExtractor
from app.services.cross_encoder_reranker import CrossEncoderReranker


SYSTEM_PROMPT = """\
You are an AI assistant that answers questions using indexed documents.

IMPORTANT: You MUST use the available tools to answer questions. Do not make up information.

Available tools:
- search_documents: ALWAYS call this first when the user asks a question. Pass the user's question as the query.
- index_pdf: Call when user wants to index/add a PDF file.
- delete_document: Call when user wants to remove a document.
- clear_index: Call when user wants to clear all documents.

After receiving search results, summarize the relevant information and cite sources.
If no results are found, tell the user no relevant documents were found.
"""


@dataclass(slots=True)
class AiAssistantCli:
    client: AsyncOpenAI
    model: str
    rag_plugin: RagPlugin
    temperature: float = 0.2

    async def run(self) -> None:
        print("=== AI RAG Assistant (FastAPI-style tools) ===\n")

        messages: list[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        print("Assistant initialized! You can:")
        print("1. Index PDFs in the Data directory: Type 'index <pdf-name>' to add documents")
        print("2. Ask questions: The assistant will search indexed documents")
        print("3. Type 'exit' to quit\n")

        while True:
            user_input = await self._ainput("You: ")
            if not user_input.strip():
                continue

            if user_input.strip().lower() == "exit":
                print("Goodbye!")
                return

            # Convenience: CLI command -> tool call directly (same UX as your C#)
            if user_input.strip().lower().startswith("index "):
                pdf_name = user_input.strip()[len("index ") :].strip()
                msg = await self.rag_plugin.index_pdf(pdf_file_name=pdf_name)
                print(f"Assistant: {msg}\n")
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": msg})
                continue

            # Normal chat message
            messages.append({"role": "user", "content": user_input})

            try:
                answer = await self._chat_with_tools(messages)
                print("\n")  # spacing
                messages.append({"role": "assistant", "content": answer})
            except Exception as ex:
                print(f"\nError: {ex}\n")

    async def _chat_with_tools(self, messages: list[Dict[str, Any]]) -> str:
        """
        Runs one assistant "turn" with tool calling:
        1) Ask model with tools enabled
        2) If it requests tool calls, execute them and append tool outputs
        3) Ask model again (optionally streaming) for final response
        """

        tools = rag_tool_schemas()

        # First call: let the model decide tools (auto)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.temperature,
        )

        msg = resp.choices[0].message

        # If no tool calls, just return content (may be empty if model did nothing)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            content = (msg.content or "").strip()
            print(f"Assistant: {content}", end="", flush=True)
            return content

        # Add the assistant tool-call message to history
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
        )

        # Execute all tool calls
        for tc in tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            tool_output = await dispatch_rag_tool(self.rag_plugin, tool_name, args)

            # Append tool result (role=tool) with tool_call_id
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                }
            )

        # Second call: ask model to produce final answer using tool outputs
        print("Assistant: ", end="", flush=True)

        final = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )

        full = []
        async for event in final:
            delta = event.choices[0].delta
            chunk = getattr(delta, "content", None)
            if chunk:
                print(chunk, end="", flush=True)
                full.append(chunk)

        return "".join(full).strip()

    async def _ainput(self, prompt: str) -> str:
        # Non-blocking input so it plays nicely with asyncio
        return await asyncio.to_thread(input, prompt)


# -------------------------
# Entry point
# -------------------------

async def main() -> None:
    # ---- OpenAI Client ----
    client = AsyncOpenAI(
        api_key=""
    )

    model = "gpt-4o-mini"   # or your Azure deployment name

    # ---- Create RAG components ----

    qdrant_client = AsyncQdrantClient(url="http://localhost:6333")
    tf_vectorizer = TfSparseVectorizer(tokenizer=TextTokenizer())
    embedding_store = QdrantHybridIdfEmbeddingStore(client=qdrant_client, collection_name="documents", dense_vector_size=1536, tf_vectorizer=tf_vectorizer)  # client will be set internally

    await embedding_store.ensure_collection()

    settings = EmbeddingSettings(
        endpoint="",
        api_key="",
        deployment_name="text-embedding-3-small",
        type=EmbeddingType.azure_openai,
        max_chunk_tokens=512,
        chunk_overlap_tokens=50,
        vector_size=1536,
        store_type=EmbeddingStoreType.qdrant
    )
    embedding_generator = AzureOpenAIEmbeddingGenerator(settings=settings)

    document_extractor = DocumentExtractor(max_chunk_tokens=512, chunk_overlap_tokens=50)

    print("Initializing embedding store (Qdrant collection)...")

    reranker = CrossEncoderReranker(endpoint="http://localhost:8001/")  # your reranker service URL

    rag_service = RagService(
        embedding_store=embedding_store,
        embedding_generator=embedding_generator,
        reranker=None,
        document_extractor=document_extractor,
        adjacent_chunk_count=1
    )

    rag_plugin = RagPlugin(rag_service)

    assistant = AiAssistantCli(
        client=client,
        model=model,
        rag_plugin=rag_plugin,
        temperature=0.2
    )

    await assistant.run()

if __name__ == "__main__":
    asyncio.run(main())