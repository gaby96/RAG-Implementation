from __future__ import annotations

from typing import Any, Dict

from app.plugins.rag_plugin import RagPlugin


def rag_tool_schemas() -> list[Dict[str, Any]]:
    """
    OpenAI tool (function-calling) schemas for RagPlugin.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search documents for information. Call this for any user question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "index_pdf",
                "description": "Add a PDF file to the search index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_file_name": {
                            "type": "string",
                            "description": "PDF filename to index",
                        }
                    },
                    "required": ["pdf_file_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_document",
                "description": "Remove a document from the index.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_name": {
                            "type": "string",
                            "description": "Document name to delete",
                        }
                    },
                    "required": ["document_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clear_index",
                "description": "Delete all documents from the index.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


async def dispatch_rag_tool(plugin: RagPlugin, tool_name: str, args: Dict[str, Any]) -> str:
    """
    Executes a tool call from the LLM and returns the tool output as a string.
    """
    if tool_name == "search_documents":
        return await plugin.search_documents(
            query=str(args.get("query", "")),
            top_k=int(args.get("top_k", 5)),
        )

    if tool_name == "index_pdf":
        return await plugin.index_pdf(
            pdf_file_name=str(args.get("pdf_file_name", "")),
        )

    if tool_name == "delete_document":
        return await plugin.delete_document(
            document_name=str(args.get("document_name", "")),
        )

    if tool_name == "clear_index":
        return await plugin.clear_index()

    raise ValueError(f"Unknown tool: {tool_name}")