"""
Main Entry of TableRAG.
"""
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from chat_utils import *
from tools.retriever import *
from tools.sql_tool import *
from utils.config import *
from utils.utils import read_in, read_in_lines
from typing import Dict, Tuple, Any
import threading
import traceback
import copy
import time
from prompt import *


MAX_ITER = 5
ASSISTANT = "assistant"
FUNCTION = "function"


class TableRAG() :
    """
    Agent of TableRAG.
    """
    def __init__(self, _args: Any) -> None:
        self.config = _args
        self.max_iter = min(_args.max_iter, MAX_ITER)
        self.cnt = 0
        self.retriever = MixedDocRetriever(
            doc_dir_path="",
            excel_dir_path="",
            llm_path="",
            reranker_path="",
            save_path=""
        )

        self.function_lock = threading.Lock()

    def relate_to_table(self, doc_name: str) -> str :
        """
        Find the excel file according to json file.
        """
        if "json" in doc_name :
            table_file_name = doc_name.replace("json", "xlsx")
        if os.path.exists() :
            run_name = doc_name.replace(".json", "_sheet1.xlsx")
            return f"[\"{run_name}\"]"
        return ""

    def create_tools(self) :
        tools = [{
            "type": "function",
            "function": {
                "name": "solve_subquery",
                "description": "Return answer for the decomposed subquery."
                "paramters": {
                    "type": "object",
                    "properties": {
                        "subquery": {
                            "type": "string",
                            "description": "The subquery to be solved"
                        }
                    },
                    "required": [
                        "subquery"
                    ],
                    "additionalProperties": False
                }
            }
        }]
        return tools

    def extract_subquery(self, response: Any, backbone: str = 'openai') -> Tuple[str, str] :
        """
        Extract the subquery and reasoning process.
        """
        subquery, tool_call_id = [], []
        if isinstance(response, dict) :
            if "tool_calls" 