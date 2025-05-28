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
        self.repo_id = self.config.get("repo_id", "")
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
            if "tool_calls" in response and response["tool_calls"] :
                for call in response["tool_calls"] :
                    tool_call_id.append(call["id"])
                    arguments = call["function"]["arguments"]
                    subquery.append(json.loads(arguments)["subquery"])
                return response['content'], subquery, tool_call_id
            else :
                return response['content'], None, None
        
        reasoning = response.content
        try :
            for call in response.tool_calls :
                arguments = call.function.arguments
                subquery.append(json.loads(arguments)["subquery"])
                tool_call_id.append(call.id)
            return reasoning, subquery, tool_call_id
        except :
            return reasoning, None, None

    def extract_answer(self, response: str) -> str :
        ans = response[response.index("<Answer>") + len("<Answer>"): ] 
        return ans

    def extract_content(self, response: Any) -> str :
        try :
            return response.content
        except :
            return response['content']

    def get_llm_response(self, text_messages: object, tools: object, backbone: str, selct_config: object) :
        if tools :
            response = get_chat_result(messages=text_messages, tools=tools, llm_config=selct_config)   
        else :
            response = get_chat_result(messages=text_messages, tools=None, llm_config=selct_config)   

        return response
                        

    def _run(self, case: dict, backbone: str, tmp: Any = None) :
        """
        Single iteration of TableRAG inference.
        """
        query = case["question"]
        table_id = case["table_id"]

        # TO BE FIXED
        query_with_suffix = case['question'] + f"The given table is in {table_id}"

        _, _, doc_filenaems = self.retriever.retrieve(query_with_suffix, 30, 5)

        top1_table_name = doc_filenaems[0].replace(".json", "").replace(".xlsx", "")
        related_table_name_list = [top1_table_name + "_Sheet1.xlsx"]


        tools = self.create_tools()
        current_iter = self.max_iter
        text_messages = self.construct_initial_prompt(case, top1_table_name)

        logger.info(f"Processing query: {query}")
        select_config = config_mapping[backbone]

        while current_iter :
            current_iter -= 1
            response = self.get_llm_response(text_messages=text_messages, tools=tools, backbone=backbone, select_config=select_config)

            reasoning, sub_queries, tool_call_ids = self.extract_subquery(response, backbone=backbone)
            logger.info(f"Step {self.max_iter - current_iter}: {sub_queries}")

            if not sub_queries and "<Answer>" in reasoning and current_iter != self.max_iter - 1 :
                answer = self.extract_answer(reasoning)
                logger.info(f"Answer: {answer}")
                return answer
            
            if not sub_queries :
                text_messages.append({
                    "role": "user",
                    "content": "ERROR: Did not output a suquery!"
                })
                continue

            messages = response
            text_messages.append(messages)

            for sub_query, tool_call_id in zip(sub_queries, tool_call_ids) :
                reranked_docs, _, _ = self.retriever.retrieve(sub_query, 30, 5)
                unique_retriebed_docs = list(set(reranked_docs))
                doc_content = "\n".join([r for r in unique_retriebed_docs[:3]])

                excel_rag_response_dict = get_excel_rag_response_plain(str(related_table_name_list), sub_query, self.repo_id)
                excel_rag_response = copy.deepcopy(excel_rag_response_dict)
                logger.info(f"Requesting ExcelRAG, source file {str(related_table_name_list)}, with query {sub_query}")

                try :
                    sql_str = excel_rag_response['sql_str']
                    sql_execute_result = excel_rag_response['sql_execute_result']
                    schema  = get_excel_rag_response['nl2sql_prompt'].split('基于上面的schema，请使用MySQL语法解决下面的问题')[0].strip()
                except :
                    sql_str, sql_execute_result, schema = "ExcelRAG execute fails, key does not exists."

                combine_prompt_formatted = COMBINE_PROMPT.format(
                    docs=doc_content, 
                    schema=schema, 
                    nl2sql_model_response=sql_str, 
                    sql_execute_result=sql_execute_result,
                    query=sub_query
                )

                final_prompt = combine_prompt_formatted