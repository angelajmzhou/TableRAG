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
from config import *
from utils.utils import read_in, read_in_lines
from utils.tool_utils import excel_to_markdown
from typing import Dict, Tuple, Any
import threading
import traceback
import copy
import time
from prompt import *
import pandas as pd
from typing import Any, Tuple, List, Optional
from google.generativeai.types import FunctionDeclaration
import re


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
            doc_dir_path=_args.doc_dir,
            excel_dir_path=_args.excel_dir,
            llm_path=os.path.join(_args.bge_dir, "bge-m3"),
            reranker_path=os.path.join(_args.bge_dir, "bge-reranker-v2-m3"),
            save_path="./embedding.pkl"
        )
        self.function_lock = threading.Lock()


    from google.generativeai.types import FunctionDeclaration

    def create_gemini_tools(self):
        return [
            FunctionDeclaration(
                name="solve_subquery",
                description="Return answer for the decomposed subquery.",
                parameters={
                    "type": "object",
                    "properties": {
                        "subquery": {
                            "type": "string",
                            "description": "The subquery to be solved, only take natural language as input."
                        }
                    },
                    "required": ["subquery"]
                }
            )
        ]

    def create_v3_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "solve_subquery",
                "description": "Return answer for the decomposed subquery.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subquery": {
                            "type": "string",
                            "description": "The subquery to be solved, only take natural language as input."
                        }
                    },
                    "required": [
                        "subquery"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

    def extract_subquery(self, response: Any, backbone: str = 'gemini') -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
        """
        Extract the subquery and reasoning process from Gemini or OpenAI-style responses.
        """
        reasoning = ""
        subquery = None
        tool_call_id = []
        print(response)
        if backbone == "gemini":
            try:
                if not response._result.candidates:
                    return "No valid response from model.", None, None

                parts = response._result.candidates[0].content.parts

                for part in parts:
                    if hasattr(part, "text"):
                        reasoning = part.text
                    if hasattr(part, "function_call") and part.function_call is not None:
                        try:
                            subquery = part.function_call.args["subquery"]
                        except Exception as e:
                            print(f"Error reading function args: {e}")

                return reasoning.strip(), [subquery] if subquery else None, tool_call_id if tool_call_id else None

            except Exception as e:
                print(f"Error extracting Gemini tool call: {e}")
                return reasoning.strip(), None, None

        elif backbone == "v3":
            try:
                # Step 1: get raw content (from ChatCompletionMessage or str)
                if hasattr(response, "content"):
                    raw = response.content
                elif isinstance(response, str):
                    raw = response
                else:
                    return "", None, None

                # Step 2: extract JSON from code block
                match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
                if not match:
                    print("No JSON block found in response.")
                    return "", None, None

                parsed = json.loads(match.group(1))

                reasoning = parsed.get("content", "")
                tool_calls = parsed.get("tool_calls", [])

                if tool_calls:
                    subquery = tool_calls[0]["function"]["arguments"].get("subquery")
                    return reasoning, [subquery], [tool_calls[0].get("id", "tool_call_0")]

                return reasoning, None, None

            except Exception as e:
                print(f"[extract_subquery] failed: {e}")
                return "", None, None

        else:
            print(f"Unknown backbone '{backbone}' in extract_subquery")
            return "", None, None

    def extract_answer(self, response: str) -> str :
        ans = response[response.index("<Answer>") + len("<Answer>"): ] 
        return ans

    def extract_content(self, response):
        """
        Extracts plain text from a Vertex Gemini GenerateContentResponse.
        """
        if isinstance(response, str):
            return response

        elif isinstance(response, dict):
            return response.get("content", "")

        elif hasattr(response, "candidates"):
            try:
                return response.candidates[0].content.parts[0].text
            except Exception as e:
                print(f"[extract_content] Failed to extract from Gemini response: {e}")
                return ""
        else:
            try:  
                return response.content
            except:
                try:       
                    return response['content']
                except:
                    raise ValueError(f"Unexpected response format: {type(response)} → {response}")



    def get_llm_response(self, text_messages: object, tools: object, backbone: str, select_config: object) :
        if tools :
            response = get_chat_result(messages=text_messages, tools=tools, llm_config=select_config)   
        else :
            response = get_chat_result(messages=text_messages, tools=None, llm_config=select_config)   

        return response
                        


    def construct_initial_prompt(self, query: str, top1_table_name: str) -> Any :

        table_id = top1_table_name + ".xlsx"
        print("top table: ", table_id)
        csv_file_path = os.path.join(self.config.excel_dir, table_id)
        if os.path.exists(csv_file_path) :
            df = pd.read_excel(csv_file_path)
            df = df.fillna("") 
            markdown_text = df.to_markdown(index=False)
        else:
            print("table does not exist.")
        if(self.config.backbone == "gemini"):
            inital_prompt = SYSTEM_PROMPT.format(name=top1_table_name, query=query, table_content=markdown_text)
        else:
            inital_prompt = SYSTEM_PROMPT_V3.format(name=top1_table_name, query=query, table_content=markdown_text)
        logger.info(f"Inital prompt: {inital_prompt}")

        intial_msg = [{"role": "user", "content": inital_prompt}]
        return intial_msg
    
    def _run(self, query_or_case: Union[str, dict], backbone: str, tmp: Any = None):
        """
        Single iteration of TableRAG inference.
        Accepts either a raw string query or a preset of cases.
        """
        if isinstance(query_or_case, dict):
            query = query_or_case["question"]
            table_id = query_or_case["table_id"]
            suffix = f" The given table is in {table_id}"
            query_with_suffix = query + suffix
        # pure-string path
        else:
            query = query_or_case
            query_with_suffix = query

        _, _, doc_filenames = self.retriever.retrieve(query_with_suffix, 30, 5)
        top1_table_name = doc_filenames[0].replace(".json", "").replace(".xlsx", "")
        related_table_name_list = [top1_table_name]
        if backbone == "v3":
            tools = self.create_v3_tools()
        elif backbone == "gemini":
            tools = self.create_gemini_tools()
        current_iter = self.max_iter
        text_messages = self.construct_initial_prompt(query, top1_table_name)
        logger.info(f"Processing query: {query}")
        select_config = config_mapping[backbone]

        while current_iter:
            current_iter -= 1
            response = self.get_llm_response(
                text_messages=text_messages,
                tools=tools,
                backbone=backbone,
                select_config=select_config
            )

            reasoning, sub_queries, tool_call_ids = self.extract_subquery(response, backbone=backbone)
            logger.info(f"Step {self.max_iter - current_iter}: {sub_queries}")

            if not sub_queries and "<Answer>" in reasoning and current_iter != self.max_iter - 1:
                answer = self.extract_answer(reasoning)
                logger.info(f"Answer: {answer}")
                return answer, text_messages

            if not sub_queries:
                text_messages.append({
                    "role": "user",
                    "content": "ERROR: Did not call tool with a query!"
                })
                continue

            # ✅ Append model message properly based on backend
            if backbone == "gemini":
                gemini_parts = response._result.candidates[0].content.parts
                flattened_content = []
                for part in gemini_parts:
                    if hasattr(part, "text"):
                        flattened_content.append(part.text)
                    elif hasattr(part, "function_call"):
                        flattened_content.append({
                            "function_call": {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args)
                            }
                        })
                text_messages.append({
                    "role": "model",
                    "content": [x for x in flattened_content if x]  # remove empty strings
                })
            else:
                # Handle response.content for v3/deepseek/etc
                if hasattr(response, "content"):
                    model_response_content = response.content
                elif isinstance(response, str):
                    model_response_content = response
                else:
                    model_response_content = str(response)

                text_messages.append({
                    "role": "model",
                    "content": [model_response_content]
                })

            for sub_query in sub_queries:
                reranked_docs, _, _ = self.retriever.retrieve(sub_query, 30, 5)
                unique_retriebed_docs = list(set(reranked_docs))
                doc_content = "\n".join(unique_retriebed_docs[:3])

                excel_rag_response_dict = get_excel_rag_response_plain(related_table_name_list, sub_query)
                excel_rag_response = copy.deepcopy(excel_rag_response_dict)
                logger.info(f"Requesting ExcelRAG, source file {related_table_name_list}, with query {sub_query}")

                try:
                    sql_str = excel_rag_response['sql_str']
                    sql_execute_result = excel_rag_response['sql_execution_result']
                    schema = excel_rag_response['nl2sql_prompt'].split(
                        'Based on the schemas above, please use MySQL syntax to solve the following problem'
                    )[0].strip()
                except:
                    sql_str, sql_execute_result, schema = "", "", "ExcelRAG execute fails, key does not exist."

                combine_prompt_formatted = COMBINE_PROMPT.format(
                    docs=doc_content,
                    schema=schema,
                    nl2sql_model_response=sql_str,
                    sql_execute_result=sql_execute_result,
                    query=sub_query
                )

                msg = [{"role": "user", "content": combine_prompt_formatted}]
                answer = self.get_llm_response(text_messages=msg, backbone=backbone, select_config=select_config, tools=None)
                answer = self.extract_content(answer) or ""

                logger.info(f"LLM Subquery Answer: {answer}")
                text_messages.append({
                    "role": "user",
                    "content": "Subquery Answer: " + answer
                })

        return None, text_messages
    def run(
    self,
    file_path: Optional[str],
    save_file_path: str,
    backbone: str,
    rerun: bool = False,
    max_workers: int = 1,
) -> None:
        live_mode = file_path is None or not file_path or file_path == ""
        if rerun and (not live_mode):
            pre_data = read_in_lines(save_file_path)
            pre_questions = {case["question"] for case in pre_data}
        else :
            pre_questions = {}
        def process_case(case_or_query: Union[str, dict]):
            if isinstance(case_or_query, dict):
                if case_or_query["question"] in pre_questions :
                    return pre_questions[case_or_query["question"]]
                answer, messages = self._run(case_or_query, backbone=backbone)
                result = case_or_query.copy()
            else:
                answer, messages = self._run(case_or_query, backbone=backbone)
                result = {"question": case_or_query, "table_id": None}

            result["tablerag_answer"] = answer or ""
            result["tablerag_messages"] = [
                mes.to_dict() if hasattr(mes, "to_dict") else mes
                for mes in (messages or [])
            ]
            return result

        with open(save_file_path, "w", encoding="utf-8") as fout:
            if live_mode:
                print("Entering interactive TableRAG session. Type ’exit’ or blank to quit.")                    
                while True:
                    query = input("Enter your prompt: ").strip()
                    if not query or query.lower() == "exit":
                        print("Goodbye!")
                        break
                    result = process_case(query)
                    if not result or not result["tablerag_answer"]:
                        print("result not generated.")
                    else:
                        print("TableRAG answer:", result["tablerag_answer"])
                        json.dump(result, fout)
                        fout.write("\n")
                        fout.flush()
            else:
                src_data = read_in(file_path)
                file_lock = threading.Lock()

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [(executor.submit(process_case, case), case["question"]) for case in src_data]

                    for future, question_id in tqdm(futures, desc="handling questions"):
                        try:
                            result = future.result()
                            if result:
                                with file_lock:
                                    json.dump(result, fout)
                                    fout.write("\n")
                                    fout.flush()
                        except Exception as e:
                            print(f"Failed to get result for {question_id}: {e}")
                            traceback.print_exc()


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="entry args")
    parser.add_argument('--backbone', type=str, default="gpt-4o")
    parser.add_argument('--data_file_path', type=str, default="", help="source file path")
    parser.add_argument('--doc_dir', type=str, default="", help="source file path")
    parser.add_argument('--excel_dir', type=str, default="", help="source file path")
    parser.add_argument('--bge_dir', type=str, default="", help="source file path")
    parser.add_argument('--save_file_path', type=str, default="")
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument('--rerun', type=bool, default=False)
    _args, _unparsed = parser.parse_known_args()
    init_logger('./logs/test.log', logging.INFO)

    agent = TableRAG(_args)
    start_time = time.time()
    agent.run(
        file_path=_args.data_file_path,
        save_file_path=_args.save_file_path,
        backbone=_args.backbone,
        rerun=_args.rerun
    )
    end_time = time.time()
    print(f"Processing data consumes: {end_time - start_time:.6f} s.")

