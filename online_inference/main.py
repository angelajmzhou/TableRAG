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
        #self.repo_id = self.config.get("repo_id", "")
        self.function_lock = threading.Lock()


    from google.generativeai.types import FunctionDeclaration

    def create_tools(self):
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
                    # ❌ Do NOT include "additionalProperties"
                }
            )
        ]


    def extract_subquery(self, response: Any, backbone: str = 'gemini') -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
        """
        Extract the subquery and reasoning process from Gemini or OpenAI responses.
        """
        subquery, tool_call_id = [], []

        if backbone == 'openai':
            if isinstance(response, dict):
                if "tool_calls" in response and response["tool_calls"]:
                    for call in response["tool_calls"]:
                        tool_call_id.append(call["id"])
                        arguments = call["function"]["arguments"]
                        subquery.append(json.loads(arguments)["subquery"])
                    return response.get("content", ""), subquery, tool_call_id
                else:
                    return response.get("content", ""), None, None
            else:
                reasoning = getattr(response, "content", "")
                try:
                    for call in response.tool_calls:
                        arguments = call.function.arguments
                        subquery.append(json.loads(arguments)["subquery"])
                        tool_call_id.append(call.id)
                    return reasoning, subquery, tool_call_id
                except:
                    return reasoning, None, None

        elif backbone == 'gemini':
            reasoning = getattr(response.candidates[0].content, 'parts', [])[0].text if response.candidates else ""
            try:
                tool_calls = response.candidates[0].content.tool_calls
                for call in tool_calls:
                    args_dict = call.function.arguments  # Already parsed dict
                    subquery.append(args_dict["subquery"])
                    tool_call_id.append(call.id)
                return reasoning, subquery, tool_call_id
            except:
                return reasoning, None, None


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
            raise ValueError(f"Unexpected response format: {type(response)} → {response}")



    def get_llm_response(self, text_messages: object, tools: object, backbone: str, select_config: object) :
        if tools :
            response = get_chat_result(messages=text_messages, tools=tools, llm_config=gemini_config)   
        else :
            response = get_chat_result(messages=text_messages, tools=None, llm_config=gemini_config)   

        return response
                        

    def _run(self, case: dict, backbone: str, tmp: Any = None) :
        """
        Single iteration of TableRAG inference.
        """
        #this looks at pre-existing single case.
        query = case["question"]
        table_id = case["table_id"]

        query_with_suffix = case['question'] + f"The given table is in {table_id}"
    
        _, _, doc_filenames = self.retriever.retrieve(query_with_suffix, 30, 5)

        top1_table_name = doc_filenames[0].replace(".json", "").replace(".xlsx", "")
        related_table_name_list = [top1_table_name]

        tools = self.create_tools()
        current_iter = self.max_iter
        text_messages = self.construct_initial_prompt(case["question"], top1_table_name)

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
                return answer, text_messages
            
            if not sub_queries :
                text_messages.append({
                    "role": "user",
                    "content": "ERROR: Did not call tool with a query!"
                })
                continue

            messages = response
            text_messages.append(messages)

            for sub_query, tool_call_id in zip(sub_queries, tool_call_ids) :
                reranked_docs, _, _ = self.retriever.retrieve(sub_query, 30, 5)
                unique_retriebed_docs = list(set(reranked_docs))
                doc_content = "\n".join([r for r in unique_retriebed_docs[:3]])

                excel_rag_response_dict = get_excel_rag_response_plain(related_table_name_list, sub_query)
                excel_rag_response = copy.deepcopy(excel_rag_response_dict)
                logger.info(f"Requesting ExcelRAG, source file {str(related_table_name_list)}, with query {sub_query}")

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

                final_prompt = combine_prompt_formatted

                msg = [{"role": "user", "content": final_prompt}]
                answer = self.get_llm_response(text_messages=msg, backbone=backbone, select_config=select_config, tools=None)
                answer = self.extract_content(answer)

                if not answer :
                    answer = ""
                
                logger.info(f"LLM Subquery Answer: {answer}")
                execution_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Subquery Answer: " + answer
                }
                text_messages.append(execution_message)

        return None, text_messages


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
        inital_prompt = SYSTEM_PROMPT.format(name=top1_table_name, query=query, table_content=markdown_text)
        logger.info(f"Inital prompt: {inital_prompt}")

        intial_msg = [{"role": "user", "content": inital_prompt}]
        return intial_msg
    
    def run(
        self,
        file_path: str,
        save_file_path: str,
        backbone: str,
        rerun: bool = False,
        max_workers: int = 1
    ) -> None :
        """
        Experimental Entry.
        """
        if rerun :
            pre_data = read_in_lines(save_file_path)
            pre_questions = {case["question"] for case in pre_data}

        else :
            pre_questions = {}
        src_data = read_in(file_path)

        def process_data(case) :
            if case["question"] in pre_questions :
                return pre_questions[case["question"]]
            answer, messages = self._run(case, backbone=backbone)
            
            result = case.copy()
            if answer == None :
                result["tablerag_answer"] = ""
                result["tablerag_messages"] = []
            else :
                new_messages = []
                for mes in messages :
                    if not isinstance(mes, dict) :
                        new_messages.append(mes.to_dict())
                    else :
                        new_messages.append(mes)
                result["tablerag_answer"] = answer
                result["tablerag_messages"] = new_messages

            return result

        if max_workers >= 1 :
            file_lock = threading.Lock()
            with open(save_file_path, "w", encoding="utf-8") as fout :
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor :
                    futures = []
                    for case in src_data :
                        future = executor.submit(process_data, case)
                        futures.append((future, case["question"]))
                    
                    for future, question_id in tqdm(futures, desc="handling questions") :
                        try :
                            result = future.result()
                            with file_lock :
                                json.dump(result, fout)
                                fout.write("\n")
                                fout.flush()
                        except Exception as e :
                            print(f"Failed to get result for {question_id}: {e}")
                            traceback.print_exc()
    def liverun(self,
        save_file_path: str,
        backbone: str,
        rerun: bool = False,
        max_workers: int = 1
    ) -> None :

        question = input("Enter your prompt:")
        if max_workers >= 1 :
            with open(save_file_path, "w", encoding="utf-8") as fout :
                answer, messages = self._queryrun(question, backbone=backbone)

                

                if answer == None :
                    print("result not generated.")
                else :
                    new_messages = []
                    for mes in messages :
                        if not isinstance(mes, dict) :
                            new_messages.append(mes.to_dict())
                        else :
                            new_messages.append(mes)
                    result = {"query": question, "tablerag answer": answer, "tablerag message": new_messages}
                    print("tablerag answer: ", answer)
                    print("tablerag message: ", new_messages)
                    json.dump(result, fout)
                    fout.write("\n")
                    fout.flush()
  

    def _queryrun(self, query: str, backbone: str, tmp: Any = None) :
        """
        Single iteration of TableRAG inference.
        """
        #query with suffix
        #30 from faiss, 5 from reranking
        _, _, doc_filenames = self.retriever.retrieve(query, 30, 5)
        top1_table_name = doc_filenames[0].replace(".json", "").replace(".xlsx", "")
        related_table_name_list = [top1_table_name]

        tools = self.create_tools()
        current_iter = self.max_iter
        text_messages = self.construct_initial_prompt(query, top1_table_name)

        logger.info(f"Processing query: {query}")
        select_config = config_mapping[backbone]

        while current_iter :
            current_iter -= 1
            response = self.get_llm_response(text_messages=text_messages, tools=tools, backbone=backbone, select_config=select_config)
            print(response)
            reasoning, sub_queries, tool_call_ids = self.extract_subquery(response, backbone=backbone)
            logger.info(f"Step {self.max_iter - current_iter}: {sub_queries}")
            if sub_queries is None:
                print(reasoning)

            if not sub_queries and "<Answer>" in reasoning and current_iter != self.max_iter - 1 :
                answer = self.extract_answer(reasoning)
                logger.info(f"Answer: {answer}")
                return answer, text_messages
            
            if not sub_queries :
                text_messages.append({
                    "role": "user",
                    "content": "ERROR: Did not call tool with a query!"
                })
                continue

            messages = response
            text_messages.append({
                "role": "model",
                "content": [response["content"]]

            })


            for sub_query, tool_call_id in zip(sub_queries, tool_call_ids) :
                reranked_docs, _, _ = self.retriever.retrieve(sub_query, 30, 5)
                unique_retriebed_docs = list(set(reranked_docs))
                doc_content = "\n".join([r for r in unique_retriebed_docs[:3]])

                excel_rag_response_dict = get_excel_rag_response_plain(related_table_name_list, sub_query)
                excel_rag_response = copy.deepcopy(excel_rag_response_dict)
                logger.info(f"Requesting ExcelRAG, source file {str(related_table_name_list)}, with query {sub_query}")

                try :
                    sql_str = excel_rag_response['sql_str']
                    sql_execute_result = excel_rag_response['sql_execution_result']
                    schema  = get_excel_rag_response['nl2sql_prompt'].split('Based on the schemas above, please use MySQL syntax to solve the following problem')[0].strip()
                except :
                    sql_str, sql_execute_result, schema = "", "", "ExcelRAG execute fails, key does not exist."

                combine_prompt_formatted = COMBINE_PROMPT.format(
                    docs=doc_content, 
                    schema=schema, 
                    nl2sql_model_response=sql_str, 
                    sql_execute_result=sql_execute_result,
                    query=sub_query
                )

                final_prompt = combine_prompt_formatted

                msg = [{"role": "user", "content": final_prompt}]
                answer = self.get_llm_response(text_messages=msg, backbone=backbone, select_config=select_config, tools=None)
                answer = self.extract_content(answer)

                if not answer :
                    answer = ""
                
                logger.info(f"LLM Subquery Answer: {answer}")
                execution_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Subquery Answer: " + answer
                }
                text_messages.append(execution_message)

        return None, text_messages

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
    # agent.run(
    #     file_path=_args.data_file_path,
    #     save_file_path=_args.save_file_path,
    #     backbone=_args.backbone,
    #     rerun=_args.rerun
    # )

    agent.liverun(
        save_file_path=_args.save_file_path,
        backbone=_args.backbone,
        rerun=_args.rerun
    )
    end_time = time.time()
    print(f"Processing data consumes: {end_time - start_time:.6f} s.")

