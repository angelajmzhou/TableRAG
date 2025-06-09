SYSTEM_EXPLORE_PROMPT = """Next, you will complete a table-related question answering task. Based on the provided materials such as the table content(in Markdown format), you need to analyze the Question.
And try to decide whether the Question should be broken down into subqueries. After you have collected sufficient information, you need to generate comprehensive answers.
You have a "solve_subquery" tool, it only takes natural language question as input.

Instructions:
1. Carefully analyze each user query through step-by-step reasoning.
2. If the query needs information more than the given table content：
    - Decompose the query into subqueries.
    - Process one subquery at a time.
    - Use "solve_subquery" tool to get answers for each subquey.
3. If a query can be answered by table content, do not decompose it. And directly put the orignal query into the "solve_subquery" tool.
    The "solve_subquery" tool utilizes SQL execution inside, it can solve complex subquery on table through one tool call.
4. Generate exactly ONE subquery at a time.
5. Write out all terms completely - avoid using abbreviations.
6. When you have sufficient information, provide the final answer in the following format:
    <Answer>: [your complete response]

Table Contet: {table_content}
Question: {question}
Please start!
"""

COMBINE_PROMPT = """You are about to complete a table-based question answernig task using the following two types of reference materials:

# Content 1: Original content (table content is provided in Markdown format):
{docs}

$ Content 2: NL2SQL related Question information and SQL execution result in the database:
# the user given table schema
{schema}

# SQL generated based on the schema and the user question:
{nl2sql_model_resopnse}

# SQL execution results
{sql_execute_result}

Please answer the user's question based on the materials above.
User question: {query}

Note:
1. The markdown table content in Content 1 may be incomplete.
2. You should cross-validate the given two materials:
    - if the answers are the same, directly output the answer.
    - if the "SQL execution result" contains error, you should try to answer based on the Content 1.
    - if the two materials shows conflit, you should think about each of them, and finally give an answer.
"""

