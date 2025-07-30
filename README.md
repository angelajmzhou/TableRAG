# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning

Repo for _[TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://github.com/yxh-y/TableRAG/)_  

![Main Architecture](./figures/Main%20structure.png)

# 📌 Introduction

- We identify two key limitations of existing RAG approaches in the context of heterogeneous document question answering: structural information loss and lack of global view. 
- We propose **TableRAG**, an **Hybrid (SQL Execution and Textual Retrieval) framework** that unifies textual understanding and complex manipulations over tabular data. TableRAG comprises an offline database construction phase and a four-step online iterative reasoning process.
- We develop **HeteQA**, a benchmark for evaluating multi-hop heterogeneous reasoning capabilities. Experimental results show that TableRAG outperforms RAG and programmatic approaches on HeteQA and public benchmarks, establishing a state-of-the-art solution.

# 🔎 Setup

## Environment
```
conda create -n your_env python=3.10

git clone https://github.com/angelajmzhou/TableRAG/
cd TableRAG

pip install -r linux_requirements.txt
```

# 🛠 How to Run?

## Dataset Preparation
1. Download dev_excel.zip, dev_doc.zip, my_dev.json from [Google Drive](https://drive.google.com/drive/folders/1Pea6kiUZv0UP8k7Ohv19KorBdBaUrouE?usp=drive_link).

## Offline Workflow

### Step 1: Setup MySQL Database

1. Download MySQL
Reach https://downloads.mysql.com/archives/community/ and find MySQL 8.0.24 and downloads for your appropriate environment.
(In my experience, later versions also worked.)

2. Install MySQL
```
tar -zxvf mysql-8.0.24-linux-glibc2.12-x86_64.tar.gz
cd mysql-8.0.24-linux-glibc2.12-x86_64
sudo mkdir /usr/local/mysql && sudo mv * /usr/local/mysql/
sudo groupadd mysql
sudo useradd -r -g mysql mysql
cd /usr/local/mysql
sudo bin/mysqld --initialize --user=mysql --basedir=/usr/local/mysql --datadir=/usr/local/mysql/data
sudo cp support-files/mysql.server /etc/init.d/mysql
sudo systemctl enable mysql
sudo systemctl start mysql
```
3. Create Database for TableRAG
```sql
CREATE DATABASE TableRAG;
```

### Step 2: Offline data Ingestion

1. Setup database config 
Edit offline_backend/config/database_config.json and update it with your own MySQL config.

2. Prepare table files to be ingested
Extract dev_doc and dev_excel into offline/backend/data/hybridqa, and extract my_dev.json into root directory.
optionally, use `clean.py` to reduce dataset to only sports-related data.

4. Execute data ingestion pipeline
```
cd offline_backend/src/
python data_persistent.py
```

### Step 3: Start Database query service

1. Setup LLM config
Edit 'offline_backend/src/handle_requests.py' and substitute your llm request url and apikey into model_request_config.

2. Start service to provide SQL query interface

```
python interface.py
```

## Online Workflow

### Step 1: Setup Config and Data Source

1. Edit 'online_inference/config.py' to set the LLM infering url and key, and the query service url.
   
3. Unzip the dev_excel.zip and put it into "/data" directory.

### Step 2: Run Main Experiment

From original repository:
```
cd online_inference
python3 main.py
  --backbone <backbone_llm>
  --data_file_path ./data/my_dev.json
  --save_file_path <path to save file>
  --max_iter <max iterations of TableRAG, default to 5>
  --rerun <True if some cases fail at the previous run, default to False> 
```

I used this to run:
```
python3 main.py   --backbone gemini   --save_file_path ../results/results.json   --doc_dir ../offline_backend/dataset/hybridqa/dev_doc   --excel_dir ../offline_backend/dataset/hybridqa/dev_excel   --bge_dir ../models --data_file_path ../my_dev.json --rerun True
```

If data_file_path is not provided as an argument, TableRAG will run in interactive mode, where users can live-prompt and receive answers until 'exit' is input.
