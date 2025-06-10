# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning

Repo for _[TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://github.com/yxh-y/TableRAG/)_  

![Main Architecture](./figures/Main%20structure.png)

# 📌 Introduction

- We identify two key limitations of existing RAG approaches in the context of heterogeneous document question answering: structural information loss and lack of global view. 
- We propose **TableRAG**, an **SQL-based framework** that unifies textual understanding and complex manipulations over tabular data. TableRAG comprises an offline database construction phase and a four-step online iterative reasoning process.
- We develop **HeteQA**, a benchmark for evaluating multi-hop heterogeneous reasoning capabilities. Experimental results show that TableRAG outperforms RAG and programmatic approaches on HeteQA and public benchmarks, establishing a state-of-the-art solution.

# 🔎 Setup

## Environment
```
conda create -n webwalker python=3.10

git clone https://github.com/yxh-y/TableRAG/
cd TableRAG

pip install -r requirements.txt
```

# 🛠 How to Run?


## Step 1: Setup MySQL Database

### Download MySQL
reach https://downloads.mysql.com/archives/community/
find MySQL 8.0.24 and downloads for your appropriate environment

### Install MySQL
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

### Create Database for TableRAG
```sql
CREATE DATABASE TableRAG;
```


## Step 2: Set Config and Arguments

### Offline data ingestion

#### Setup database config 
edit offline_data_ingestion_and_query_interface/config/database_config.json
update it with your own MySQL config

#### Prepare table files to be ingested
unzip offline_data_ingestion_and_query_interface/dataset/hybridqa/dev_excel.zip
to offline_data_ingestion_and_query_interface/dataset/hybridqa/dev_excel/

#### Execute data ingestion pipeline
```
cd offline_data_ingestion_and_query_interface/src/
python data_persistent.py
```

## Step 3: Example Usage Command

### Start Database query service

#### setup LLM config
edit offline_data_ingestion_and_query_interface/src/handle_requests.py
substitute your llm request url and apikey into model_request_config

#### start service to provide SQL query interface

```
python interface.py
```



