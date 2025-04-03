import os
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pydantic import BaseModel
import google.generativeai as genai
from google.cloud import bigquery
import traceback

# Configurações
BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = BASE_DIR / "credentials" / "gc-pr-dl-001-2d9103cd4951.json"
PROJECT_ID = "gc-pr-dl-001"
DATASET_ID = "dtl_bl_rag"
GEMINI_MODEL = "gemini-1.5-pro"
GEMINI_TEMPERATURE = 0

# Prompts
QUESTION_ANALYSIS_PROMPT = '''Você é um analista de dados que pode ajudar a resumir tabelas SQL e analisar perguntas sobre um banco de dados.
Dada a pergunta e o esquema do banco de dados, identifique as tabelas e colunas relevantes.
Se a pergunta não for relevante para o banco de dados ou se não houver informações suficientes para respondê-la, defina is_relevant como false.

Sua resposta deve estar no seguinte formato JSON:
{
    "is_relevant": boolean,
    "relevant_tables": [
        {
            "table_name": string,
            "columns": [string],
            "noun_columns": [string]
        }
    ]
}'''

SQL_GENERATION_PROMPT = '''Você é um assistente de IA que gera consultas SQL específicas para o BigQuery.
Gere uma consulta SQL válida para responder à pergunta do usuário.

IMPORTANTE: Use apenas sintaxe compatível com BigQuery:
- Para datas/timestamps, use:
  * FORMAT_DATETIME('%Y-%m', DATETIME(campo_data)) para extrair ano e mês
  * EXTRACT(MONTH FROM campo_data) para extrair o mês
  * EXTRACT(YEAR FROM campo_data) para extrair o ano
  * DATE_SUB(CURRENT_DATE(), INTERVAL N DAY/MONTH/YEAR) para subtrair datas
  * DATE_ADD(CURRENT_DATE(), INTERVAL N DAY/MONTH/YEAR) para adicionar datas
- NÃO use funções PostgreSQL como to_char ou INTERVAL '1 month'
- Use crases (`) para nomes de tabelas e colunas

Se não houver informações suficientes para escrever uma consulta SQL, responda com "NOT_ENOUGH_INFO".

Aqui estão alguns exemplos:

1. Quantos convênios existem por estado?
Resposta: SELECT `estado`, COUNT(*) as total FROM `dtl_bl_rag.convenios` GROUP BY `estado`

2. Qual é a média de cartões por convênio?
Resposta: SELECT c.`nome`, COUNT(ca.`id`) as total_cartoes 
FROM `dtl_bl_rag.convenios` c 
LEFT JOIN `dtl_bl_rag.cartoes` ca ON c.`id` = ca.`convenio_id` 
GROUP BY c.`nome`

3. Qual é a distribuição de status dos convênios?
Resposta: SELECT `status`, COUNT(*) as total FROM `dtl_bl_rag.convenios` GROUP BY `status`'''

SQL_VALIDATION_PROMPT = '''Você é um assistente de IA que valida e corrige consultas SQL específicas para o BigQuery.

FUNÇÕES VÁLIDAS DO BIGQUERY:
1. Funções de Data/Hora:
   - FORMAT_DATETIME('%Y-%m', DATETIME(campo))
   - FORMAT_DATE('%Y-%m', DATE(campo))
   - DATETIME(campo)
   - DATE(campo)
   - EXTRACT(YEAR FROM campo)
   - EXTRACT(MONTH FROM campo)
   - DATE_SUB(CURRENT_DATE(), INTERVAL N DAY/MONTH/YEAR)
   - DATE_ADD(CURRENT_DATE(), INTERVAL N DAY/MONTH/YEAR)
   - TIMESTAMP_DIFF(timestamp1, timestamp2, INTERVAL)

2. Funções de Agregação:
   - COUNT(*)
   - SUM(campo)
   - AVG(campo)
   - MAX(campo)
   - MIN(campo)

3. Funções de String:
   - CONCAT(str1, str2)
   - LOWER(string)
   - UPPER(string)

EXEMPLOS DE QUERIES VÁLIDAS:
1. Consulta com data:
   SELECT SUM(VALOR)
   FROM `dtl_bl_rag.TB_TRANSACOES_OPFIN`
   WHERE FORMAT_DATETIME('%Y-%m', DATETIME(DATA_HORA_TRANSACAO)) = '2024-03'

2. Consulta com intervalo de datas:
   SELECT COUNT(*)
   FROM `dtl_bl_rag.TB_TRANSACOES_OPFIN`
   WHERE DATA_HORA_TRANSACAO >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)

3. Consulta com extração de data:
   SELECT EXTRACT(MONTH FROM DATA_HORA_TRANSACAO) as mes,
          SUM(VALOR) as total
   FROM `dtl_bl_rag.TB_TRANSACOES_OPFIN`
   GROUP BY mes

REGRAS DE VALIDAÇÃO:
1. Verificar se a consulta SQL é válida para o BigQuery
2. Garantir que todos os nomes de tabelas e colunas:
   - Existam no esquema fornecido
   - Estejam entre crases (`)
   - Incluam o prefixo do dataset (dtl_bl_rag)
3. NÃO converter funções válidas do BigQuery para outras sintaxes
4. NÃO usar funções de outros bancos (PostgreSQL, MySQL, etc)
5. Se houver problemas, corrigi-los mantendo a sintaxe do BigQuery
6. Se nenhum problema for encontrado, retornar a consulta original

Responda apenas com um JSON no seguinte formato:
{
    "valid": boolean,
    "issues": string ou null,
    "corrected_query": string
}'''

RESPONSE_FORMULATION_PROMPT = '''Você é um assistente de IA que formata os resultados de consultas de banco de dados em uma resposta legível para humanos. Dê uma conclusão à pergunta do usuário com base nos resultados da consulta. Não dê a resposta em formato markdown. Apenas dê a resposta em uma linha.'''

VISUALIZATION_CHOICE_PROMPT = '''Você é um assistente de IA que recomenda visualizações de dados apropriadas. Com base na pergunta do usuário, consulta SQL e resultados da consulta, sugira o tipo mais adequado de gráfico ou tabela para visualizar os dados. Se nenhuma visualização for apropriada, indique isso.

Tipos de gráficos disponíveis e seus casos de uso:
- Gráficos de Barras: Melhor para comparar dados categóricos ou mostrar mudanças ao longo do tempo quando as categorias são discretas e o número de categorias é maior que 2.
- Gráficos de Barras Horizontais: Melhor para comparar dados categóricos ou mostrar mudanças ao longo do tempo quando o número de categorias é pequeno ou a disparidade entre categorias é grande.
- Gráficos de Linha: Melhor para mostrar tendências e distribuições ao longo do tempo.
- Gráficos de Pizza: Ideal para mostrar proporções ou porcentagens dentro de um todo.
- Gráficos de Dispersão: Útil para identificar relacionamentos ou correlações entre duas variáveis numéricas.

Forneça sua resposta no seguinte formato:
Visualização Recomendada: [Tipo de gráfico ou "None"]. APENAS use os seguintes nomes: bar, horizontal_bar, line, pie, scatter, none
Razão: [Breve explicação para sua recomendação]'''

# Classes
class InputState(TypedDict):
    """Estado de entrada do workflow"""
    question: str

class OutputState(TypedDict):
    """Estado de saída do workflow"""
    answer: str
    visualization: str
    visualization_reason: str
    formatted_data_for_visualization: Optional[Dict[str, Any]]

class WorkflowState(TypedDict):
    """Estado interno do workflow"""
    question: str
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    sql_valid: bool
    sql_issues: Optional[str]
    results: List[Dict[str, Any]]
    visualization: str
    visualization_reason: str
    answer: str
    formatted_data_for_visualization: Optional[Dict[str, Any]]

class QuestionRequest(BaseModel):
    """Modelo para a requisição de pergunta"""
    question: str

class DatabaseManager:
    def __init__(self):
        """Inicializa o gerenciador do banco de dados"""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)
        self.client = bigquery.Client(project=PROJECT_ID)
    
    def get_schema(self) -> Dict[str, Any]:
        """Obtém o esquema do banco de dados diretamente do BigQuery"""
        tables = []
        
        # Lista todas as tabelas no dataset
        dataset_ref = self.client.dataset(DATASET_ID)
        for table in self.client.list_tables(dataset_ref):
            table_ref = table.reference
            table_obj = self.client.get_table(table_ref)
            
            # Obtém as colunas da tabela
            columns = [field.name for field in table_obj.schema]
            
            # Identifica colunas que podem conter substantivos (baseado no nome da coluna)
            noun_columns = [
                col for col in columns 
                if any(noun in col.lower() for noun in ['nome', 'descricao', 'estado', 'status', 'tipo', 'categoria'])
            ]
            
            tables.append({
                "table_name": f"{DATASET_ID}.{table.table_id}",
                "columns": columns,
                "noun_columns": noun_columns
            })
        
        return {"tables": tables}
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Executa uma query no BigQuery"""
        query_job = self.client.query(query)
        return [dict(row.items()) for row in query_job]

class LLMManager:
    def __init__(self):
        """Inicializa o gerenciador do LLM"""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
    
    def invoke(self, prompt_template: str, **kwargs) -> str:
        """Invoca o modelo com um prompt"""
        try:
            # Formata o prompt substituindo as variáveis
            formatted_prompt = prompt_template
            for key, value in kwargs.items():
                formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))
            
            # Gera a resposta
            response = self.model.generate_content(formatted_prompt)
            return response.text
        except Exception as e:
            print(f"Erro ao invocar o modelo: {str(e)}")
            raise

class DataFormatter:
    def __init__(self):
        """Inicializa o formatador de dados"""
        self.llm_manager = LLMManager()
        self.output_parser = JsonOutputParser()
    
    def format_data_for_visualization(
        self,
        data: List[Dict[str, Any]],
        visualization_type: str,
        x_column: str = None,
        y_column: str = None,
        title: str = None
    ) -> Dict[str, Any]:
        """Formata os dados para visualização"""
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df = df.dropna()
        
        if visualization_type == 'bar':
            return {
                'x': df[x_column].tolist() if x_column else df.index.tolist(),
                'y': df[y_column].tolist() if y_column else df.iloc[:, 0].tolist(),
                'title': title
            }
        elif visualization_type == 'horizontal_bar':
            return {
                'x': df[y_column].tolist() if y_column else df.iloc[:, 0].tolist(),
                'y': df[x_column].tolist() if x_column else df.index.tolist(),
                'title': title
            }
        elif visualization_type == 'line':
            return {
                'x': df[x_column].tolist() if x_column else df.index.tolist(),
                'y': df[y_column].tolist() if y_column else df.iloc[:, 0].tolist(),
                'title': title
            }
        elif visualization_type == 'pie':
            return {
                'labels': df[x_column].tolist() if x_column else df.index.tolist(),
                'values': df[y_column].tolist() if y_column else df.iloc[:, 0].tolist(),
                'title': title
            }
        elif visualization_type == 'scatter':
            return {
                'x': df[x_column].tolist() if x_column else df.index.tolist(),
                'y': df[y_column].tolist() if y_column else df.iloc[:, 0].tolist(),
                'title': title
            }
        else:
            return None

class SQLAgent:
    def __init__(self):
        """Inicializa o agente SQL"""
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        self.data_formatter = DataFormatter()
    
    def parse_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa a pergunta e identifica tabelas e colunas relevantes"""
        question = state["question"]
        schema = self.db_manager.get_schema()
        
        prompt = f"""Você é um analista de dados que pode ajudar a resumir tabelas SQL e analisar perguntas sobre um banco de dados.
Dada a pergunta e o esquema do banco de dados, identifique as tabelas e colunas relevantes.

Schema do banco de dados:
{schema}

Pergunta do usuário:
{question}

Responda apenas com um JSON no seguinte formato, sem usar marcadores de código:
{{
    "is_relevant": boolean,
    "relevant_tables": [
        {{
            "table_name": string,
            "columns": [string],
            "noun_columns": [string]
        }}
    ]
}}"""
        
        try:
            response = self.llm_manager.invoke(prompt)
            print("\nResposta original do modelo:")  # Log para debug
            print("="*50)
            print(response)
            print("="*50)
            
            # Limpa a resposta removendo marcadores de código e espaços em branco
            cleaned_response = response
            
            # Remove marcadores de código se existirem
            if "```json" in cleaned_response:
                parts = cleaned_response.split("```json")
                if len(parts) > 1:
                    cleaned_response = parts[1]
            
            if "```" in cleaned_response:
                parts = cleaned_response.split("```")
                cleaned_response = parts[0]
            
            # Remove espaços em branco extras e caracteres especiais
            cleaned_response = cleaned_response.strip()
            
            print("\nResposta após limpeza:")  # Log para debug
            print("="*50)
            print(cleaned_response)
            print("="*50)
            
            # Tenta fazer o parse do JSON
            try:
                parsed_response = json.loads(cleaned_response)
                print("\nJSON parseado com sucesso:")  # Log para debug
                print("="*50)
                print(json.dumps(parsed_response, indent=2))
                print("="*50)
                return {"parsed_question": parsed_response}
            except json.JSONDecodeError as e:
                print(f"\nErro ao fazer parse do JSON: {str(e)}")  # Log para debug
                print(f"Posição do erro: caractere {e.pos}")
                print(f"Linha do erro: {e.lineno}")
                print(f"Coluna do erro: {e.colno}")
                # Retorna um resultado padrão em caso de erro
                return {
                    "parsed_question": {
                        "is_relevant": False,
                        "relevant_tables": []
                    }
                }
        except Exception as e:
            print(f"\nErro geral no parse_question: {str(e)}")  # Log para debug
            return {
                "parsed_question": {
                    "is_relevant": False,
                    "relevant_tables": []
                }
            }
    
    def get_unique_nouns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Obtém substantivos únicos das colunas relevantes"""
        parsed_question = state["parsed_question"]
        
        if not parsed_question["is_relevant"]:
            return {"unique_nouns": []}
        
        unique_nouns = []
        for table in parsed_question["relevant_tables"]:
            for column in table["noun_columns"]:
                query = f"SELECT DISTINCT `{column}` FROM `{table['table_name']}` WHERE `{column}` IS NOT NULL AND `{column}` != '' AND `{column}` != 'N/A' LIMIT 10"
                try:
                    results = self.db_manager.execute_query(query)
                    unique_nouns.extend([str(row[column]) for row in results])
                except Exception as e:
                    print(f"Erro ao obter substantivos únicos para coluna {column}: {str(e)}")
        
        return {"unique_nouns": list(set(unique_nouns))}
    
    def generate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Gera uma query SQL baseada na pergunta e nos substantivos únicos"""
        question = state["question"]
        parsed_question = state["parsed_question"]
        unique_nouns = state["unique_nouns"]
        
        if not parsed_question["is_relevant"]:
            print("Pergunta não é relevante para o banco de dados.")  # Log
            return {"sql_query": "NOT_RELEVANT"}
        
        schema = self.db_manager.get_schema()
        
        prompt = f"""Você é um assistente de IA que gera consultas SQL com base em perguntas do usuário.
Gere uma consulta SQL válida para o BigQuery para responder à pergunta do usuário.

Schema do banco de dados:
{schema}

Pergunta do usuário:
{question}

Tabelas e colunas relevantes:
{parsed_question}

Substantivos únicos encontrados:
{unique_nouns}

Importante:
- Use FORMAT_DATETIME('%Y-%m', DATETIME(campo_data)) para extrair ano e mês
- Use EXTRACT(MONTH FROM campo_data) para extrair o mês
- Use EXTRACT(YEAR FROM campo_data) para extrair o ano

Se não houver informações suficientes, responda apenas com "NOT_ENOUGH_INFO".
Caso contrário, forneça apenas a query SQL, sem explicações adicionais.

Para a pergunta sobre o número de registros por tabela, use:
SELECT table_name, COUNT(*) as total_records 
FROM `dtl_bl_rag.INFORMATION_SCHEMA.PARTITIONS` 
GROUP BY table_name 
ORDER BY total_records DESC 
LIMIT 3"""
        
        try:
            response = self.llm_manager.invoke(prompt)
            response = response.strip()
            
            print("\nResposta do modelo (query gerada):")  # Log
            print("="*50)  # Log
            print(response)  # Log
            print("="*50)  # Log
            
            if response == "NOT_ENOUGH_INFO" or not response:
                print("Informações insuficientes para gerar query.")  # Log
                return {"sql_query": "NOT_RELEVANT"}
            
            # Remove marcadores de código SQL se existirem
            if response.startswith("```sql"):
                response = response.replace("```sql", "", 1)
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            if not response:
                print("Query vazia após limpeza.")  # Log
                return {"sql_query": "NOT_RELEVANT"}
            
            print("\nQuery final após limpeza:")  # Log
            print("="*50)  # Log
            print(response)  # Log
            print("="*50)  # Log
                
            return {"sql_query": response}
        except Exception as e:
            print(f"Erro ao gerar SQL: {str(e)}")  # Log
            return {"sql_query": "NOT_RELEVANT"}
    
    def validate_and_fix_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Valida e corrige a query SQL gerada"""
        sql_query = state.get("sql_query", "NOT_RELEVANT")
        
        if sql_query == "NOT_RELEVANT" or not sql_query or sql_query == "null":
            return {
                "sql_query": "NOT_RELEVANT",
                "sql_valid": False,
                "results": "NOT_RELEVANT"  # Garantindo que results existe no estado
            }
        
        schema = self.db_manager.get_schema()
        
        prompt = f"""Você é um assistente de IA que valida e corrige consultas SQL.
Sua tarefa é verificar se a query é válida e se todos os nomes de tabelas e colunas existem no schema.

Schema do banco de dados:
{schema}

Query SQL gerada:
{sql_query}

Responda apenas com um JSON no seguinte formato, sem usar marcadores de código:
{{
    "valid": boolean,
    "issues": string ou null,
    "corrected_query": string
}}"""
        
        try:
            response = self.llm_manager.invoke(prompt)
            print(f"Resposta original: {response}")  # Log para debug
            
            # Limpa a resposta removendo marcadores de código e espaços em branco extras
            cleaned_response = response.strip()
            
            # Remove marcadores de código se existirem
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.replace("```json", "", 1)
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Remove espaços em branco extras e caracteres especiais
            cleaned_response = cleaned_response.strip()
            
            # Remove quebras de linha dentro das strings
            cleaned_response = cleaned_response.replace('\n', ' ')
            
            print(f"Resposta limpa: {cleaned_response}")  # Log para debug
            
            try:
                result = json.loads(cleaned_response)
                print(f"JSON parseado com sucesso: {result}")  # Log para debug
                
                if result["valid"] and result["issues"] is None:
                    if not result["corrected_query"] or result["corrected_query"] == "null":
                        return {
                            "sql_query": "NOT_RELEVANT",
                            "sql_valid": False,
                            "results": "NOT_RELEVANT"
                        }
                    return {
                        "sql_query": sql_query,
                        "sql_valid": True
                    }
                else:
                    if not result["corrected_query"] or result["corrected_query"] == "null":
                        return {
                            "sql_query": "NOT_RELEVANT",
                            "sql_valid": False,
                            "results": "NOT_RELEVANT"
                        }
                    return {
                        "sql_query": result["corrected_query"],
                        "sql_valid": result["valid"],
                        "sql_issues": result["issues"]
                    }
            except json.JSONDecodeError as e:
                print(f"Erro ao fazer parse do JSON: {str(e)}")  # Log para debug
                print(f"Posição do erro: caractere {e.pos}")
                print(f"Linha do erro: {e.lineno}")
                print(f"Coluna do erro: {e.colno}")
                # Se houver erro no parse, mas a query original parece válida, prossegue com ela
                if sql_query and sql_query != "NOT_RELEVANT":
                    return {
                        "sql_query": sql_query,
                        "sql_valid": True
                    }
                return {
                    "sql_query": "NOT_RELEVANT",
                    "sql_valid": False,
                    "results": "NOT_RELEVANT"
                }
        except Exception as e:
            print(f"Erro ao validar SQL: {str(e)}")
            # Se houver erro na validação, mas a query original parece válida, prossegue com ela
            if sql_query and sql_query != "NOT_RELEVANT":
                return {
                    "sql_query": sql_query,
                    "sql_valid": True
                }
            return {
                "sql_query": "NOT_RELEVANT",
                "sql_valid": False,
                "results": "NOT_RELEVANT"
            }
    
    def execute_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Executa a query SQL e retorna os resultados"""
        query = state["sql_query"]
        
        if query == "NOT_RELEVANT":
            return {"results": "NOT_RELEVANT"}
        
        if not query or query == "null":  # Adicionando verificação para query vazia ou null
            return {"results": "NOT_RELEVANT"}
            
        try:
            results = self.db_manager.execute_query(query)
            return {"results": results}
        except Exception as e:
            print(f"Erro ao executar query: {str(e)}")  # Log do erro
            return {"results": "NOT_RELEVANT", "error": str(e)}
    
    def format_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Formata os resultados em uma resposta legível"""
        question = state["question"]
        results = state["results"]
        
        if results == "NOT_RELEVANT":
            return {"answer": "Desculpe, só posso dar respostas relevantes para o banco de dados."}
        
        prompt = f"""Você é um assistente de IA que formata resultados de consultas de banco de dados em respostas legíveis.
Formate os resultados da consulta em uma resposta clara e concisa para o usuário.

Pergunta do usuário:
{question}

Resultados da consulta:
{results}

Forneça apenas a resposta em uma linha, sem formatação markdown ou explicações adicionais."""
        
        response = self.llm_manager.invoke(prompt)
        return {"answer": response}
    
    def choose_visualization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Escolhe uma visualização apropriada para os dados"""
        question = state["question"]
        results = state["results"]
        sql_query = state["sql_query"]
        
        if results == "NOT_RELEVANT":
            return {
                "visualization": "none",
                "visualization_reason": "Nenhuma visualização necessária para perguntas irrelevantes."
            }
        
        prompt = f"""Você é um assistente de IA que recomenda visualizações de dados apropriadas.
Com base na pergunta, query SQL e resultados, sugira o tipo mais adequado de gráfico.

Tipos de gráficos disponíveis:
- bar: Gráficos de barras para comparar dados categóricos
- horizontal_bar: Gráficos de barras horizontais para categorias com nomes longos
- line: Gráficos de linha para tendências temporais
- pie: Gráficos de pizza para proporções
- scatter: Gráficos de dispersão para correlações
- none: Sem visualização necessária

Pergunta do usuário:
{question}

Query SQL:
{sql_query}

Resultados:
{results}

Responda em duas linhas exatamente neste formato (sem colchetes ou outros caracteres especiais):
Visualização Recomendada: tipo
Razão: explicação breve"""
        
        response = self.llm_manager.invoke(prompt)
        lines = response.split('\n')
        
        # Remove colchetes e espaços extras do tipo de visualização
        visualization = lines[0].split(': ')[1].strip().strip('[]')
        reason = lines[1].split(': ')[1].strip()
        
        print("\nTipo de visualização escolhido:")  # Debug
        print(f"Original: {lines[0].split(': ')[1].strip()}")
        print(f"Limpo: {visualization}")
        
        return {"visualization": visualization, "visualization_reason": reason}
    
    def format_data_for_visualization(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Formata os dados para a visualização escolhida"""
        visualization = state["visualization"]
        results = state["results"]
        
        if visualization == "none" or results == "NOT_RELEVANT":
            return {"formatted_data_for_visualization": None}
        
        try:
            print("\nDados para visualização:")  # Debug
            print("Tipo de visualização:", visualization)
            print("Resultados:", results)
            
            # Para o caso específico de contagem de registros por tabela
            if isinstance(results, str) and "registro" in results:
                # Extrai os dados da string de resposta
                data = []
                for part in results.split(", "):
                    table_info = part.strip("() ").split(" ")
                    table_name = table_info[0]
                    count = int(table_info[1].strip("()").split()[0])  # Pega o número antes de "registro"
                    data.append({"table_name": table_name, "count": count})
                
                # Cria DataFrame
                df = pd.DataFrame(data)
                
                # Formata os dados para visualização
                formatted_data = {
                    'x': df['table_name'].tolist(),
                    'y': df['count'].tolist(),
                    'title': 'Número de Registros por Tabela'
                }
                
                return {"formatted_data_for_visualization": formatted_data}
            
            # Para resultados do BigQuery (lista de dicionários)
            if isinstance(results, list) and len(results) > 0:
                df = pd.DataFrame(results)
                print("\nDataFrame criado:")  # Debug
                print(df)
                
                if len(df.columns) >= 2:
                    # Pega as duas primeiras colunas
                    x_column = df.columns[0]
                    y_column = df.columns[1]
                    
                    formatted_data = {
                        'x': df[x_column].tolist(),
                        'y': df[y_column].tolist(),
                        'title': f'{y_column} por {x_column}'
                    }
                    
                    # Se for um gráfico de pizza, ajusta o formato
                    if visualization == "pie":
                        formatted_data = {
                            'labels': formatted_data['x'],
                            'values': formatted_data['y'],
                            'title': formatted_data['title']
                        }
                    
                    print("\nDados formatados:")  # Debug
                    print(formatted_data)
                    
                    return {"formatted_data_for_visualization": formatted_data}
            
            print("Nenhum dado válido para visualização")  # Debug
            return {"formatted_data_for_visualization": None}
            
        except Exception as e:
            print(f"Erro ao formatar dados para visualização: {str(e)}")
            print("Traceback:", traceback.format_exc())  # Debug completo
            return {"formatted_data_for_visualization": None}

class WorkflowManager:
    def __init__(self):
        """Inicializa o gerenciador do workflow"""
        self.sql_agent = SQLAgent()
    
    def run(self, question: str) -> Dict[str, Any]:
        """Executa o workflow com uma pergunta"""
        try:
            print("1. Iniciando processamento da pergunta...")  # Log
            
            # Estado inicial
            state = {"question": question}
            
            print("2. Analisando a pergunta...")  # Log
            state.update(self.sql_agent.parse_question(state))
            
            print("3. Obtendo substantivos únicos...")  # Log
            state.update(self.sql_agent.get_unique_nouns(state))
            
            print("4. Gerando SQL...")  # Log
            state.update(self.sql_agent.generate_sql(state))
            
            print("5. Validando SQL...")  # Log
            state.update(self.sql_agent.validate_and_fix_sql(state))
            
            print("6. Executando SQL...")  # Log
            state.update(self.sql_agent.execute_sql(state))
            
            print("7. Escolhendo visualização...")  # Log
            state.update(self.sql_agent.choose_visualization(state))
            
            print("8. Formatando resultados...")  # Log
            state.update(self.sql_agent.format_results(state))
            
            print("9. Formatando dados para visualização...")  # Log
            state.update(self.sql_agent.format_data_for_visualization(state))
            
            # Retorna o resultado final usando get() para evitar KeyError
            result = {
                "answer": state.get("answer", "Desculpe, não foi possível gerar uma resposta."),
                "visualization": state.get("visualization", "none"),
                "visualization_reason": state.get("visualization_reason", "Não foi possível determinar uma visualização."),
                "formatted_data_for_visualization": state.get("formatted_data_for_visualization", None)
            }
            
            print(f"10. Workflow concluído. Resultado: {result}")  # Log
            return result
            
        except Exception as e:
            print(f"Erro no workflow: {str(e)}")  # Log
            print(f"Traceback do workflow: {traceback.format_exc()}")  # Log
            raise

# Configuração do FastAPI
app = FastAPI(
    title="Chat With Data",
    description="API para processamento de perguntas em linguagem natural e visualização de dados do BigQuery",
    version="1.0.0"
)

# Configuração dos templates e arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Inicializa o gerenciador do workflow
workflow_manager = WorkflowManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Rota principal que renderiza a página inicial"""
    return templates.TemplateResponse(
        "chat.html",
        {"request": request}
    )

@app.post("/query")
async def process_query(request: QuestionRequest):
    """Rota para processar perguntas e retornar respostas"""
    try:
        print(f"Recebendo pergunta: {request.question}")  # Log
        
        print("Iniciando workflow...")  # Log
        result = workflow_manager.run(request.question)
        print(f"Resultado do workflow: {result}")  # Log
        
        return result
    except Exception as e:
        print(f"Erro ao processar pergunta: {str(e)}")  # Log
        print(f"Traceback completo: {traceback.format_exc()}")  # Log detalhado do erro
        return {"error": str(e), "answer": f"Erro ao processar sua pergunta: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 