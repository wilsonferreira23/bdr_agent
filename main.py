# -*- coding: utf-8 -*-

# Importações de bibliotecas padrão do Python
import asyncio
import io
import json
import logging
import os
import time
import uuid
from typing import List, Optional, Dict

# Importações de bibliotecas de terceiros (instalar via requirements.txt)
import aiohttp
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# --- CONFIGURAÇÃO INICIAL E LOGGING ---
# Configura o sistema de logging para registrar informações úteis e erros durante a execução.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- ARMAZENAMENTO DE TRABALHOS EM MEMÓRIA ---
# Dicionário global para armazenar o status e o resultado dos trabalhos.
# ATENÇÃO: Para um ambiente de produção robusto, o ideal é usar um banco de dados
# ou um serviço de cache como o Redis, para não perder os resultados se a API reiniciar.
job_storage: Dict[str, dict] = {}

# --- MODELOS DE DADOS (PYDANTIC) ---
# Define a estrutura de dados que a API espera receber. Garante a validação automática.

class EmpresaInput(BaseModel):
    """Define a estrutura de uma única empresa na lista de entrada."""
    cnpj: str
    razao_social: str

class EnrichmentRequest(BaseModel):
    """Define a estrutura do corpo (body) da requisição para o endpoint /enrich."""
    empresas: List[EmpresaInput]

# --- LÓGICA DE NEGÓCIO: ENRIQUECIMENTO DE DADOS ---

def extract_target_data(data: dict) -> dict:
    """
    Extrai os dados de interesse da resposta da API da Casa dos Dados
    e formata os contatos de forma limpa para o CSV final.
    """
    # Procura pelo primeiro sócio com a qualificação "Sócio-Administrador"
    socio_admin = next((s for s in data.get("quadro_societario", []) if s.get("qualificacao_socio") == "Sócio-Administrador"), None)
    
    # Extrai e formata os telefones em uma única string separada por vírgulas
    phones_list = data.get("contato_telefonico", [])
    telefones_formatados = ", ".join([
        f"{phone.get('ddd', '')}{phone.get('numero', '')}"
        for phone in phones_list if phone.get('ddd') and phone.get('numero')
    ])

    # Extrai e formata os e-mails em uma única string separada por vírgulas
    emails_list = data.get("contato_email", [])
    emails_formatados = ", ".join([
        email.get("email") 
        for email in emails_list if email.get("email")
    ])

    return {
        "cnpj_consultado": data.get("cnpj"),
        "razao_social": data.get("razao_social"),
        "nome_socio_admin": socio_admin.get("nome") if socio_admin else None,
        "telefones": telefones_formatados,
        "emails": emails_formatados,
    }

# Adiciona um decorador para tentar novamente em caso de falhas de rede ou do servidor
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=lambda retry_state: logging.error(f"Falha ao buscar CNPJ após {retry_state.attempt_number} tentativas: {retry_state.outcome.exception()}")
)
async def fetch_cnpj_details(session: aiohttp.ClientSession, cnpj: str) -> Optional[dict]:
    """Busca os detalhes de um CNPJ de forma assíncrona, com retentativas."""
    API_URL_TEMPLATE = "https://api.casadosdados.com.br/v4/cnpj/{}"
    url = API_URL_TEMPLATE.format(cnpj)
    try:
        async with session.get(url, timeout=20) as response:
            response.raise_for_status()
            data = await response.json()
            return extract_target_data(data)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.warning(f"Erro de rede/timeout ao buscar CNPJ {cnpj}. Tentando novamente...")
        raise  # Re-lança a exceção para que o 'tenacity' possa tentar novamente.
    except Exception as e:
        logging.error(f"Erro inesperado para o CNPJ {cnpj}: {e}")
        return None

async def process_enrichment_task(job_id: str, empresas: List[EmpresaInput], api_key: str):
    """Função principal que roda em segundo plano para fazer todo o trabalho pesado."""
    start_time = time.monotonic()
    try:
        total_empresas = len(empresas)
        logging.info(f"Job {job_id}: Iniciando tarefa para {total_empresas} empresas.")

        headers = {"accept": "application/json", "api-key": api_key}
        connector = aiohttp.TCPConnector(limit=50)
        
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = [fetch_cnpj_details(session, empresa.cnpj) for empresa in empresas]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        successful_results = [res for res in results if res is not None]
        duration_seconds = time.monotonic() - start_time
        logging.info(f"Job {job_id}: Tarefa concluída em {duration_seconds:.2f}s. Sucesso: {len(successful_results)}/{total_empresas}.")

        if successful_results:
            df = pd.DataFrame(successful_results)
            csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            job_storage[job_id] = {"status": "complete", "data": csv_bytes, "duration_seconds": duration_seconds}
        else:
            job_storage[job_id] = {"status": "complete", "data": None, "message": "Nenhum dado pôde ser enriquecido.", "duration_seconds": duration_seconds}

    except Exception as e:
        duration_seconds = time.monotonic() - start_time
        logging.error(f"Job {job_id}: Falha crítica na tarefa de background após {duration_seconds:.2f}s: {e}")
        job_storage[job_id] = {"status": "failed", "message": str(e), "duration_seconds": duration_seconds}

# --- APLICAÇÃO WEB (FastAPI) ---
# Cria a instância principal da API com documentação automática.
app = FastAPI(
    title="Agente Enriquecedor de Empresas",
    description="API para enriquecer dados de empresas em massa. Envie uma lista para /enrich e busque o resultado em /results/{job_id}",
    version="2.2.0"
)

@app.post("/enrich", status_code=202, summary="Inicia o Processo de Enriquecimento")
async def start_enrichment(request: EnrichmentRequest, background_tasks: BackgroundTasks):
    """Recebe uma lista de empresas, inicia o trabalho em background e retorna um ID de trabalho."""
    API_KEY = os.getenv("CASA_DOS_DADOS_API_KEY")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Erro interno: Chave de API não configurada.")
    
    if not request.empresas:
        raise HTTPException(status_code=400, detail="A lista de 'empresas' não pode estar vazia.")

    job_id = str(uuid.uuid4())
    job_storage[job_id] = {"status": "processing"}
    
    background_tasks.add_task(process_enrichment_task, job_id, request.empresas, API_KEY)
    
    status_url = app.url_path_for('get_result', job_id=job_id)
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Trabalho recebido. Verifique a URL de status em alguns minutos para baixar o resultado.",
        "status_url": status_url
    }

@app.get("/results/{job_id}", summary="Verifica o Status e Baixa o Resultado")
def get_result(job_id: str):
    """Verifica o status de um trabalho. Se estiver completo, retorna o arquivo CSV para download."""
    job = job_storage.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="ID de trabalho não encontrado.")
    
    status = job.get("status")
    if status == "processing":
        return {"status": "processing", "message": "Seu arquivo ainda está sendo processado. Tente novamente em alguns instantes."}
    
    if status == "failed":
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar seu trabalho: {job.get('message')}")
        
    if status == "complete" and job.get("data"):
        csv_bytes = job.get("data")
        duration = job.get("duration_seconds", 0)
        
        response = StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=resultado_{job_id}.csv"
        response.headers["X-Processing-Time-Seconds"] = f"{duration:.2f}"
        
        return response
    
    return {"status": "complete", "message": "Processamento concluído, mas sem dados para retornar."}
