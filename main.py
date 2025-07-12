import asyncio
import io
import json
import logging
import os
import time
import uuid
from typing import List, Optional, Dict, Any

# Importações de bibliotecas de terceiros (instalar via requirements.txt)
import aiohttp
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

# --- CONFIGURAÇÃO INICIAL E LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- ARMAZENAMENTO DE TRABALHOS EM MEMÓRIA ---
job_storage: Dict[str, dict] = {}

# --- MODELOS DE DADOS (PYDANTIC) ---

# Modelos para o endpoint original /enrich
class EmpresaInput(BaseModel):
    """Define a estrutura de uma única empresa na lista de entrada."""
    cnpj: str
    razao_social: str

    class Config:
        extra = "ignore"

class EnrichmentRequest(BaseModel):
    """Define a estrutura do corpo da requisição para o endpoint /enrich."""
    empresas: List[EmpresaInput]
    api_key: str

    @validator('empresas', pre=True)
    def parse_empresas_from_str(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("O campo 'empresas' foi enviado como uma string, mas não é um JSON válido.")
        return v

# Modelos para a API combinada (/search-and-enrich)
class CapitalSocialInput(BaseModel):
    minimo: int = Field(50000, description="Valor mínimo do capital social.")

class MeiInput(BaseModel):
    optante: bool = False
    excluir_optante: bool = True

class MaisFiltrosInput(BaseModel):
    com_email: bool = True
    com_telefone: bool = True
    somente_celular: bool = True
    somente_fixo: bool = False
    somente_matriz: bool = False
    excluir_empresas_visualizadas: bool = False
    excluir_email_contab: bool = True

class SearchAndEnrichRequest(BaseModel):
    """
    Define a estrutura do corpo da requisição para o novo endpoint /search-and-enrich.
    Combina os filtros de busca da API da Casa dos Dados com os parâmetros necessários.
    """
    # Chave da API (usada para busca e enriquecimento)
    api_key: str = Field(..., description="Sua chave da API da Casa dos Dados.")

    # Filtros de busca
    codigo_atividade_principal: List[str] = Field(..., description="Lista de códigos CNAE principal. Ex: ['5611201']")
    municipio: List[str] = Field(..., description="Lista de municípios. Ex: ['sao paulo']")
    situacao_cadastral: List[str] = Field(["ATIVA"], description="Situação cadastral da empresa.")
    
    # Parâmetros variáveis (agora aceitam string ou int)
    capital_social_minimo: int = Field(50000, description="Valor mínimo do capital social para o filtro.")
    limite: int = Field(5, description="Número máximo de CNPJs a serem buscados e enriquecidos.", le=1000)
    pagina: int = Field(1, description="Página da consulta.")

    # Filtros fixos ou com valores padrão
    incluir_atividade_secundaria: bool = False
    codigo_atividade_secundaria: List[str] = []
    mei: MeiInput = Field(default_factory=MeiInput)
    mais_filtros: MaisFiltrosInput = Field(default_factory=MaisFiltrosInput)
    resposta: str = "completo"

    # ▼▼▼ VALIDADOR ADICIONADO AQUI ▼▼▼
    @validator('capital_social_minimo', 'limite', 'pagina', pre=True)
    def validate_str_to_int(cls, v):
        """
        Permite que os campos sejam enviados como strings e os converte para inteiros.
        """
        if isinstance(v, str):
            try:
                # Remove espaços em branco e converte para inteiro
                return int(v.strip())
            except ValueError:
                raise ValueError(f"O valor '{v}' não é um número inteiro válido.")
        # Se já for um inteiro (ou outro tipo), retorna o valor original para validação padrão
        return v

# --- LÓGICA DE NEGÓCIO: ENRIQUECIMENTO DE DADOS ---

def extract_target_data(data: dict) -> dict:
    """Extrai e formata os dados de interesse do JSON de resposta."""
    socio_admin = next((s for s in data.get("quadro_societario", []) if s.get("qualificacao_socio") == "Sócio-Administrador"), None)
    phones_list = data.get("contato_telefonico", [])
    telefones_formatados = ", ".join([f"{p.get('ddd', '')}{p.get('numero', '')}" for p in phones_list if p.get('ddd') and p.get('numero')])
    emails_list = data.get("contato_email", [])
    emails_formatados = ", ".join([e.get("email") for e in emails_list if e.get("email")])
    return {
        "cnpj_consultado": data.get("cnpj"), "razao_social": data.get("razao_social"),
        "nome_socio_admin": socio_admin.get("nome") if socio_admin else None,
        "telefones": telefones_formatados, "emails": emails_formatados,
    }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=lambda retry_state: logging.error(f"Falha ao buscar CNPJ após {retry_state.attempt_number} tentativas: {retry_state.outcome.exception()}")
)
async def fetch_cnpj_details(session: aiohttp.ClientSession, cnpj: str) -> Optional[dict]:
    """Busca detalhes individuais de um único CNPJ."""
    API_URL_TEMPLATE = "https://api.casadosdados.com.br/v4/cnpj/{}"
    url = API_URL_TEMPLATE.format(cnpj)
    try:
        async with session.get(url, timeout=20) as response:
            response.raise_for_status()
            data = await response.json()
            return extract_target_data(data)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.warning(f"Erro de rede/timeout ao buscar CNPJ {cnpj}. Tentando novamente...")
        raise
    except Exception as e:
        logging.error(f"Erro inesperado para o CNPJ {cnpj}: {e}")
        return None

async def enrich_empresas_list(empresas: List[EmpresaInput], api_key: str) -> Optional[bytes]:
    """
    Processo de enriquecimento para uma lista de empresas. Retorna os bytes do CSV.
    """
    total_empresas = len(empresas)
    headers = {"accept": "application/json", "api-key": api_key}
    connector = aiohttp.TCPConnector(limit=50) # Conexões simultâneas

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [fetch_cnpj_details(session, empresa.cnpj) for empresa in empresas]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    successful_results = [res for res in results if res is not None]
    logging.info(f"Enriquecimento concluído. Sucesso: {len(successful_results)}/{total_empresas}.")

    if successful_results:
        df = pd.DataFrame(successful_results)
        csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        return csv_bytes
    return None

async def process_enrichment_task(job_id: str, empresas: List[EmpresaInput], api_key: str):
    """Tarefa de background para o endpoint original /enrich."""
    start_time = time.monotonic()
    try:
        logging.info(f"Job {job_id}: Iniciando tarefa de enriquecimento para {len(empresas)} empresas.")
        csv_bytes = await enrich_empresas_list(empresas, api_key)
        duration_seconds = time.monotonic() - start_time

        if csv_bytes:
            job_storage[job_id] = {"status": "complete", "data": csv_bytes, "duration_seconds": duration_seconds}
        else:
            job_storage[job_id] = {"status": "complete", "data": None, "message": "Nenhum dado pôde ser enriquecido.", "duration_seconds": duration_seconds}

    except Exception as e:
        duration_seconds = time.monotonic() - start_time
        logging.error(f"Job {job_id}: Falha crítica na tarefa de enriquecimento: {e}")
        job_storage[job_id] = {"status": "failed", "message": str(e), "duration_seconds": duration_seconds}

async def process_search_and_enrich_task(job_id: str, request: SearchAndEnrichRequest):
    """
    Tarefa de background que primeiro busca os CNPJs e depois os enriquece.
    """
    start_time = time.monotonic()
    SEARCH_API_URL = "https://api.casadosdados.com.br/v5/cnpj/pesquisa"
    
    try:
        # 1. Montar o corpo (payload) para a API de busca
        search_payload = {
            "codigo_atividade_principal": request.codigo_atividade_principal,
            "incluir_atividade_secundaria": request.incluir_atividade_secundaria,
            "codigo_atividade_secundaria": request.codigo_atividade_secundaria,
            "municipio": request.municipio,
            "situacao_cadastral": request.situacao_cadastral,
            "capital_social": {"minimo": request.capital_social_minimo},
            "mei": request.mei.dict(),
            "mais_filtros": request.mais_filtros.dict(),
            "limite": request.limite,
            "pagina": request.pagina,
            "resposta": request.resposta,
        }
        
        headers = {"accept": "application/json", "api-key": request.api_key}

        # 2. Fazer a chamada para a API de Busca
        logging.info(f"Job {job_id}: Buscando CNPJs com os filtros fornecidos...")
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(SEARCH_API_URL, json=search_payload, timeout=60) as response:
                response.raise_for_status()
                search_result = await response.json()

        cnpjs_data = search_result.get("cnpjs", [])
        if not cnpjs_data:
            duration_seconds = time.monotonic() - start_time
            logging.warning(f"Job {job_id}: A busca não retornou nenhum CNPJ.")
            job_storage[job_id] = {"status": "complete", "data": None, "message": "A busca com os filtros fornecidos não retornou nenhum CNPJ.", "duration_seconds": duration_seconds}
            return

        # Converter a resposta da busca para o formato EmpresaInput
        empresas_para_enriquecer = [EmpresaInput(**item) for item in cnpjs_data]
        logging.info(f"Job {job_id}: Busca retornou {len(empresas_para_enriquecer)} CNPJs. Iniciando enriquecimento...")

        # 3. Chamar a lógica de enriquecimento com a lista obtida
        csv_bytes = await enrich_empresas_list(empresas_para_enriquecer, request.api_key)
        duration_seconds = time.monotonic() - start_time
        
        # 4. Salvar o resultado final
        if csv_bytes:
            job_storage[job_id] = {"status": "complete", "data": csv_bytes, "duration_seconds": duration_seconds}
            logging.info(f"Job {job_id}: Tarefa de busca e enriquecimento concluída em {duration_seconds:.2f}s.")
        else:
            job_storage[job_id] = {"status": "complete", "data": None, "message": "Os CNPJs foram encontrados, mas nenhum dado pôde ser enriquecido.", "duration_seconds": duration_seconds}

    except aiohttp.ClientResponseError as e:
        duration_seconds = time.monotonic() - start_time
        error_message = f"Erro na API da Casa dos Dados (Busca): {e.status} - {e.message}"
        logging.error(f"Job {job_id}: {error_message}")
        job_storage[job_id] = {"status": "failed", "message": error_message, "duration_seconds": duration_seconds}
    except Exception as e:
        duration_seconds = time.monotonic() - start_time
        logging.error(f"Job {job_id}: Falha crítica na tarefa de busca e enriquecimento: {e}")
        job_storage[job_id] = {"status": "failed", "message": str(e), "duration_seconds": duration_seconds}

# --- APLICAÇÃO WEB (FastAPI) ---
app = FastAPI(
    title="Agente Enriquecedor e Buscador de Empresas",
    description="API para buscar CNPJs por filtros e enriquecê-los em massa, ou para enriquecer uma lista pré-existente.",
    version="4.1.0" # Versão incrementada
)

# ▼▼▼ ENDPOINT COMBINADO ▼▼▼
@app.post("/search-and-enrich", status_code=202, summary="Busca e Inicia o Processo de Enriquecimento")
async def start_search_and_enrich(request: SearchAndEnrichRequest, background_tasks: BackgroundTasks):
    """
    Recebe filtros de busca, encontra os CNPJs correspondentes, inicia o trabalho
    de enriquecimento em background e retorna um ID de trabalho.
    """
    job_id = str(uuid.uuid4())
    job_storage[job_id] = {"status": "processing"}
    
    background_tasks.add_task(process_search_and_enrich_task, job_id, request)
    
    status_url = app.url_path_for('get_result', job_id=job_id)
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Trabalho de busca e enriquecimento recebido. Verifique a URL de status para baixar o resultado.",
        "status_url": status_url
    }

# Endpoint original, mantido para retrocompatibilidade
@app.post("/enrich", status_code=202, summary="Inicia o Processo de Enriquecimento (Lista Manual)")
async def start_enrichment(request: EnrichmentRequest, background_tasks: BackgroundTasks):
    """
    Recebe uma lista manual de empresas, inicia o trabalho em background e 
    retorna um ID de trabalho.
    """
    if not request.empresas:
        raise HTTPException(status_code=400, detail="A lista de 'empresas' não pode estar vazia.")

    job_id = str(uuid.uuid4())
    job_storage[job_id] = {"status": "processing"}
    
    background_tasks.add_task(process_enrichment_task, job_id, request.empresas, request.api_key)
    
    status_url = app.url_path_for('get_result', job_id=job_id)
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Trabalho recebido. Verifique a URL de status para baixar o resultado.",
        "status_url": status_url
    }

# Endpoint de resultados, comum para ambos os fluxos
@app.get("/results/{job_id}", summary="Verifica o Status e Baixa o Resultado")
def get_result(job_id: str):
    """
    Verifica o status de um trabalho e retorna o arquivo CSV se estiver concluído.
    """
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
        
    return {"status": "complete", "message": job.get("message", "Processamento concluído, mas sem dados para retornar.")}
