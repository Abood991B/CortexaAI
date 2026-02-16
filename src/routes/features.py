"""Feature endpoints: templates, batch, marketplace, builder, plugins, regression,
similarity, complexity, language, auth, finetuning, webhooks."""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from pydantic import BaseModel, Field

from src.deps import (
    logger,
    coordinator,
    template_engine,
    batch_processor,
    marketplace,
    prompt_builder,
    plugin_manager,
    regression_runner,
    similarity_engine,
    complexity_analyzer,
    language_processor,
    auth_manager,
    finetuning_manager,
    webhook_manager,
)
from core.auth import require_api_key

router = APIRouter()


# ── Admin authentication dependency ───────────────────────────────────────
async def _require_admin(request: Request):
    """Dependency that enforces admin scope when auth is enabled."""
    key_info = await require_api_key(request)
    # If auth is disabled (key_info is None), allow through
    if key_info is None:
        return key_info
    if isinstance(key_info, dict) and key_info.get("error") == "rate_limited":
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if isinstance(key_info, dict) and "admin" not in key_info.get("scopes", []):
        raise HTTPException(status_code=403, detail="Admin scope required for this operation")
    return key_info


# ═══════════════════════════════════════════════════════════════════════════
#  Batch Processing
# ═══════════════════════════════════════════════════════════════════════════

class BatchRequest(BaseModel):
    prompts: List[Dict[str, Any]]
    concurrency: int = Field(default=3, ge=1, le=10)


@router.post("/api/batch")
async def create_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    job = batch_processor.create_batch(req.prompts, req.concurrency)
    batch_id = job["batch_id"]

    async def _processor(prompt_text, prompt_type="auto"):
        return await coordinator.process_prompt(prompt=prompt_text, prompt_type=prompt_type)

    background_tasks.add_task(batch_processor.run_batch, batch_id, _processor)
    return job


@router.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    status = batch_processor.get_status(batch_id)
    if not status:
        raise HTTPException(404, "Batch not found")
    return status


@router.get("/api/batch/{batch_id}/results")
async def get_batch_results(batch_id: str):
    results = batch_processor.get_results(batch_id)
    if not results:
        raise HTTPException(404, "Batch not found")
    return results


@router.get("/api/batches")
async def list_batches():
    return batch_processor.list_batches()


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════

class TemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    domain: str = Field(..., min_length=1, max_length=100)
    template_text: str = Field(..., min_length=1, max_length=50000)
    description: str = Field(default="", max_length=2000)
    variables: Optional[List[str]] = None
    is_public: bool = True


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    domain: Optional[str] = None
    template_text: Optional[str] = None
    description: Optional[str] = None
    variables: Optional[List[str]] = None
    is_public: Optional[bool] = None


class TemplateRenderRequest(BaseModel):
    template_id: str
    variables: Dict[str, str] = {}


@router.get("/api/templates")
async def list_templates(domain: Optional[str] = None, query: Optional[str] = None):
    if query:
        return template_engine.search(query)
    return template_engine.list_all(domain)


@router.post("/api/templates")
async def create_template(req: TemplateCreateRequest):
    return template_engine.create(
        name=req.name,
        domain=req.domain,
        template_text=req.template_text,
        description=req.description,
        variables=req.variables,
        is_public=req.is_public,
    )


@router.get("/api/templates/{template_id}")
async def get_template(template_id: str):
    t = template_engine.get(template_id)
    if not t:
        raise HTTPException(404, "Template not found")
    return t


@router.put("/api/templates/{template_id}")
async def update_template(template_id: str, req: TemplateUpdateRequest):
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    if "template_text" in data:
        data["template"] = data.pop("template_text")
    result = template_engine.update_template(template_id, data)
    if result is None:
        raise HTTPException(404, "Template not found")
    if "error" in result:
        raise HTTPException(403, result["error"])
    return result


@router.delete("/api/templates/{template_id}")
async def delete_template(template_id: str):
    success = template_engine.delete_template(template_id)
    if not success:
        raise HTTPException(404, "Template not found or is a system template")
    return {"status": "deleted", "id": template_id}


@router.post("/api/templates/render")
async def render_template(req: TemplateRenderRequest):
    result = template_engine.render(req.template_id, req.variables)
    if not result:
        raise HTTPException(404, "Template not found")
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Complexity Scoring
# ═══════════════════════════════════════════════════════════════════════════

class ComplexityRequest(BaseModel):
    text: str = Field(..., max_length=100000)


@router.post("/api/complexity")
async def analyze_complexity(req: ComplexityRequest):
    return complexity_analyzer.analyze(req.text)


@router.post("/api/complexity/pipeline-config")
async def get_pipeline_config(req: ComplexityRequest):
    return complexity_analyzer.get_pipeline_config(req.text)


# ═══════════════════════════════════════════════════════════════════════════
#  Language Detection & Processing
# ═══════════════════════════════════════════════════════════════════════════

class LanguageRequest(BaseModel):
    text: str = Field(..., max_length=100000)


@router.post("/api/language/detect")
async def detect_language(req: LanguageRequest):
    return language_processor.analyze(req.text)


@router.get("/api/language/supported")
async def supported_languages():
    return language_processor.get_supported_languages()


# ═══════════════════════════════════════════════════════════════════════════
#  API Key Auth Management
# ═══════════════════════════════════════════════════════════════════════════

class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    scopes: Optional[List[str]] = None
    rate_limit_rpm: Optional[int] = Field(default=None, ge=1, le=10000)


@router.post("/api/auth/keys")
async def create_api_key(req: APIKeyCreateRequest, _admin=Depends(_require_admin)):
    # Enforce auth strictly - even if auth is disabled, creating keys requires intent
    if _admin is None:
        raise HTTPException(403, "API key management requires authentication to be enabled")
    return auth_manager.create_key(req.name, req.scopes, req.rate_limit_rpm)


@router.get("/api/auth/keys")
async def list_api_keys(_admin=Depends(_require_admin)):
    return auth_manager.list_keys()


@router.delete("/api/auth/keys/{name}")
async def revoke_api_key(name: str, _admin=Depends(_require_admin)):
    auth_manager.revoke_key(name)
    return {"status": "revoked", "name": name}


@router.post("/api/auth/verify")
async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    result = auth_manager.verify_key(key)
    if not result:
        raise HTTPException(401, "Invalid or inactive API key")
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Marketplace
# ═══════════════════════════════════════════════════════════════════════════

class MarketplacePublishRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., min_length=1, max_length=5000)
    prompt_text: str = Field(..., min_length=1, max_length=50000)
    domain: str = Field(..., min_length=1, max_length=100)
    author: str = Field(default="anonymous", max_length=200)
    tags: Optional[List[str]] = None
    price: float = Field(default=0.0, ge=0.0)


class MarketplaceRateRequest(BaseModel):
    stars: int = Field(ge=1, le=5)


@router.get("/api/marketplace")
async def marketplace_search(
    query: Optional[str] = None,
    domain: Optional[str] = None,
    sort_by: str = "downloads",
    limit: int = 20,
    offset: int = 0,
):
    return marketplace.search(query, domain, sort_by, limit, offset)


@router.post("/api/marketplace")
async def marketplace_publish(req: MarketplacePublishRequest, _admin=Depends(_require_admin)):
    return marketplace.publish(
        title=req.title,
        description=req.description,
        prompt_text=req.prompt_text,
        domain=req.domain,
        author=req.author,
        tags=req.tags,
        price=req.price,
    )


@router.get("/api/marketplace/{item_id}")
async def marketplace_download(item_id: str):
    item = marketplace.download(item_id)
    if not item:
        raise HTTPException(404, "Marketplace item not found")
    return item


@router.post("/api/marketplace/{item_id}/rate")
async def marketplace_rate(item_id: str, req: MarketplaceRateRequest):
    result = marketplace.rate(item_id, req.stars)
    if not result:
        raise HTTPException(404, "Marketplace item not found")
    return result


@router.get("/api/marketplace/stats/overview")
async def marketplace_stats():
    return marketplace.stats()


@router.get("/api/marketplace/featured/list")
async def marketplace_featured(limit: int = 10):
    return marketplace.featured(limit)


# ═══════════════════════════════════════════════════════════════════════════
#  Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════

class FineTuneJobRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini-2024-07-18"
    hyperparameters: Optional[Dict[str, Any]] = None


class TrainingDataRequest(BaseModel):
    prompts: List[Dict[str, str]]
    format_type: str = "openai"


@router.post("/api/finetuning/prepare")
async def prepare_training_data(req: TrainingDataRequest):
    return finetuning_manager.prepare_training_data(req.prompts, req.format_type)


@router.post("/api/finetuning/jobs")
async def create_finetune_job(req: FineTuneJobRequest):
    return finetuning_manager.create_job(req.provider, req.model, hyperparameters=req.hyperparameters)


@router.get("/api/finetuning/jobs")
async def list_finetune_jobs():
    return finetuning_manager.list_jobs()


@router.get("/api/finetuning/jobs/{job_id}")
async def get_finetune_job(job_id: str):
    job = finetuning_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Fine-tuning job not found")
    return job


@router.post("/api/finetuning/jobs/{job_id}/simulate")
async def simulate_finetune(job_id: str):
    result = finetuning_manager.simulate_run(job_id)
    if not result:
        raise HTTPException(404, "Fine-tuning job not found")
    return result


@router.get("/api/finetuning/models/{provider}")
async def get_finetune_models(provider: str):
    return finetuning_manager.get_supported_models(provider)


@router.get("/api/finetuning/estimate")
async def estimate_finetune_cost(provider: str = "openai", samples: int = 100, epochs: int = 3):
    return finetuning_manager.estimate_cost(provider, samples, epochs)


# ═══════════════════════════════════════════════════════════════════════════
#  Visual Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════

class BuilderBlockRequest(BaseModel):
    block_type: str
    content: str
    label: Optional[str] = None
    order: Optional[int] = None


class BuilderAssembleRequest(BaseModel):
    variables: Optional[Dict[str, str]] = None


class BuilderReorderRequest(BaseModel):
    block_ids: List[str]


@router.post("/api/builder/sessions")
async def create_builder_session(domain: Optional[str] = None):
    return prompt_builder.create_session(domain)


@router.get("/api/builder/sessions/{session_id}")
async def get_builder_session(session_id: str):
    s = prompt_builder.get_session(session_id)
    if not s:
        raise HTTPException(404, "Builder session not found")
    return s


@router.post("/api/builder/sessions/{session_id}/blocks")
async def add_builder_block(session_id: str, req: BuilderBlockRequest):
    result = prompt_builder.add_block(session_id, req.block_type, req.content, req.label, req.order)
    if not result:
        raise HTTPException(404, "Session not found")
    return result


@router.put("/api/builder/sessions/{session_id}/blocks/{block_id}")
async def update_builder_block(session_id: str, block_id: str, req: BuilderBlockRequest):
    result = prompt_builder.update_block(session_id, block_id, req.content)
    if not result:
        raise HTTPException(404, "Block or session not found")
    return result


@router.delete("/api/builder/sessions/{session_id}/blocks/{block_id}")
async def remove_builder_block(session_id: str, block_id: str):
    if not prompt_builder.remove_block(session_id, block_id):
        raise HTTPException(404, "Block or session not found")
    return {"status": "removed"}


@router.post("/api/builder/sessions/{session_id}/reorder")
async def reorder_builder_blocks(session_id: str, req: BuilderReorderRequest):
    result = prompt_builder.reorder_blocks(session_id, req.block_ids)
    if not result:
        raise HTTPException(404, "Session not found")
    return result


@router.post("/api/builder/sessions/{session_id}/assemble")
async def assemble_prompt(session_id: str, req: BuilderAssembleRequest):
    result = prompt_builder.assemble(session_id, req.variables)
    if not result:
        raise HTTPException(404, "Session not found")
    return result


@router.get("/api/builder/presets")
async def list_builder_presets():
    return prompt_builder.list_presets()


@router.get("/api/builder/presets/{domain}")
async def get_builder_preset(domain: str):
    preset = prompt_builder.get_preset(domain)
    if not preset:
        raise HTTPException(404, "Preset not found")
    return preset


# ═══════════════════════════════════════════════════════════════════════════
#  Plugin System
# ═══════════════════════════════════════════════════════════════════════════

class PluginRegisterRequest(BaseModel):
    name: str
    version: str = "1.0.0"
    plugin_type: str = "expert"
    description: str = ""
    author: str = ""
    config: Optional[Dict[str, Any]] = None


@router.get("/api/plugins")
async def list_plugins():
    return plugin_manager.list_plugins()


@router.post("/api/plugins")
async def register_plugin(req: PluginRegisterRequest):
    return plugin_manager.register(
        req.name, req.version, req.plugin_type, req.description, req.author, req.config,
    )


@router.get("/api/plugins/{name}")
async def get_plugin(name: str):
    p = plugin_manager.get_plugin(name)
    if not p:
        raise HTTPException(404, "Plugin not found")
    return p


@router.delete("/api/plugins/{name}")
async def unregister_plugin(name: str):
    if not plugin_manager.unregister(name):
        raise HTTPException(404, "Plugin not found")
    return {"status": "removed"}


@router.post("/api/plugins/{name}/enable")
async def enable_plugin(name: str):
    if not plugin_manager.enable(name):
        raise HTTPException(404, "Plugin not found")
    return {"status": "enabled"}


@router.post("/api/plugins/{name}/disable")
async def disable_plugin(name: str):
    if not plugin_manager.disable(name):
        raise HTTPException(404, "Plugin not found")
    return {"status": "disabled"}


# ═══════════════════════════════════════════════════════════════════════════
#  Regression Testing
# ═══════════════════════════════════════════════════════════════════════════

class RegressionSuiteRequest(BaseModel):
    name: str
    domain: str
    description: str = ""
    test_cases: Optional[List[Dict[str, Any]]] = None


class RegressionCaseRequest(BaseModel):
    input_prompt: str
    expected_keywords: Optional[List[str]] = None
    min_score: float = 0.7


@router.get("/api/regression/suites")
async def list_regression_suites():
    return regression_runner.list_suites()


@router.post("/api/regression/suites")
async def create_regression_suite(req: RegressionSuiteRequest):
    return regression_runner.create_suite(req.name, req.domain, req.description, req.test_cases)


@router.get("/api/regression/suites/{suite_id}")
async def get_regression_suite(suite_id: str):
    s = regression_runner.get_suite(suite_id)
    if not s:
        raise HTTPException(404, "Suite not found")
    return s


@router.delete("/api/regression/suites/{suite_id}")
async def delete_regression_suite(suite_id: str):
    regression_runner.delete_suite(suite_id)
    return {"status": "deleted"}


@router.post("/api/regression/suites/{suite_id}/cases")
async def add_regression_case(suite_id: str, req: RegressionCaseRequest):
    result = regression_runner.add_test_case(suite_id, req.input_prompt, req.expected_keywords, req.min_score)
    if not result:
        raise HTTPException(404, "Suite not found")
    return result


@router.post("/api/regression/suites/{suite_id}/run")
async def run_regression_suite(suite_id: str):
    # Validate suite_id format
    suite = regression_runner.get_suite(suite_id)
    if not suite:
        raise HTTPException(404, "Suite not found")

    async def _processor(prompt_text):
        result = await coordinator.process_prompt(prompt=prompt_text, prompt_type="auto")
        return {
            "improved_prompt": result.get("output", {}).get("optimized_prompt", ""),
            "evaluation_score": result.get("output", {}).get("quality_score", 0),
        }

    return await regression_runner.run_suite(suite_id, _processor)


@router.post("/api/regression/suites/{suite_id}/baseline")
async def save_regression_baseline(suite_id: str, run_results: Dict[str, Any]):
    regression_runner.save_baseline(suite_id, run_results)
    return {"status": "baseline_saved"}


# ═══════════════════════════════════════════════════════════════════════════
#  Similarity Search
# ═══════════════════════════════════════════════════════════════════════════

class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., max_length=50000)
    top_k: int = Field(default=5, ge=1, le=50)
    domain: Optional[str] = None
    min_score: float = 0.0


class SimilarityIndexRequest(BaseModel):
    doc_id: str = Field(..., max_length=200)
    text: str = Field(..., max_length=100000)
    domain: str = ""
    metadata: Optional[Dict[str, Any]] = None


@router.post("/api/similarity/search")
async def similarity_search(req: SimilaritySearchRequest):
    return similarity_engine.search(req.query, req.top_k, req.domain, req.min_score)


@router.post("/api/similarity/index")
async def similarity_index(req: SimilarityIndexRequest):
    return similarity_engine.add_document(req.doc_id, req.text, req.domain, req.metadata)


@router.get("/api/similarity/duplicates")
async def find_duplicates(threshold: float = 0.85):
    return similarity_engine.find_duplicates(threshold)


@router.post("/api/similarity/reindex")
async def reindex_similarity(limit: int = 500):
    count = similarity_engine.index_from_history(limit)
    return {"indexed": count, "corpus_size": similarity_engine.stats()["corpus_size"]}


@router.get("/api/similarity/stats")
async def similarity_stats():
    return similarity_engine.stats()


# ═══════════════════════════════════════════════════════════════════════════
#  Webhook Management
# ═══════════════════════════════════════════════════════════════════════════

class WebhookSubscribeRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    events: Optional[List[str]] = None
    secret: Optional[str] = Field(default=None, max_length=500)
    name: str = Field(default="", max_length=200)


@router.post("/api/webhooks")
async def subscribe_webhook(req: WebhookSubscribeRequest, _admin=Depends(_require_admin)):
    return webhook_manager.subscribe(req.url, req.events, req.secret, req.name)


@router.get("/api/webhooks")
async def list_webhooks():
    return webhook_manager.list_subscriptions()


@router.delete("/api/webhooks/{sub_id}")
async def unsubscribe_webhook(sub_id: str, _admin=Depends(_require_admin)):
    if not webhook_manager.unsubscribe(sub_id):
        raise HTTPException(404, "Subscription not found")
    return {"status": "removed"}


@router.get("/api/webhooks/log")
async def webhook_delivery_log(limit: int = 50):
    return webhook_manager.get_delivery_log(limit)
