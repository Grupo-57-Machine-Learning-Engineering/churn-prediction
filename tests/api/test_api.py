"""Testes da API FastAPI (Etapa 3).

O carregamento do modelo é substituído via monkeypatch para que a suíte
não dependa de `models/champion_model.joblib` (não versionado) nem de
rede. A fixture do campeão sintético está em `tests/conftest.py`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import THRESHOLD_DECISAO
from src.models.predict import Campeao, ModeloIndisponivelError

ORIGEM_DE_TESTE = "mlflow:churn_champion/42"


@pytest.fixture
def client(campeao_sintetico, monkeypatch):
    """TestClient com o campeão sintético injetado no lugar do artefato real."""
    monkeypatch.setattr(
        "src.api.main.carregar_campeao",
        lambda: Campeao(campeao_sintetico, ORIGEM_DE_TESTE),
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_sem_modelo(monkeypatch):
    """TestClient simulando ambiente sem joblib local e sem MLflow."""

    def _falha():
        raise ModeloIndisponivelError("sem artefato e sem MLflow no ambiente de teste")

    monkeypatch.setattr("src.api.main.carregar_campeao", _falha)
    with TestClient(app) as client:
        yield client


# --------------------------------------------------------------------------
# GET /health
# --------------------------------------------------------------------------


def test_health_retorna_ok(client):
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_health_nao_depende_do_modelo(client_sem_modelo):
    """Liveness continua de pé mesmo sem modelo (Contrato 3)."""
    resposta = client_sem_modelo.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# POST /predict
# --------------------------------------------------------------------------


def test_predict_payload_valido(client, payload_valido):
    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"churn", "probability", "model_version", "model_source"}
    assert 0.0 <= corpo["probability"] <= 1.0
    assert corpo["churn"] == (corpo["probability"] >= THRESHOLD_DECISAO)
    assert isinstance(corpo["model_version"], str) and corpo["model_version"]
    assert corpo["model_source"] == ORIGEM_DE_TESTE


def test_predict_cliente_sem_internet(client, payload_valido):
    """`services_internet_type: null` é entrada válida (Contrato 3)."""
    payload_valido["services_internet_type"] = None
    payload_valido["services_avg_monthly_gb_download"] = 0

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 200


def test_predict_aceita_categoria_nova(client, payload_valido):
    """Categoria que o modelo nunca viu segue pontuando (ADR-006).

    O score alimenta outros sistemas, então plano novo da operadora não pode
    virar indisponibilidade. Quem absorve o valor desconhecido é o
    `handle_unknown` do OneHotEncoder.
    """
    payload_valido["services_contract"] = "Weekly"

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 200
    assert 0.0 <= resposta.json()["probability"] <= 1.0


def test_predict_aceita_categoria_nova_em_qualquer_coluna(client, payload_valido):
    payload_valido["services_payment_method"] = "Pix"
    payload_valido["services_internet_type"] = "5G"

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 200


def test_predict_aceita_campo_ausente(client, payload_valido):
    """Ficha incompleta continua sendo pontuada (ADR-006)."""
    payload_valido.pop("demographics_age")
    payload_valido.pop("services_online_security")

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 200
    assert 0.0 <= resposta.json()["probability"] <= 1.0


def test_predict_aceita_payload_vazio(client):
    """Caso extremo do opcional: sem nenhuma informação ainda sai um número."""
    resposta = client.post("/predict", json={})

    assert resposta.status_code == 200
    assert 0.0 <= resposta.json()["probability"] <= 1.0


def test_predict_campo_ausente_equivale_a_nulo_explicito(client, payload_valido):
    """Omitir e mandar null são a mesma informação, então dão o mesmo número."""
    omitido = {k: v for k, v in payload_valido.items() if k != "demographics_gender"}
    explicito = dict(payload_valido, demographics_gender=None)

    r_omitido = client.post("/predict", json=omitido)
    r_explicito = client.post("/predict", json=explicito)

    assert r_omitido.status_code == r_explicito.status_code == 200
    assert r_omitido.json()["probability"] == r_explicito.json()["probability"]


def test_predict_valor_numerico_invalido_da_422(client, payload_valido):
    """Flexível com categoria nova, rígido com número que não descreve cliente."""
    payload_valido["demographics_age"] = -5

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 422


def test_predict_cobranca_negativa_da_422(client, payload_valido):
    payload_valido["services_monthly_charge"] = -10.0

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 422


def test_predict_tipo_errado_da_422(client, payload_valido):
    payload_valido["services_tenure_in_months"] = "doze"

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 422


def test_predict_rejeita_services_offer(client, payload_valido):
    """Coluna em quarentena por suspeita de vazamento nunca é input da API."""
    payload_valido["services_offer"] = "Offer E"

    resposta = client.post("/predict", json=payload_valido)

    assert resposta.status_code == 422


def test_predict_sem_modelo_da_503(client_sem_modelo, payload_valido):
    resposta = client_sem_modelo.post("/predict", json=payload_valido)

    assert resposta.status_code == 503
    assert "champion_model.joblib" in resposta.json()["detail"]


# --------------------------------------------------------------------------
# GET /sample
# --------------------------------------------------------------------------


def test_sample_retorna_amostra_pontuada(client):
    resposta = client.get("/sample")

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["total"] == len(corpo["clientes"]) > 0
    assert corpo["threshold"] == THRESHOLD_DECISAO

    for cliente in corpo["clientes"]:
        assert cliente["nome"] and cliente["descricao"]
        assert 0.0 <= cliente["resultado"]["probability"] <= 1.0
        assert cliente["resultado"]["churn"] == (
            cliente["resultado"]["probability"] >= THRESHOLD_DECISAO
        )


def test_sample_reporta_a_origem_do_campeao(client):
    """`model_source` diz de onde o modelo veio, no topo e em cada cliente.

    Com o registry na frente do joblib (ADR-006), o modelo servido pode mudar
    sem o pacote mudar de versão, então `model_version` sozinho não identifica
    quem respondeu.
    """
    corpo = client.get("/sample").json()

    assert corpo["model_source"] == ORIGEM_DE_TESTE
    assert {c["resultado"]["model_source"] for c in corpo["clientes"]} == {ORIGEM_DE_TESTE}


def test_sample_cobre_os_perfis_documentados(client):
    nomes = {c["nome"] for c in client.get("/sample").json()["clientes"]}

    assert {"ficha_incompleta", "categoria_nova", "sem_internet"} <= nomes


def test_sample_bate_com_predict(client):
    """A validação que o grupo pediu: o número do GET tem que sair igual no POST.

    Vale para todos os perfis da amostra, incluindo o de ficha incompleta e o
    de categoria desconhecida, que são os casos onde o caminho do dado é menos
    óbvio.
    """
    amostra = client.get("/sample").json()

    for cliente in amostra["clientes"]:
        resposta = client.post("/predict", json=cliente["payload"])

        assert resposta.status_code == 200, cliente["nome"]
        assert resposta.json() == cliente["resultado"], cliente["nome"]


def test_sample_sem_modelo_da_503(client_sem_modelo):
    resposta = client_sem_modelo.get("/sample")

    assert resposta.status_code == 503
    assert "champion_model.joblib" in resposta.json()["detail"]
