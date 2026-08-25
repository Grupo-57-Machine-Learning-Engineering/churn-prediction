# API de inferência de churn (Etapa 3), imagem para deploy em container.
#
# O modelo NÃO entra na imagem. O .dockerignore exclui models/, e a API busca o
# campeão no Model Registry do MLflow/DagsHub durante o startup (ADR-006). Com
# isso a imagem carrega só código: promover um campeão novo é reiniciar o
# container, não reconstruir a imagem.
#
# Credenciais entram como variáveis de ambiente da plataforma, nunca no build.
# Ver ADR-009 e a seção "Rodando com Docker" do README.

FROM python:3.11-slim

# O uv instala a partir do uv.lock, então o container fica com exatamente as
# mesmas versões que o CI e a máquina de quem desenvolve. Vale trocar `latest`
# pela versão que o grupo usa (`uv --version`) quando quiserem build reproduzível
# também na ferramenta.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Usuário não-root com uid 1000: é o que o Hugging Face Spaces espera. O HOME
# precisa ser gravável porque o MLflow escreve o download do modelo em cache
# dentro dele no startup.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/api \
    && chown -R app:app /home/app

USER app
WORKDIR /home/app/api

ENV HOME=/home/app \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/home/app/api/.venv/bin:$PATH

# Dependências numa camada própria, antes do código: editar src/ não reinstala
# scikit-learn e mlflow de novo. O README.md vai junto porque o pyproject
# declara `readme = "README.md"` e o hatchling exige o arquivo no build.
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app src/ ./src/
RUN uv sync --frozen --no-dev

# 7860 é o padrão do Hugging Face Spaces e continua sendo o default. Plataforma que
# sorteia a porta (Render, Fly, Cloud Run) injeta PORT no ambiente, e aí a porta vem
# de lá. Ver ADR-009.
ENV PORT=7860
EXPOSE 7860

# Carência alta de propósito: o startup importa mlflow, sklearn e pandas e baixa o
# campeão do registry. Medido em 2,4s de CPU num core inteiro, o que numa instância
# de 0,1 vCPU vira dezenas de segundos. Sem essa folga a plataforma mata o container
# antes dele terminar de subir, e entra em loop de restart.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health')"

# Shell form com exec de propósito: sem o exec o uvicorn vira filho do /bin/sh e não
# recebe o SIGTERM da plataforma, o que transforma todo restart em kill por timeout.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
