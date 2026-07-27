#!/usr/bin/env bash
# Hook de pre-push: recusa qualquer push cujo destino seja a branch `main`.
#
# O git envia, via stdin, uma linha por ref sendo empurrada:
#   <local_ref> <local_sha> <remote_ref> <remote_sha>
# O pre-commit encaminha esse stdin para este script.
#
# Fluxo correto: develop -> PR -> main. Para emergência, dá pra pular com
# `git push --no-verify` (o workflow "Guard: no direct push to main" denuncia).
set -euo pipefail

protected="refs/heads/main"
blocked=0

while read -r _local_ref _local_sha remote_ref _remote_sha; do
  if [ "${remote_ref:-}" = "$protected" ]; then
    blocked=1
  fi
done

if [ "$blocked" -eq 1 ]; then
  echo "" >&2
  echo "Push direto na 'main' bloqueado." >&2
  echo "   Fluxo correto: feature/* -> develop  e  develop -> PR -> main." >&2
  echo "   (Emergência: 'git push --no-verify' -- mas o CI vai denunciar.)" >&2
  echo "" >&2
  exit 1
fi

exit 0
