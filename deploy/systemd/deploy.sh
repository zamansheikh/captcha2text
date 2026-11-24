#!/usr/bin/env bash
# Deploy the committed code, and optionally model folders, to a server set up as in
# docs/DEPLOY.md (a virtualenv under systemd in /opt/captcha2text).
#
#   deploy/systemd/deploy.sh                    # code only
#   deploy/systemd/deploy.sh math-v1            # code + these folders from models/
#
# The server is the SSH host in $HOST, or the first line of .deploy-host (gitignored):
#   echo myserver > .deploy-host
# Only committed code is deployed (git archive of HEAD). To change the default model:
#   ssh <host> 'ln -sfn <folder> /opt/captcha2text/models/current && systemctl restart captcha2text'
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
HOST="${HOST:-$(head -n 1 .deploy-host 2>/dev/null || true)}"
if [ -z "$HOST" ]; then
  echo "Which server? Run HOST=<ssh host> $0, or put the SSH host name in .deploy-host." >&2
  exit 1
fi
ROOT=/opt/captcha2text

if [ -n "$(git status --porcelain)" ]; then
  echo "Commit your changes first: only committed code is deployed." >&2
  exit 1
fi
REV=$(git rev-parse --short HEAD)

for model in "$@"; do
  [ -f "models/$model/model.json" ] || { echo "No models/$model/model.json" >&2; exit 1; }
done

echo "Uploading code at $REV"
git archive --format=tar HEAD | ssh "$HOST" \
  "set -e; rm -rf $ROOT/app.new; mkdir -p $ROOT/app.new; tar -x -C $ROOT/app.new; echo $REV > $ROOT/app.new/REVISION"

for model in "$@"; do
  echo "Uploading model $model"
  rsync -a --delete --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "models/$model/" "$HOST:$ROOT/models/$model/"
done

ssh "$HOST" "set -e
  $ROOT/venv/bin/pip install --quiet --disable-pip-version-check $ROOT/app.new
  $ROOT/venv/bin/pip install --quiet --disable-pip-version-check --force-reinstall --no-deps $ROOT/app.new
  install -m 0755 $ROOT/app.new/deploy/systemd/c2t-keys /usr/local/bin/c2t-keys
  rm -rf $ROOT/app.old
  if [ -d $ROOT/app ]; then mv $ROOT/app $ROOT/app.old; fi
  mv $ROOT/app.new $ROOT/app
  systemctl restart captcha2text
  for i in \$(seq 1 30); do
    if curl -sf http://127.0.0.1:7030/v1/health; then echo; echo 'Deployed $REV'; exit 0; fi
    sleep 1
  done
  echo 'Health check failed:' >&2
  journalctl -u captcha2text -n 30 --no-pager >&2
  exit 1"
