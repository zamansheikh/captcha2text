# Deploying Captcha2Text

Two ways to run the server in production: a Python virtualenv under systemd (what
[captcha.silifton.com](https://captcha.silifton.com) uses), or Docker. Either way, run
**one** server process: requests run on threads, and the per-key rate limiter lives in
memory.

## systemd + virtualenv

Good for a server that also runs other services: no Docker daemon, and no changes to the
firewall rules. The files are in [deploy/systemd/](../deploy/systemd/).

| | |
|---|---|
| Folders | `/opt/captcha2text/`: `app/` (code), `venv/`, `models/`, `data/` (keys, admins, usage; SQLite) |
| Models | `models/current` is the default model, `models/math` the `type=math` model (both symlinks) |
| Service | `captcha2text.service` on 127.0.0.1:7030, capped at 3 CPU cores and 3 GB |
| Proxy | nginx site `captcha2text`, HTTPS from certbot; Cloudflare in front is optional |
| Keys and admins | `c2t-keys ...` as root on the server |
| Logs | `journalctl -u captcha2text -f` |

### First-time setup

On the server (Ubuntu 24.04 shown), as root:

```bash
apt-get install -y python3-venv nginx certbot python3-certbot-nginx
useradd --system --home-dir /opt/captcha2text --shell /usr/sbin/nologin captcha2text
mkdir -p /opt/captcha2text/models /opt/captcha2text/data
chown captcha2text:captcha2text /opt/captcha2text/data && chmod 750 /opt/captcha2text/data
python3 -m venv /opt/captcha2text/venv
ln -sfn ddddocr-beta-v1 /opt/captcha2text/models/current
ln -sfn math-v1 /opt/captcha2text/models/math
```

On your machine, in this repo, with the server's SSH host name in `.deploy-host`
(gitignored) or `HOST`:

```bash
echo myserver > .deploy-host
scp deploy/systemd/captcha2text.service myserver:/etc/systemd/system/
ssh myserver 'systemctl daemon-reload && systemctl enable captcha2text'
deploy/systemd/deploy.sh ddddocr-beta-v1 math-v1     # code + the two model folders
```

On the server again, for nginx, the certificate and the first admin (put your domain in
`server_name` first):

```bash
cp /opt/captcha2text/app/deploy/systemd/captcha2text.nginx /etc/nginx/sites-available/captcha2text
ln -s /etc/nginx/sites-available/captcha2text /etc/nginx/sites-enabled/captcha2text
nginx -t && systemctl reload nginx
certbot --nginx -d captcha.example.com --redirect
c2t-keys admin-create <name>        # prints the admin password once
```

Back up `/opt/captcha2text/data`: it holds the API keys, admin logins, access requests
and the usage log.

### Updates

Only committed code is deployed (`git archive HEAD`). The script uploads the code and any
model folders you name, reinstalls, restarts and checks `/v1/health`:

```bash
deploy/systemd/deploy.sh                    # code only
deploy/systemd/deploy.sh acme-v1            # code + models/acme-v1
```

Point a symlink at another model folder, then restart:

```bash
ssh myserver 'ln -sfn <folder> /opt/captcha2text/models/current && systemctl restart captcha2text'
```

### Behind Cloudflare

If the DNS record is proxied by Cloudflare, use SSL mode "Full" or "Full (strict)", and
issue the certificate while "Always Use HTTPS" is off (certbot's HTTP check goes through
the proxy). The proxy also hides the server's IP address, so keep that address out of
public files.

## Docker

```bash
docker compose up -d --build
docker compose exec api c2t-keys admin-create <name>
```

This serves the API on 127.0.0.1:7030 with `./models` mounted read-only (put your model
folders and the `current` / `math` symlinks there) and the database in the `c2t-data`
volume. Put a reverse proxy with HTTPS in front; with nginx, set
`client_max_body_size 4m` (a 2 MB image as base64 is about 2.7 MB).
