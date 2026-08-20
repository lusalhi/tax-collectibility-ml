# 🚀 Deploy

Aplikasi Streamlit + model artefak, dikemas Docker dengan reverse proxy TLS.
App **tidak pernah** di-publish langsung; satu-satunya pintu adalah Caddy.

```
Petugas ──HTTPS──▶ Caddy (:443) ──▶ app:8501 (jaringan internal compose saja)
                    │
                    └─(opsional, profile "sso") oauth2-proxy untuk SSO/OIDC
```

## Prasyarat

- Docker Engine + Compose v2
- (Produksi) DNS internal, mis. `ketertagihan.internal.example.go.id` → IP server

## Uji coba lokal

```bash
docker compose up -d --build
docker compose exec caddy caddy trust   # trust CA internal Caddy (sekali per mesin klien)
# buka https://localhost
```

## Deploy internal (produksi)

1. Siapkan `.env` di root repo (jangan di-commit):

   ```env
   DOMAIN=ketertagihan.internal.example.go.id
   # hanya jika mengaktifkan SSO:
   OIDC_ISSUER=https://sso.internal.example.go.id/realms/djp
   OIDC_CLIENT_ID=ketertagihan
   OIDC_CLIENT_SECRET=...
   COOKIE_SECRET=<openssl rand -base64 32>
   ```

2. Jalankan:

   ```bash
   docker compose --profile sso up -d --build
   ```

3. Aktifkan blok `@public`/`forward_auth` di `deploy/Caddyfile` agar selain
   callback `/oauth2/*` semua request wajib lolos sesi oauth2-proxy.

## Operasional

| Hal | Perintah |
|---|---|
| Log app | `docker compose logs -f app` |
| Audit akses (JSON) | `docker compose exec caddy tail -f /data/access.log` |
| Update versi | `git pull && docker compose up -d --build` |
| Backup | artefak model ada di image; data upload tidak persisten by design |

## Catatan keamanan

- `models/` ikut ke dalam image — **jangan push image ke registry publik**.
- RAM limit 4 GB diset untuk agregasi kohir ±200rb baris; naikkan bila OOM.
- `no-store` + HSTS di level proxy; Streamlit juga tidak menyimpan upload.
- Sertifikat: Caddy auto-TLS (ACME untuk domain publik, CA internal untuk
  hostname `.internal`/IP pribadi).

## Tanpa Docker (fallback)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py   # bind 127.0.0.1:8501 sesuai .streamlit/config.toml
```
