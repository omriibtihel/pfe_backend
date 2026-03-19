#!/bin/sh
set -e

echo "Waiting for PostgreSQL..."
python - <<'EOF'
import time, sys, os
from urllib.parse import urlparse
import psycopg2

raw = os.environ["DATABASE_URL"]
if "+" in raw.split("://")[0]:
    raw = raw.split("+")[0] + "://" + raw.split("://", 1)[1]

p = urlparse(raw)
kwargs = dict(
    host=p.hostname,
    port=p.port or 5432,
    user=p.username,
    password=p.password,
    dbname=p.path.lstrip("/"),
    connect_timeout=3,
)

for i in range(30):
    try:
        psycopg2.connect(**kwargs).close()
        print("DB ready!")
        sys.exit(0)
    except Exception as e:
        print(f"  attempt {i+1}/30: {e}")
        time.sleep(2)

print("DB never became ready.")
sys.exit(1)
EOF

echo "Running Alembic migrations..."
alembic upgrade head

echo "Seeding admin user..."
python - <<'EOF'
import os, psycopg2
from urllib.parse import urlparse
from passlib.context import CryptContext

raw = os.environ["DATABASE_URL"]
if "+" in raw.split("://")[0]:
    raw = raw.split("+")[0] + "://" + raw.split("://", 1)[1]

p = urlparse(raw)
conn = psycopg2.connect(
    host=p.hostname, port=p.port or 5432,
    user=p.username, password=p.password,
    dbname=p.path.lstrip("/"),
)
cur = conn.cursor()
cur.execute("SELECT 1 FROM users WHERE email = 'admin@medicalvision.com'")
if cur.fetchone() is None:
    pwd = CryptContext(schemes=["bcrypt"]).hash("Admin@1234")
    cur.execute(
        "INSERT INTO users (full_name, email, password_hash, role, status) VALUES (%s,%s,%s,%s,%s)",
        ("Admin", "admin@medicalvision.com", pwd, "ADMIN", "APPROVED"),
    )
    conn.commit()
    print("Admin created: admin@medicalvision.com / Admin@1234")
else:
    print("Admin already exists, skipping.")
conn.close()
EOF

echo "Starting server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
