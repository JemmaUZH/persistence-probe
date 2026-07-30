#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${MINERL_BUILD_ROOT:-/tmp/minerl_codespaces_build}"
JDK8_DIR="${JDK8_DIR:-$HOME/.jdks/jdk8}"
MAVEN_ROOT="${MINERL_MAVEN_ROOT:-/tmp/minerl-maven}"
MIXIN_REPO="${MIXIN_REPO:-/tmp/MixinGradle-dcfaf61}"
ENV_FILE="$ROOT/.minerl-codespaces-env"
ASSET_PROXY="/tmp/mc_asset_proxy.py"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 8):
    raise SystemExit(f"MineRL 0.4.4 setup expects Python 3.8, got {sys.version.split()[0]}")
PY

python -m pip install --user --force-reinstall "pip==21.3.1" "setuptools==57.5.0" "wheel==0.37.1"
python -m pip install --user "gym==0.19.0"

if [ ! -x "$JDK8_DIR/bin/java" ]; then
    mkdir -p "$JDK8_DIR"
    curl -L "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jdk/hotspot/normal/eclipse" \
        -o /tmp/jdk8.tar.gz
    tar -xzf /tmp/jdk8.tar.gz -C "$JDK8_DIR" --strip-components=1
fi

if [ ! -d "$MIXIN_REPO/.git" ]; then
    git clone --depth 1 https://github.com/verityw/MixinGradle-dcfaf61.git "$MIXIN_REPO"
fi

mkdir -p "$MAVEN_ROOT/com/github/SpongePowered/MixinGradle/dcfaf61"
cp "$MIXIN_REPO/MixinGradle/dcfaf61/MixinGradle-dcfaf61.jar" \
    "$MAVEN_ROOT/com/github/SpongePowered/MixinGradle/dcfaf61/MixinGradle-dcfaf61.jar"
cat > "$MAVEN_ROOT/com/github/SpongePowered/MixinGradle/dcfaf61/MixinGradle-dcfaf61.pom" <<'POM'
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.github.SpongePowered</groupId>
  <artifactId>MixinGradle</artifactId>
  <version>dcfaf61</version>
</project>
POM

cat > "$ASSET_PROXY" <<'PY'
import socket, socketserver, ssl, threading, time

UPSTREAM_HOST = "mojang-resourcesdownloadminecra.azureedge.net"
ORIGIN_HOST = "resources.download.minecraft.net"
limit = threading.BoundedSemaphore(6)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        data = b""
        self.request.settimeout(180)
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            data += chunk
        try:
            first = data.split(b"\r\n", 1)[0].decode("latin1")
            method, path, _version = first.split(" ", 2)
        except Exception:
            return
        if path.startswith("http://"):
            i = path.find(ORIGIN_HOST)
            path = path[i + len(ORIGIN_HOST):] if i >= 0 else "/"
            if not path.startswith("/"):
                path = "/" + path
        req = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: {UPSTREAM_HOST}\r\n"
            "User-Agent: Java/1.8.0 MineRL-asset-proxy\r\n"
            "Accept: */*\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin1")
        last = None
        with limit:
            for attempt in range(30):
                try:
                    ctx = ssl.create_default_context()
                    with socket.create_connection((UPSTREAM_HOST, 443), timeout=30) as raw:
                        raw.settimeout(180)
                        with ctx.wrap_socket(raw, server_hostname=UPSTREAM_HOST) as upstream:
                            upstream.settimeout(180)
                            upstream.sendall(req)
                            sent = 0
                            while True:
                                chunk = upstream.recv(65536)
                                if not chunk:
                                    break
                                sent += len(chunk)
                                self.request.sendall(chunk)
                            if sent:
                                return
                except Exception as exc:
                    last = exc
                    time.sleep(min(5, 0.5 + attempt * 0.2))
        msg = (
            "HTTP/1.1 504 Gateway Timeout\r\n"
            "Connection: close\r\n"
            "Content-Type: text/plain\r\n\r\n"
            + repr(last)
        ).encode("latin1")
        try:
            self.request.sendall(msg)
        except Exception:
            pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


with Server(("0.0.0.0", 80), Handler) as server:
    print(f"proxy listening on :80 -> https://{UPSTREAM_HOST}", flush=True)
    server.serve_forever()
PY

if ! grep -q "127.0.0.1 resources.download.minecraft.net" /etc/hosts; then
    echo "127.0.0.1 resources.download.minecraft.net" | sudo tee -a /etc/hosts >/dev/null
fi
sudo fuser -k 80/tcp >/dev/null 2>&1 || true
sudo nohup python3 "$ASSET_PROXY" >/tmp/mc_asset_proxy.log 2>&1 &
sleep 2
curl -I --max-time 30 \
    http://resources.download.minecraft.net/9e/9ea8a9e105321891bda18b9007b383b40aa7c076 >/tmp/mc_asset_proxy_probe.log

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"
python -m pip download --no-deps --no-binary :all: minerl==0.4.4 -d .
tar -xzf minerl-0.4.4.tar.gz

python - <<PY
from pathlib import Path
path = Path("$BUILD_ROOT/minerl-0.4.4/minerl/Malmo/Minecraft/build.gradle")
text = path.read_text()
needle = "buildscript {\\n    repositories {\\n"
insert = needle + "        maven { url \\"file://$MAVEN_ROOT\\" }\\n"
if "file://$MAVEN_ROOT" not in text:
    text = text.replace(needle, insert, 1)
path.write_text(text)
PY

export JAVA_HOME="$JDK8_DIR"
export PATH="$JAVA_HOME/bin:$HOME/.local/bin:$PATH"
cd "$BUILD_ROOT/minerl-0.4.4"
python setup.py build

BUILD_LIB="$(python - <<'PY'
import sysconfig, sys
print(f"build/lib.{sysconfig.get_platform()}-{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
cat > "$ENV_FILE" <<EOF
export JAVA_HOME="$JDK8_DIR"
export PATH="\\$JAVA_HOME/bin:\\$HOME/.local/bin:\\$PATH"
export PYTHONPATH="$BUILD_ROOT/minerl-0.4.4/$BUILD_LIB:\\${PYTHONPATH:-}"
export LIBGL_ALWAYS_SOFTWARE=1
export JAVA_TOOL_OPTIONS="-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
EOF

cd "$ROOT"
source "$ENV_FILE"
python - <<'PY'
import gym
import minerl
print("gym", gym.__version__)
print("minerl", getattr(minerl, "__version__", "unknown"))
PY

echo "wrote $ENV_FILE"
echo "acceptance:"
echo "source .minerl-codespaces-env"
echo 'xvfb-run -a -s "-screen 0 1280x720x24 +extension RANDR +extension GLX +render" python scripts/minerl_smoke.py --env MineRLBasaltFindCave-v0 --steps 3 --output scratch/minerl_smoke'
