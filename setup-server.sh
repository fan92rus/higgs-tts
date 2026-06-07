#!/usr/bin/env bash
#===============================================================================
# setup-server.sh — полная настройка сервера для Higgs TTS
# Запускать от root или через sudo на сервере 192.168.2.96
#
# Использование:
#   curl -fsSL https://your-gitea/.../raw/branch/main/setup-server.sh | bash
#   или:
#   ssh dkrenev@192.168.2.96 'bash -s' < setup-server.sh
#===============================================================================
set -euo pipefail

# ─── Конфигурация ────────────────────────────────────────────────────────────
GITEA_URL="${GITEA_URL:-http://192.168.2.96:3000}"          # URL Gitea
GITEA_RUNNER_TOKEN="${GITEA_RUNNER_TOKEN:-}"                 # Токен регистрации раннера
HIGGS_REPO="${HIGGS_REPO:-higgs-tts}"                        # Имя репозитория
PORTAINER_PORT="${PORTAINER_PORT:-9000}"                     # Порт Portainer
HOST_IP="192.168.2.96"                                       # IP сервера

# ─── Цвета для вывода ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[✗]${NC} $1"; }
section() { echo; echo -e "${YELLOW}═══ $1 ═══${NC}"; }

# ─── Проверка прав ───────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    err "Запусти от root: sudo bash setup-server.sh"
    exit 1
fi

# ==============================================================================
# 1. Установка Docker
# ==============================================================================
section "1. Установка Docker"

if command -v docker &>/dev/null; then
    info "Docker уже установлен: $(docker --version)"
else
    warn "Docker не найден. Устанавливаю..."
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg lsb-release
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker
    info "Docker установлен: $(docker --version)"
fi

# Добавляем пользователя dkrenev в группу docker
if id "dkrenev" &>/dev/null; then
    usermod -aG docker dkrenev
    info "Пользователь dkrenev добавлен в группу docker"
fi

# ==============================================================================
# 2. Установка nvidia-container-toolkit
# ==============================================================================
section "2. Установка nvidia-container-toolkit (GPU)"

if command -v nvidia-smi &>/dev/null; then
    if docker info 2>/dev/null | grep -q "nvidia"; then
        info "nvidia-container-toolkit уже настроен"
    else
        warn "Устанавливаю nvidia-container-toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update -qq
        apt-get install -y -qq nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker
        info "nvidia-container-toolkit установлен"
    fi
else
    warn "NVIDIA драйверы не найдены. GPU-функции будут недоступны."
    warn "Установи драйверы: https://www.nvidia.com/download/index.aspx"
fi

# ==============================================================================
# 3. Создание Docker volumes
# ==============================================================================
section "3. Docker volumes"

for vol in higgs-models higgs-outputs higgs-cache; do
    if docker volume inspect "$vol" &>/dev/null; then
        info "Volume $vol уже существует"
    else
        docker volume create "$vol"
        info "Volume $vol создан"
    fi
done

# ==============================================================================
# 4. Установка Portainer (если не установлен)
# ==============================================================================
section "4. Portainer"

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "portainer"; then
    PORTAINER_PORT=$(docker port portainer 2>/dev/null | grep -oP '\d+' | head -1)
    info "Portainer уже запущен на порту ${PORTAINER_PORT:-9000}"
else
    warn "Устанавливаю Portainer..."
    docker volume create portainer_data 2>/dev/null || true
    docker run -d \
        --name portainer \
        --restart unless-stopped \
        -p "$PORTAINER_PORT:9000" \
        -p 9443:9443 \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v portainer_data:/data \
        portainer/portainer-ce:latest
    info "Portainer запущен на https://${HOST_IP}:9443 или http://${HOST_IP}:${PORTAINER_PORT}"
    info "При первом заходе создай пароль администратора"
fi

# ==============================================================================
# 5. Установка Gitea Runner
# ==============================================================================
section "5. Gitea Runner"

if [ -f /usr/local/bin/act_runner ]; then
    info "act_runner уже установлен"
else
    if [ -z "$GITEA_RUNNER_TOKEN" ]; then
        warn "GITEA_RUNNER_TOKEN не задан. Пропускаю установку раннера."
        warn "Создай токен в Gitea: Administration → Actions → Runners → Create New Runner"
        warn "и перезапусти скрипт с: GITEA_RUNNER_TOKEN='токен' bash setup-server.sh"
    else
        warn "Устанавливаю Gitea Runner..."
        
        # Скачиваем act_runner
        curl -sL "https://dl.gitea.com/act_runner/act_runner-linux-amd64" -o /usr/local/bin/act_runner
        chmod +x /usr/local/bin/act_runner
        
        # Создаём service-файл
        cat > /etc/systemd/system/act_runner.service << 'SERVICE'
[Unit]
Description=Gitea Actions Runner
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/local/bin/act_runner daemon --config /etc/act_runner/config.yaml
Restart=always
RestartSec=5
User=dkrenev
Group=docker

[Install]
WantedBy=multi-user.target
SERVICE
        
        # Регистрируем раннер
        mkdir -p /etc/act_runner
        /usr/local/bin/act_runner register \
            --instance "$GITEA_URL" \
            --token "$GITEA_RUNNER_TOKEN" \
            --name "higgs-gpu-runner" \
            --labels "self-hosted,linux,x64,gpu" \
            --no-interactive \
            -o /etc/act_runner/config.yaml
        
        systemctl daemon-reload
        systemctl enable --now act_runner
        info "Gitea Runner установлен и запущен"
    fi
fi

# ==============================================================================
# 6. Проверка итогов
# ==============================================================================
section "6. Итоговая проверка"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  СЕРВЕР ГОТОВ К РАБОТЕ"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Docker:         $(docker --version 2>/dev/null || echo 'N/A')"
echo "  GPU доступен:   $(docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi -L 2>/dev/null | head -1 || echo 'нет')"
echo "  Portainer:      http://${HOST_IP}:${PORTAINER_PORT}"
echo "  Gitea:          ${GITEA_URL}"
echo ""
echo "  Далее:"
echo "  1. Создай репозиторий в Gitea → ${GITEA_URL}/user/repositories"
echo "  2. Запусти на локальной машине:"
echo "     cd Higgs-Audio-v3-TTS-Portable-by-Neurogen"
echo "     git remote add origin ${GITEA_URL}/dkrenev/${HIGGS_REPO}.git"
echo "     git push -u origin main"
echo ""
echo "  3. В CI нужно настроить secrets (Settings → Actions → Secrets):"
echo "     PORTAINER_URL=http://${HOST_IP}:${PORTAINER_PORT}"
echo "     PORTAINER_USER=admin"
echo "     PORTAINER_PASSWORD=<твой пароль Portainer>"
echo "     PORTAINER_STACK_ID=<ID стека в Portainer>"
echo "     HOST_IP=${HOST_IP}"
echo "     REGISTRY_USER=<твой логин ghcr.io>"
echo "     REGISTRY_PASSWORD=<токен ghcr.io>"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
