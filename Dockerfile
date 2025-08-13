# 使用轻量级 Python 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖，只在第一次构建需要编译 LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt 并安装依赖（利用 Docker 缓存）
COPY requirements.txt .

# 使用阿里云镜像加速安装
RUN pip install --upgrade pip \
    && pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r requirements.txts

# 再复制整个项目代码，方便实时挂载调试
COPY . .

# 默认工作命令，可用 docker-compose 覆盖为交互模式
CMD ["python", "main.py"]
