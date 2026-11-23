FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app /home/user/.cache/huggingface \
    && chown -R user:user /home/user

ENV HOME=/home/user \
    PATH="/home/user/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/tmp/mpl \
    DEEPFAKE_NO_BROWSER=1 \
    VERIFAKE_DETECTOR=auto \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PORT=7860

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.15.0 torchvision==0.29.2 \
    && grep -vE '^(torch|torchvision)(==|$)' requirements.txt > /tmp/req_nocuda.txt \
    && pip install --no-cache-dir -r /tmp/req_nocuda.txt \
    && rm -f /tmp/req_nocuda.txt

USER user

RUN python -c "import timm; timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True, num_classes=0)"

COPY --chown=user:user . /home/user/app

RUN python -c "from pathlib import Path; p = Path('models/convnext_v2.pt'); assert p.exists() and p.stat().st_size > 1_000_000, f'Git LFS weights missing or unhydrated: {p} ({p.stat().st_size if p.exists() else 0} bytes)'"

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
