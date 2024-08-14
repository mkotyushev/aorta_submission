FROM --platform=linux/amd64 pytorch/pytorch


ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Add the directory containing the scripts to PATH
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


COPY --chown=user:user requirements.txt /opt/app/

### mkotyushev deps

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

### rostepifanov deps

USER root
RUN apt-get update && apt-get install -y git
USER user

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-deps \
    git+https://github.com/rostepifanov/nnspt.git@release_v0.0.2

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --upgrade \
    timm==0.9.2 connected-components-3d==3.18.0

### end deps

### mkotyushev codes

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user convert_2d_to_3d.py /opt/app/

### rostepifanov codes

COPY --chown=user:user convertors.py /opt/app/
 
### end codes

# ENTRYPOINT ["python", "-m", "pip", "list", "--format", "freeze"]
ENTRYPOINT ["python", "inference.py"]
