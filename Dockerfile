FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

WORKDIR /tmp/unique_for_apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
WORKDIR /workspace

RUN conda install --quiet --yes \ 
    tqdm \
    numpy \
    pandas \
    seaborn \
    && conda clean -ya

RUN conda install --quiet --yes -c conda-forge \ 
    opencv \
    tensorboardx \
    && conda clean -ya

RUN pip install \
    fairseq \ 
    viseq \
    fastBPE \
    sacremoses \
    sacrebleu==1.4.5 \

    COPY ./src ./src/

CMD [ "python", "src/rosita.py", "--help" ]
