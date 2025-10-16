FROM sebffischer/anvil-cpu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN R -q -e "install.packages(c('torch','targets', 'data.table', 'batchtools', 'mirai', 'bench'))"

# Install CPU libtorch binaries for R torch
RUN R -q -e "torch::install_torch()"

CMD ["bash"]


