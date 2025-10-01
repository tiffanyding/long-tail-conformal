mkdir train_models/data
cd train_models/data

# Download files
gdown 1k_PPQV3VJT44hz02CcnbqPstjQo70vGr # PlantNet-300K (0.42 GB)
gdown 1W8R8Jj2bhS2PbR-3X9vEw-WkanbOk6mq # iNaturalist-2018 (4.64 GB)
gdown 1a0SF6xbMDwxmAde2VgBoP4q_qYX2hslm # PlantNet-300K-truncated  (0.12 GB)
gdown 1patL2K450vwiI4DGugWlCPXh-j6y6uCF # iNaturalist-2018-truncated  (0.62 GB)
gdown 1rky60OUmL9imLBuThe4quNIVUWoqeSwp # Focal loss versions for PlantNet300K and iNaturalist-2018 (5.03 GB)

# Unzip files
unzip plantnet.zip
unzip plantnet-trunc.zip
unzip inaturalist.zip
unzip inaturalist-trunc.zip
unzip focal_loss.zip
cd ../.. 
