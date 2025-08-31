from huggingface_hub import snapshot_download

# You can change the save path as you want.
snapshot_download("cmots/UniSS", local_dir="pretrained_models/UniSS")