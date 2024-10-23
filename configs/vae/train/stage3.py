num_frames = 24
image_size = (256, 256)

# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=num_frames,
    frame_interval=1,
    image_size=image_size,
)

# Define acceleration
num_workers = 16
dtype = "fp16" # change from "bf16" to "fp16"
grad_checkpoint = True
plugin = "zero2"

# Define model
model = dict(
    type="OpenSoraVAE_V1_2",
    freeze_vae_2d=False,
    # from_pretrained="outputs/vae_stage2", # OSError: outputs/vae_stage2 is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    cal_loss=True,
)

# loss weights
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
kl_loss_weight = 1e-6

mixed_strategy = "mixed_video_random"
use_real_rec_loss = True
use_z_rec_loss = False
use_image_identity_loss = False

# Others
seed = 42
outputs = "outputs/vae_stage3"
wandb = False

epochs = 1000  # NOTE: adjust accordingly w.r.t dataset size
log_every = 100
ckpt_every = 2000
load = None

batch_size = 2
# lr = 1e-5 # batch_size = 1
lr = 2*1e-5
grad_clip = 1.0
