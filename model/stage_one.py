import torch
from torch import nn
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from model.noise_scheduler import NoiseSchedule_Scalar, NoiseSchedule_Fixed
import numpy as np


class DistillModel(nn.Module):
    def __init__(self,
                 **config):
        """
        Initialize the student model with the teacher model  weights.

        :param pipeline: the DiffusionPipeline from HuggingFace. the teacher model
            corresponds to the pipeline.unet attribute.
        """
        super().__init__()

        # Load pretrained teacher model
        pipeline = StableDiffusionPipeline.from_pretrained(config['teacher_model_id'])
        # initialize the student model
        teacher_model = pipeline.unet
        student_config = teacher_model.config.copy()
        # student_config['in_channels'] *= 3  # additional channels for Fourier features
        student_config['time_cond_proj_dim'] = 256  # for adding guidance weight as a condition
        self.model = UNet2DConditionModel(**student_config)

        # initialize the student model with the teacher model weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in teacher_model.state_dict():
                    if name == 'conv_in.weight':
                        # do not inherit the weights corresponding to Fourier features
                        param[:, :4, :, :].copy_(teacher_model.state_dict()[name])
                    else:
                        param.copy_(teacher_model.state_dict()[name])

            # in the stable diffusion case, the conv_in has 4 input channels
            nn.init.xavier_uniform_(self.model.conv_in.weight.data[:, 4:, :, :])
            nn.init.zeros_(self.model.conv_in.bias.data)

        self.min_guidance = config['min_guidance']
        self.max_guidance = config['max_guidance']
        self.teacher_pipeline = pipeline
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.vae = pipeline.vae
        self.noise_schedule = NoiseSchedule_Fixed(config)
        self.config = config

    def forward(self, img, context):
        """
        :param z: the noisy input in the diffusion process
        :param w: the guidance scale
        :param t: the time step
        :param context: the original text prompts
        """
        # get latent state
        z = self.vae.encode(img).latent_dist.sample()

        # sample time, guidance, and noise
        t = torch.rand(())
        w = torch.rand((1)) * (self.max_guidance - self.min_guidance) + self.min_guidance
        noise = torch.normal(0, 1, z.shape, device=z.device)

        # postprocess the noise schedule to get alpha and sigma
        alpha_sq = self.noise_schedule(t)
        sigma_sq = 1 - alpha_sq

        # add noise to the input
        z = torch.sqrt(alpha_sq) * z + torch.sqrt(sigma_sq) * noise

        # log-SNR
        log_snr = torch.log(alpha_sq / sigma_sq)

        # if self.config['with_fourier_features']:
        #     # apply Fourier features to the input
        #     sin_features = torch.sin(z * 2 ** w * torch.pi)
        #     cos_features = torch.cos(z * 2 ** w * torch.pi)
        #     z = torch.cat([z, sin_features, cos_features], dim=1)

        # convert the text prompts to hidden states
        tokens = self.tokenizer(context, return_tensors='pt', padding=True)[
            'input_ids'].to(z.device)

        c = self.text_encoder(tokens)['last_hidden_state']

        # concatenate Fourier-transformed w to the hidden states
        w_fourier = self.get_w_embedding(w, embedding_dim=256, dtype=c.dtype)

        z_denoised = self.model(sample=z,
                                timestep=t,
                                timestep_cond=w_fourier,
                                encoder_hidden_states=c).sample

        recon = self.vae.decode(z_denoised).sample

        # constant loss weighting for stage one distillation
        lambda_t = torch.exp(log_snr)

        # compute target
        with torch.no_grad():
            teacher_output = self.teacher_pipeline(prompt=context,
                                                   latents=z,
                                                   num_inference_steps=self.config['num_inference_step'],
                                                   guidance_scale=w,
                                                   output_type=np)

        # make target match the dimensions (batch, channel, height, width)
        target = torch.tensor(teacher_output.images).permute(0, 3, 1, 2).to(recon.device)

        # filter out invalid pairs
        target = target[~np.array(teacher_output.nsfw_content_detected)]
        recon = recon[~np.array(teacher_output.nsfw_content_detected)]

        return recon, lambda_t, target

    def loss_function(self, recon, lambda_t, target):
        try:
            assert recon.shape == target.shape
        except AssertionError:
            print(f'recon shape {recon.shape} does not match with target shape {target.shape}')

        loss = torch.nn.functional.mse_loss(recon, target) * lambda_t
        return {'loss': loss, 'batch_size': recon.shape[0]}

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings

        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
