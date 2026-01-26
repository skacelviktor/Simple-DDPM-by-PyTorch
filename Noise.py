import torch


class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.device = device

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])[:, None, None, None]
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        return noisy_samples
    
    def sample_prev_timestep(self, noisy_sample, noise, timestep):
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timestep])

        original_samples = (noisy_sample - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod
        original_samples = torch.clamp(original_samples, -1.0, 1.0)

        mean = noisy_sample - ((self.betas[timestep] * noise) / sqrt_one_minus_alpha_cumprod)
        mean = mean / torch.sqrt(self.alphas[timestep])

        if timestep == 0:
            return mean, original_samples
        else:
            variance = (1 - self.alphas_cumprod[timestep - 1]) / (1 - self.alphas_cumprod[timestep])
            variance = variance * self.betas[timestep]
            sigma = torch.sqrt(variance)
            noise = torch.randn(noisy_sample.shape).to(noisy_sample.device)
            prev_noisy_sample = mean + sigma * noise
            return prev_noisy_sample, original_samples