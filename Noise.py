"""
Noise Scheduler for DDPM (Denoising Diffusion Probabilistic Models)

This module implements the mathematical framework for the diffusion process:
- Forward process: Gradually add noise to images
- Reverse process: Iteratively remove noise to generate new images

The noise schedule defines how much noise is added at each timestep,
which is crucial for the model's performance.
"""

import torch


class NoiseScheduler:
    """
    Manages the noise addition and removal processes in DDPM.

    The forward diffusion process gradually transforms clean data x_0 into
    pure noise x_T by adding Gaussian noise according to a predefined
    beta schedule.

    The reverse process learns to progressively remove noise, reconstructing
    the original data from pure noise.

    Mathematical formulation:

    - Forward diffusion (noise addition):
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε

    - Estimation of the original sample:
        x̂_0 = (x_t - sqrt(1 - α̅_t) * ε) / sqrt(α̅_t)

    - Mean of the reverse process:
        μ(x_t, t) = (1 / sqrt(α_t)) * (
            x_t - (β_t / sqrt(1 - α_{t-1})) * ε
        )

    - Variance of the reverse process:
        σ_t² = ((1 - α̅_{t-1}) / (1 - α̅_t)) * β_t

    Where:
    - β_t defines the noise schedule
    - α_t = 1 - β_t
    - α̅_t = ∏_{i=1}^t α_i (cumulative product)
    - ε ~ N(0, I) is Gaussian noise
    - x_0 is the clean data sample
    - x_t is the noisy sample at timestep t
    - t denotes the diffusion timestep
    """

    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):        
        # Create linearly spaced betas from beta_start to beta_end
        # Higher betas = more noise added to the time step
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        
        # α_t = 1 - β_t:
        self.alphas = 1.0 - self.betas
        
        # α̅_t = ∏α_i: cumulative product of all alphas up to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        
        # Store device for later tensor operations
        self.device = device

    def add_noise(self, original_samples, noise, timesteps):        
        # Extract alpha_cumprod values for the given timesteps
        # Shape: (batch,) -> (batch, 1, 1, 1)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])[:, None, None, None]
        
        # Calculate sqrt(1 - alpha_cumprod)
        # At t=0: sqrt(1 - 1) = 0 (no noise)
        # At t=T: sqrt(1 - 0) ≈ 1 (mostly noise)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])[:, None, None, None]
        
        # Linear combination of original signal and noise
        # Signal component gets smaller, noise component gets larger over time
        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise
        
        return noisy_samples
    
    def sample_prev_timestep(self, noisy_sample, noise, timestep):       
        # Get alpha values for current timestep
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timestep])

        # Reverse the forward process: solve for x_0 given x_t and noise
        # This estimates what the original clean image was
        original_samples = (noisy_sample - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod
        
        # Clamp to [-1, 1] to keep values in valid range
        original_samples = torch.clamp(original_samples, -1.0, 1.0)

        # Calculate mean of the reverse process distribution
        # This is the direction we should move in denoising space
        mean = noisy_sample - ((self.betas[timestep] * noise) / sqrt_one_minus_alpha_cumprod)
        mean = mean / torch.sqrt(self.alphas[timestep])

        # At timestep 0, there's no more noise to add - return mean directly
        if timestep == 0:
            return mean, original_samples
        
        # For all other timesteps, add stochasticity via variance schedule
        else:
            # Compute variance for this step (posterior variance)
            variance = (1 - self.alphas_cumprod[timestep - 1]) / (1 - self.alphas_cumprod[timestep])
            variance = variance * self.betas[timestep]
            
            # Standard deviation from variance
            sigma = torch.sqrt(variance)
            
            # Sample fresh noise for this step
            noise = torch.randn(noisy_sample.shape).to(noisy_sample.device)
            
            # Combine mean and variance to get x_{t-1}
            # This adds controlled stochasticity to the denoising process
            prev_noisy_sample = mean + sigma * noise
            
            return prev_noisy_sample, original_samples