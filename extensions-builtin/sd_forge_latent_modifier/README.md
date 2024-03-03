This extension is compiled from https://github.com/Clybius
Original Licence GPL V3


## Latent Diffusion Mega Modifier (sampler_mega_modifier.py)
### Adds multiple parameters to control the diffusion process towards a quality the user expects.
* Sharpness: utilizes code from Fooocus's sampling process to sharpen the noise in the middle of the diffusion process.
This can lead to more perceptual detail, especially at higher strengths.

* Tonemap: Clamps conditioning noise (CFG) using a user-chosen method, which can allow for the use of higher CFG values.

* Rescale: Scales the CFG by comparing the standard deviation to the existing latent to dynamically lower the CFG.

* Extra Noise: Adds extra noise in the middle of the diffusion process to conditioning, and do the inverse operation on unconditioning, if chosen.

* Contrast: Adjusts the contrast of the conditioning, can lead to more pop-style results. Essentially functions as a secondary CFG slider for stylization, without changing subject pose and location much, if at all.

* Combat CFG Drift: As we increase CFG, the mean will slightly drift away from 0. This subtracts the mean or median of the latent. Can lead to potentially sharper and higher frequency results, but may result in discoloration.

* Divisive Norm: Normalizes the latent using avg_pool2d, and can reduce noisy artifacts, due in part to features such as sharpness.

* Spectral Modulation: Converts the latent to frequencies, and clamps higher frequencies while boosting lower ones, then converts it back to an image latent. This effectively can be treated as a solution to oversaturation or burning as a result of higher CFG values, while not touching values around the median.

### Tonemapping Methods Explanation:
* Reinhard: <p>Uses the reinhard method of tonemapping (from comfyanonymous' ComfyUI Experiments) to clamp the CFG if the difference is too strong.

  Lower `tonemap_multiplier` clamps more noise, and a lower `tonemap_percentile` will increase the calculated standard deviation from the original noise. Play with it!</p>
* Arctan: <p>Clamps the values dynamically using a simple arctan curve. [Link to interactive Desmos visualization](https://www.desmos.com/calculator/e4nrcdpqbl).

  Recommended values for testing: tonemap_multiplier of 5, tonemap_percentile of 90.</p>
* Quantile: <p>Clamps the values using torch.quantile for obtaining the highest magnitudes, and clamping based on the result.


  `Closer to 100 percentile == stronger clamping`. Recommended values for testing: tonemap_multiplier of 1, tonemap_percentile of 99.</p>
* Gated: <p>Clamps the values using torch.quantile, only if above a specific floor value, which is set by `tonemapping_multiplier`. Clamps the noise prediction latent based on the percentile.


  `Closer to 100 percentile == stronger clamping, lower tonemapping_multiplier == stronger clamping`. Recommended values for testing: tonemap_multiplier of 0.8-1, tonemap_percentile of 99.995.</p>
* CFG-Mimic: <p>Attempts to mimic a lower or higher CFG based on `tonemapping_multiplier`, and clamps it using `tonemapping_percentile` with torch.quantile.


  `Closer to 100 percentile == stronger clamping, lower tonemapping_multiplier == stronger clamping`. Recommended values for testing: tonemap_multiplier of 0.33-1.0, tonemap_percentile of 100.</p>
* Spatial-Norm: <p>Clamps the values according to the noise prediction's absolute mean in the spectral domain. `tonemap_multiplier` adjusts the strength of the clamping.


  `Lower tonemapping_multiplier == stronger clamping`. Recommended value for testing: tonemap_multiplier of 0.5-2.0.</p>

### Contrast Explanation:
<p>Scales the pixel values by the standard deviation, achieving a more contrasty look. In practice, this can effectively act as a secondary CFG slider for stylization. It doesn't modify subject poses much, if at all, which can be great for those looking to get more oomf out of their low-cfg setups.

Using a negative value will apply the inverse of the operation to the latent.</p>

### Spectral Modification Explanation:
<p>We boost the low frequencies (low rate of change in the noise), and we lower the high frequencies (high rates of change in the noise). 

Change the low/high frequency range using `spectral_mod_percentile` (default of 5.0, which is the upper and lower 5th percentiles.)

Increase/Decrease the strength of the adjustment by increasing `spectral_mod_multiplier`

Beware of percentile values higher than 15 and multiplier values higher than 5, especially for hard clamping. Here be dragons, as large values may cause it to "noise-out", or become full of non-sensical noise, especially earlier in the diffusion process.</p>


#### Current Pipeline:
>##### Add extra noise to conditioning -> Sharpen conditioning -> Convert to Noise Prediction -> Tonemap Noise Prediction -> Spectral Modification -> Modify contrast of noise prediction -> Rescale CFG -> Divisive Normalization -> Combat CFG Drift

#### Why use this over `x` node?
Since the `set_model_sampler_cfg_function` hijack in ComfyUI can only utilize a single function, we bundle many latent modification methods into one large function for processing. This is simpler than taking an existing hijack and modifying it, which may be possible, but my (Clybius') lack of Python/PyTorch knowledge leads to this being the optimal method for simplicity. If you know how to do this, feel free to reach out through any means!

#### Can you implement `x` function?
Depends. Is there existing code for such a function, with an open license for possible use in this repository? I could likely attempt adding it! Feel free to start an issue or to reach out for ideas you'd want implemented.
