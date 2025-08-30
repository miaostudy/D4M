```python
conda create -n d4m python=3.12
conda activate d4m
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 transformers scikit-learn ipdb --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/huggingface/diffusers.git
cd diffusers && python setup.py install

mv scripts/pipeline_stable_diffusion_gen_latents.py diffusers/src/diffusers/pipelines/stable_diffusion
mv scripts/pipeline_stable_diffusion_latents2img.py diffusers/src/diffusers/pipelines/stable_diffusion
```
