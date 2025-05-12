import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import torch.multiprocessing as mp
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from diffusers import DDIMScheduler, DDPMScheduler, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, PromptProcessorOutput, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *
from threestudio.utils.misc import barrier, cleanup, get_rank


@threestudio.register("stable-diffusion-3-prompt-processor")
class StableDiffusionV3PromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, cache_dir_2=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pipe = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name_or_path,
        ).to("cuda")
        encode_prompt = pipe.encode_prompt
        # process text.
        for prompt, negative_prompt in prompts:
            with torch.no_grad():
                (
                    prompt_embeds,
                    ng_prompt_embeds,
                    pooled_prompt_embeds,
                    ng_pooled_prompt_embeds
                ) = encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt,
                    prompt_3=prompt,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt,
                    negative_prompt_3=negative_prompt,
                )
            prompt_embeds = torch.cat([ng_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([ng_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

            torch.save(
                prompt_embeds,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )
            torch.save(
                pooled_prompt_embeds,
                os.path.join(
                    cache_dir_2,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del pipe
        
    @rank_zero_only
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._cache_dir_2, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + self.prompts_vd
        )
        all_neg_prompts = (
            [self.negative_prompt]
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for prompt, neg_prompt in zip(all_prompts, all_neg_prompts):
            if self.cfg.use_cache:
                # some text embeddings are already in cache
                # do not process them
                cache_path = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                if os.path.exists(cache_path):
                    threestudio.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue
            prompts_to_process.append((prompt, neg_prompt))

        if len(prompts_to_process) > 0:
            if self.cfg.spawn:
                raise NotImplementedError("Multiprocessing is not supported in this version.")
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    ),
                )
                subprocess.start()
                subprocess.join()
                assert subprocess.exitcode == 0, "prompt embedding process failed!"
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                    self._cache_dir_2,
                )
            cleanup()

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.text_embeddings = self.load_from_cache(self.prompt, cache_dir=self._cache_dir)[None, ...]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt, cache_dir=self._cache_dir) for prompt in self.prompts_vd], dim=0
        )
        self.pooled_text_embeddings = self.load_from_cache(self.prompt, cache_dir=self._cache_dir_2)[None, ...]
        self.pooled_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt, cache_dir=self._cache_dir_2) for prompt in self.prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")

    def load_from_cache(self, prompt, cache_dir):
        cache_path = os.path.join(
            cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)
    
    def __call__(self) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.pooled_text_embeddings,
            prompt=self.prompt,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.pooled_text_embeddings_vd,
            prompts_vd=self.prompts_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )
