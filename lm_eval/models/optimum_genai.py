import logging
from importlib.util import find_spec
import numpy as np
from typing import List, Optional
import copy
from tqdm import tqdm

from openvino_genai import TokenizedInputs, GenerationConfig
from openvino import Tensor
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@register_model("openvino_genai")
class OptimumGenAILM(HFLM):
    """
    Using the OpenVINO GenAI backend from optimum-intel for accelerated inference
    on Intel architectures. This leverages the OpenVINO GenAI library for optimized
    text generation.
    """

    def __init__(
        self,
        device="cpu",
        config=None,
        **kwargs,
    ) -> None:
        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only OpenVINOGenAIModelForCausalLM is supported."
            )

        self.openvino_device = device

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.pop("backend", "causal"),
            **kwargs,
        )

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=False,
        gpus=0,
        offload_folder='./offload',
        autogptq=False,
        gptqmodel=False,
        parallelize=False,
        **kwargs,
    ) -> None:
        if not find_spec("optimum"):
            raise ModuleNotFoundError(
                "package `optimum` is not installed. Please install it via `pip install optimum-intel[openvino-genai]`"
            )
        else:
            from optimum.intel.openvino_genai.modeling_base import OpenVINOGenAIModelForCausalLM

        model_kwargs = {}

        # FIXME: add a transfer function to convert all relevant args to GenAI config
        model_kwargs["MAX_PROMPT_LEN"] = kwargs.pop("max_prompt_len", 1024)
        model_kwargs["MIN_RESPONSE_LEN"] = kwargs.pop("min_response_len", 150)
        for key, value in kwargs.items():
            model_kwargs[key] = value
        
        self._model = OpenVINOGenAIModelForCausalLM(
            model_path=pretrained,
            device=self.openvino_device,
            **model_kwargs,
        )

        self.ov_tokenizer = self._model.ov_tokenizer
        self._model_config = model_kwargs
    
    # FIXME: extract loglikelihood_token 
    def loglikelihood(self, requests):
        res = []
        for request in requests:
            context, continuation = request.args
            # FIXME:set max prompt length
            generation_config = GenerationConfig(echo=True,
                                               max_new_tokens=0,
                                               do_sample=True)

            whole_enc = self.ov_tokenizer.encode(context + continuation)
            inp_ids = whole_enc.input_ids
            whole_enc_len = inp_ids.shape[1]

            # Note: in latest OV, there is no need to fix the tokenizer input length for npu
            # if self.openvino_device == "NPU":
            #     whole_enc = self.ov_tokenizer.encode(context + continuation, max_length=self._model_config["MAX_PROMPT_LEN"], pad_to_max_length=True)

            context_enc = self.ov_tokenizer.encode(context)
            context_enc_len = context_enc.input_ids.shape[1]

            output, score, logprobs = self._model(whole_enc, generation_config=generation_config)

            cont_logits = logprobs[context_enc_len: whole_enc_len]    
            print('cont_logits: ', cont_logits)            
            # MultipleChoiceTask process_results discard is_greedy anyway
            res.append((sum(cont_logits), False))

        return res

    def loglikelihood_rolling(self, requests):
        """
        Return fake rolling loglikelihood values for evaluation purposes.
        OpenVINO GenAI models are focused on generation and don't support loglikelihood calculation.
        """
        eval_logger.warning(
            "OpenVINO GenAI models don't support loglikelihood calculation. Returning fake values."
        )
        
        res = []
        for request in requests:
            context, continuation = request.args
            # Return fake token loglikelihoods - one value per token in continuation
            fake_token_loglikelihoods = [-1.0] * len(continuation)  # Fake values
            res.append(fake_token_loglikelihoods)
        
        return res 

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text using OpenVINO GenAI's pipeline until a specified stopping criteria is met.
        Maintains compatibility with the HFLM implementation.
        
        Args:
            requests: List of Instance objects containing generation requests
            disable_tqdm: Whether to disable the progress bar
        
        Returns:
            List of generated strings
        """
        res = []
        
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running OpenVINO GenAI generation requests",
        )
        
        # Process each request individually
        for request in requests:
            context, gen_kwargs = request.args
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                stop_strings = kwargs.pop("until", None)
                # Extract max_gen_toks if provided, otherwise use default
                if "max_gen_toks" in kwargs.keys() and "max_new_tokens" in kwargs.keys():
                    logging.warning("Set max_gen_toks and max_new_tokens in meantime, will set with max_new_tokens")
                max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)

                if "max_new_tokens" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_new_tokens")
            else:
                raise ValueError(f"Expected kwargs to be of type dict but got {type(gen_kwargs)}")
            
            generation_kwargs = {
                "max_new_tokens": max_gen_toks,
                "stop_strings": set(stop_strings),
            }
            
            # Add any other parameters from kwargs that are supported by OpenVINO GenAI
            for k, v in kwargs.items():
                if k not in generation_kwargs:
                    generation_kwargs[k] = v

            generated_text = self._model.generate(
                context,
                **generation_kwargs
            )
                
            res.append(generated_text)
            
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
            
            pbar.update(1)
        
        pbar.close()
        return res 