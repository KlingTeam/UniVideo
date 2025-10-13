import math
from typing import List, Optional

from numpy import true_divide
import torch
from torch import nn
from torchvision import transforms as v2

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
)
import os
# from .transformer_encoder import Qwen2Encoder

def hf_local_snapshot(repo_id: str, revision: str = "main") -> str:
    repo_dir = repo_id.replace("/", "--")
    base = "/m2v_intern/weicong/cache/huggingface/hub"
    ref_file = os.path.join(base, f"models--{repo_dir}", "refs", revision)
    with open(ref_file) as f:
        rev = f.read().strip()
    return os.path.join(base, f"models--{repo_dir}", "snapshots", rev)


def _find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i+len(sub)] == sub:
            return i
    return -1

def compute_user_start_drop_idx(tokenizer, system_prompt: str) -> int:
    """
    Returns the token index where user content starts (just after `<|im_start|>user\n`).
    Works with Qwen2.5-VL apply_chat_template.
    """
    # 1) Build a minimal conversation using the same template path you already use
    conv = []
    if system_prompt is not None:
        conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    SENTINEL = "<<<__SENTINEL_USER_TEXT__>>>"

    conv.append({"role": "user", "content": [{"type": "text", "text": SENTINEL}]})

    # 2) Render with apply_chat_template (same flags as in your tokenize())
    rendered = tokenizer.apply_chat_template(conv, add_generation_prompt=True)

    # 3) Tokenize both the full string and just the sentinel
    full_ids = tokenizer(text=rendered, return_tensors="pt", padding=False).input_ids[0].tolist()
    sent_ids = tokenizer(text=SENTINEL, return_tensors="pt", padding=False).input_ids[0].tolist()

    # 4) Find sentinel start in the full sequence
    start = _find_subseq(full_ids, sent_ids)
    if start == -1:
        # Very rare: if the sentinel got split weirdly, fall back to string search and re-tokenize prefix
        # to compute a robust boundary.
        prefix = rendered.split(SENTINEL)[0]
        start = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
    return int(start)


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "Qwen2.5-VL",
        num_metaqueries: int = 64,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 1024,
        connector_num_hidden_layers: int = 24,
        connector_out_dim: Optional[int] = None,
        system_prompt: str = "You will be given a video or its caption. Please describe the content of the video in detail in your own words.",
        connector_method: str = "qwen2+mlp",
        connector_mlp_hidden_dim: int = 3076,
        use_chat_template: bool = True,
        crop_system_tokens: bool = True,
        crop_vision_tokens: bool = True,
        system_tokens_drop_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.connector_out_dim = connector_out_dim
        self.system_prompt = system_prompt
        self.connector_method = connector_method
        self.connector_mlp_hidden_dim = connector_mlp_hidden_dim
        self.use_chat_template = use_chat_template
        self.crop_system_tokens = crop_system_tokens
        self.crop_vision_tokens = crop_vision_tokens
        self.system_tokens_drop_idx = system_tokens_drop_idx


class MLLMInContext(PreTrainedModel):
    config_class = MLLMInContextConfig

    def __init__(
        self,
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        if "Qwen2.5-VL" in config.mllm_id:
            self.mllm_type = "qwenvl"
        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")
        
        if self.mllm_type == "qwenvl":
            print(f"Using Qwen MLLM {config.mllm_id}")
            self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                hf_local_snapshot(config.mllm_id, "main"), attn_implementation="sdpa", 
                # hf_local_snapshot(config.mllm_id, "main"), attn_implementation="flash_attention_2", 
                # torch_dtype=torch.bfloat16
            )
            # TODO: should we activate this in the CoT setting?
            # self.mllm_backbone.model.config.use_sliding_window = False
            # self.mllm_backbone.model.config.sliding_window = None

            # If use metaquery
            if config.num_metaqueries > 0:
                num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
                self.num_embeddings = num_embeddings
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

                def freeze_hook(grad):
                    print(f"  [Query] Original tokens (frozen): {self.num_embeddings}")
                    print(f"  Total tokens: {grad.shape[0]}")
                    print(f"  Gradient shape: {grad.shape}")
                    print(f"  Pre-zero grad norm: {grad.norm().item():.6f}")
                    print(f"  Pre-zero original token grad norm: {grad[:self.num_embeddings].norm().item():.6f}")
                    print(f"  Pre-zero new token grad norm: {grad[self.num_embeddings:].norm().item():.6f}")
                    if grad is None:
                        print(f"  ❌ Gradient is None!")
                    elif torch.isnan(grad).any():
                        print(f"  ❌ Gradient contains NaN!")
                    elif grad.norm().item() == 0.0:
                        print(f"  ❌ All gradients are exactly zero - gradient flow broken!")
                    
                    # Zero out gradients for original tokens
                    grad[: self.num_embeddings].zero_()
                    
                    print(f"  Post-zero original token grad norm: {grad[:self.num_embeddings].norm().item():.6f}")
                    print(f"  Post-zero new token grad norm: {grad[self.num_embeddings:].norm().item():.6f}")
                    
                    return grad
                self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)
                
            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            # max_pixels = 480 * 854 
            self.tokenizer = AutoProcessor.from_pretrained(
                hf_local_snapshot(config.mllm_id, "main"), min_pixels=min_pixels, max_pixels=max_pixels
            ) # Qwen2_5_VLProcessor
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            # 3B 2048
            # 7B 3584

        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")

        self.tokenizer.mllm_type = self.mllm_type
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.tokenizer.connector_method = config.connector_method
        self.tokenizer.use_chat_template = getattr(config, 'use_chat_template', True)
        self.tokenizer.crop_system_tokens = getattr(config, 'crop_system_tokens', True)

        # Auto-detect drop index if cropping is on and not explicitly provided
        if self.tokenizer.use_chat_template and self.tokenizer.crop_system_tokens:
            if getattr(config, 'system_tokens_drop_idx', 0) > 0:
                drop_idx = config.system_tokens_drop_idx
                print(f"[AUTO-CROP] Using provided system_tokens_drop_idx={drop_idx}")
            else:
                drop_idx = compute_user_start_drop_idx(self.tokenizer, config.system_prompt)
                print(f"[AUTO-CROP] Detected system_tokens_drop_idx={drop_idx}")
            self.tokenizer.system_tokens_drop_idx = drop_idx
        else:
            self.tokenizer.system_tokens_drop_idx = 0

        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id

        # If use Metaqueies we need to add special token
        if config.num_metaqueries > 0:
            print(f"Using metaqueries with {config.num_metaqueries} query")
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<pad_token_{i}>"
                        for i in range(num_embeddings - len(tokenizer))
                    ]
                }
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")

        self.connector_in_dim = self.mllm_hidden_size
        if config.connector_out_dim is not None:
            self.connector_out_dim = config.connector_out_dim
        else:
            self.connector_out_dim = 4096 # Default to 4096 for 1b transformer cross_attention_dim

        # Create connector based on method
        if config.connector_method == "mlp":
            # Simple 2-layer MLP connector for all tokens: input_dim -> hidden_dim -> output_dim
            print(f"Using mlp connector: {self.connector_in_dim} -> {config.connector_mlp_hidden_dim} -> {self.connector_out_dim}")
            self.connector = nn.Sequential(
                nn.Linear(self.connector_in_dim, config.connector_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(config.connector_mlp_hidden_dim, self.connector_out_dim),
            )
        elif config.connector_method == "qwen2+mlp":
            # Original complex connector with Qwen2Encoders
            print(f"Using qwen2+mlp connector with {config.connector_num_hidden_layers} layers")
            
            # norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
            # # Initialize norm weight based on original MetaQuery (math.sqrt(5.5) for Sana-like models)
            # import math
            # input_scale = math.sqrt(5.5)  # Original MetaQuery default for Sana
            # with torch.no_grad():
            #     norm.weight.fill_(input_scale)

            # encoder = Qwen2Encoder(
            #     Qwen2Config(
            #         hidden_size=self.connector_in_dim,
            #         intermediate_size=self.connector_in_dim * 4,
            #         num_hidden_layers=config.connector_num_hidden_layers,
            #         num_attention_heads=self.connector_in_dim // 64,
            #         num_key_value_heads=self.connector_in_dim // 64,
            #         initializer_range=0.014,
            #         use_cache=False,
            #         rope=true_divide,
            #         qk_norm=True,
            #     ),
            # )
            # self.connector = nn.Sequential(
            #     encoder,
            #     nn.Linear(self.connector_in_dim, self.connector_out_dim),
            #     nn.GELU(approximate="tanh"),
            #     nn.Linear(self.connector_out_dim, self.connector_out_dim),
            #     # norm,
            # )
        elif config.connector_method == "none":
            # No connector - use raw MLLM hidden states directly
            print(f"Using no connector - raw MLLM hidden states will be used directly")
            self.connector = None
        else:
            raise ValueError(f"Unknown connector_method: {config.connector_method}")
        
        # Only register hooks if connector exists
        if self.connector is not None:
            def log_grad_hook(name):
                def hook(grad):
                    print(f"[HOOK] Gradient for {name} | shape: {grad.shape} | norm: {grad.norm():.6f}")
                return hook
            for name, param in self.connector.named_parameters():
                if param.requires_grad:
                    param.register_hook(log_grad_hook(name))

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
                print("Enable Gradient Checkpoint for MLLM backbone")
            except:
                pass
            # if self.connector is not None and not isinstance(self.connector, nn.Identity):
            #     for module in self.connector:
            #         if isinstance(module, Qwen2Encoder):
            #             module.gradient_checkpointing_enable({"use_reentrant": False})

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize_fn

    def get_resize_fn(self):
        return self.resize_fn
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract hidden states using attention mask, similar to QwenImage pipeline"""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def _crop_system_tokens(self, hidden_states_list: List[torch.Tensor], drop_idx: int = 0):
        """Crop system prompt tokens from the beginning of sequences"""
        if drop_idx > 0:
            return [h[drop_idx:] for h in hidden_states_list]
        return hidden_states_list
    
    def _repad_to_max_length(self, hidden_states_list: List[torch.Tensor]):
        """Re-pad sequences to maximum length after cropping"""
        if not hidden_states_list:
            return None, None
            
        # Create attention masks for each sequence
        attn_mask_list = [torch.ones(h.size(0), dtype=torch.long, device=h.device) for h in hidden_states_list]
        
        # Find maximum sequence length
        max_seq_len = max([h.size(0) for h in hidden_states_list])
        
        # Pad sequences to max length
        padded_hidden_states = torch.stack([
            torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))]) 
            for h in hidden_states_list
        ])
        
        # Pad attention masks
        padded_attention_mask = torch.stack([
            torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))]) 
            for mask in attn_mask_list
        ])
        
        return padded_hidden_states, padded_attention_mask

    @staticmethod
    @torch.no_grad()
    def tokenize_fn(
        tokenizer, 
        texts,         # ["" x b] one sentence per example
        images=None,   # [[PIL.Image.Image x num] x b]
        videos=None,   # [[torch.tensor (f h w c) 0-255 x num] x b]
        text_response=None, 
        add_generation_prompt=True
    ):
        if not isinstance(texts, List):
            texts = [texts]

        # Check if we should use chat template or direct tokenization
        if not tokenizer.use_chat_template:
            assert not images
            print(f"[DEBUG] Using direct tokenization (no chat template)")
            print(f"[DEBUG] texts(s) before tokenization: {texts}")
            # Direct tokenization - no images, no chat template
            text_inputs = tokenizer(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.max_input_text_tokens,
            )
            
            print(f"[DEBUG] Direct tokenization - input_ids shape: {text_inputs['input_ids'].shape}")
            return text_inputs.values()

        # Chat template mode (original behavior)
        print(f"[DEBUG] Using chat template mode")
        
        prefix = (
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": tokenizer.system_prompt}],
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = ""
        else:  # metauqery token
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )

        texts = [
            tokenizer.decode(
                tokenizer(text=text, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            )
            for text in texts
        ]

        if images is not None and len(images) == 0:
            images = None
        if images is not None:
            # If images is not a list, wrap it in a list
            if not isinstance(images, list):
                images = [images]
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, img in enumerate(images):
                if img and not isinstance(img, list):
                    images[i] = [img]
        
        if videos is not None and len(videos) == 0:
            videos = None
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            for i, vids in enumerate(videos):
                if vids and not isinstance(vids, list):
                    videos[i] = [vids]

        batch_size = len(texts)
        if images is not None and len(images) != batch_size:
            raise ValueError(f"images batch ({len(images)}) must match texts ({batch_size})")
        if videos is not None and len(videos) != batch_size:
            raise ValueError(f"videos batch ({len(videos)}) must match texts ({batch_size})")

        # Build conversations: images first, then videos, then text
        # If a sample has no images/videos, it’s just the text.
        conversations = []
        for i in range(batch_size):
            content = []
            imgs = images[i] if images is not None else None
            vids = videos[i] if videos is not None else None
            if imgs:
                content.extend([{"type": "image"} for _ in imgs])
            if vids:
                content.extend([{"type": "video"} for _ in vids])
            content.append({"type": "text", "text": texts[i]})

            conversations.append(
                prefix
                + [
                    {
                        "role": "user",
                        "content": content,
                    },
                ]
            )

        kwargs = {}
        if images is not None:
            kwargs["images"] = images
        if videos is not None:
            kwargs["videos"] = videos

        prompts = [
            tokenizer.apply_chat_template(
                conv, 
                add_generation_prompt=True
            )
            for conv in conversations
        ]
        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]
        if tokenizer.num_metaqueries > 0:
            prompts = [p + suffix for p in prompts]

        # DEBUG PRINT
        print(f"[DEBUG] prompts:{prompts}")
        
        # Adjust max_length for chat template mode if cropping is enabled
        # max_len = tokenizer.max_input_text_tokens
        # if getattr(tokenizer, 'crop_system_tokens', False):
        #     drop_idx = getattr(tokenizer, 'system_tokens_drop_idx', 0)
        #     max_len = max_len + drop_idx
        #     print(f"[DEBUG] Chat template: Adjusted max_length from {tokenizer.max_input_text_tokens} to {max_len} (drop_idx={drop_idx})")
        
        inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            # truncation=True,  # we don't want to truncate image token
            # max_length=max_len,
            **kwargs,
        )

        # DEBUG PRINT
        # if getattr(tokenizer, 'crop_system_tokens', False):
        #     drop_idx = getattr(tokenizer, 'system_tokens_drop_idx', 0)
        #     ids, attn = inputs["input_ids"], inputs["attention_mask"]

        #     for b in range(min(2, ids.size(0))):  # limit debug spam
        #         first_valid = (attn[b] == 1).nonzero(as_tuple=False).min().item()
        #         cut = max(first_valid, min(first_valid + drop_idx, ids.size(1)))
        #         toks = ids[b, max(first_valid, cut - 5):min(ids.size(1), cut + 5)].tolist()
        #         remaining_decoded = tokenizer.decode(
        #             ids[b, cut:cut + 16], skip_special_tokens=False, clean_up_tokenization_spaces=False
        #         )
        #         print(f"[CROP-DEBUG b{b}] first_valid={first_valid}, drop_idx={drop_idx}, cut={cut}, "
        #                     f"window_ids={toks} | remaining_after_crop={repr(remaining_decoded)}")
        # DEBUG PRINT
        if "input_ids" in inputs:
            decoded_inputs = tokenizer.batch_decode(
                inputs["input_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            for i, decoded in enumerate(decoded_inputs):
                print(f"[DEBUG] \n--- Decoded input {i} ---\n{repr(decoded)}")
        else:
            print("[DEBUG] No input_ids found in inputs.")
       
        # DEBUG: Log the keys returned by QwenVL tokenizer
        print(f"[DEBUG] QwenVL tokenizer returned keys: {list(inputs.keys())}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"[DEBUG] {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"[DEBUG] {key}: {type(value)}")
        
        return inputs

    def _tok_id(self, s: str):
        tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        try:
            tid = tok.convert_tokens_to_ids(s)
            return tid if isinstance(tid, int) and tid != -1 else None
        except Exception:
            return None

    def _crop_hidden_bs1(self,
                    input_ids: torch.Tensor,        # [1, T]
                    attention_mask: torch.Tensor,   # [1, T]
                    last_hidden: torch.Tensor       # [1, T, D]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        B=1. If vision markers exist, keep tokens strictly AFTER the last <|vision_end|>.
        Otherwise, crop system tokens using tokenizer.system_tokens_drop_idx.
        Returns: (prompt_embeds [1, L, D], new_attn [1, L])
        """
        assert input_ids.shape[0] == 1 and attention_mask.shape[0] == 1 and last_hidden.shape[0] == 1
        ids  = input_ids[0]           # [T]
        attn = attention_mask[0]      # [T]
        hs   = last_hidden[0]         # [T, D]
        assert ids.shape[0] == attn.shape[0] == hs.shape[0]
        T, D = hs.shape

        valid = (attn == 1).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            # nothing valid; return a single zero token for shape sanity
            print("[KEEP-TEXT] ERROR ! No valid tokens in attention_mask, returning dummy zero.")
            return hs.new_zeros(1, 1, D), attn.new_zeros(1, 1)

        start_idx = None
        if self.config.crop_vision_tokens:
            ve_id = self._tok_id("<|vision_end|>")
            if ve_id is not None:
                ve_pos = (ids == ve_id).nonzero(as_tuple=False).flatten()
                if ve_pos.numel() > 0:
                    # vision present: keep AFTER the vision block
                    start_idx = int(ve_pos.max().item()) + 1
                    print(f"[KEEP-TEXT] Found <|vision_end|> at positions {ve_pos.tolist()}, using start_idx={start_idx}")

        if start_idx is None:
            # no vision: crop system tokens
            drop_idx = int(getattr(self.tokenizer, "system_tokens_drop_idx", 0))
            start_idx = int(valid.min().item() + drop_idx)
            print(f"[KEEP-TEXT] No <|vision_end|> found → using system_tokens_drop_idx={drop_idx}, start_idx={start_idx}")

        # end at last valid token
        end_idx = int(valid.max().item()) + 1
        start_idx = max(0, min(start_idx, end_idx))  # clamp + guard
        print(f"[KEEP-TEXT] Final slice: start={start_idx}, end={end_idx}, total_len={T}")

        kept = hs[start_idx:end_idx]                 # [L, D]
        if kept.numel() == 0:
            print("[KEEP-TEXT] Slice resulted in empty tensor, returning dummy zero.")
            return hs.new_zeros(1, 1, D), attn.new_zeros(1, 1)

        # --- DEBUG: show a small decoded window after crop ---
        try:
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            window_ids = ids[start_idx : end_idx].tolist()
            window_text = tok.decode(window_ids, skip_special_tokens=False)
            print(f"[KEEP-TEXT] Preview after crop → {repr(window_text)}")
        except Exception as e:
            print(f"[KEEP-TEXT] Preview decode failed: {e}")

        new_attn = attn.new_ones(kept.shape[0])      # [L]
        print(f"[KEEP-TEXT] Kept hidden states shape={kept.shape}, new_attn shape={new_attn.shape}")
        return kept.unsqueeze(0), new_attn.unsqueeze(0)

    @torch.no_grad()   # TODO: fix this 
    def encode_condition(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts
    ):
        if self.mllm_type == "qwenvl":
            outputs = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                output_hidden_states=True,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts
            )
            last_hidden = outputs.hidden_states[-1]  # Last layer hidden states
            print(f"[MLLM] QwenVL hidden states shape: {last_hidden.shape}")
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")

        # keep only text per your rule
        prompt_embeds, attention_mask = self._crop_hidden_bs1(input_ids, attention_mask, last_hidden)
        print(f"[TEXT-ONLY per rule] {prompt_embeds.shape}")

        # Return raw
        return prompt_embeds, attention_mask
    

        # # Handle different token cropping strategies
        # if self.tokenizer.num_metaqueries > 0:
        #     # MetaQuery mode: Extract tokens between BOI and EOI
        #     try:
        #         boi_pos = torch.where(input_ids == self.boi_token_id)[1]
        #         eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]
        #     except RuntimeError as e:
        #         # Diagnose the real issue: missing BOI/EOI tokens
        #         print(f"[ERROR] CUDA error in token position finding, investigating root cause:")
        #         print(f"  Original error: {e}")
        #         print(f"  input_ids shape: {input_ids.shape}, device: {input_ids.device}, dtype: {input_ids.dtype}")
        #         print(f"  boi_token_id: {self.boi_token_id}, eoi_token_id: {self.eoi_token_id}")
                
        #         # Check if BOI/EOI tokens exist
        #         boi_match = (input_ids == self.boi_token_id).nonzero(as_tuple=False)
        #         eoi_match = (input_ids == self.eoi_token_id).nonzero(as_tuple=False)
                
        #         print(f"  BOI token matches: {boi_match.numel()}")
        #         print(f"  EOI token matches: {eoi_match.numel()}")
                
        #         if boi_match.numel() == 0:
        #             print(f"  [ROOT CAUSE] BOI token {self.boi_token_id} not found in any sequence!")
        #             print(f"  input_ids sample: {input_ids[0][:50].tolist()}...")
                    
        #         if eoi_match.numel() == 0:
        #             print(f"  [ROOT CAUSE] EOI token {self.eoi_token_id} not found in any sequence!")
        #             print(f"  input_ids sample: {input_ids[0][:50].tolist()}...")
                    
        #         raise

        #     # Create mask for selecting tokens between BOI and EOI
        #     batch_size, seq_len = input_ids.shape
        #     indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
        #         batch_size, -1
        #     )
        #     mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])

        #     prompt_embeds = prompt_embeds[mask].view(
        #         batch_size, -1, prompt_embeds.size(-1)
        #     )
        #     attention_mask = attention_mask[mask].view(batch_size, -1)
            
        # elif getattr(self.tokenizer, 'crop_system_tokens', False):
        #     # QwenImage-style cropping: Remove system tokens and repad
        #     print(f"[CROP] Applying system token cropping with drop_idx={getattr(self.tokenizer, 'system_tokens_drop_idx', 0)}")
            
        #     # Extract valid sequences using attention mask
        #     hidden_states_list = self._extract_masked_hidden(prompt_embeds, attention_mask)
            
        #     # Crop system tokens from the beginning
        #     cropped_hidden_states = self._crop_system_tokens(
        #         hidden_states_list, 
        #         drop_idx=getattr(self.tokenizer, 'system_tokens_drop_idx', 0)
        #     )
            
        #     # Re-pad to maximum length
        #     prompt_embeds, attention_mask = self._repad_to_max_length(cropped_hidden_states)
            
        #     print(f"[CROP] After cropping and repadding: prompt_embeds shape={prompt_embeds.shape}")
            
        # # If neither metaqueries nor cropping, use full sequences as-is
        # # Log connector weights (first layer as example) if connector exists
        # if self.connector is not None:
        #     with torch.no_grad():
        #         for name, param in self.connector.named_parameters():
        #             print(f"[CONNECTOR WEIGHT] {name} | shape: {param.shape} | mean: {param.data.mean():.6f} | std: {param.data.std():.6f}")
        
        # # for i in range(min(3, attention_mask.size(0))):  # Show at most 3 samples
        # #     print(f"[MASK] attention_mask[{i}] : {attention_mask[i]}")
        
        # # Apply connector if it exists, otherwise return raw embeddings
        # if self.connector is not None:
        #     return self.connector(prompt_embeds), attention_mask
        # else:
        #     print(f"[NO CONNECTOR] Returning raw MLLM embeddings with shape: {prompt_embeds.shape}")
        #     return prompt_embeds, attention_mask
