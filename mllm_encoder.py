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
        system_prompt: str = "You will be given a video or its caption. Please describe the content of the video in detail in your own words.",
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
        self.system_prompt = system_prompt
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
                config.mllm_id, 
                attn_implementation="sdpa", 
                # attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16
            )
            # self.mllm_backbone.model.config.use_sliding_window = False
            # self.mllm_backbone.model.config.sliding_window = None

            # If use metaquery
            if config.num_metaqueries > 0:
                print(f"Before resize embed_tokens: {self.mllm_backbone.model.embed_tokens.weight.shape}")
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
                print(f"After resize embed_tokens: {self.mllm_backbone.model.embed_tokens.weight.shape}")

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
            # max_pixels = 1280 * 28 * 28
            max_pixels = 480 * 854 
            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
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

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
                print("Enable Gradient Checkpoint for MLLM backbone")
            except:
                pass

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
        add_queires=True,  # For video/image generation we add queires otherwise for text generation we don't add them.
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
        if tokenizer.num_metaqueries > 0 and add_queires:
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


    def _extract_text_and_queries_bs1(
        self,
        input_ids: torch.Tensor,        # [1, T]
        attention_mask: torch.Tensor,   # [1, T]
        last_hidden: torch.Tensor       # [1, T, D]
    ):
        """
        Returns:
            embeds : [1, L, D]   (text first, then query tokens)
            attn   : [1, L]
        Assumes bs=1.
        """
        assert input_ids.shape[0] == 1 and attention_mask.shape[0] == 1 and last_hidden.shape[0] == 1

        ids  = input_ids[0]       # [T]
        attn = attention_mask[0]  # [T]
        hs   = last_hidden[0]     # [T, D]
        T, D = hs.shape

        # --- valid span (handles left padding) ---
        valid = (attn == 1).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            print("[TEXT+QUERY] No valid tokens; returning empty.")
            return hs.new_zeros(1, 0, D), attn.new_zeros(1, 0)

        first_valid = int(valid.min().item())
        end_idx = int(valid.max().item()) + 1

        # --- choose start_idx: vision crop > system crop ---
        start_idx = None

        def _tok_id(token_str: str):
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            try:
                tid = tok.convert_tokens_to_ids(token_str)
                return tid if isinstance(tid, int) and tid != -1 else None
            except Exception:
                return None

        # always crop vision token
        ve_id = _tok_id("<|vision_end|>")
        if ve_id is not None:
            ve_pos = (ids == ve_id).nonzero(as_tuple=False).flatten()
            if ve_pos.numel() > 0:
                start_idx = int(ve_pos.max().item()) + 1
                print(f"[TEXT+QUERY] vision_end at {ve_pos.tolist()} → start_idx={start_idx}")

        if start_idx is None:
            drop_idx = int(getattr(self.tokenizer, "system_tokens_drop_idx", 0))
            start_idx = first_valid + drop_idx
            print(f"[TEXT+QUERY] no vision_end → drop system drop_idx={drop_idx}, start_idx={start_idx}")

        start_idx = max(0, min(start_idx, end_idx))

        kept_hs  = hs[start_idx:end_idx]     # [L, D]
        kept_ids = ids[start_idx:end_idx]    # [L]
        L = kept_hs.shape[0]

        if L == 0:
            print("[TEXT+QUERY] crop produced empty; returning empty.")
            return hs.new_zeros(1, 0, D), attn.new_zeros(1, 0)

        try:
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            window_text = tok.decode(
                kept_ids.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            print(f"[TEXT+QUERY] Preview after crop → {repr(window_text)}")
        except Exception as e:
            print(f"[TEXT+QUERY] Preview decode failed: {e}")

        # --- split text vs query ---
        device = kept_hs.device
        text_mask  = torch.ones(L, dtype=torch.bool, device=device)
        query_mask = torch.zeros(L, dtype=torch.bool, device=device)

        if getattr(self.tokenizer, "num_metaqueries", 0) > 0:
            boi = getattr(self, "boi_token_id", None)
            eoi = getattr(self, "eoi_token_id", None)

            if boi is not None and eoi is not None:
                boi_pos = (kept_ids == boi).nonzero(as_tuple=False).flatten()
                eoi_pos = (kept_ids == eoi).nonzero(as_tuple=False).flatten()

                if boi_pos.numel() > 0 and eoi_pos.numel() > 0:
                    boi_i = int(boi_pos[0].item())
                    eoi_i = int(eoi_pos[0].item())

                    if eoi_i > boi_i + 1:
                        query_mask[boi_i + 1 : eoi_i] = True

                    text_mask[boi_i : eoi_i + 1] = False
                else:
                    print("[TEXT+QUERY] BOI/EOI not found → all tokens treated as text.")
            else:
                print("[TEXT+QUERY] missing BOI/EOI ids → all tokens treated as text.")

        # --- concat text then queries ---
        text_hs  = kept_hs[text_mask]     # [Lt, D]
        query_hs = kept_hs[query_mask]    # [Lq, D]

        concat_hs = torch.cat([text_hs, query_hs], dim=0)
        concat_attn = torch.ones(concat_hs.shape[0], device=device, dtype=attn.dtype)

        print(
            f"[TEXT+QUERY] final concat shape={concat_hs.shape} "
            f"(text={text_hs.shape}, query={query_hs.shape})"
        )

        return concat_hs.unsqueeze(0), concat_attn.unsqueeze(0)

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

        if self.tokenizer.num_metaqueries > 0:
            prompt_embeds, attention_mask = self._extract_text_and_queries_bs1(
                input_ids, attention_mask, last_hidden
            )
        else:
            prompt_embeds, attention_mask = self._crop_hidden_bs1(input_ids, attention_mask, last_hidden)
            print(f"[TEXT-ONLY per rule] {prompt_embeds.shape}")
        
        # Return raw
        return prompt_embeds, attention_mask
    
    def generation(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts
    ):
        if self.mllm_type == "qwenvl":
            generated_ids = self.mllm_backbone.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                max_new_tokens=1000,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")
        return output_text