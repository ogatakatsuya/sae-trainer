import json
import os

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import Trainer

from sae.models.topk_sae.layer import TopKSaeLayer


class SaeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_sae = None
        self._teacher_activations = {}
        self._teacher_hook_handle = None

        if getattr(self.args, "distillation_sae_path", None):
            self._init_teacher_sae(self.args.distillation_sae_path)

    # ── Teacher SAE setup ─────────────────────────────────────────────────

    def _init_teacher_sae(self, sae_path: str):
        with open(os.path.join(sae_path, "adapter_config.json")) as f:
            config = json.load(f)

        target_module = config["target_modules"]
        prefix = f"base_model.model.{target_module}"

        tensors = {}
        with safe_open(os.path.join(sae_path, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        self._teacher_sae = {
            "k": config["k"],
            "target_module": target_module,
            "encoder_weight": tensors[f"{prefix}.sae_encoder.weight"],  # (num_latents, hidden)
            "b_dec": tensors[f"{prefix}.sae_b_dec"],                     # (hidden,)
        }

        # Register hook on teacher's target layer to capture its output
        base_model = self.model.base_model.model
        module = base_model
        for part in target_module.split("."):
            module = getattr(module, part)

        # If wrapped by SAE, hook the inner base_layer
        hook_target = module.base_layer if isinstance(module, TopKSaeLayer) else module

        def _teacher_hook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            self._teacher_activations["x"] = x.flatten(0, 1).float().detach()

        self._teacher_hook_handle = hook_target.register_forward_hook(_teacher_hook)
        print(f"[distillation] teacher SAE loaded from {sae_path}, hooking {target_module}")

    # ── Distillation loss ─────────────────────────────────────────────────

    def _compute_distillation_loss(self) -> torch.Tensor:
        teacher_x = self._teacher_activations.get("x")  # (T_t, hidden_t)
        if teacher_x is None:
            return torch.tensor(0.0)

        t = self._teacher_sae
        enc_w = t["encoder_weight"].to(teacher_x.device)
        b_dec = t["b_dec"].to(teacher_x.device)

        # Teacher pre-activation: (T_t, num_latents)
        teacher_pre_act = (teacher_x - b_dec) @ enc_w.T
        teacher_mean = teacher_pre_act.mean(0)  # (num_latents,)

        # Collect student pre-activations across all SAE layers
        student_pre_acts = []
        for _, module in self.model.named_modules():
            if isinstance(module, TopKSaeLayer):
                for pre_act in module._last_pre_act.values():
                    flat = pre_act.flatten(0, 1) if pre_act.dim() == 3 else pre_act
                    student_pre_acts.append(flat.float())

        if not student_pre_acts:
            return torch.tensor(0.0)

        student_pre_act = torch.cat(student_pre_acts, dim=0)  # (T_s, num_latents)
        student_mean = student_pre_act.mean(0)  # (num_latents,)

        if teacher_mean.shape != student_mean.shape:
            raise ValueError(
                f"Teacher num_latents ({teacher_mean.shape[0]}) != "
                f"student num_latents ({student_mean.shape[0]}). "
                "Set the same --num_latents for both SAEs."
            )

        return (teacher_mean - student_mean).pow(2).mean()

    # ── Training step / loss ──────────────────────────────────────────────

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not getattr(self, "_static_graph_set", False):
            if isinstance(model, nn.parallel.DistributedDataParallel):
                model._set_static_graph()
            self._static_graph_set = True
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)

        output_hidden_dict = self.model.base_model.output_hidden_dict
        input_hidden_dict = self.model.base_model.input_hidden_dict

        per_layer_loss = {}
        total_loss = 0

        for layer, output_hidden_states in output_hidden_dict.items():
            input_hidden_states = input_hidden_dict[layer]
            e = output_hidden_states - input_hidden_states
            total_variance = (
                (input_hidden_states - input_hidden_states.mean(0)).pow(2).sum()
            )
            l2_loss = e.pow(2).sum()
            fvu = l2_loss / total_variance
            per_layer_loss[layer] = fvu.item()
            total_loss += fvu

        aux_log_info = self.model.base_model.get_aux_log_info()
        if aux_log_info:
            for key, value in aux_log_info.items():
                per_layer_loss[key] = value

        self.log(per_layer_loss)

        # Distillation loss — logged separately so it appears as its own metric in WandB
        if self._teacher_sae is not None:
            distil_loss = self._compute_distillation_loss()
            total_loss = total_loss + self.args.distillation_alpha * distil_loss
            self.log({
                "distillation_loss": distil_loss.item(),
                "distillation_loss_weighted": (self.args.distillation_alpha * distil_loss).item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss
