from typing import Dict
import torch
from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_MESSAGE = """
You are a helpful AI Assistant who solves problems. Solve each problem step by step, ensuring that:
1. Each reasoning step is separated by a blank line for clarity.
2. The final answer is formatted as: \\boxed{<answer>}.
Always follow this format when solving problems.
"""

problem = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
answer = "\\left( 3, \\frac{\\pi}{2} \\right)"

class BeamSearch:
    def __init__(
        self,
        number_of_beams=4,
        width=5,
        max_new_tokens=2500,
        max_generation_rounds=10,
        device=None,
    ):
        self.number_of_beams = number_of_beams # N
        self.width = width # M
        self.max_new_tokens = max_new_tokens
        self.max_generation_rounds = max_generation_rounds

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.generator = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=quant_config,
            token="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",
        )

        self.prm = AutoModelForCausalLM.from_pretrained(
            "mtzig/prm800k_llama_debug_full",
            token="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",
            quantization_config=quant_config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token="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",
        )

        self.step_strs = []
        self.step_ids = []

        for t, i in self.tokenizer.vocab.items():
            if "\n\n" in t or "ĊĊ" in t:
                self.step_strs.append(t)
                self.step_ids.append(i)

        self.prm_step_id = self.tokenizer.encode(" ки", add_special_tokens=False)[0]
        self.eos_id = self.tokenizer.encode(
            "<|end_of_text|>", add_special_tokens=False
        )[0]

        self.candidate_ids = [648, 387]
        self.pad_id = self.tokenizer.eos_token_id
        self.generator.generation_config.pad_token_id = self.pad_id
        self.generator.generation_config.eos_token_id = self.step_ids + [
            self.eos_id,
            self.generator.generation_config.eos_token_id,
        ]

    def encode(self, question: str) -> Tensor:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": problem},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        
        return input_ids.repeat(self.number_of_beams, 1).to(
            self.device
        )  # start with `self.width` beams initialized to the prompt

    def convert_to_left_padding(self, gen_outputs: Dict) -> Dict:
        """
        Realign sequences and their past key-value (kv) caches based on left-padding.

        Parameters:
            gen_outputs (Dict): A dictionary containing:
                - 'sequences': A tensor of shape [batch_size, sequence_length]
                - 'past_key_values': A tuple where each element is a tuple
                                    containing (key, value) of dimensions
                                    [batch_size, num_heads, seq_len, head_dim].

        Returns:
            Dict: The updated dictionary with realigned sequences and past_key_values.
        """
        converted_sequences = []
        realigned_past_key_values = []

        # Realign sequences
        for batch_idx, row in enumerate(gen_outputs["sequences"]):
            pad_idxs = row == self.pad_id
            # Align the sequences with padding on the left
            converted_sequences.append(torch.cat((row[pad_idxs], row[~pad_idxs])))

        # Realign kv cache
        for layer in gen_outputs["past_key_values"]:
            key, value = layer  # Unpack key and value for this layer
            realigned_key = []
            realigned_value = []

            # Realign each batch's key and value
            for batch_idx, row in enumerate(gen_outputs["sequences"]):
                values_to_check = torch.tensor(self.step_ids + [self.eos_id], device=row.device)  # Combine step IDs and eos_id
                matches = torch.isin(row, values_to_check)  # Boolean mask for matching elements
                indices = torch.nonzero(matches, as_tuple=True)[0]  # Get indices of matching elements
                last_idx_step_or_eos = indices[-1].item()  # Get the largest index (last match)

                pad_ids_cache = torch.zeros(len(row) - 1, dtype=torch.bool, device=row.device)
                pad_ids_cache[:last_idx_step_or_eos] |= row[:last_idx_step_or_eos] == self.pad_id
                pad_ids_cache[last_idx_step_or_eos:] = True

                # Realign keys and values for the batch
                realigned_key.append(torch.cat((key[batch_idx, :, pad_ids_cache, :], 
                                                key[batch_idx, :, ~pad_ids_cache, :]), dim=1)) # (H, S, D)
                realigned_value.append(torch.cat((value[batch_idx, :, pad_ids_cache, :], 
                                                value[batch_idx, :, ~pad_ids_cache, :]), dim=1)) # (H, S, D)

            # Stack the realigned keys and values for this layer
            realigned_past_key_values.append((
                torch.stack(realigned_key),
                torch.stack(realigned_value)
            ))

        # Update the gen_outputs dictionary
        gen_outputs["sequences"] = torch.stack(converted_sequences)
        gen_outputs["past_key_values"] = tuple(realigned_past_key_values)

        return gen_outputs

    def prepare_prm_inputs(self, output_ids: Tensor, initial_prompt_len: int) -> Tensor:
        input_ids = output_ids.clone()

        mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        for i in self.step_ids:
            mask |= input_ids == i

        mask |= input_ids == self.eos_id

        ignore_prefix_lens = (
            torch.sum((input_ids == self.pad_id).long(), dim=1) + initial_prompt_len
        )
        idxs = torch.arange(input_ids.shape[1], device=input_ids.device)
        idxs = idxs.repeat(len(mask)).view(len(mask), -1)
        response_mask = (idxs.T > ignore_prefix_lens).T

        mask &= response_mask
        input_ids[mask] = self.prm_step_id
        return input_ids

    def compute_scores(self, logits: Tensor) -> list[float]:
        return logits.softmax(dim=-1)[:, -1, 0].cpu().tolist()

    def __call__(self, question: str) -> str:
        gen_input_ids = self.encode(question)
        initial_prompt_len = gen_input_ids.shape[1]

        width = self.width
        completed_beams: list[tuple[Tensor, float]] = []

        for _ in range(self.max_generation_rounds):
            attention_mask = (gen_input_ids != self.pad_id).long()

            # gen_output_ids = self.generator.generate(
            gen_outputs = self.generator.generate(
                input_ids=gen_input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                max_length=None,
                eos_token_id=self.step_ids + [self.eos_id],
                pad_token_id=self.pad_id,
                use_cache=True,
                return_dict_in_generate=True,
                past_key_values = gen_past_key_values if "gen_past_key_values" in locals() else None,
            )

            gen_outputs = self.convert_to_left_padding(gen_outputs)
            gen_output_ids = gen_outputs["sequences"]
            gen_past_key_values = gen_outputs["past_key_values"]

            prm_input_ids = self.prepare_prm_inputs(gen_output_ids, initial_prompt_len)

            with torch.no_grad():
                logits = self.prm(
                    prm_input_ids,
                    # Todo: input kv cache if exists
                ).logits[:, :, self.candidate_ids]
                scores = self.compute_scores(logits)
            # Todo: left padding for kv cache (self.prm)

            sorted_scored_idxs = sorted(
                enumerate(scores), key=lambda t: t[1], reverse=True
            )
            best_idxs = [i for i, _ in sorted_scored_idxs]
            
            # filter M best beams for both reasoning paths, kv cache for generator
            # Todo: filter M best beams for kv cache for PRM
            gen_input_ids = gen_output_ids[best_idxs[:width]]
            gen_past_key_values = tuple(  # Iterate over each element of the KV cache tuple
                (
                    k[best_idxs[:width], ...],  # Filter the key tensor along batch axis using best_idxs[:width]
                    v[best_idxs[:width], ...]   # Filter the value tensor along batch axis using best_idxs[:width]
                )
                for k, v in gen_past_key_values  # Unpack each key-value pair from the tuple
            )

            # inflating batch to N (by multiplying N/M times for each)
            # Todo: inflating for PRM
            inflator = self.number_of_beams/self.width
            gen_input_ids = gen_input_ids.repeat(int(inflator), 1)
            gen_past_key_values = tuple(
                (
                    k.repeat(int(inflator), *([1] * (k.ndim - 1))),  # Repeat batch dimension for key
                    v.repeat(int(inflator), *([1] * (v.ndim - 1)))   # Repeat batch dimension for value
                )
                for k, v in gen_past_key_values  # Iterate over each key-value pair in the tuple
            )


        return [self.tokenizer.decode(x) for x in gen_input_ids]

def main():
    outputs = BeamSearch(width=2, number_of_beams=4, max_generation_rounds=7)(problem)

    for output in outputs:
        print(output)

if __name__ == "__main__":
    main()