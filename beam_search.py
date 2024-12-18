from typing import Dict, Tuple
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
        self.width = width # M
        self.number_of_beams = number_of_beams # N
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
        self.pad_id = 128248
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
        
        return input_ids

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
        gen_prev_sequence_pad_ids = []  
        realigned_past_key_values = []

        # Realign sequences
        for batch_idx, row in enumerate(gen_outputs["sequences"]):
            pad_idxs = row == self.pad_id
            # Align the sequences with padding on the left
            converted_sequences.append(torch.cat((row[pad_idxs], row[~pad_idxs])))
            gen_prev_sequence_pad_ids.append(pad_idxs)

        # Realign kv cache
        for layer in gen_outputs["past_key_values"]:
            key, value = layer 
            realigned_key, realigned_value = [], []

            # Realign each batch's key and value
            for batch_idx, row in enumerate(gen_outputs["sequences"]):
                values_to_check = torch.tensor(self.step_ids + [self.eos_id], device=row.device)  # Combine step IDs and eos_id
                matches = torch.isin(row, values_to_check)  # Boolean mask for matching elements
                indices = torch.nonzero(matches, as_tuple=True)[0]  # Get indices of matching elements
                last_idx_step_or_eos = indices[-1].item()  # Get the largest index (last match)

                pad_ids_cache = torch.zeros(len(row) - 1, dtype=torch.bool, device=row.device)
                pad_ids_cache[:last_idx_step_or_eos] |= row[:last_idx_step_or_eos] == self.pad_id
                pad_ids_cache[last_idx_step_or_eos:] = True

                realigned_key.append(align_kv_cache_left_paddding(key[batch_idx], pad_ids_cache)) # (H, S, D)
                realigned_value.append(align_kv_cache_left_paddding(value[batch_idx], pad_ids_cache)) # (H, S, D)

            realigned_past_key_values.append((
                torch.stack(realigned_key),
                torch.stack(realigned_value)
            ))

        # Update the gen_outputs dictionary
        gen_outputs["sequences"] = torch.stack(converted_sequences)
        gen_outputs["past_key_values"] = tuple(realigned_past_key_values)
        gen_prev_sequence_pad_ids = torch.stack(gen_prev_sequence_pad_ids)

        return gen_outputs, gen_prev_sequence_pad_ids

    def prepare_prm_inputs(self, output_ids: Tensor, initial_prompt_len: int, gen_prev_sequence_pad_ids: Tensor, prm_past_key_values: Tuple) -> Tensor:
        input_ids = output_ids.clone()

        mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        for i in self.step_ids:
            mask |= input_ids == i
        mask |= input_ids == self.eos_id

        if prm_past_key_values is None: # First reasoning step
            num_pad_prev_step = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device) 
            seq_len_prev = 0
            cache_position = None
            prm_past_key_values = None
        
        else: # Subsequent reasoning steps
            num_pad_prev_step = torch.sum(
                input_ids[:, :prm_past_key_values[0][0].size(2)] == self.pad_id, 
                dim=1
            ) 
            seq_len_prev = prm_past_key_values[0][0].size(2)
            cache_position = torch.arange(seq_len_prev, input_ids.size(1), device=input_ids.device)

            realigned_past_key_values = []
            for layer in prm_past_key_values:
                key, value = layer  # Unpack key and value for this layer
                realigned_key, realigned_value = [], []

                for batch_idx, pad_ids_cache in enumerate(gen_prev_sequence_pad_ids):
                    realigned_key.append(align_kv_cache_left_paddding(key[batch_idx], pad_ids_cache)) # (H, S, D)
                    realigned_value.append(align_kv_cache_left_paddding(value[batch_idx], pad_ids_cache)) # (H, S, D)

                realigned_past_key_values.append((
                    torch.stack(realigned_key),
                    torch.stack(realigned_value)
                ))

            # Update the prm_past_key_values tuple
            prm_past_key_values = tuple(realigned_past_key_values)

        ignore_prefix_lens = (
            num_pad_prev_step + initial_prompt_len
        )

        idxs = torch.arange(input_ids.shape[1], device=input_ids.device)
        idxs = idxs.repeat(len(mask)).view(len(mask), -1)
        response_mask = (idxs.T > ignore_prefix_lens).T

        mask &= response_mask
        input_ids[mask] = self.prm_step_id
        
        prm_inputs = self.prm.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=prm_past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        return prm_inputs
        
    
    def compute_scores(self, logits: Tensor) -> list[float]:
        return logits.softmax(dim=-1)[:, -1, 0].cpu().tolist()

    def __call__(self, question: str) -> str:
        gen_input_ids = self.encode(question)
        initial_prompt_len = gen_input_ids.shape[1]
        
        # start with `number_of_beams` beams initialized to the prompt
        gen_input_ids = gen_input_ids.repeat(self.number_of_beams, 1).to(self.device) 


        for _round in range(self.max_generation_rounds):
            print(f"Round {_round + 1}"+"#"*50) # [TEMP]
            attention_mask = (gen_input_ids != self.pad_id).long()

            print(f"gen input shape: {gen_input_ids.shape}") # [TEMP]

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
            
            print(f"gen output shape: {gen_outputs['sequences'].shape}") # [TEMP]
            
            prm_inputs = self.prepare_prm_inputs(
                gen_outputs["sequences"], 
                initial_prompt_len, 
                gen_prev_sequence_pad_ids if "gen_prev_sequence_pad_ids" in locals() else None, 
                prm_past_key_values if "prm_past_key_values" in locals() else None,
            )
            
            print(f"prm input shape: {prm_inputs['input_ids'].shape}") # [TEMP]
            gen_outputs, gen_prev_sequence_pad_ids = self.convert_to_left_padding(gen_outputs)
            gen_output_ids = gen_outputs["sequences"]
            gen_past_key_values = gen_outputs["past_key_values"]

            print(f"gen kv cache shape: {gen_past_key_values[0][0].shape}") # [TEMP]
            print(f"gen prev sequence pad shape: {gen_prev_sequence_pad_ids.shape}") # [TEMP]

            with torch.no_grad():
                prm_outputs = self.prm(
                    **prm_inputs,
                    return_dict=True,
                )
                prm_logits = prm_outputs.logits[:, :, self.candidate_ids]
                prm_past_key_values = prm_outputs.past_key_values
                scores = self.compute_scores(prm_logits)

                print(f"prm_past_key_values shape: {prm_past_key_values[0][0].shape}") # [TEMP]

            sorted_scored_idxs = sorted(
                enumerate(scores), key=lambda t: t[1], reverse=True
            )
            best_idxs = [i for i, _ in sorted_scored_idxs]
            # best_idxs.reverse() # [TEMP]: to test PRM works correctly
            
            # filter M best beams for both reasoning paths, kv cache for generator
            gen_input_ids = gen_output_ids[best_idxs[:self.width]]
            gen_past_key_values = filter_kv_cache(gen_past_key_values, best_idxs[:self.width])
            prm_past_key_values = filter_kv_cache(prm_past_key_values, best_idxs[:self.width])

            if _round == self.max_generation_rounds - 1:
                break
            
            # inflating batch to N (by multiplying N/M times for each) 
            # (Note: num_return_sequences=N/M without separately inflating input could be an option, but I'm not sure if it works correctly with kv cache)
            inflator = int(self.number_of_beams/self.width)
            gen_input_ids = gen_input_ids.repeat(inflator, 1)
            gen_past_key_values = inflate_kv_cache(gen_past_key_values, inflator)
            prm_past_key_values = inflate_kv_cache(prm_past_key_values, inflator)


        return [self.tokenizer.decode(x) for x in gen_input_ids]

def align_kv_cache_left_paddding(tensor: Tensor, pad_ids_cache: Tensor) -> Tuple:
    """
    tensor: (H, S, D) or (S,)
    pad_ids_cache: (S,)
    """
    return torch.cat((tensor[:, pad_ids_cache, ...], 
                        tensor[:, ~pad_ids_cache, ...]), dim=1) # (H, S, D)

def inflate_kv_cache(past_key_values: Tuple, inflator: int) -> Tuple:
    return tuple(
        (
            k.repeat(inflator, *([1] * (k.ndim - 1))),
            v.repeat(inflator, *([1] * (v.ndim - 1)))
        )
        for k, v in past_key_values
    )

def filter_kv_cache(past_key_values: Tuple, best_idxs: list[int]) -> Tuple:
    return tuple(
        (
            k[best_idxs, ...],
            v[best_idxs, ...]
        )
        for k, v in past_key_values
    )

def main():
    outputs = BeamSearch(width=2, number_of_beams=4, max_generation_rounds=7)(problem)

    for output in outputs:
        print(output)

if __name__ == "__main__":
    main()