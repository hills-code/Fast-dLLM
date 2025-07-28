import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor

# 在文件开头添加颜色常量
MASK_COLOR = "#FF4444"  # 红色用于mask
TOKEN_COLOR = "#44FF44"  # 绿色用于已解码的token

# 检查可用的GPU
device_baseline = 'cuda:4' if torch.cuda.is_available() and torch.cuda.device_count() > 4 else 'cuda' if torch.cuda.is_available() else 'cpu'
device_accelerated = 'cuda:5' if torch.cuda.is_available() and torch.cuda.device_count() > 5 else 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Baseline model using device: {device_baseline}")
print(f"Accelerated model using device: {device_accelerated}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

# 加载两个模型实例到不同的GPU
model_baseline = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                  torch_dtype=torch.bfloat16).to(device_baseline)

model_accelerated = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                  torch_dtype=torch.bfloat16).to(device_accelerated)

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 126336  # The token ID of [MASK] in LLaDA
question_ai = '''Write a piece of code to implement quick sort.'''
# question_poem = '''莲动下渔舟上一句是什么，介绍一下这首诗。'''

question_gsm8k = '''Question: Skyler has 100 hats on his hand with the colors red, blue, and white. Half of the hats are red, 3/5 of the remaining hats are blue, and the rest are white. How many white hats does Skyler have?'''

# Removed parse_constraints function - no longer needed

def format_chat_history(history):
    """
    Format chat history for the LLaDA model
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature <= 0:
        return logits
        
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

# 添加线程生成函数
def generate_baseline_thread(messages, gen_length, steps, temperature, block_length, remasking):
    """在线程中生成baseline模型的响应"""
    with torch.no_grad():
        return generate_response_with_visualization(
            model_baseline, tokenizer, device_baseline,
            messages, gen_length, steps, temperature, block_length, remasking
        )

def generate_accelerated_thread(messages, gen_length, steps, temperature, block_length, remasking, threshold):
    """在线程中生成加速模型的响应"""
    with torch.no_grad():
        return generate_response_with_visualization_cache_and_parallel(
            model_accelerated, tokenizer, device_accelerated,
            messages, gen_length, steps, temperature, block_length, remasking, threshold
        )

@torch.no_grad()
def generate_response_with_visualization_cache_and_parallel(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         temperature=0.0, block_length=32,
                                         remasking='low_confidence', threshold=0.9):
    """
    Generate text with LLaDA model with visualization using the same sampling as in generate.py
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        temperature: Sampling temperature
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Returns:
        List of visualization states showing the progression and final text
    """
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response part
    visualization_states = []
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, MASK_COLOR) for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Constraints functionality removed
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    # Process each block
    for num_block in range(num_blocks):
        current_block_start = prompt_length + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == MASK_ID)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        # Create visualization state only for the response part
        current_state = []
        for i in range(gen_length):
            pos = prompt_length + i  # Absolute position in the sequence
            
            if x[0, pos] == MASK_ID:
                # Still masked
                current_state.append((MASK_TOKEN, MASK_COLOR))  # 红色用于mask
            else:
                # Previously revealed
                token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                current_state.append((token, TOKEN_COLOR))  # 绿色用于已解码的token
        
        visualization_states.append(current_state)
        i = 1
        while True:
            mask_index = (x[:, current_block_start:] == MASK_ID)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            # Create visualization state only for the response part
            current_state = []
            for i in range(gen_length):
                pos = prompt_length + i  # Absolute position in the sequence
                
                if x[0, pos] == MASK_ID:
                    # Still masked
                    current_state.append((MASK_TOKEN, MASK_COLOR))  # 红色用于mask
                else:
                    # Previously revealed
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    current_state.append((token, TOKEN_COLOR))  # 绿色用于已解码的token
            
            visualization_states.append(current_state)
            if (x[:, current_block_start:current_block_end] == MASK_ID).sum() == 0:
                break
            i += 1
    
    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, 
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

@torch.no_grad()
def generate_response_with_visualization(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         temperature=0.0, block_length=32,
                                         remasking='low_confidence'):
    """
    Generate text with LLaDA model with visualization using the same sampling as in generate.py
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        temperature: Sampling temperature
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Returns:
        List of visualization states showing the progression and final text
    """
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response part
    visualization_states = []
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, MASK_COLOR) for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Constraints functionality removed
    
    # Mark prompt positions to exclude them from masking during classifier-free guidance
    prompt_index = (x != MASK_ID)
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    # Process each block
    for num_block in range(num_blocks):
        # Calculate the start and end indices for the current block
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
        
        # Get mask indices for the current block
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        
        # Skip if no masks in this block
        if not block_mask_index.any():
            continue
        
        # Calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        # Process each step
        for i in range(steps_per_block):
            # Get all mask positions in the current sequence
            mask_index = (x == MASK_ID)
            
            # Skip if no masks
            if not mask_index.any():
                break
            
            # Get logits from model
            logits = model(x).logits
            
            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Calculate confidence scores for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            
            # Don't consider positions beyond the current block
            x0_p[:, block_end:] = -float('inf')
            
            # Apply predictions where we have masks
            old_x = x.clone()
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))
            
            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                # Only consider positions within the current block for unmasking
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:  # Not the last step
                    # Take top-k confidences
                    _, select_indices = torch.topk(block_confidence, 
                                                  k=min(num_transfer_tokens[j, i].item(), 
                                                       block_confidence.numel()))
                    # Adjust indices to global positions
                    select_indices = select_indices + block_start
                    transfer_index[j, select_indices] = True
                else:  # Last step - unmask everything remaining
                    transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]
            
            # Apply the selected tokens
            x = torch.where(transfer_index, x0, x)
            
            # Constraints functionality removed
            
            # Create visualization state only for the response part
            current_state = []
            for i in range(gen_length):
                pos = prompt_length + i  # Absolute position in the sequence
                
                if x[0, pos] == MASK_ID:
                    # Still masked
                    current_state.append((MASK_TOKEN, MASK_COLOR))  # 红色用于mask
                else:
                    # Previously revealed
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    current_state.append((token, TOKEN_COLOR))  # 绿色用于已解码的token
            
            visualization_states.append(current_state)
    
    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, 
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text

css = '''
.category-legend{display:none}
.message, .bubble, .chatbot .message, .chatbot .bubble {
    max-width: 80% !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    box-sizing: border-box !important;
}
'''
def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding")
        gr.Markdown("[code](https://github.com/NVlabs/Fast-dLLM), [project page](https://nvlabs.github.io/Fast-dLLM/)")
        
        # STATE MANAGEMENT
        chat_history_baseline = gr.State([])
        chat_history_cache = gr.State([])
        
        # UI COMPONENTS
                
        # Duplicate conversation interface
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui_copy = gr.Chatbot(label="Conversation (Accelerated)", height=500)
            with gr.Column(scale=2):
                output_vis_copy = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
                generation_time_copy = gr.Textbox(
                    label="Generation Time",
                    value="wait for generation",
                    interactive=False
                )
                throughput_copy = gr.Textbox(
                    label="Generation Speed",
                    value="wait for generation",
                    interactive=False
                )
        
        # Add separator line
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=500)
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
                generation_time = gr.Textbox(
                    label="Generation Time",
                    value="wait for generation",
                    interactive=False
                )
                throughput = gr.Textbox(
                    label="Generation Speed",
                    value="wait for generation",
                    interactive=False
                )
                
        # Move input area below the duplicate conversation interface
        with gr.Group():
            user_input = gr.Textbox(
                label="Your Message", 
                placeholder="Type your message here...",
                show_label=False
            )
            send_btn = gr.Button("Send")
            gr.Examples(
                examples=[
                    [question_ai],
                    # [question_poem],
                    [question_gsm8k]
                ],
                inputs=user_input,
                label="Example Inputs"
            )
        
        # Advanced generation settings
        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=64,
                    label="Generation Length"
                )
                steps = gr.Slider(
                    minimum=8, maximum=1024, value=256, step=4,
                    label="Denoising Steps"
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                    label="Temperature"
                )
                threshold = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.9, step=0.1,
                    label="Threshold"
                )
            with gr.Row():
                block_length = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="Block Length"
                )
                remasking_strategy = gr.Radio(
                    choices=["low_confidence", "random"],
                    value="low_confidence",
                    label="Remasking Strategy"
                )
            with gr.Row():
                visualization_delay = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                    label="Visualization Delay (seconds)"
                )
        
        # Current response text box (hidden)
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False
        )
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        
        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append([message, response])
            return history
            
        def user_message_submitted(message, history_baseline, history_cache, gen_length, steps, delay):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_baseline_for_display = history_baseline.copy()
                history_cache_for_display = history_cache.copy()
                return history_baseline, history_cache, history_baseline_for_display, history_cache_for_display, "", [], [], "", "wait for generation", "wait for generation", "wait for generation", "wait for generation"
                
            # Add user message to both histories
            history_baseline = add_message(history_baseline, message, None)
            history_cache = add_message(history_cache, message, None)
            
            # Format for display - temporarily show user message with empty response
            history_baseline_for_display = history_baseline.copy()
            history_cache_for_display = history_cache.copy()
            
            # Clear the input
            message_out = ""
            
            # Return immediately to update UI with user message
            return history_baseline, history_cache, history_baseline_for_display, history_cache_for_display, message_out, [], [], "", "processing...", "processing...", "processing...", "processing..."
            
        def baseline_response(history_baseline, gen_length, steps, delay, temperature, block_length, remasking):
            """Generate baseline model response independently"""
            if not history_baseline:
                return history_baseline, [], "", "wait for generation", "wait for generation"
                
            # Get the last user message
            last_user_message = history_baseline[-1][0]
            
            try:
                # Format all messages except the last one (which has no response yet)
                messages = format_chat_history(history_baseline[:-1])
                
                # Add the last user message
                messages.append({"role": "user", "content": last_user_message})
                
                # Start timing
                start_time = time.time()
                
                # Generate with baseline model
                with torch.no_grad():
                    vis_states, response_text = generate_response_with_visualization(
                        model_baseline, tokenizer, device_baseline,
                        messages, gen_length, steps, temperature, block_length, remasking
                    )
                
                baseline_complete_time = time.time() - start_time
                generation_time_str = f"{baseline_complete_time:.2f}s"
                
                # Calculate throughput
                response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
                num_tokens = len(response_tokens)
                throughput = num_tokens / baseline_complete_time if baseline_complete_time > 0 else 0
                throughput_str = f"{throughput:.2f} tokens/s"
                
                # Update history
                history_baseline[-1][1] = response_text
                
                # Output results
                yield history_baseline, vis_states[0], response_text, generation_time_str, throughput_str
                
                # Animate generation process
                for state in vis_states[1:]:
                    time.sleep(delay)
                    yield history_baseline, state, response_text, generation_time_str, throughput_str
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                error_vis = [(error_msg, "red")]
                yield history_baseline, error_vis, error_msg, "Error", "Error"
        
        def accelerated_response(history_cache, gen_length, steps, delay, temperature, block_length, remasking, threshold):
            """Generate accelerated model response independently"""
            if not history_cache:
                return history_cache, [], "", "wait for generation", "wait for generation"
                
            # Get the last user message
            last_user_message = history_cache[-1][0]
            
            try:
                # Format all messages except the last one (which has no response yet)
                messages = format_chat_history(history_cache[:-1])
                
                # Add the last user message
                messages.append({"role": "user", "content": last_user_message})
                
                # Start timing
                start_time = time.time()
                
                # Generate with accelerated model
                with torch.no_grad():
                    cache_vis_states, cache_response_text = generate_response_with_visualization_cache_and_parallel(
                        model_accelerated, tokenizer, device_accelerated,
                        messages, gen_length, steps, temperature, block_length, remasking, threshold
                    )
                
                accelerated_complete_time = time.time() - start_time
                cache_generation_time_str = f"{accelerated_complete_time:.2f}s"
                
                # Calculate throughput
                cache_response_tokens = tokenizer.encode(cache_response_text, add_special_tokens=False)
                cache_num_tokens = len(cache_response_tokens)
                cache_throughput = cache_num_tokens / accelerated_complete_time if accelerated_complete_time > 0 else 0
                cache_throughput_str = f"{cache_throughput:.2f} tokens/s"
                
                # Update history
                history_cache[-1][1] = cache_response_text
                
                # Output results
                yield history_cache, cache_vis_states[0], cache_response_text, cache_generation_time_str, cache_throughput_str
                
                # Animate generation process
                for state in cache_vis_states[1:]:
                    time.sleep(delay)
                    yield history_cache, state, cache_response_text, cache_generation_time_str, cache_throughput_str
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                error_vis = [(error_msg, "red")]
                yield history_cache, error_vis, error_msg, "Error", "Error"
        
        def clear_conversation():
            """Clear the conversation history"""
            empty_history = []
            empty_response = ""
            empty_vis = []
            time_str = "wait for generation"
            throughput_str = "wait for generation"
            
            return (
                empty_history,  # chat_history_baseline
                empty_history,  # chat_history_cache
                empty_history,  # chatbot_ui
                empty_history,  # chatbot_ui_copy
                empty_response,  # current_response
                empty_vis,      # output_vis
                time_str,       # generation_time
                throughput_str, # throughput
                empty_vis,      # output_vis_copy
                time_str,       # generation_time_copy
                throughput_str  # throughput_copy
            )
        
        # EVENT HANDLERS
        
        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history_baseline, chat_history_cache, chatbot_ui, chatbot_ui_copy, current_response, output_vis, generation_time, throughput, output_vis_copy, generation_time_copy, throughput_copy]
        )
        
        # User message submission flow (2-step process)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_baseline, chat_history_cache, gen_length, steps, visualization_delay],
            outputs=[chat_history_baseline, chat_history_cache, chatbot_ui, chatbot_ui_copy, user_input, output_vis, output_vis_copy, current_response, generation_time, throughput, generation_time_copy, throughput_copy]
        )
        
        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_baseline, chat_history_cache, gen_length, steps, visualization_delay],
            outputs=[chat_history_baseline, chat_history_cache, chatbot_ui, chatbot_ui_copy, user_input, output_vis, output_vis_copy, current_response, generation_time, throughput, generation_time_copy, throughput_copy]
        )
        
        # Step 2: Generate bot responses independently
        # Baseline model response
        msg_submit.then(
            fn=baseline_response,
            inputs=[
                chat_history_baseline, gen_length, steps, 
                visualization_delay, temperature, block_length, remasking_strategy
            ],
            outputs=[chatbot_ui, output_vis, current_response, generation_time, throughput]
        )
        
        send_click.then(
            fn=baseline_response,
            inputs=[
                chat_history_baseline, gen_length, steps, 
                visualization_delay, temperature, block_length, remasking_strategy
            ],
            outputs=[chatbot_ui, output_vis, current_response, generation_time, throughput]
        )
        
        # Accelerated model response
        msg_submit.then(
            fn=accelerated_response,
            inputs=[
                chat_history_cache, gen_length, steps, 
                visualization_delay, temperature, block_length, remasking_strategy, threshold
            ],
            outputs=[chatbot_ui_copy, output_vis_copy, current_response, generation_time_copy, throughput_copy]
        )
        
        send_click.then(
            fn=accelerated_response,
            inputs=[
                chat_history_cache, gen_length, steps, 
                visualization_delay, temperature, block_length, remasking_strategy, threshold
            ],
            outputs=[chatbot_ui_copy, output_vis_copy, current_response, generation_time_copy, throughput_copy]
        )
        
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_chatbot_demo()
    demo.queue().launch(server_port=10086)